import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from unet import ProbUNet
from analysis.paths import hadgem, weights


parser = argparse.ArgumentParser()
parser.add_argument("--test-ppe", choices=["GA7", "GA8", "GA9"], required=True)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument(
    "--input-path",
    default=str(hadgem / "GA789_PR_bilinear_rg128.nc"),
)
parser.add_argument(
    "--target-path",
    default=str(hadgem / "GA789_dPdK_bilinear_rg128.nc"),
)
parser.add_argument(
    "--weights-path",
    default=str(weights / "direct_dpdk_bilinear_cv"),
)
parser.add_argument("--legacy-100-channel", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

inputpath = Path(args.input_path)
targetpath = Path(args.target_path)
weightspath = Path(args.weights_path)
landpath = hadgem / "hadgem_landmask_rg128.nc"

channels = args.channels
dropout = args.dropout
kernel = 3
groups = 1
bins = 64
minimum = -700.0
maximum = 1200.0

if args.legacy_100_channel:
    channels = 100
    dropout = 0.0
    modelname = (
        f"unet_direct_dpdk_cv_flat_ch{channels}_k{kernel}_bins{bins}_"
        f"min{minimum:g}_max{maximum:g}"
    )
else:
    modelname = (
        f"unet_direct_dpdk_cv_flat_ch{channels}_k{kernel}_bins{bins}_"
        f"min{minimum:g}_max{maximum:g}_sigma0.6_dropout{dropout:g}"
    )
foldpath = weightspath / modelname / f"fold_{args.test_ppe}"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
print(f"Held out PPE: {args.test_ppe}")

with xr.open_dataset(inputpath) as dataset:
    inputs = dataset["PR"].values.astype(np.float32)[:, None]
    inputlabels = dataset.realization.values.astype(str)
    latitude = dataset.latitude.values

with xr.open_dataset(targetpath) as dataset:
    targets = dataset["dPdK"].values.astype(np.float32)
    targetlabels = dataset.realization.values.astype(str)

with xr.open_dataset(landpath) as dataset:
    landmask = dataset["land_mask"].values.astype(bool)

if not np.array_equal(inputlabels, targetlabels):
    raise ValueError("Input and target realization labels do not match")

splits = np.load(foldpath / "data_splits.npz")
trainindices = splits["train"]
validationindices = splits["val"]
testindices = splits["test"]

with open(foldpath / "norm_stats.json") as file:
    normalization = json.load(file)

with open(foldpath / "born_bins.json") as file:
    bininformation = json.load(file)

inputmean = float(normalization["x_mean"][0])
inputstd = float(normalization["x_std"][0])
targetmean = float(normalization["y_mean"])
targetstd = float(normalization["y_std"])
normalizedcenters = np.asarray(
    bininformation["bin_centers_norm"], dtype=np.float32
)

modelfiles = sorted(foldpath.glob(f"{modelname}_member*.pth"))
if len(modelfiles) != 3:
    raise ValueError(f"Expected 3 final models, found {len(modelfiles)}")

latitudeweights = np.cos(np.deg2rad(latitude)).astype(np.float32)
latitudeweights /= latitudeweights.mean()
landdenominator = float((landmask * latitudeweights[:, None]).sum())
normalizedinputs = (inputs - inputmean) / inputstd
centertensor = torch.as_tensor(
    normalizedcenters, dtype=torch.float32, device=device
)[None, :, None, None]


def loadmodel(path):
    model = ProbUNet(
        1, channels, kernel, dropout, bins, gn_groups=groups
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict(model, indices):
    prediction = np.empty(
        (len(indices), targets.shape[1], targets.shape[2]), dtype=np.float32
    )
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_size):
            stop = min(start + args.batch_size, len(indices))
            batchindices = indices[start:stop]
            inputbatch = torch.as_tensor(
                normalizedinputs[batchindices], dtype=torch.float32, device=device
            )
            probabilities = model(inputbatch)
            mean = (probabilities * centertensor).sum(dim=1)
            prediction[start:stop] = (mean * targetstd + targetmean).cpu().numpy()
    return prediction


def globalrmse(prediction, truth):
    squarederror = (prediction - truth) ** 2
    return np.sqrt(
        (squarederror * latitudeweights[None, :, None]).mean(axis=(1, 2))
    )


def landrmse(prediction, truth):
    squarederror = (prediction - truth) ** 2
    numerator = (
        squarederror * landmask[None] * latitudeweights[None, :, None]
    ).sum(axis=(1, 2))
    return np.sqrt(numerator / landdenominator)


if args.dry_run:
    model = loadmodel(modelfiles[0])
    prediction = predict(model, validationindices[:2])
    print(f"Dry run successful. Prediction shape: {prediction.shape}")
    raise SystemExit(0)

validationrmse = []
for seed, path in enumerate(modelfiles):
    print(f"Checking seed {seed} on the validation set", flush=True)
    model = loadmodel(path)
    prediction = predict(model, validationindices)
    validationrmse.append(
        float(globalrmse(prediction, targets[validationindices]).mean())
    )
    del model, prediction
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

validationrmse = np.asarray(validationrmse)
if not np.all(np.isfinite(validationrmse)):
    raise RuntimeError("At least one completed seed has nonfinite validation RMSE")

goodseeds = np.arange(3)
print("Including all 3 completed seeds")

testseedglobalrmse = np.empty((3, len(testindices)), dtype=np.float32)
testseedlandrmse = np.empty((3, len(testindices)), dtype=np.float32)
predictiontotal = np.zeros_like(targets[testindices], dtype=np.float32)

for seed, path in enumerate(modelfiles):
    print(f"Evaluating seed {seed} on held out {args.test_ppe}", flush=True)
    model = loadmodel(path)
    prediction = predict(model, testindices)
    testseedglobalrmse[seed] = globalrmse(
        prediction, targets[testindices]
    )
    testseedlandrmse[seed] = landrmse(prediction, targets[testindices])
    if seed in goodseeds:
        predictiontotal += prediction
    del model, prediction
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

seedmeanprediction = predictiontotal / len(goodseeds)
seedmeanglobalrmse = globalrmse(seedmeanprediction, targets[testindices])
seedmeanlandrmse = landrmse(seedmeanprediction, targets[testindices])

baseline = targets[trainindices].mean(axis=0)
baselineprediction = np.broadcast_to(baseline, targets[testindices].shape)
baselineglobalrmse = globalrmse(baselineprediction, targets[testindices])
baselinelandrmse = landrmse(baselineprediction, targets[testindices])

globalimprovement = (1 - seedmeanglobalrmse / baselineglobalrmse) * 100
landimprovement = (1 - seedmeanlandrmse / baselinelandrmse) * 100

summary = {
    "held_out_ppe": args.test_ppe,
    "channels": channels,
    "dropout": dropout,
    "members": int(len(testindices)),
    "good_seeds": goodseeds.tolist(),
    "excluded_seeds": np.setdiff1d(np.arange(3), goodseeds).tolist(),
    "validation_seed_rmse": validationrmse.tolist(),
    "baseline_global_rmse": float(baselineglobalrmse.mean()),
    "seed_mean_global_rmse": float(seedmeanglobalrmse.mean()),
    "mean_global_improvement_percent": float(
        100 * (1 - seedmeanglobalrmse.mean() / baselineglobalrmse.mean())
    ),
    "median_global_improvement_percent": float(np.median(globalimprovement)),
    "baseline_land_rmse": float(baselinelandrmse.mean()),
    "seed_mean_land_rmse": float(seedmeanlandrmse.mean()),
    "mean_land_improvement_percent": float(
        100 * (1 - seedmeanlandrmse.mean() / baselinelandrmse.mean())
    ),
    "median_land_improvement_percent": float(np.median(landimprovement)),
}

with open(foldpath / "test_results.json", "w") as file:
    json.dump(summary, file, indent=2)

np.savez_compressed(
    foldpath / "test_results.npz",
    good_seeds=goodseeds,
    validation_seed_rmse=validationrmse,
    test_seed_global_rmse=testseedglobalrmse,
    test_seed_land_rmse=testseedlandrmse,
    seed_mean_global_rmse=seedmeanglobalrmse,
    seed_mean_land_rmse=seedmeanlandrmse,
    baseline_global_rmse=baselineglobalrmse,
    baseline_land_rmse=baselinelandrmse,
    train_indices=trainindices,
    validation_indices=validationindices,
    test_indices=testindices,
)

print(json.dumps(summary, indent=2))

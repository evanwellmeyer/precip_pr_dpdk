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
parser.add_argument("--channels", type=int, required=True)
parser.add_argument("--dropout", type=float, required=True)
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
    default=str(weights / "direct_dpdk_bilinear"),
)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

inputpath = Path(args.input_path)
targetpath = Path(args.target_path)
weightspath = Path(args.weights_path)
landpath = hadgem / "hadgem_landmask_rg128.nc"

kernel = 3
groups = 1
bins = 64
minimum = -700.0
maximum = 1200.0
sigmascale = 0.6

dropouttext = f"{args.dropout:g}"
modelname = (
    f"unet_direct_dpdk_flat_ch{args.channels}_k{kernel}_bins{bins}_"
    f"min{minimum:g}_max{maximum:g}_sigma{sigmascale:g}_dropout{dropouttext}"
)
modelpath = weightspath / modelname
resultpath = modelpath / "test_results.npz"
summarypath = modelpath / "test_results.json"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
print(f"Run: {modelname}")

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

splits = np.load(modelpath / "data_splits.npz")
trainindices = splits["train"]
validationindices = splits["val"]
testindices = splits["test"]

with open(modelpath / "norm_stats.json") as file:
    normalization = json.load(file)

with open(modelpath / "born_bins.json") as file:
    bininformation = json.load(file)

inputmean = float(normalization["x_mean"][0])
inputstd = float(normalization["x_std"][0])
targetmean = float(normalization["y_mean"])
targetstd = float(normalization["y_std"])
normalizedcenters = np.asarray(
    bininformation["bin_centers_norm"], dtype=np.float32
)

modelfiles = sorted(modelpath.glob(f"{modelname}_member*.pth"))
if len(modelfiles) != 10:
    raise ValueError(f"Expected 10 final models, found {len(modelfiles)}")

latitudeweights = np.cos(np.deg2rad(latitude)).astype(np.float32)
latitudeweights /= latitudeweights.mean()
landdenominator = float((landmask * latitudeweights[:, None]).sum())
normalizedinputs = (inputs - inputmean) / inputstd
centertensor = torch.as_tensor(
    normalizedcenters, dtype=torch.float32, device=device
)[None, :, None, None]


def loadmodel(path):
    model = ProbUNet(
        1,
        args.channels,
        kernel,
        args.dropout,
        bins,
        gn_groups=groups,
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
            mean = mean * targetstd + targetmean
            prediction[start:stop] = mean.cpu().numpy()
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

# The validation set is used only to check for failed or unusually poor seeds.
validationseedrmse = []
for seed, path in enumerate(modelfiles):
    print(f"Checking seed {seed} on the validation set", flush=True)
    model = loadmodel(path)
    prediction = predict(model, validationindices)
    validationseedrmse.append(
        float(globalrmse(prediction, targets[validationindices]).mean())
    )
    del model, prediction
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

validationseedrmse = np.asarray(validationseedrmse)
if not np.all(np.isfinite(validationseedrmse)):
    raise RuntimeError("At least one completed seed has nonfinite validation RMSE")

goodseeds = np.arange(10)
print("Including all 10 completed seeds")

# Each seed is loaded separately so model memory is released before the next seed.
seedglobalrmse = np.full((10, len(targets)), np.nan, dtype=np.float32)
seedlandrmse = np.full((10, len(targets)), np.nan, dtype=np.float32)
predictiontotal = np.zeros_like(targets, dtype=np.float32)

for seed in range(10):
    print(f"Evaluating seed {seed} on all members", flush=True)
    model = loadmodel(modelfiles[seed])
    prediction = predict(model, np.arange(len(targets)))
    seedglobalrmse[seed] = globalrmse(prediction, targets)
    seedlandrmse[seed] = landrmse(prediction, targets)
    if seed in goodseeds:
        predictiontotal += prediction
    del model, prediction
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

seedmeanprediction = predictiontotal / len(goodseeds)
seedmeanglobalrmse = globalrmse(seedmeanprediction, targets)
seedmeanlandrmse = landrmse(seedmeanprediction, targets)

baseline = targets[trainindices].mean(axis=0)
baselineprediction = np.broadcast_to(baseline, targets.shape)
baselineglobalrmse = globalrmse(baselineprediction, targets)
baselinelandrmse = landrmse(baselineprediction, targets)


def summarize(indices):
    baselineglobal = float(baselineglobalrmse[indices].mean())
    baselineland = float(baselinelandrmse[indices].mean())
    modelglobal = float(seedmeanglobalrmse[indices].mean())
    modelland = float(seedmeanlandrmse[indices].mean())
    return {
        "members": int(len(indices)),
        "baseline_global_rmse": baselineglobal,
        "seed_mean_global_rmse": modelglobal,
        "global_improvement_percent": 100 * (1 - modelglobal / baselineglobal),
        "baseline_land_rmse": baselineland,
        "seed_mean_land_rmse": modelland,
        "land_improvement_percent": 100 * (1 - modelland / baselineland),
    }


summary = {
    "channels": args.channels,
    "dropout": args.dropout,
    "target_file": targetpath.name,
    "good_seeds": goodseeds.tolist(),
    "excluded_seeds": np.setdiff1d(np.arange(10), goodseeds).tolist(),
    "validation_seed_rmse": validationseedrmse.tolist(),
    "train": summarize(trainindices),
    "validation": summarize(validationindices),
    "test": summarize(testindices),
}

with open(summarypath, "w") as file:
    json.dump(summary, file, indent=2)

np.savez_compressed(
    resultpath,
    good_seeds=goodseeds,
    validation_seed_rmse=validationseedrmse,
    seed_global_rmse=seedglobalrmse,
    seed_land_rmse=seedlandrmse,
    seed_mean_global_rmse=seedmeanglobalrmse,
    seed_mean_land_rmse=seedmeanlandrmse,
    baseline_global_rmse=baselineglobalrmse,
    baseline_land_rmse=baselinelandrmse,
    train_indices=trainindices,
    validation_indices=validationindices,
    test_indices=testindices,
    latitude_weights=latitudeweights,
    landmask=landmask,
)

print(json.dumps(summary, indent=2))
print(f"Saved {summarypath}")
print(f"Saved {resultpath}")

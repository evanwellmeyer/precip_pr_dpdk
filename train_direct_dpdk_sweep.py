import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from unet import ProbUNet
from analysis.paths import hadgem, weights


parser = argparse.ArgumentParser()
parser.add_argument("--channels", type=int, required=True)
parser.add_argument("--dropout", type=float, required=True)
parser.add_argument("--first-seed", type=int, default=0)
parser.add_argument("--number-of-seeds", type=int, default=10)
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
parser.add_argument(
    "--split-path",
    default=(
        str(Path(__file__).resolve().parent / "data_splits.npz")
    ),
)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

inputpath = Path(args.input_path)
targetpath = Path(args.target_path)
weightspath = Path(args.weights_path)
splitpath = Path(args.split_path)

baseseed = 42
groups = 1
kernel = 3
bins = 64
minimum = -700.0
maximum = 1200.0
sigmascale = 0.6
epochs = 5000
patience = 25
gradientclip = 1.0
learningrate = 1e-3

dropouttext = f"{args.dropout:g}"
modelname = (
    f"unet_direct_dpdk_flat_ch{args.channels}_k{kernel}_bins{bins}_"
    f"min{minimum:g}_max{maximum:g}_sigma{sigmascale:g}_dropout{dropouttext}"
)
modelpath = weightspath / modelname

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
print(f"Run: {modelname}")

with xr.open_dataset(inputpath) as dataset:
    inputs = dataset["PR"].values.astype(np.float32)[:, None]
    labels = dataset.realization.values.astype(str)
with xr.open_dataset(targetpath) as dataset:
    targets = dataset["dPdK"].values.astype(np.float32)[:, None]
    targetlabels = dataset.realization.values.astype(str)

if not np.array_equal(labels, targetlabels):
    raise ValueError("Input and target realization labels do not match")

splits = np.load(splitpath)
trainindices = splits["train"]
validationindices = splits["val"]
testindices = splits["test"]
if len(trainindices) != 1060 or len(validationindices) != 151 or len(testindices) != 304:
    raise ValueError("The shared main split has unexpected sizes")

inputmean = inputs[trainindices].mean(dtype=np.float64)
inputstd = inputs[trainindices].std(dtype=np.float64)
targetmean = targets[trainindices].mean(dtype=np.float64)
targetstd = targets[trainindices].std(dtype=np.float64)

traininputs = (inputs[trainindices] - inputmean) / inputstd
validationinputs = (inputs[validationindices] - inputmean) / inputstd
traintargets = (targets[trainindices] - targetmean) / targetstd
validationtargets = (targets[validationindices] - targetmean) / targetstd
validationphysical = torch.from_numpy(targets[validationindices])

physicalcenters = np.linspace(minimum, maximum, bins, dtype=np.float32)
normalizedcenters = ((physicalcenters - targetmean) / targetstd).astype(np.float32)
spacing = float(normalizedcenters[1] - normalizedcenters[0])
normalizedsigma = np.full(bins, spacing * sigmascale, dtype=np.float32)


class climatedataset(torch.utils.data.Dataset):
    def __init__(self, inputvalues, targetvalues):
        self.inputvalues = torch.from_numpy(inputvalues.astype(np.float32))
        self.targetvalues = torch.from_numpy(targetvalues.astype(np.float32))

    def __len__(self):
        return len(self.inputvalues)

    def __getitem__(self, index):
        return self.inputvalues[index], self.targetvalues[index]


def makelabels(targetvalues, centers, sigmas):
    targetvalues = targetvalues.clamp(
        min=float(centers.min()), max=float(centers.max())
    )
    distance = targetvalues - centers
    labels = torch.exp(-0.5 * (distance / sigmas) ** 2)
    return labels / labels.sum(dim=1, keepdim=True).clamp_min(1e-12)


def negativeloglikelihood(probabilities, labels):
    return -(labels * probabilities.clamp_min(1e-6).log()).sum(dim=1).mean()


modelpath.mkdir(parents=True, exist_ok=True)
np.savez(
    modelpath / "data_splits.npz",
    train=trainindices,
    val=validationindices,
    test=testindices,
)
with open(modelpath / "norm_stats.json", "w") as file:
    json.dump(
        {
            "x_mean": [float(inputmean)],
            "x_std": [float(inputstd)],
            "y_mean": float(targetmean),
            "y_std": float(targetstd),
            "input_file": str(inputpath),
            "target_file": str(targetpath),
            "target_definition": "(warming precipitation - present precipitation) / global mean temperature change",
        },
        file,
        indent=2,
    )
with open(modelpath / "born_bins.json", "w") as file:
    json.dump(
        {
            "num_bins": bins,
            "bin_centers_norm": normalizedcenters.tolist(),
            "bin_centers_dP": physicalcenters.tolist(),
            "sigma_bins_norm": normalizedsigma.tolist(),
            "definition": "uniform_direct_dPdK",
            "dP_min": minimum,
            "dP_max": maximum,
            "sigma_scale": sigmascale,
        },
        file,
        indent=2,
    )

print(
    f"Split: {len(trainindices)} train, {len(validationindices)} validation, "
    f"{len(testindices)} test"
)
print(f"Target training mean and standard deviation: {targetmean:.4f}, {targetstd:.4f}")

centertensor = torch.as_tensor(
    normalizedcenters, dtype=torch.float32, device=device
)[None, :, None, None]
sigmatensor = torch.as_tensor(
    normalizedsigma, dtype=torch.float32, device=device
)[None, :, None, None]

traindataset = climatedataset(traininputs, traintargets)
validationloader = DataLoader(
    climatedataset(validationinputs, validationtargets),
    batch_size=args.batch_size,
    shuffle=False,
)

if args.dry_run:
    model = ProbUNet(
        1, args.channels, kernel, args.dropout, bins, gn_groups=groups
    ).to(device)
    inputbatch, targetbatch = next(iter(validationloader))
    inputbatch = inputbatch.to(device)
    targetbatch = targetbatch.to(device)
    with torch.inference_mode():
        probabilities = model(inputbatch)
        softlabels = makelabels(targetbatch, centertensor, sigmatensor)
        loss = negativeloglikelihood(probabilities, softlabels)
    print(f"Dry run successful. Output shape: {tuple(probabilities.shape)}, loss: {loss.item():.5f}")
    raise SystemExit(0)

for member in range(args.first_seed, args.first_seed + args.number_of_seeds):
    seed = baseseed + member
    finalpath = modelpath / f"{modelname}_member{member}.pth"
    bestpath = modelpath / f"best_member{member}.pth"

    if finalpath.exists():
        print(f"Seed {member} is complete; skipping")
        continue

    print(f"Training seed {member} with random seed {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    generator = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        traindataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = ProbUNet(
        1, args.channels, kernel, args.dropout, bins, gn_groups=groups
    ).to(device)
    optimizer = optim.RAdam(model.parameters(), lr=learningrate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    bestloss = float("inf")
    firstepoch = 1
    if bestpath.exists():
        checkpoint = torch.load(bestpath, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        bestloss = float(checkpoint["best_validation_loss"])
        firstepoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming from epoch {firstepoch}")

    epochswithoutimprovement = 0
    for epoch in range(firstepoch, epochs + 1):
        model.train()
        trainingloss = 0.0
        trainingcount = 0

        for inputbatch, targetbatch in trainloader:
            inputbatch = inputbatch.to(device)
            targetbatch = targetbatch.to(device)
            softlabels = makelabels(targetbatch, centertensor, sigmatensor)

            optimizer.zero_grad(set_to_none=True)
            probabilities = model(inputbatch)
            loss = negativeloglikelihood(probabilities, softlabels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradientclip)
            optimizer.step()

            trainingloss += float(loss.item()) * len(inputbatch)
            trainingcount += len(inputbatch)

        model.eval()
        validationloss = 0.0
        validationrmse = 0.0
        validationcount = 0
        validationstart = 0

        with torch.inference_mode():
            for inputbatch, targetbatch in validationloader:
                inputbatch = inputbatch.to(device)
                targetbatch = targetbatch.to(device)
                softlabels = makelabels(targetbatch, centertensor, sigmatensor)
                probabilities = model(inputbatch)
                loss = negativeloglikelihood(probabilities, softlabels)

                prediction = (probabilities * centertensor).sum(dim=1, keepdim=True)
                prediction = prediction * targetstd + targetmean
                stop = validationstart + len(inputbatch)
                actual = validationphysical[validationstart:stop].to(device)
                rmse = torch.sqrt((prediction - actual).pow(2).mean())

                validationloss += float(loss.item()) * len(inputbatch)
                validationrmse += float(rmse.item()) * len(inputbatch)
                validationcount += len(inputbatch)
                validationstart = stop

        trainingloss /= trainingcount
        validationloss /= validationcount
        validationrmse /= validationcount
        scheduler.step(validationloss)

        print(
            f"Seed {member} epoch {epoch:04d}: train NLL {trainingloss:.5f}, "
            f"validation NLL {validationloss:.5f}, RMSE {validationrmse:.3f}",
            flush=True,
        )

        if validationloss < bestloss - 1e-6:
            bestloss = validationloss
            epochswithoutimprovement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_validation_loss": bestloss,
                    "validation_rmse": validationrmse,
                    "epoch": epoch,
                },
                bestpath,
            )
        else:
            epochswithoutimprovement += 1
            if epochswithoutimprovement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(bestpath, map_location=device)
    model.load_state_dict(checkpoint["model"])
    torch.save(
        {
            "model": model.state_dict(),
            "best_validation_loss": checkpoint["best_validation_loss"],
            "validation_rmse": checkpoint["validation_rmse"],
            "epoch": checkpoint["epoch"],
            "seed": seed,
            "channels": args.channels,
            "dropout": args.dropout,
        },
        finalpath,
    )
    print(f"Saved {finalpath}", flush=True)

    del model, optimizer, scheduler, trainloader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

print("Requested sweep configuration is complete")

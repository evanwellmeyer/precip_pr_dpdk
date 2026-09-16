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
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from unet import ProbUNet
from analysis.paths import hadgem, weights


parser = argparse.ArgumentParser()
parser.add_argument("--test-ppe", choices=["GA7", "GA8", "GA9"], required=True)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--first-seed", type=int, default=0)
parser.add_argument("--number-of-seeds", type=int, default=3)
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
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

inputpath = Path(args.input_path)
targetpath = Path(args.target_path)
weightspath = Path(args.weights_path)

families = ["GA7", "GA8", "GA9"]
baseseed = 42
channels = args.channels
groups = 1
kernel = 3
dropout = args.dropout
bins = 64
minimum = -700.0
maximum = 1200.0
sigmascale = 0.6
validationfraction = 0.2
epochs = 5000
patience = 20
gradientclip = 1.0
learningrate = 1e-3
weightdecay = 0.0

modelname = (
    f"unet_direct_dpdk_cv_flat_ch{channels}_k{kernel}_bins{bins}_"
    f"min{minimum:g}_max{maximum:g}_sigma{sigmascale:g}_dropout{dropout:g}"
)
foldpath = weightspath / modelname / f"fold_{args.test_ppe}"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
print(f"Held out PPE: {args.test_ppe}")

with xr.open_dataset(inputpath) as dataset:
    inputs = dataset["PR"].values.astype(np.float32)[:, None]
    labels = dataset.realization.values.astype(str)
with xr.open_dataset(targetpath) as dataset:
    targets = dataset["dPdK"].values.astype(np.float32)[:, None]
    targetlabels = dataset.realization.values.astype(str)

if not np.array_equal(labels, targetlabels):
    raise ValueError("Input and target realization labels do not match")

testmask = np.asarray([label.startswith(args.test_ppe + "_") for label in labels])
poolindices = np.where(~testmask)[0]
testindices = np.where(testmask)[0]
trainindices, validationindices = train_test_split(
    poolindices,
    test_size=validationfraction,
    random_state=baseseed,
)
trainingfamilies = [family for family in families if family != args.test_ppe]

expectedsizes = {"GA7": 509, "GA8": 503, "GA9": 503}
if len(testindices) != expectedsizes[args.test_ppe]:
    raise ValueError("The held out PPE has an unexpected number of members")

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


foldpath.mkdir(parents=True, exist_ok=True)
np.savez(
    foldpath / "data_splits.npz",
    train=trainindices,
    val=validationindices,
    test=testindices,
    train_ppes=np.asarray(trainingfamilies),
    test_ppe=np.asarray([args.test_ppe]),
)
with open(foldpath / "norm_stats.json", "w") as file:
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
with open(foldpath / "born_bins.json", "w") as file:
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
    f"{len(testindices)} held out test"
)
print(f"Training PPEs: {', '.join(trainingfamilies)}")
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
        1, channels, kernel, dropout, bins, gn_groups=groups
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
    finalpath = foldpath / f"{modelname}_member{member}.pth"
    bestpath = foldpath / f"best_member{member}.pth"

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
        1, channels, kernel, dropout, bins, gn_groups=groups
    ).to(device)
    optimizer = optim.RAdam(
        model.parameters(), lr=learningrate, weight_decay=weightdecay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
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
            f"{args.test_ppe} seed {member} epoch {epoch:04d}: "
            f"train NLL {trainingloss:.5f}, validation NLL {validationloss:.5f}, "
            f"RMSE {validationrmse:.3f}",
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
            "test_ppe": args.test_ppe,
        },
        finalpath,
    )
    print(f"Saved {finalpath}", flush=True)

    del model, optimizer, scheduler, trainloader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

print(f"Requested {args.test_ppe} runs are complete")

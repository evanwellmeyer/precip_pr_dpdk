from IPython.display import display
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from analysis.paths import research

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from types import SimpleNamespace

sensitivity = SimpleNamespace(
    hadgempath=research / 'HadGEM',
    secondsperyear=365.25 * 24 * 60 * 60,
)
sensitivity.outputpath = Path("analysis/direct_revision/climatology")
sensitivity.figurepath = Path("analysis/direct_revision/supp_figures")
sensitivity.outputpath.mkdir(parents=True, exist_ok=True)

pd.set_option("display.float_format", lambda value: f"{value:.2f}")

import cartopy.crs as ccrs
import xarray as xr
import xesmf as xe
from cartopy.util import add_cyclic_point

amippath = research / "AMIP/PD/concatenated/pr_Amon_HadGEM3-GC31-LL_amip_r5i1p1f3_gn_197901-201412.nc"
lengths = [1, 2, 3, 4, 5, 10, 20, 30]
endyear = 2009

with xr.open_dataset(amippath) as dataset:
    amipmonthly = dataset["pr"].sel(
        time=(dataset.time.dt.year >= 1980)
        & (dataset.time.dt.year <= endyear)
    ).load()

longitude = ((amipmonthly.lon + 180) % 360) - 180
amipmonthly = amipmonthly.assign_coords(lon=longitude).sortby("lon")

with xr.open_dataset(sensitivity.hadgempath / "GA789_PR_bilinear_rg128.nc") as dataset:
    targetlatitude = dataset.latitude.values
    targetlongitude = dataset.longitude.values

destination = xr.Dataset(
    coords={"lat": targetlatitude, "lon": targetlongitude}
)
weightpath = Path("analysis/direct_revision/hadgem_amip_climatology_bilinear_weights.nc")
regridder = xe.Regridder(
    amipmonthly.isel(time=0),
    destination,
    "bilinear",
    extrap_method="nearest_s2d",
    periodic=True,
    filename=weightpath,
    reuse_weights=weightpath.exists(),
)

amipfields = {}
for length in lengths:
    startyear = endyear - length + 1
    period = amipmonthly.sel(
        time=(amipmonthly.time.dt.year >= startyear)
        & (amipmonthly.time.dt.year <= endyear)
    )
    climatology = period.mean("time") * sensitivity.secondsperyear
    climatology = xr.DataArray(
        np.ascontiguousarray(climatology.values),
        dims=("lat", "lon"),
        coords={"lat": climatology.lat, "lon": climatology.lon},
    )
    amipfields[length] = regridder(climatology).values.astype(np.float32)
    print(f"{length:>2} years: {startyear}--{endyear}, {period.sizes['time']} months")

amipmonthlyforregridding = xr.DataArray(
    np.ascontiguousarray(amipmonthly.values),
    dims=amipmonthly.dims,
    coords=amipmonthly.coords,
)
regriddedmonthly = regridder(amipmonthlyforregridding).load()

amipwindowfields = []
amipwindowlengths = []
amipwindowstarts = []

for length in lengths:
    laststart = endyear - length + 1
    for startyear in range(1980, laststart + 1):
        stopyear = startyear + length - 1
        period = regriddedmonthly.sel(
            time=(regriddedmonthly.time.dt.year >= startyear)
            & (regriddedmonthly.time.dt.year <= stopyear)
        )
        climatology = period.mean("time") * sensitivity.secondsperyear
        amipwindowfields.append(climatology.values.astype(np.float32))
        amipwindowlengths.append(length)
        amipwindowstarts.append(startyear)

amipwindowfields = np.stack(amipwindowfields)
amipwindowlengths = np.asarray(amipwindowlengths)
amipwindowstarts = np.asarray(amipwindowstarts)

for length in lengths:
    count = np.sum(amipwindowlengths == length)
    print(f"{length:>2} years: {count} contiguous windows")

allvalues = np.concatenate([amipfields[length].ravel() for length in lengths])
upper = np.ceil(np.percentile(allvalues, 99.5) / 500) * 500
projection = ccrs.Robinson()
figure, axes = plt.subplots(
    2,
    4,
    figsize=(13.5, 5.9),
    subplot_kw={"projection": projection},
)

for axis, length in zip(axes.ravel(), lengths):
    startyear = endyear - length + 1
    cyclicfield, cycliclongitude = add_cyclic_point(
        amipfields[length], coord=targetlongitude
    )
    image = axis.pcolormesh(
        cycliclongitude,
        targetlatitude,
        cyclicfield,
        transform=ccrs.PlateCarree(),
        cmap="YlGnBu",
        vmin=0,
        vmax=upper,
        shading="auto",
        rasterized=True,
    )
    axis.coastlines(linewidth=0.45)
    axis.set_global()
    axis.set_title(
        f"{length}-year mean ({startyear}--{endyear})",
        fontsize=11,
        fontweight="bold",
    )

colorbar = figure.colorbar(
    image, ax=axes, orientation="horizontal", fraction=0.055, pad=0.06
)
colorbar.set_label(r"Annual-mean precipitation (mm yr$^{-1}$)")
figure.suptitle(
    "HadGEM3-GC31-LL AMIP precipitation climatologies",
    fontsize=14,
    fontweight="bold",
)
figure.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.2, wspace=0.04)
plt.close("all")

reference = amipfields[30]
differences = {length: amipfields[length] - reference for length in lengths}
differencevalues = np.concatenate(
    [np.abs(differences[length]).ravel() for length in lengths[:-1]]
)
limit = np.ceil(np.percentile(differencevalues, 99.0) / 50) * 50

figure, axes = plt.subplots(
    2,
    4,
    figsize=(13.5, 6.1),
    subplot_kw={"projection": projection},
)

for axis, length in zip(axes.ravel(), lengths):
    cyclicfield, cycliclongitude = add_cyclic_point(
        differences[length], coord=targetlongitude
    )
    image = axis.pcolormesh(
        cycliclongitude,
        targetlatitude,
        cyclicfield,
        transform=ccrs.PlateCarree(),
        cmap="BrBG",
        vmin=-limit,
        vmax=limit,
        shading="auto",
        rasterized=True,
    )
    axis.coastlines(linewidth=0.45)
    axis.set_global()
    axis.set_title(
        f"{length}-year minus 30-year",
        fontsize=11,
        fontweight="bold",
    )

colorbar = figure.colorbar(
    image, ax=axes, orientation="horizontal", fraction=0.055, pad=0.06
)
colorbar.set_label(r"Precipitation difference (mm yr$^{-1}$)")
figure.suptitle(
    "Sensitivity to climatology length relative to the 1980--2009 mean",
    fontsize=14,
    fontweight="bold",
)
figure.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.18, wspace=0.04)
plt.close("all")

latitudeweights = np.cos(np.deg2rad(targetlatitude))[:, None]
denominator = latitudeweights.sum() * len(targetlongitude)
rows = []
for length in lengths:
    squarederror = differences[length] ** 2
    rmse = np.sqrt((squarederror * latitudeweights).sum() / denominator)
    rows.append({"length": length, "rmsefrom30year": rmse})

amipconvergence = pd.DataFrame(rows).set_index("length")
display(amipconvergence)

futureprpath = research / "AMIP/future4K/PR/cat/pr_Amon_HadGEM3-GC31-LL_amip-future4K_r5i1p1f3_gn_197901-201412.nc"
presenttaspath = research / "AMIP/PD_tas/tas_Amon_HadGEM3-GC31-LL_amip_r5i1p1f3_gn_197901-201412.nc"
futuretaspath = research / "AMIP/future4K/TAS/tas_Amon_HadGEM3-GC31-LL_amip-future4K_r5i1p1f3_gn_197901-201412.nc"

with xr.open_dataset(futureprpath) as dataset:
    futureprmonthly = dataset["pr"].sel(
        time=(dataset.time.dt.year >= 1980)
        & (dataset.time.dt.year <= endyear)
    ).load()

with xr.open_dataset(presenttaspath) as dataset:
    presenttasmonthly = dataset["tas"].sel(
        time=(dataset.time.dt.year >= 1980)
        & (dataset.time.dt.year <= endyear)
    ).load()

with xr.open_dataset(futuretaspath) as dataset:
    futuretasmonthly = dataset["tas"].sel(
        time=(dataset.time.dt.year >= 1980)
        & (dataset.time.dt.year <= endyear)
    ).load()

futurelongitude = ((futureprmonthly.lon + 180) % 360) - 180
futureprmonthly = futureprmonthly.assign_coords(lon=futurelongitude).sortby("lon")

presentpr = amipmonthly.mean("time") * sensitivity.secondsperyear
futurepr = futureprmonthly.mean("time") * sensitivity.secondsperyear

presentweights = np.cos(np.deg2rad(presenttasmonthly.lat))
futureweights = np.cos(np.deg2rad(futuretasmonthly.lat))
presenttemperature = presenttasmonthly.weighted(presentweights).mean(("time", "lat", "lon"))
futuretemperature = futuretasmonthly.weighted(futureweights).mean(("time", "lat", "lon"))
temperaturechange = float(futuretemperature - presenttemperature)

nativeresponse = (futurepr - presentpr) / temperaturechange
nativeresponse = xr.DataArray(
    np.ascontiguousarray(nativeresponse.values),
    dims=("lat", "lon"),
    coords={"lat": nativeresponse.lat, "lon": nativeresponse.lon},
)
amiptarget = regridder(nativeresponse).values.astype(np.float32)

print(f"Thirty-year global temperature change: {temperaturechange:.3f} K")
print(f"Target range: {np.nanmin(amiptarget):.1f} to {np.nanmax(amiptarget):.1f} mm yr^-1 K^-1")

import glob
import json

import torch

from unet import ProbUNet

modelname = "unet_direct_dpdk_flat_ch64_k3_bins64_min-700_max1200_sigma0.6_dropout0.1"
modelpath = research / "weights/direct_dpdk_bilinear" / modelname
modelfiles = sorted(glob.glob(str(modelpath / f"{modelname}_member*.pth")))

with open(modelpath / "norm_stats.json") as file:
    normalization = json.load(file)
with open(modelpath / "born_bins.json") as file:
    bins = json.load(file)

inputmean = np.asarray(normalization["x_mean"], dtype=np.float32)
inputstandarddeviation = np.asarray(normalization["x_std"], dtype=np.float32)
targetmean = float(normalization["y_mean"])
targetstandarddeviation = float(normalization["y_std"])
bincenters = np.asarray(bins["bin_centers_norm"], dtype=np.float32)

inputs = amipwindowfields[:, None]
inputs = (inputs - inputmean[None, :, None, None]) / inputstandarddeviation[None, :, None, None]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
centertensor = torch.as_tensor(bincenters, dtype=torch.float32, device=device)[None, :, None, None]
predictiontotal = np.zeros((len(inputs), len(targetlatitude), len(targetlongitude)), dtype=np.float64)

print(f"Using {device} and {len(modelfiles)} NN seeds")
for number, modelfile in enumerate(modelfiles, start=1):
    model = ProbUNet(1, 64, 3, 0.1, 64, gn_groups=1).to(device)
    checkpoint = torch.load(modelfile, map_location=device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    for start in range(0, len(inputs), 4):
        stop = min(start + 4, len(inputs))
        inputtensor = torch.as_tensor(inputs[start:stop], dtype=torch.float32, device=device)
        with torch.inference_mode():
            probabilities = model.forward_components(inputtensor).float()
            prediction = (probabilities * centertensor).sum(dim=1)
            prediction = prediction * targetstandarddeviation + targetmean
        predictiontotal[start:stop] += prediction.cpu().numpy()
        del inputtensor, probabilities, prediction

    del model, checkpoint, state
    if device.type == "mps":
        torch.mps.empty_cache()
    print(f"Finished seed {number} of {len(modelfiles)}")

amippredictions = (predictiontotal / len(modelfiles)).astype(np.float32)

latitudeweights = np.cos(np.deg2rad(targetlatitude))[:, None]
denominator = latitudeweights.sum() * len(targetlongitude)
rows = []
for length, startyear, field, prediction in zip(
    amipwindowlengths, amipwindowstarts, amipwindowfields, amippredictions
):
    inputerror = (field - amipfields[30]) ** 2
    inputrmse = np.sqrt(np.nansum(inputerror * latitudeweights) / denominator)

    predictionerror = (prediction - amiptarget) ** 2
    nnrmse = np.sqrt(np.nansum(predictionerror * latitudeweights) / denominator)

    rows.append(
        {
            "length": length,
            "startyear": startyear,
            "endyear": startyear + length - 1,
            "inputrmse": inputrmse,
            "nnrmse": nnrmse,
        }
    )

amiperrors = pd.DataFrame(rows)
amiperrorsummary = amiperrors.groupby("length").agg(
    windows=("nnrmse", "size"),
    inputmean=("inputrmse", "mean"),
    inputmedian=("inputrmse", "median"),
    nnmean=("nnrmse", "mean"),
    nnmedian=("nnrmse", "median"),
    nnminimum=("nnrmse", "min"),
    nnmaximum=("nnrmse", "max"),
)
errorpath = sensitivity.outputpath / "amip_climatology_length_nn_error.csv"
amiperrors.to_csv(errorpath, index=False)
summarypath = sensitivity.outputpath / "amip_climatology_length_nn_error_summary.csv"
amiperrorsummary.to_csv(summarypath)
display(amiperrorsummary)
print(f"Saved {errorpath}")
print(f"Saved {summarypath}")

inputdistributions = [
    amiperrors.loc[amiperrors["length"] == length, "inputrmse"].values
    for length in lengths
]
predictiondistributions = [
    amiperrors.loc[amiperrors["length"] == length, "nnrmse"].values
    for length in lengths
]

figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
positions = np.arange(len(lengths))

inputboxes = axes[0].boxplot(
    inputdistributions,
    positions=positions,
    widths=0.58,
    patch_artist=True,
    showfliers=False,
)
for box in inputboxes["boxes"]:
    box.set_facecolor("#d9e6f5")
    box.set_edgecolor("#3b528b")
axes[0].plot(positions, [values.mean() for values in inputdistributions], color="#3b528b", marker="o", linewidth=2)
axes[0].set_xticks(positions, lengths)
axes[0].set_xlabel("Climatology length (years)")
axes[0].set_ylabel(r"RMSE from 30-year PR (mm yr$^{-1}$)")
axes[0].set_title("(a) Present-day input sensitivity", fontweight="bold")

predictionboxes = axes[1].boxplot(
    predictiondistributions,
    positions=positions,
    widths=0.58,
    patch_artist=True,
    showfliers=False,
)
for box in predictionboxes["boxes"]:
    box.set_facecolor("#d9e6f5")
    box.set_edgecolor("#3b528b")
axes[1].plot(positions, [values.mean() for values in predictiondistributions], color="#3b528b", marker="o", linewidth=2)
axes[1].set_xticks(positions, lengths)
axes[1].set_xlabel("Climatology length (years)")
axes[1].set_ylabel(r"NN RMSE (mm yr$^{-1}$ K$^{-1}$)")
axes[1].set_title("(b) Prediction sensitivity", fontweight="bold")

for axis in axes:
    axis.grid(axis="y", alpha=0.2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

figure.tight_layout()
robustpngpath = sensitivity.figurepath / "climatology_length_window_distributions.png"
robustpdfpath = sensitivity.figurepath / "climatology_length_window_distributions.pdf"
figure.savefig(robustpngpath, dpi=300, bbox_inches="tight")
figure.savefig(robustpdfpath, bbox_inches="tight")
plt.close("all")

print(f"Saved {robustpngpath}")
print(f"Saved {robustpdfpath}")

with xr.open_dataset(sensitivity.hadgempath / "GA789_PR_bilinear_rg128.nc") as dataset:
    ppefield = dataset["PR"].isel(realization=259).load().values

comparisonfields = [ppefield, amipfields[5], amipfields[30]]
comparisontitles = [
    "GA7 PPE realization 259\n5-year mean",
    "HadGEM3 standard AMIP\n2005--2009 mean",
    "HadGEM3 standard AMIP\n1980--2009 mean",
]
comparisonvalues = np.concatenate([field.ravel() for field in comparisonfields])
comparisonupper = np.ceil(np.percentile(comparisonvalues, 99.5) / 500) * 500

figure, axes = plt.subplots(
    1,
    3,
    figsize=(12.5, 3.8),
    subplot_kw={"projection": projection},
)
for axis, field, title in zip(axes, comparisonfields, comparisontitles):
    cyclicfield, cycliclongitude = add_cyclic_point(
        field, coord=targetlongitude
    )
    image = axis.pcolormesh(
        cycliclongitude,
        targetlatitude,
        cyclicfield,
        transform=ccrs.PlateCarree(),
        cmap="YlGnBu",
        vmin=0,
        vmax=comparisonupper,
        shading="auto",
        rasterized=True,
    )
    axis.coastlines(linewidth=0.45)
    axis.set_global()
    axis.set_title(title, fontsize=11, fontweight="bold")

colorbar = figure.colorbar(
    image, ax=axes, orientation="horizontal", fraction=0.07, pad=0.08
)
colorbar.set_label(r"Annual-mean precipitation (mm yr$^{-1}$)")
figure.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.19, wspace=0.04)
plt.close("all")

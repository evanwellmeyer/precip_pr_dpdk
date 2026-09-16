from IPython.display import display
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from analysis.paths import research, project
Path("analysis/direct_revision/figures").mkdir(parents=True, exist_ok=True)
Path("analysis/direct_revision/supp_figures").mkdir(parents=True, exist_ok=True)
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

hadgempath = Path(str(research / "HadGEM"))
secondsperyear = 365.25 * 24 * 60 * 60
familyorder = ["GA7", "GA8", "GA9"]
familycolors = {"GA7": "#4477aa", "GA8": "#ee6677", "GA9": "#228833"}

configs = {
    "GA7": {"temperature": "surface_temperature", "prefix": "ts"},
    "GA8": {"temperature": "air_temperature", "prefix": "tas"},
    "GA9": {"temperature": "air_temperature", "prefix": "tas"},
}


def globalmean(field):
    weights = np.cos(np.deg2rad(field.latitude))
    return field.weighted(weights).mean(("latitude", "longitude"), skipna=True)


records = []
familydata = {}

for family in familyorder:
    config = configs[family]

    prhis = xr.open_dataset(hadgempath / f"{family}_pr_his_clim.nc")["precipitation_flux"]
    prfut = xr.open_dataset(hadgempath / f"{family}_pr_fut_clim.nc")["precipitation_flux"]
    temphis = xr.open_dataset(
        hadgempath / f"{family}_{config['prefix']}_his_clim.nc"
    )[config["temperature"]]
    tempfut = xr.open_dataset(
        hadgempath / f"{family}_{config['prefix']}_fut_clim.nc"
    )[config["temperature"]]

    prhis, prfut, temphis, tempfut = xr.align(
        prhis, prfut, temphis, tempfut, join="inner"
    )

    prhismean = globalmean(prhis * secondsperyear)
    prfutmean = globalmean(prfut * secondsperyear)
    deltat = globalmean(tempfut) - globalmean(temphis)
    deltap = prfutmean - prhismean
    absoluteresponse = deltap / deltat
    fractionalresponse = 100 * deltap / prhismean / deltat

    familydata[family] = {
        "prhis": prhismean.values,
        "prfut": prfutmean.values,
        "deltat": deltat.values,
        "deltap": deltap.values,
        "absolute": absoluteresponse.values,
        "fractional": fractionalresponse.values,
    }

    for member in range(deltat.size):
        records.append({
            "family": family,
            "deltat": float(deltat.values[member]),
            "prhis": float(prhismean.values[member]),
            "prfut": float(prfutmean.values[member]),
            "deltap": float(deltap.values[member]),
            "absolute": float(absoluteresponse.values[member]),
            "fractional": float(fractionalresponse.values[member]),
        })

data = pd.DataFrame(records)
data.groupby("family").size().rename("members")


def rangeentry(values, digits=2):
    median = np.nanmedian(values)
    lower = np.nanpercentile(values, 5)
    upper = np.nanpercentile(values, 95)
    return f"{median:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


summaryrows = []
for family in familyorder:
    values = familydata[family]
    summaryrows.append({
        "family": family,
        "members": len(values["deltat"]),
        "deltat_k": rangeentry(values["deltat"]),
        "historical_pr_mm_yr": rangeentry(values["prhis"], 1),
        "dpdk_mm_yr_k": rangeentry(values["absolute"], 1),
        "fractional_dpdk_pct_k": rangeentry(values["fractional"]),
    })

summary = pd.DataFrame(summaryrows).set_index("family")
display(summary)


figure, axes = plt.subplots(2, 2, figsize=(11, 8))

plots = [
    ("deltat", "(a) Global-mean temperature response", r"$\Delta T$ (K)"),
    ("prhis", "(b) Historical global-mean precipitation", r"PR (mm yr$^{-1}$)"),
    (
        "absolute",
        "(c) Absolute global-mean precipitation response",
        r"$\Delta P / \Delta T$ (mm yr$^{-1}$ K$^{-1}$)",
    ),
    (
        "fractional",
        "(d) Fractional global-mean precipitation response",
        r"$100(\Delta P / P) / \Delta T$ (% K$^{-1}$)",
    ),
]

for axis, (field, title, label) in zip(axes.flat, plots):
    for family in familyorder:
        sns.kdeplot(
            familydata[family][field],
            color=familycolors[family],
            label=family,
            fill=False,
            linewidth=2,
            ax=axis,
        )
    axis.set_title(title, fontsize=13)
    axis.set_xlabel(label, fontsize=11)
    axis.set_ylabel("Probability density", fontsize=11)
    axis.tick_params(labelsize=11)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=11)

figure.tight_layout()
figure.savefig(
    "analysis/direct_revision/supp_figures/figS1.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close("all")


weightspath = Path(str(research / "weights/direct_dpdk_bilinear/unet_direct_dpdk_flat_ch64_k3_bins64_min-700_max1200_sigma0.6_dropout0.1"))
targetpath = hadgempath / "GA789_dPdK_bilinear_rg128.nc"
with open(weightspath / "born_bins.json") as file:
    correctedbins = json.load(file)
lowerlimit = float(correctedbins["dP_min"])
upperlimit = float(correctedbins["dP_max"])

split = np.load(weightspath / "data_splits.npz")
trainindices = split["train"]

with xr.open_dataset(targetpath) as targetfile:
    target = targetfile["dPdK"].isel(realization=trainindices).load()

plotbins = np.linspace(-1000, 1600, 321)
unweightedcounts = np.zeros(plotbins.size - 1)
weightedcounts = np.zeros(plotbins.size - 1)
unweightedtotal = 0
weightedtotal = 0.0
unweightedoutside = 0
weightedoutside = 0.0
latitudeweights = np.cos(np.deg2rad(target.latitude.values))

for latitudeindex, latitudeweight in enumerate(latitudeweights):
    values = target.isel(latitude=latitudeindex).values.ravel()
    values = values[np.isfinite(values)]
    outside = (values < lowerlimit) | (values > upperlimit)

    counts, _ = np.histogram(values, bins=plotbins)
    unweightedcounts += counts
    weightedcounts += counts * latitudeweight
    unweightedtotal += values.size
    weightedtotal += values.size * latitudeweight
    unweightedoutside += outside.sum()
    weightedoutside += outside.sum() * latitudeweight

unweightedfraction = unweightedoutside / unweightedtotal
weightedfraction = weightedoutside / weightedtotal
binwidths = np.diff(plotbins)
bincenters = plotbins[:-1] + binwidths / 2
unweighteddensity = unweightedcounts / unweightedcounts.sum() / binwidths
weighteddensity = weightedcounts / weightedcounts.sum() / binwidths

display(pd.DataFrame({
    "members": [trainindices.size],
    "grid_values": [unweightedtotal],
    "unweighted_outside_fraction": [unweightedfraction],
    "unweighted_outside_percent": [100 * unweightedfraction],
    "latitude_weighted_outside_fraction": [weightedfraction],
    "latitude_weighted_outside_percent": [100 * weightedfraction],
}))

figure, axis = plt.subplots(figsize=(9, 5))
axis.plot(
    bincenters,
    unweighteddensity,
    color="#4477aa",
    linewidth=2,
    label="Unweighted",
)
axis.plot(
    bincenters,
    weighteddensity,
    color="#cc6677",
    linewidth=2,
    label="Cosine-latitude weighted",
)
axis.axvline(lowerlimit, color="black", linestyle="--", linewidth=1.5)
axis.axvline(upperlimit, color="black", linestyle="--", linewidth=1.5)
axis.set_xlim(plotbins[0], plotbins[-1])
axis.set_xlabel(r"Gridded dPdK target (mm yr$^{-1}$ K$^{-1}$)")
axis.set_ylabel("Probability density")
axis.set_title("Training-set gridded dPdK distribution")
axis.grid(alpha=0.2)
axis.legend()
axis.text(
    0.02,
    0.96,
    f"Outside limits: {100 * unweightedfraction:.4f}% unweighted\n"
    f"{100 * weightedfraction:.4f}% latitude weighted",
    transform=axis.transAxes,
    va="top",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95},
)

figure.tight_layout()
figure.savefig(
    "analysis/direct_revision/supp_figures/figS_target_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close("all")

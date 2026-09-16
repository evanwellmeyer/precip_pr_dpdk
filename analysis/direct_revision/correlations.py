from pathlib import Path
from analysis.paths import research, project

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import xarray as xr
from matplotlib.lines import Line2D


hadgempath = Path(str(research / "HadGEM"))
outputpath = Path("analysis/direct_revision/figures/fig1.png")

sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")

with xr.open_dataset(hadgempath / "GA789_PR_bilinear_rg128.nc") as dataset:
    precipitation = dataset["PR"].values.astype(np.float32)
    labels = dataset.realization.values.astype(str)

with xr.open_dataset(hadgempath / "GA789_dPdK_bilinear_rg128.nc") as dataset:
    response = dataset["dPdK"].values.astype(np.float32)
    responselabels = dataset.realization.values.astype(str)
    latitudes = dataset.latitude.values

if not np.array_equal(labels, responselabels):
    raise ValueError("The precipitation and response labels do not match")

weights = np.cos(np.deg2rad(latitudes))[:, None]
weights = np.broadcast_to(weights, precipitation.shape[1:]).reshape(-1)
weights = weights / weights.sum()

precipitation = precipitation.reshape(len(labels), -1)
response = response.reshape(len(labels), -1)

precipitation = precipitation - np.sum(precipitation * weights, axis=1)[:, None]
response = response - np.sum(response * weights, axis=1)[:, None]

precipitation = precipitation * np.sqrt(weights)[None, :]
response = response * np.sqrt(weights)[None, :]

precipitation = precipitation / np.sqrt(np.sum(precipitation**2, axis=1))[:, None]
response = response / np.sqrt(np.sum(response**2, axis=1))[:, None]

precipitationcorrelations = precipitation @ precipitation.T
responsecorrelations = response @ response.T

first, second = np.triu_indices(len(labels), k=1)
precipitationcorrelations = precipitationcorrelations[first, second]
responsecorrelations = responsecorrelations[first, second]
families = np.asarray([label.split("_")[0] for label in labels])

allr = stats.linregress(precipitationcorrelations, responsecorrelations).rvalue
colors = {"GA7": "#E69F00", "GA8": "#56B4E9", "GA9": "#009E73"}

figure, axis = plt.subplots(figsize=(8, 6))
axis.scatter(
    precipitationcorrelations,
    responsecorrelations,
    c="black",
    s=3,
    alpha=0.5,
    edgecolor="none",
)

legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="white",
        markerfacecolor="black",
        markersize=8,
        label=rf"HadGEM PPE ($r={allr:.2f}$)",
    )
]

for family, color in colors.items():
    mask = (families[first] == family) & (families[second] == family)
    familyr = stats.linregress(
        precipitationcorrelations[mask], responsecorrelations[mask]
    ).rvalue
    axis.scatter(
        precipitationcorrelations[mask],
        responsecorrelations[mask],
        c=color,
        s=2,
        alpha=0.5,
        edgecolor="none",
    )
    legend.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=color,
            markersize=8,
            label=rf"{family} ($r={familyr:.2f}$)",
        )
    )

axis.set_xlabel(r"More similar Climatology $\longrightarrow$", fontweight="bold")
axis.set_ylabel(r"More similar Change $\longrightarrow$", fontweight="bold")
axis.set_xlim(0.65, 1.0)
axis.set_ylim(0, 1.0)
axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
axis.legend(handles=legend, loc="upper left", frameon=False)
sns.despine(trim=True)
figure.tight_layout()
figure.savefig(outputpath, dpi=300, bbox_inches="tight")

print(f"HadGEM PPE r: {allr:.3f}")
for family in colors:
    mask = (families[first] == family) & (families[second] == family)
    familyr = stats.linregress(
        precipitationcorrelations[mask], responsecorrelations[mask]
    ).rvalue
    print(f"{family} r: {familyr:.3f}")

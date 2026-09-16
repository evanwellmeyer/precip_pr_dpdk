import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# ==============================================================================
# 12. Publication-quality figure: AMIP vs HadGEM PPE member-level statistics
# ==============================================================================
# Self-contained, matplotlib-only. Reads AMIP and HadGEM PPE fields, computes
# per-member statistics, and draws a compact GRL-style supporting figure.

from pathlib import Path
from analysis.paths import research, project

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# ------------------------------------------------------------------------------
# Step 1: Declare input file locations and the output directory.
# ------------------------------------------------------------------------------
amip_pr_candidates = [Path("analysis/direct_revision/amip_direct.nc")]
amip_dpdk_candidates = [Path("analysis/direct_revision/amip_direct.nc")]
hg_pr_path = Path(str(research / "HadGEM/GA789_PR_bilinear_rg128.nc"))
hg_dpdk_path = Path(str(research / "HadGEM/GA789_dPdK_bilinear_rg128.nc"))

out_dir = project / "analysis/direct_revision/figures"
out_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Step 2: Helpers for locating files and loading fields/coords out of a dataset.
# ------------------------------------------------------------------------------
def _first_existing(paths):
    """Return the first path in `paths` that exists on disk."""
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exist: " + ", ".join(str(p) for p in paths))


def _load_field(path, preferred_vars):
    """Load a (members, lat, lon) field plus lat/lon coords from a NetCDF file."""
    with xr.open_dataset(path) as ds:
        # Pick the first preferred variable that exists; otherwise take the first data_var.
        var_name = next((name for name in preferred_vars if name in ds.data_vars), None)
        if var_name is None:
            var_name = list(ds.data_vars)[0]
        arr = ds[var_name].values.astype(np.float32)

        # Extract latitude coordinate under any of the common names.
        lat = None
        for name in ("latitude", "lat", "y"):
            if name in ds.coords:
                lat = ds[name].values.astype(np.float32)
                break

        # Extract longitude coordinate under any of the common names.
        lon = None
        for name in ("longitude", "lon", "x"):
            if name in ds.coords:
                lon = ds[name].values.astype(np.float32)
                break

    return arr, lat, lon, var_name


# ------------------------------------------------------------------------------
# Step 3: Resolve the input file paths (first one that exists wins).
# ------------------------------------------------------------------------------
amip_pr_path = _first_existing(amip_pr_candidates)
amip_dpdk_path = _first_existing(amip_dpdk_candidates)


# ------------------------------------------------------------------------------
# Step 4: Load the four fields and keep the first available lat/lon coords.
# ------------------------------------------------------------------------------
amip_pr, lats, lons, amip_pr_var = _load_field(
    amip_pr_path, ["PR", "pr", "precipitation_flux"]
)
amip_dpdk, _, _, amip_dpdk_var = _load_field(
    amip_dpdk_path, ["dPdK", "dP_dK", "dPdP"]
)
hg_pr, hg_lats, hg_lons, hg_pr_var = _load_field(
    hg_pr_path, ["PR", "pr", "precipitation_flux"]
)
hg_dpdk, _, _, hg_dpdk_var = _load_field(
    hg_dpdk_path, ["dPdK", "dP_dK", "dPdP"]
)

# Fall back to HadGEM lats if the AMIP file didn't carry coordinates.
if lats is None:
    lats = hg_lats
if lats is None:
    raise ValueError("Could not infer latitude coordinates from AMIP or HadGEM files.")


# ------------------------------------------------------------------------------
# Step 5: Build area weights from cos(latitude) for area-weighted statistics.
# ------------------------------------------------------------------------------
lat_weights = np.cos(np.deg2rad(lats)).astype(np.float64)
lat_weights = np.clip(lat_weights, 0.0, None)


# ------------------------------------------------------------------------------
# Step 6: Statistical helpers used by every panel.
# ------------------------------------------------------------------------------
def area_weighted_mean_per_member(x, weights):
    """Global area-weighted mean, computed separately for each member."""
    x = np.asarray(x, dtype=np.float64)
    denom = weights.sum() * x.shape[2]
    return (x * weights[None, :, None]).sum(axis=(1, 2)) / denom


def area_weighted_spatial_std_per_member(x, weights):
    """Area-weighted spatial standard deviation, computed separately for each member."""
    x = np.asarray(x, dtype=np.float64)
    mu = area_weighted_mean_per_member(x, weights)[:, None, None]
    denom = weights.sum() * x.shape[2]
    var = ((x - mu) ** 2 * weights[None, :, None]).sum(axis=(1, 2)) / denom
    return np.sqrt(np.maximum(var, 0.0))


def leave_one_out_rmse_global(x, weights, batch_size=128):
    """Per-member global RMSE against the mean of every other member (leave-one-out)."""
    x = np.asarray(x, dtype=np.float64)
    n, _, width = x.shape

    # Total across members, used to form each LOO mean as (total - xb) / (n - 1).
    total = x.sum(axis=0, dtype=np.float64)
    denom = weights.sum() * width
    out = np.empty(n, dtype=np.float64)

    # Process in batches so the (batch, lat, lon) tensors stay small.
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xb = x[start:stop]
        loo = (total[None, :, :] - xb) / max(n - 1, 1)
        se = (loo - xb) ** 2
        out[start:stop] = np.sqrt((se * weights[None, :, None]).sum(axis=(1, 2)) / denom)

    return out


# ------------------------------------------------------------------------------
# Step 7: Compute the per-member statistics that drive the panels.
# ------------------------------------------------------------------------------
amip_pr_mean = area_weighted_mean_per_member(amip_pr, lat_weights)
hg_pr_mean = area_weighted_mean_per_member(hg_pr, lat_weights)

amip_dpdk_mean = area_weighted_mean_per_member(amip_dpdk, lat_weights)
hg_dpdk_mean = area_weighted_mean_per_member(hg_dpdk, lat_weights)

amip_dpdk_std = area_weighted_spatial_std_per_member(amip_dpdk, lat_weights)
hg_dpdk_std = area_weighted_spatial_std_per_member(hg_dpdk, lat_weights)


# ------------------------------------------------------------------------------
# Step 8: Try to add LOO RMSE as a fourth panel; degrade gracefully if it fails.
# ------------------------------------------------------------------------------
loo_available = True
try:
    amip_loo_rmse = leave_one_out_rmse_global(amip_dpdk, lat_weights, batch_size=16)
    hg_loo_rmse = leave_one_out_rmse_global(hg_dpdk, lat_weights, batch_size=128)
except Exception as exc:
    print(f"LOO RMSE could not be computed; making 3-panel figure instead. Reason: {exc}")
    loo_available = False


# ------------------------------------------------------------------------------
# Step 9: Report what was loaded so the cell output is self-documenting.
# ------------------------------------------------------------------------------
n_amip = int(amip_dpdk.shape[0])
n_hg = int(hg_dpdk.shape[0])
print(f"Loaded AMIP {amip_dpdk_var} from {amip_dpdk_path.name}; N={n_amip}")
print(f"Loaded HadGEM {hg_dpdk_var} from {hg_dpdk_path.name}; N={n_hg}")


# ------------------------------------------------------------------------------
# Step 10: Assemble panel specifications (title, ylabel, and archive values).
# ------------------------------------------------------------------------------
panel_specs = [
    (
        "(a) Global-mean PR",
        "PR (mm yr$^{-1}$)",
        amip_pr_mean,
        hg_pr_mean,
    ),
    (
        "(b) Global-mean dPdK",
        "dPdK (mm yr$^{-1}$ K$^{-1}$)",
        amip_dpdk_mean,
        hg_dpdk_mean,
    ),
    (
        "(c) Spatial std. of dPdK",
        "Spatial std. (mm yr$^{-1}$ K$^{-1}$)",
        amip_dpdk_std,
        hg_dpdk_std,
    ),
]
if loo_available:
    panel_specs.append(
        (
            "(d) Leave-one-out global RMSE",
            "LOO RMSE (mm yr$^{-1}$ K$^{-1}$)",
            amip_loo_rmse,
            hg_loo_rmse,
        )
    )


# ------------------------------------------------------------------------------
# Step 11: Figure-wide style choices.
# ------------------------------------------------------------------------------
C_AMIP = "#D55E00"   # orange, used for AMIP
C_HG = "#0072B2"     # blue, used for HadGEM PPE
rng = np.random.default_rng(42)


def _jitter(n, width):
    """Uniform x-axis jitter of half-width `width`, for overlaid AMIP points."""
    return rng.uniform(-width, width, size=n)


# ------------------------------------------------------------------------------
# Step 12: Single-panel drawing routine: boxplots plus AMIP points only.
# ------------------------------------------------------------------------------
def _draw_distribution(ax, amip_values, hg_values, ylabel, title):
    data = [
        np.asarray(amip_values, dtype=np.float64),
        np.asarray(hg_values, dtype=np.float64),
    ]
    positions = [0, 1]
    colors = [C_AMIP, C_HG]

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.44,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.20)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)
    for whisker, color in zip(box["whiskers"], [C_AMIP, C_AMIP, C_HG, C_HG]):
        whisker.set_color(color)
        whisker.set_linewidth(1.1)
    for cap, color in zip(box["caps"], [C_AMIP, C_AMIP, C_HG, C_HG]):
        cap.set_color(color)
        cap.set_linewidth(1.1)
    for median in box["medians"]:
        median.set_color("0.15")
        median.set_linewidth(1.6)

    # AMIP has only 11 models, so show those individual values explicitly.
    # HadGEM has 1515 members; its raw points are omitted to keep the panel clean.
    ax.scatter(
        positions[0] + _jitter(len(data[0]), 0.055),
        data[0],
        s=18,
        facecolor="white",
        edgecolor=C_AMIP,
        linewidth=0.9,
        alpha=0.95,
        zorder=3,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([
        "AMIP",
        "HadGEM PPE",
    ])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ------------------------------------------------------------------------------
# Step 13: Set readable figure-wide rcParams before building the figure.
# ------------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12.5,
    "axes.titlesize": 13.5,
    "axes.labelsize": 13,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12,
})


# ------------------------------------------------------------------------------
# Step 14: Build the figure as a single row of boxplot panels.
# ------------------------------------------------------------------------------
n_panels = len(panel_specs)
fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 4.0), squeeze=False)
axes = axes.ravel()


# ------------------------------------------------------------------------------
# Step 15: Draw each panel.
# ------------------------------------------------------------------------------
for ax, (title, ylabel, amip_values, hg_values) in zip(axes, panel_specs):
    _draw_distribution(ax, amip_values, hg_values, ylabel, title)


# ------------------------------------------------------------------------------
# Step 16: Tighten layout; the manuscript caption carries the interpretation.
# ------------------------------------------------------------------------------
fig.tight_layout()


# ------------------------------------------------------------------------------
# Step 17: Save to the manuscript figure path so recompiling updates Figure 6.
# ------------------------------------------------------------------------------
png_path = out_dir / "fig6.png"
pdf_path = out_dir / "fig6.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

plt.close("all")


# ==============================================================================
# 13. Manuscript Figure 6: AMIP vs separated HadGEM PPEs
# ==============================================================================
# Run the preceding combined AMIP/HadGEM figure cell first. This cell reuses the
# loaded arrays and helper functions, then splits the HadGEM archive into GA7,
# GA8, and GA9 using the realization coordinate in the NetCDF files. The figure
# is saved to fig6 so recompiling the manuscript uses this split-PPE version.

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

C_AMIP_SPLIT = "#CC79A7"  # reddish purple, distinct from the PPE colors
C_PPE_SPLIT = {"GA7": "#E69F00", "GA8": "#56B4E9", "GA9": "#009E73"}
rng_split = np.random.default_rng(43)


def _split_jitter(n, width):
    """Uniform x-axis jitter of half-width `width`, for overlaid AMIP points."""
    return rng_split.uniform(-width, width, size=n)


required_names = [
    "amip_pr_mean",
    "amip_dpdk_mean",
    "amip_dpdk_std",
    "hg_pr",
    "hg_dpdk",
    "amip_dpdk",
    "hg_dpdk_path",
    "lat_weights",
    "out_dir",
    "area_weighted_mean_per_member",
    "area_weighted_spatial_std_per_member",
    "leave_one_out_rmse_global",
]
missing = [name for name in required_names if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the preceding combined AMIP/HadGEM figure cell first; missing: "
        + ", ".join(missing)
    )


# ------------------------------------------------------------------------------
# Step 1: Read HadGEM realization labels and split the archive by PPE prefix.
# ------------------------------------------------------------------------------
with xr.open_dataset(hg_dpdk_path) as ds:
    if "realization" not in ds.coords:
        raise ValueError(f"{hg_dpdk_path.name} does not contain a realization coordinate.")
    hg_realizations = np.asarray(ds["realization"].values, dtype=str)

if hg_realizations.shape[0] != hg_dpdk.shape[0]:
    raise ValueError(
        "The number of HadGEM realization labels does not match the dPdK array: "
        f"{hg_realizations.shape[0]} labels vs {hg_dpdk.shape[0]} members."
    )

ppe_labels = ["GA7", "GA8", "GA9"]
ppe_masks = {}
for label in ppe_labels:
    mask = np.char.startswith(hg_realizations, f"{label}_")
    if not np.any(mask):
        raise ValueError(f"No HadGEM members found for prefix {label}_")
    ppe_masks[label] = mask

print("HadGEM PPE member counts:")
for label in ppe_labels:
    print(f"  {label}: {int(ppe_masks[label].sum())}")


# ------------------------------------------------------------------------------
# Step 2: Compute per-PPE statistics using the same definitions as Figure 6.
# ------------------------------------------------------------------------------
ppe_stats = {}
for label in ppe_labels:
    mask = ppe_masks[label]
    ppe_stats[label] = {
        "pr_mean": area_weighted_mean_per_member(hg_pr[mask], lat_weights),
        "dpdk_mean": area_weighted_mean_per_member(hg_dpdk[mask], lat_weights),
        "dpdk_std": area_weighted_spatial_std_per_member(hg_dpdk[mask], lat_weights),
    }

split_loo_available = True
try:
    amip_split_loo = (
        amip_loo_rmse
        if "amip_loo_rmse" in globals()
        else leave_one_out_rmse_global(amip_dpdk, lat_weights, batch_size=16)
    )
    for label in ppe_labels:
        ppe_stats[label]["loo_rmse"] = leave_one_out_rmse_global(
            hg_dpdk[ppe_masks[label]], lat_weights, batch_size=128
        )
except Exception as exc:
    print(f"Split-PPE LOO RMSE could not be computed; making 3-panel figure instead. Reason: {exc}")
    split_loo_available = False


# ------------------------------------------------------------------------------
# Step 3: Assemble panel specifications.
# ------------------------------------------------------------------------------
def _series_for(key):
    return [("AMIP", globals()[f"amip_{key}"], C_AMIP_SPLIT)] + [
        (label, ppe_stats[label][key], C_PPE_SPLIT[label])
        for label in ppe_labels
    ]

split_panel_specs = [
    (
        "(a) Global-mean PR",
        "PR (mm yr$^{-1}$)",
        _series_for("pr_mean"),
    ),
    (
        "(b) Global-mean dPdK",
        "dPdK (mm yr$^{-1}$ K$^{-1}$)",
        _series_for("dpdk_mean"),
    ),
    (
        "(c) Spatial std. of dPdK",
        "Spatial std. (mm yr$^{-1}$ K$^{-1}$)",
        _series_for("dpdk_std"),
    ),
]
if split_loo_available:
    split_panel_specs.append(
        (
            "(d) Leave-one-out global RMSE",
            "LOO RMSE (mm yr$^{-1}$ K$^{-1}$)",
            [("AMIP", amip_split_loo, C_AMIP_SPLIT)] + [
                (label, ppe_stats[label]["loo_rmse"], C_PPE_SPLIT[label])
                for label in ppe_labels
            ],
        )
    )


# ------------------------------------------------------------------------------
# Step 4: Draw the split-PPE version of the same boxplot figure.
# ------------------------------------------------------------------------------
def _draw_split_distribution(ax, series, ylabel, title):
    labels = [item[0] for item in series]
    data = [np.asarray(item[1], dtype=np.float64) for item in series]
    colors = [item[2] for item in series]
    positions = np.arange(len(data))

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.20)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.35)
    for i, color in enumerate(colors):
        for whisker in box["whiskers"][2 * i: 2 * i + 2]:
            whisker.set_color(color)
            whisker.set_linewidth(1.05)
        for cap in box["caps"][2 * i: 2 * i + 2]:
            cap.set_color(color)
            cap.set_linewidth(1.05)
    for median in box["medians"]:
        median.set_color("0.15")
        median.set_linewidth(1.55)

    # AMIP has only 11 models, so keep its individual values visible.
    ax.scatter(
        positions[0] + _split_jitter(len(data[0]), 0.055),
        data[0],
        s=18,
        facecolor="white",
        edgecolor=colors[0],
        linewidth=0.9,
        alpha=0.95,
        zorder=3,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


n_split_panels = len(split_panel_specs)
fig, axes = plt.subplots(1, n_split_panels, figsize=(3.6 * n_split_panels, 4.15), squeeze=False)
axes = axes.ravel()

for ax, (title, ylabel, series) in zip(axes, split_panel_specs):
    _draw_split_distribution(ax, series, ylabel, title)

fig.tight_layout()

fig6_png_path = out_dir / "fig6.png"
fig6_pdf_path = out_dir / "fig6.pdf"
split_png_path = out_dir / "amip_vs_hadgem_ppe_split.png"
split_pdf_path = out_dir / "amip_vs_hadgem_ppe_split.pdf"
fig.savefig(fig6_png_path, dpi=300, bbox_inches="tight")
fig.savefig(fig6_pdf_path, bbox_inches="tight")
fig.savefig(split_png_path, dpi=300, bbox_inches="tight")
fig.savefig(split_pdf_path, bbox_inches="tight")
print(f"Saved: {fig6_png_path}")
print(f"Saved: {fig6_pdf_path}")
print(f"Saved: {split_png_path}")
print(f"Saved: {split_pdf_path}")
plt.close("all")

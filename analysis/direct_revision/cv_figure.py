import os
from pathlib import Path
from analysis.paths import research, project

MPLCONFIGDIR = Path("/tmp/matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

XDG_CACHE_HOME = Path("/tmp/fontconfig-cache")
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import xarray as xr


PPE_FAMILIES = ["GA7", "GA8", "GA9"]
TRUTH_FILE = Path(str(research / "HadGEM/GA789_dPdK_bilinear_rg128.nc"))
WEIGHTS_BASE = Path(str(research / "weights/direct_dpdk_bilinear_cv"))

# This run matches the leave-one-PPE-out values reported in the manuscript table.
CV_RUN = "unet_direct_dpdk_cv_flat_ch64_k3_bins64_min-700_max1200_sigma0.6_dropout0.1"
OUT_PATH = Path("analysis/direct_revision/supp_figures/figS4.png")


def rmse_improvement(baseline_rmse, model_rmse):
    baseline_rmse = np.asarray(baseline_rmse, dtype=np.float64)
    model_rmse = np.asarray(model_rmse, dtype=np.float64)
    return (baseline_rmse - model_rmse) / (baseline_rmse + 1e-12) * 100.0


def extend_rows(rows, values, fold, region):
    rows.extend(
        {
            "RMSE Improvement (%)": float(value),
            "Held-out PPE": fold,
            "Region": region,
        }
        for value in np.asarray(values)
    )


def load_truth():
    with xr.open_dataset(TRUTH_FILE) as ds:
        return ds["dPdK"].values.astype(np.float32)


def fold_improvements(fold, y_all):
    arr_path = WEIGHTS_BASE / CV_RUN / f"fold_{fold}" / "test_results.npz"
    with np.load(arr_path) as arr:
        ppe_rmse_global = arr["baseline_global_rmse"]
        ens_rmse_global = arr["seed_mean_global_rmse"]
        ppe_rmse_land = arr["baseline_land_rmse"]
        ens_rmse_land = arr["seed_mean_land_rmse"]
        y_test = arr["test_indices"]

    table_like = {
        "Fold": fold,
        "N": len(y_test),
        "Global PPE RMSE": float(ppe_rmse_global.mean()),
        "Global NN RMSE": float(ens_rmse_global.mean()),
        "Global mean-RMSE improvement (%)": float(
            rmse_improvement(ppe_rmse_global.mean(), ens_rmse_global.mean())
        ),
        "Land PPE RMSE": float(ppe_rmse_land.mean()),
        "Land NN RMSE": float(ens_rmse_land.mean()),
        "Land mean-RMSE improvement (%)": float(
            rmse_improvement(ppe_rmse_land.mean(), ens_rmse_land.mean())
        ),
    }

    return (
        rmse_improvement(ppe_rmse_global, ens_rmse_global),
        rmse_improvement(ppe_rmse_land, ens_rmse_land),
        table_like,
    )


def main():
    sns.set_context("paper", font_scale=1.35)
    sns.set_style("ticks")

    y_all = load_truth()
    rows = []
    table_rows = []

    for fold in PPE_FAMILIES:
        global_imp, land_imp, table_like = fold_improvements(fold, y_all)
        extend_rows(rows, global_imp, fold, "Global")
        extend_rows(rows, land_imp, fold, "Land Only")
        table_rows.append(table_like)

    df = pd.DataFrame(rows)
    fold_order = PPE_FAMILIES
    palette = {
        "GA7": "#E69F00",
        "GA8": "#56B4E9",
        "GA9": "#009E73",
    }

    summary = (
        df.groupby(["Region", "Held-out PPE"])["RMSE Improvement (%)"]
        .agg(["mean", "median"])
        .round(2)
    )
    print(pd.DataFrame(table_rows).round(2).to_string(index=False))
    print()
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, region, letter, show_legend in zip(
        axes, ["Global", "Land Only"], ["A", "B"], [False, True]
    ):
        subset = df[df["Region"] == region]

        sns.kdeplot(
            data=subset,
            x="RMSE Improvement (%)",
            hue="Held-out PPE",
            hue_order=fold_order,
            palette=palette,
            fill=True,
            alpha=0.15,
            linewidth=2,
            common_norm=False,
            legend=show_legend,
            ax=ax,
        )

        for fold in fold_order:
            values = subset.loc[
                subset["Held-out PPE"] == fold, "RMSE Improvement (%)"
            ].values
            median_val = np.median(values)
            kde = stats.gaussian_kde(values)
            kde_at_median = kde(median_val)[0]
            ax.vlines(
                median_val,
                0,
                kde_at_median,
                color=palette[fold],
                linestyle="--",
                alpha=0.85,
                linewidth=1.4,
            )

        ax.axvline(0, color=".3", linestyle="-", linewidth=1)
        ax.set_xlim(-40, 55)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{letter}. {region}", fontweight="bold", pad=15)
        ax.set_xlabel("RMSE Improvement (%)")
        ax.set_ylabel("Density" if region == "Global" else "")

        if show_legend:
            sns.move_legend(ax, "upper right", frameon=False, title=None)

    sns.despine(trim=True)
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()

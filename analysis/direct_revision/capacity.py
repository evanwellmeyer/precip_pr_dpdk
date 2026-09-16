from pathlib import Path
from analysis.paths import research, project

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


WEIGHTS_BASE = Path(str(research / "weights/direct_dpdk_bilinear"))
OUT_DIR = Path("analysis/direct_revision/figures")

CHANNELS = [8, 16, 32, 64, 128]
DROP_FORMATS = [
    {"suffix": "", "label": "0", "color": "#3B73B9", "marker": "o"},
    {"suffix": "_dr0.1", "label": "0.10", "color": "#D77927", "marker": "s"},
]

RUN_TEMPLATE = "unet_direct_dpdk_flat_ch{ch}_k3_bins64_min-700_max1200_sigma0.6_dropout{suffix}"


def run_dir(ch, suffix):
    return WEIGHTS_BASE / RUN_TEMPLATE.format(ch=ch, suffix="0.1" if suffix else "0")


def pct_improve(baseline_rmse, model_rmse):
    baseline_rmse = np.asarray(baseline_rmse, dtype=float)
    model_rmse = np.asarray(model_rmse, dtype=float)
    return (1.0 - model_rmse / (baseline_rmse + 1e-12)) * 100.0


def load_result(ch, suffix):
    path = run_dir(ch, suffix) / "test_results.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path)
    idx = data["test_indices"]
    good = set(data["good_seeds"].astype(int).tolist())

    ppe = data["baseline_global_rmse"][idx]
    ppe_land = data["baseline_land_rmse"][idx]
    ens = data["seed_mean_global_rmse"][idx]
    ens_land = data["seed_mean_land_rmse"][idx]
    members = data["seed_global_rmse"][:, idx]
    members_land = data["seed_land_rmse"][:, idx]

    return {
        "ens_global": float(np.nanmedian(pct_improve(ppe, ens))),
        "ens_land": float(np.nanmedian(pct_improve(ppe_land, ens_land))),
        "seed_global": np.nanmedian(pct_improve(ppe[None, :], members), axis=1),
        "seed_land": np.nanmedian(pct_improve(ppe_land[None, :], members_land), axis=1),
        "good": good,
        "n_seed": members.shape[0],
    }


def load_sweep():
    return {
        cfg["suffix"]: {ch: load_result(ch, cfg["suffix"]) for ch in CHANNELS}
        for cfg in DROP_FORMATS
    }


def configure_axis(ax, ylabel=None):
    ax.axhline(0, color="0.55", linestyle=":", linewidth=0.9, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CHANNELS)
    ax.set_xticklabels([str(ch) for ch in CHANNELS])
    ax.set_xlim(7, 145)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Base channel width")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.55, alpha=0.22)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.12)
    ax.tick_params(axis="both", length=3.2, width=0.75, color="0.25")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_main_figure(records):
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9.0,
        "legend.title_fontsize": 9.0,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.55), sharey=True)
    offsets = {"": -0.035, "_dr0.1": 0.035}
    seed_jitter = np.linspace(-0.015, 0.015, 10)
    panel_specs = [
        (axes[0], "seed_global", "ens_global", "Global"),
        (axes[1], "seed_land", "ens_land", "Land only"),
    ]

    for ax, seed_key, ens_key, title in panel_specs:
        for cfg in DROP_FORMATS:
            xs = []
            vals = []
            suffix = cfg["suffix"]
            for ch in CHANNELS:
                rec = records[suffix][ch]
                x_line = ch * (2 ** offsets[suffix])
                xs.append(x_line)
                vals.append(rec[ens_key])
                for i, val in enumerate(rec[seed_key]):
                    jitter = seed_jitter[i] if i < len(seed_jitter) else 0.0
                    ax.scatter(
                        ch * (2 ** (offsets[suffix] + jitter)),
                        val,
                        s=15,
                        marker=cfg["marker"],
                        facecolor=cfg["color"],
                        edgecolor=cfg["color"],
                        linewidth=0.65,
                        alpha=0.33,
                        zorder=2,
                    )

            ax.plot(
                xs,
                vals,
                color=cfg["color"],
                marker=cfg["marker"],
                label=cfg["label"],
                linewidth=2.0,
                markersize=5.7,
                markeredgewidth=0.7,
                solid_capstyle="round",
                zorder=4,
            )

        ax.set_title(title, pad=5)
        configure_axis(ax, "Median per-member RMSE improvement (%)" if ax is axes[0] else None)

    dropout_handles = [
        Line2D(
            [0],
            [0],
            color=cfg["color"],
            marker=cfg["marker"],
            lw=2.0,
            ms=5.7,
            label=cfg["label"],
        )
        for cfg in DROP_FORMATS
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="o",
            lw=2.0,
            ms=5.7,
            label="NN seed mean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.25",
            markerfacecolor="0.25",
            linestyle="None",
            ms=4.8,
            label="individual seed",
        ),
    ]

    legend = axes[1].legend(
        handles=dropout_handles,
        title="Dropout",
        frameon=False,
        loc="lower right",
        handlelength=1.8,
        borderaxespad=0.35,
        labelspacing=0.45,
    )
    legend._legend_box.align = "left"
    axes[0].legend(
        handles=seed_handles,
        frameon=False,
        loc="lower left",
        borderaxespad=0.35,
        labelspacing=0.35,
    )

    fig.tight_layout(pad=0.45, w_pad=1.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig3.png"
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_summary(records):
    print("Channel sweep median RMSE improvement (%)")
    for cfg in DROP_FORMATS:
        print(f"\nDropout {cfg['label']}")
        for ch in CHANNELS:
            rec = records[cfg["suffix"]][ch]
            print(
                f"  {ch:>3} ch: global={rec['ens_global']:5.2f}, "
                f"land={rec['ens_land']:5.2f}, included={len(rec['good'])}/{rec['n_seed']}"
            )


def main():
    records = load_sweep()
    print_summary(records)
    fig3 = make_main_figure(records)
    print(f"\nSaved {fig3}")


if __name__ == "__main__":
    main()

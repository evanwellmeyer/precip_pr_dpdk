# Precipitation response prediction

This repository contains the code used to construct the direct dPdK target, train the probabilistic U-Net, evaluate the baselines and model, and regenerate the paper's figures. The HadGEM perturbed parameter ensemble (PPE) data and the paired CMIP6 amip/amip-future4K data are not included. Access to those source datasets is required to rerun the analyses.

The fixed random split for the main experiment is [data_splits.npz](data_splits.npz). It contains the `train`, `val`, and `test` realization indices (1060, 151, and 304 members). Keep the input realization order unchanged when using this split.

## Setup

Create the environment with `conda env create -f environment.yml`, then activate it with `conda activate precip-pr-dpdk`. The original runs used Python 3.11 and Apple MPS; the Python training scripts can also select CPU. Training the full sweep takes substantial time and storage.

Set `PRECIP_RESEARCH_DIR` to a directory containing `HadGEM/`, `AMIP/`, and a writable `weights/` directory. If it is unset, the code uses the parent directory of this repository. The shell runners use the current `python`; set `PRECIP_PYTHON` to another interpreter if needed.

```bash
export PRECIP_RESEARCH_DIR=/path/to/research
```

The HadGEM source climatologies expected in `HadGEM/` are:

```text
GA7_pr_his_clim.nc   GA7_pr_fut_clim.nc   GA7_ts_his_clim.nc   GA7_ts_fut_clim.nc
GA8_pr_his_clim.nc   GA8_pr_fut_clim.nc   GA8_tas_his_clim.nc  GA8_tas_fut_clim.nc
GA9_pr_his_clim.nc   GA9_pr_fut_clim.nc   GA9_tas_his_clim.nc  GA9_tas_fut_clim.nc
hadgem_landmask_rg128.nc
```

The AMIP analysis uses paired present day and future4K precipitation and temperature climatologies for 11 models. Their expected directory structure and filenames are in `analysis/direct_revision/amip_target.py`. The climatology length analysis also uses monthly HadGEM3-GC31-LL amip and amip-future4K fields listed in `analysis/direct_revision/climatology.py`.

## Reproduce the results

Run these commands from the repository root, in order:

```bash
python make_direct_dpdk_dataset.py \
  --source-dir "$PRECIP_RESEARCH_DIR/HadGEM" \
  --output-dir "$PRECIP_RESEARCH_DIR/HadGEM"
./run_bilinear_dpdk_sweep.sh
./run_bilinear_dpdk_cv.sh 64 0.1
python analysis/reproduce_manuscript.py
```

The dataset builder forms dPdK from the difference between warming and present precipitation climatologies divided by each member's cosine latitude weighted global mean temperature change. It then regrids both PR and dPdK to the same 128 by 192 grid with periodic bilinear interpolation. The resulting files are `GA789_PR_bilinear_rg128.nc` and `GA789_dPdK_bilinear_rg128.nc`.

The main sweep trains ten seeds for each channel width (8, 16, 32, 64, 128) and dropout rate (0 or 0.1), then evaluates every configuration. The chosen 64 channel, 0.1 dropout model has the lowest median validation RMSE. The PPE holdout workflow trains three new seeds for each withheld family. Both runners resume completed training runs using the saved weights directory; inspect their logs before rerunning a partial experiment.

The last command runs the baseline and figure scripts in dependency order, including calibration before the uncertainty maps and AMIP target construction before transfer evaluation. Outputs and a source hash manifest appear under `analysis/direct_revision/`. It needs the trained checkpoints, HadGEM data, AMIP data, and land mask. The scripts use the fixed model configuration and do not refit it using the test set. The Gaussian comparison selects its scale using test results, as described in the study.

The model is defined in `unet.py`. The dataset builder is in `analysis/uniform_revision/make_uniform_hadgem_data.py`; the top-level builder forwards to it. The training and evaluation programs are the four `*_direct_dpdk_*` files at the repository root. The figure and benchmark programs are the `.py` files in `analysis/direct_revision/`.

This code repository does not contain the restricted source data or trained checkpoints, so a fresh clone cannot regenerate the numeric results until those inputs are supplied.

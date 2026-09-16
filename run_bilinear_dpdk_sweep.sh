#!/bin/zsh

set -euo pipefail

project="${0:A:h}"
python="${PRECIP_PYTHON:-python}"
research="${PRECIP_RESEARCH_DIR:-${project:h}}"
input="$research/HadGEM/GA789_PR_bilinear_rg128.nc"
target="$research/HadGEM/GA789_dPdK_bilinear_rg128.nc"
split="$project/data_splits.npz"
weights="$research/weights/direct_dpdk_bilinear"
logs="$project/analysis/uniform_revision/logs"

if [[ ! -f "$input" || ! -f "$target" ]]; then
    echo "The bilinear input or target dataset is missing"
    exit 1
fi

mkdir -p "$logs"
cd "$project"

for dropout in 0.0 0.1; do
    for channels in 8 16 32 64 128; do
        label="channels${channels}_dropout${dropout}"
        for seed in {0..9}; do
            echo "Starting $label seed $seed"
            "$python" train_direct_dpdk_sweep.py \
                --channels "$channels" \
                --dropout "$dropout" \
                --first-seed "$seed" \
                --number-of-seeds 1 \
                --input-path "$input" \
                --target-path "$target" \
                --weights-path "$weights" \
                --split-path "$split" \
                2>&1 | tee -a "$logs/$label.log"
        done
    done
done

for dropout in 0.0 0.1; do
    for channels in 8 16 32 64 128; do
        label="channels${channels}_dropout${dropout}"
        echo "Evaluating $label"
        "$python" evaluate_direct_dpdk_sweep.py \
            --channels "$channels" \
            --dropout "$dropout" \
            --input-path "$input" \
            --target-path "$target" \
            --weights-path "$weights" \
            2>&1 | tee "$logs/test_${label}.log"
    done
done

echo "The bilinear capacity sweep and evaluation are complete"
echo "Select the model configuration before running the leave one PPE out stage"

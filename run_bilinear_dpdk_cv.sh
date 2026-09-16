#!/bin/zsh

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: ./run_bilinear_dpdk_cv.sh CHANNELS DROPOUT"
    echo "Example: ./run_bilinear_dpdk_cv.sh 64 0.1"
    exit 1
fi

channels="$1"
dropout="$2"
project="${0:A:h}"
python="${PRECIP_PYTHON:-python}"
research="${PRECIP_RESEARCH_DIR:-${project:h}}"
input="$research/HadGEM/GA789_PR_bilinear_rg128.nc"
target="$research/HadGEM/GA789_dPdK_bilinear_rg128.nc"
weights="$research/weights/direct_dpdk_bilinear_cv"
logs="$project/analysis/uniform_revision/logs"

if [[ ! -f "$input" || ! -f "$target" ]]; then
    echo "The bilinear input or target dataset is missing"
    exit 1
fi

mkdir -p "$logs"
cd "$project"

for ppe in GA7 GA8 GA9; do
    for seed in {0..2}; do
        echo "Starting leave one PPE out fold $ppe seed $seed"
        "$python" train_direct_dpdk_cv.py \
            --test-ppe "$ppe" \
            --channels "$channels" \
            --dropout "$dropout" \
            --first-seed "$seed" \
            --number-of-seeds 1 \
            --input-path "$input" \
            --target-path "$target" \
            --weights-path "$weights" \
            2>&1 | tee -a "$logs/cv_channels${channels}_dropout${dropout}_${ppe}.log"
    done

    echo "Evaluating leave one PPE out fold $ppe"
    "$python" evaluate_direct_dpdk_cv.py \
        --test-ppe "$ppe" \
        --channels "$channels" \
        --dropout "$dropout" \
        --input-path "$input" \
        --target-path "$target" \
        --weights-path "$weights" \
        2>&1 | tee "$logs/cv_test_channels${channels}_dropout${dropout}_${ppe}.log"
done

echo "The bilinear leave one PPE out training and evaluation are complete"

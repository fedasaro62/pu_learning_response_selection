#!/usr/bin/env bash
# Baseline mixture-proportion estimators (EN, Scott, BBE, DedPU). Trains a
# positive-versus-unlabeled classifier and estimates alpha from its scores.
# Usage: scripts/baseline_estimators.sh [ubuntu|e_commerce] [rPU]
set -euo pipefail

TASK="${1:-ubuntu}"
RPU="${2:-0.8}"

python -u PvU/train_pvu.py \
    --task "$TASK" \
    --rPU "$RPU" \
    --is_training \
    --num_workers 8 \
    --num_estimate_per_epoch 5 \
    --subsample 0.2

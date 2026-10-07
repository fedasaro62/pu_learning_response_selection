#!/usr/bin/env bash
# PU or supervised fine-tuning (stage 2) and test-set evaluation.
# Usage: scripts/finetune.sh [ubuntu|e_commerce] [pu|supervised] [exact|zero|estimated] [seed] [rPU]
set -euo pipefail

TASK="${1:-ubuntu}"
OBJECTIVE="${2:-pu}"
PRIOR="${3:-exact}"
SEED="${4:-0}"
RPU="${5:-0.8}"

python -u FineTuning/Response_selection.py \
    --task "$TASK" \
    --rPU "$RPU" \
    --post-train positive_only \
    --objective "$OBJECTIVE" \
    --prior "$PRIOR" \
    --seed "$SEED" \
    --is_training

#!/usr/bin/env bash
# PU fine-tuning (stage 2) and test-set evaluation.
#   statistical_prior -> PU loss with the exact class prior alpha*
#   pos_neg_finetune  -> PU loss with alpha=0 (unlabeled treated as negative)
# Usage: scripts/finetune.sh [ubuntu|e_commerce] [statistical_prior|pos_neg_finetune] [rPU]
set -euo pipefail

TASK="${1:-ubuntu}"
STRATEGY="${2:-statistical_prior}"
RPU="${3:-0.8}"

python -u FineTuning/Response_selection.py \
    --task "$TASK" \
    --rPU "$RPU" \
    --post_train positive_only \
    --strategy "$STRATEGY" \
    --is_training

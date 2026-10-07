#!/usr/bin/env bash
# PU post-training (stage 1). Trains MLM + URC on the labeled positives only.
# Usage: scripts/post_train.sh [ubuntu|e_commerce] [rPU]
set -euo pipefail

TASK="${1:-ubuntu}"
RPU="${2:-0.8}"

case "$TASK" in
    ubuntu)     SCRIPT="FPT/BERT_post_training/positive_only/ubuntu_final.py";     EPOCHS=25; BERT="bert-base-uncased" ;;
    e_commerce) SCRIPT="FPT/BERT_post_training/positive_only/e-commerce_final.py"; EPOCHS=34; BERT="bert-base-chinese"  ;;
    *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

python -u "$SCRIPT" \
    --data_dir "./data/${TASK}_data/" \
    --output_dir "./FPT/PT_checkpoint/positive_only/${TASK}" \
    --bert_model "$BERT" \
    --rPU "$RPU" \
    --train_batch_size 64 \
    --num_train_epochs "$EPOCHS"

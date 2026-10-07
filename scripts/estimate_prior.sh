#!/usr/bin/env bash
# Our class prior estimator: extract post-trained cosine similarities, then run
# the stability-based threshold selection over them.
# Usage: scripts/estimate_prior.sh [ubuntu|e_commerce] [rPU]
set -euo pipefail

TASK="${1:-ubuntu}"
RPU="${2:-0.8}"

for SPLIT in train dev; do
    python -u FineTuning/estimate_prior_map.py \
        --task "$TASK" \
        --split "$SPLIT" \
        --rPU "$RPU" \
        --post-train positive_only

    python -u tools/estimate_alpha.py \
        --task "$TASK" \
        --split "$SPLIT" \
        --rpu "$RPU"
done

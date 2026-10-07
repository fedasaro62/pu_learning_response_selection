#!/usr/bin/env bash
# Matched vs unmatched cosine similarity distributions, zero-shot and post-trained.
# Usage: scripts/plot_similarities.sh [ubuntu|e_commerce] [rPU]
set -euo pipefail

TASK="${1:-ubuntu}"
RPU="${2:-0.8}"

python -u tools/plot_similarity_distributions.py --task "$TASK" --rPU "$RPU" --zero-shot
python -u tools/plot_similarity_distributions.py --task "$TASK" --rPU "$RPU"

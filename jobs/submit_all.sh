#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs/estimate_prior logs/train_main logs/train_estimated logs/equivalence

submitted=0
summary=""

main_id=$(sbatch --parsable jobs/train_main.job)
submitted=$((submitted + 1))
summary="${summary}  train_main.job       array $main_id"$'
'

prior_files=("resources/prior/alpha_hat_ubuntu_rPU0.8_train.json" "resources/prior/alpha_hat_e_commerce_rPU0.8_train.json")
missing=0
for f in "${prior_files[@]}"; do
    [[ -f "$f" ]] || missing=$((missing + 1))
done

if [[ $missing -eq 0 ]]; then
    echo "prior files present: submitting train_estimated.job with no dependency"
    estimated_id=$(sbatch --parsable jobs/train_estimated.job)
else
    echo "prior files missing ($missing): submitting estimate_prior.job first, train_estimated.job waits on it"
    estimate_id=$(sbatch --parsable jobs/estimate_prior.job)
    submitted=$((submitted + 1))
    summary="${summary}  estimate_prior.job   array $estimate_id"$'
'
    estimated_id=$(sbatch --parsable --dependency=afterok:"$estimate_id" jobs/train_estimated.job)
fi
submitted=$((submitted + 1))
summary="${summary}  train_estimated.job  array $estimated_id"$'
'

echo ""
echo "Summary: $submitted array jobs submitted"
printf "%s" "$summary"
echo "equivalence.job is not submitted here, launch it separately: sbatch jobs/equivalence.job"

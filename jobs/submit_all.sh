#!/usr/bin/env bash
set -euo pipefail

# Change to repo root so relative paths work
cd "$(dirname "$0")/.."

# Create all required log directories
mkdir -p "logs/pu_estimated"
mkdir -p "logs/pu_exact"
mkdir -p "logs/pu_zero"
mkdir -p "logs/supervised_ft"

submitted=0
skipped=0
skipped_jobs=

sbatch "jobs/supervised_ft/ubuntu_seed0.job"
submitted=$((submitted + 1))
sbatch "jobs/supervised_ft/ubuntu_seed1.job"
submitted=$((submitted + 1))
sbatch "jobs/supervised_ft/ubuntu_seed2.job"
submitted=$((submitted + 1))
sbatch "jobs/supervised_ft/e_commerce_seed0.job"
submitted=$((submitted + 1))
sbatch "jobs/supervised_ft/e_commerce_seed1.job"
submitted=$((submitted + 1))
sbatch "jobs/supervised_ft/e_commerce_seed2.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_zero/ubuntu_seed0.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_zero/ubuntu_seed1.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_zero/ubuntu_seed2.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_zero/e_commerce_seed0.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_zero/e_commerce_seed1.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_zero/e_commerce_seed2.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_exact/ubuntu_seed0.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_exact/ubuntu_seed1.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_exact/ubuntu_seed2.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_exact/e_commerce_seed0.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_exact/e_commerce_seed1.job"
submitted=$((submitted + 1))
sbatch "jobs/pu_exact/e_commerce_seed2.job"
submitted=$((submitted + 1))
if [[ -f "resources/prior/alpha_hat_ubuntu_rPU0.8_train.json" ]]; then
    sbatch "jobs/pu_estimated/ubuntu_seed0.job"
    submitted=$((submitted + 1))
else
    echo "SKIP jobs/pu_estimated/ubuntu_seed0.job: missing prior file resources/prior/alpha_hat_ubuntu_rPU0.8_train.json"
    echo "      produce it with: scripts/estimate_prior.sh ubuntu 0.8"
    skipped=$((skipped + 1))
    skipped_jobs="${skipped_jobs} jobs/pu_estimated/ubuntu_seed0.job"
fi
if [[ -f "resources/prior/alpha_hat_ubuntu_rPU0.8_train.json" ]]; then
    sbatch "jobs/pu_estimated/ubuntu_seed1.job"
    submitted=$((submitted + 1))
else
    echo "SKIP jobs/pu_estimated/ubuntu_seed1.job: missing prior file resources/prior/alpha_hat_ubuntu_rPU0.8_train.json"
    echo "      produce it with: scripts/estimate_prior.sh ubuntu 0.8"
    skipped=$((skipped + 1))
    skipped_jobs="${skipped_jobs} jobs/pu_estimated/ubuntu_seed1.job"
fi
if [[ -f "resources/prior/alpha_hat_ubuntu_rPU0.8_train.json" ]]; then
    sbatch "jobs/pu_estimated/ubuntu_seed2.job"
    submitted=$((submitted + 1))
else
    echo "SKIP jobs/pu_estimated/ubuntu_seed2.job: missing prior file resources/prior/alpha_hat_ubuntu_rPU0.8_train.json"
    echo "      produce it with: scripts/estimate_prior.sh ubuntu 0.8"
    skipped=$((skipped + 1))
    skipped_jobs="${skipped_jobs} jobs/pu_estimated/ubuntu_seed2.job"
fi
if [[ -f "resources/prior/alpha_hat_e_commerce_rPU0.8_train.json" ]]; then
    sbatch "jobs/pu_estimated/e_commerce_seed0.job"
    submitted=$((submitted + 1))
else
    echo "SKIP jobs/pu_estimated/e_commerce_seed0.job: missing prior file resources/prior/alpha_hat_e_commerce_rPU0.8_train.json"
    echo "      produce it with: scripts/estimate_prior.sh e_commerce 0.8"
    skipped=$((skipped + 1))
    skipped_jobs="${skipped_jobs} jobs/pu_estimated/e_commerce_seed0.job"
fi
if [[ -f "resources/prior/alpha_hat_e_commerce_rPU0.8_train.json" ]]; then
    sbatch "jobs/pu_estimated/e_commerce_seed1.job"
    submitted=$((submitted + 1))
else
    echo "SKIP jobs/pu_estimated/e_commerce_seed1.job: missing prior file resources/prior/alpha_hat_e_commerce_rPU0.8_train.json"
    echo "      produce it with: scripts/estimate_prior.sh e_commerce 0.8"
    skipped=$((skipped + 1))
    skipped_jobs="${skipped_jobs} jobs/pu_estimated/e_commerce_seed1.job"
fi
if [[ -f "resources/prior/alpha_hat_e_commerce_rPU0.8_train.json" ]]; then
    sbatch "jobs/pu_estimated/e_commerce_seed2.job"
    submitted=$((submitted + 1))
else
    echo "SKIP jobs/pu_estimated/e_commerce_seed2.job: missing prior file resources/prior/alpha_hat_e_commerce_rPU0.8_train.json"
    echo "      produce it with: scripts/estimate_prior.sh e_commerce 0.8"
    skipped=$((skipped + 1))
    skipped_jobs="${skipped_jobs} jobs/pu_estimated/e_commerce_seed2.job"
fi

echo ""
echo "Summary: $submitted submitted, $skipped skipped"
if [[ $skipped -gt 0 ]]; then
    echo "Skipped jobs:$skipped_jobs"
fi

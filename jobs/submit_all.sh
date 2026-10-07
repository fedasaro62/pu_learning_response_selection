#!/usr/bin/env bash
set -euo pipefail

sbatch jobs/supervised_ft/ubuntu_seed1.job
sbatch jobs/supervised_ft/ubuntu_seed2.job
sbatch jobs/supervised_ft/e_commerce_seed1.job
sbatch jobs/supervised_ft/e_commerce_seed2.job
sbatch jobs/pu_zero/ubuntu_seed1.job
sbatch jobs/pu_zero/ubuntu_seed2.job
sbatch jobs/pu_zero/e_commerce_seed1.job
sbatch jobs/pu_zero/e_commerce_seed2.job
sbatch jobs/pu_exact/ubuntu_seed1.job
sbatch jobs/pu_exact/ubuntu_seed2.job
sbatch jobs/pu_exact/e_commerce_seed1.job
sbatch jobs/pu_exact/e_commerce_seed2.job
sbatch jobs/pu_estimated/ubuntu_seed0.job
sbatch jobs/pu_estimated/ubuntu_seed1.job
sbatch jobs/pu_estimated/ubuntu_seed2.job
sbatch jobs/pu_estimated/e_commerce_seed0.job
sbatch jobs/pu_estimated/e_commerce_seed1.job
sbatch jobs/pu_estimated/e_commerce_seed2.job

#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

# shell expansion written verbatim into the job files: PU_VENV overrides the default repo-root .venv at submit time
VENV_ACTIVATE = '"${PU_VENV:-.venv}/bin/activate"'
RPU = 0.8
# matches both the account's MaxJobs limit and the number of 3g.40gb gpus on the cluster
MAX_CONCURRENT_ARRAY_ELEMENTS = 4

HEADER = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time={time}
#SBATCH --partition=A100
#SBATCH --gpus=3g.40gb:1
{array}#SBATCH --output=logs/{group}/{log_name}.out

# under sbatch $0 is a spooled copy, so rely on the submit dir (submit_all.sh cds to the repo root)
cd "${{SLURM_SUBMIT_DIR:-$(dirname "$0")/..}}"
"""

ARRAY_BODY = """source {venv_activate}
CONFIGS=(
{configs}
)
read -r {fields} <<< "${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}"
echo "config: {echo}"
{command}
"""

TRAIN_FIELDS = ("TASK", "OBJECTIVE", "PRIOR", "SEED")
TRAIN_COMMAND = f'scripts/finetune.sh "$TASK" "$OBJECTIVE" "$PRIOR" "$SEED" {RPU}'
ESTIMATE_PRIOR_TIME = "12:00:00"
EQUIVALENCE_TIME = "03:00:00"
TRAIN_TIME = "48:00:00"

PLAN = [
    ("supervised_ft", "supervised", "exact", (0, 1, 2)),
    ("pu_zero", "pu", "zero", (0, 1, 2)),
    ("pu_exact", "pu", "exact", (0, 1, 2)),
    ("pu_estimated", "pu", "estimated", (0, 1, 2)),
]
TASKS = ("ubuntu", "e_commerce")



def train_configs(rows: set[str]) -> list[tuple[str, str, str, int]]:
    """Return (task, objective, prior, seed) for the given PLAN rows, in PLAN order."""
    return [
        (task, objective, prior, seed)
        for row, objective, prior, seeds in PLAN if row in rows
        for task in TASKS
        for seed in seeds
    ]


def render_array(group: str, name: str, time: str, fields: tuple[str, ...], configs: list[tuple], command: str) -> str:
    """Render an array job script whose index selects a line of the CONFIGS bash array."""
    header = HEADER.format(name=name, time=time, group=group, log_name="%A_%a",
                           array=f"#SBATCH --array=0-{len(configs) - 1}%{MAX_CONCURRENT_ARRAY_ELEMENTS}\n")
    body = ARRAY_BODY.format(
        venv_activate=VENV_ACTIVATE,
        configs="\n".join(f'  "{" ".join(map(str, c))}"' for c in configs),
        fields=" ".join(fields),
        echo=" ".join(f"{f.lower()}=${f}" for f in fields),
        command=command,
    )
    return header + body


def render_equivalence() -> str:
    header = HEADER.format(name="equiv_check", time=EQUIVALENCE_TIME, group="equivalence", log_name="%j", array="")
    return header + "scripts/check_equivalence.sh\n"


def generate_jobs() -> dict[str, str]:
    """Map each generated job path (relative to jobs/) to its content."""
    main_rows = {row for row, _, prior, _ in PLAN if prior != "estimated"}
    estimated_rows = {row for row, _, prior, _ in PLAN if prior == "estimated"}
    return {
        "estimate_prior.job": render_array(
            "estimate_prior", "est_prior", ESTIMATE_PRIOR_TIME, ("TASK",), [(t,) for t in TASKS],
            f'scripts/estimate_prior.sh "$TASK" {RPU}'),
        "train_main.job": render_array(
            "train_main", "train_main", TRAIN_TIME, TRAIN_FIELDS, train_configs(main_rows), TRAIN_COMMAND),
        "train_estimated.job": render_array(
            "train_estimated", "train_est", TRAIN_TIME, TRAIN_FIELDS, train_configs(estimated_rows), TRAIN_COMMAND),
        "equivalence.job": render_equivalence(),
        "submit_all.sh": render_submit_script(),
    }


def main() -> None:
    jobs = Path("jobs")
    jobs.mkdir(exist_ok=True)
    for name, content in generate_jobs().items():
        (jobs / name).write_text(content)
    (jobs / "submit_all.sh").chmod(0o755)
    for group in ("estimate_prior", "train_main", "train_estimated", "equivalence"):
        Path("logs", group).mkdir(parents=True, exist_ok=True)
    print(f"wrote {len(generate_jobs())} files under {jobs}")


def render_submit_script() -> str:
    """Render submit_all.sh: train_main always, train_estimated after the prior files exist or are estimated."""
    prior_files = " ".join(f'"resources/prior/alpha_hat_{t}_rPU{RPU}_train.json"' for t in TASKS)
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs/estimate_prior logs/train_main logs/train_estimated logs/equivalence

submitted=0
summary=""

main_id=$(sbatch --parsable jobs/train_main.job)
submitted=$((submitted + 1))
summary="${{summary}}  train_main.job       array $main_id"$'\n'

prior_files=({prior_files})
missing=0
for f in "${{prior_files[@]}}"; do
    [[ -f "$f" ]] || missing=$((missing + 1))
done

if [[ $missing -eq 0 ]]; then
    echo "prior files present: submitting train_estimated.job with no dependency"
    estimated_id=$(sbatch --parsable jobs/train_estimated.job)
else
    echo "prior files missing ($missing): submitting estimate_prior.job first, train_estimated.job waits on it"
    estimate_id=$(sbatch --parsable jobs/estimate_prior.job)
    submitted=$((submitted + 1))
    summary="${{summary}}  estimate_prior.job   array $estimate_id"$'\n'
    estimated_id=$(sbatch --parsable --dependency=afterok:"$estimate_id" jobs/train_estimated.job)
fi
submitted=$((submitted + 1))
summary="${{summary}}  train_estimated.job  array $estimated_id"$'\n'

echo ""
echo "Summary: $submitted array jobs submitted"
printf "%s" "$summary"
echo "equivalence.job is not submitted here, launch it separately: sbatch jobs/equivalence.job"
"""


if __name__ == "__main__":
    main()

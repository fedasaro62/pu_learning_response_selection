#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

VENV_PATH = "/nfs/home/dasaro/research/innonext-BERT_FP/.venv/bin/activate"
RPU = 0.8

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={task}_{row}_seed{seed}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=msteams:dasaro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=A100
#SBATCH --gpus=3g.40gb:1
#SBATCH --output=logs/{row}/{task}_seed{seed}.out

source {venv_path}
scripts/finetune.sh {task} {objective} {prior} {seed} {rpu}
"""

# seed 0 is reused from the previous campaign for the three pre-existing rows
PLAN = [
    ("supervised_ft", "supervised", "exact", (1, 2)),
    ("pu_zero", "pu", "zero", (1, 2)),
    ("pu_exact", "pu", "exact", (1, 2)),
    ("pu_estimated", "pu", "estimated", (0, 1, 2)),
]
TASKS = ("ubuntu", "e_commerce")


def main() -> None:
    written = []
    log_dirs = set()
    for row, objective, prior, seeds in PLAN:
        for task in TASKS:
            for seed in seeds:
                path = Path("jobs") / row / f"{task}_seed{seed}.job"
                path.parent.mkdir(parents=True, exist_ok=True)
                log_dir = f"logs/{row}"
                log_dirs.add(log_dir)
                path.write_text(TEMPLATE.format(task=task, row=row, objective=objective,
                                                prior=prior, seed=seed, venv_path=VENV_PATH,
                                                rpu=RPU))
                written.append((path, row, task, objective, prior, seed))

    # verify the file count matches the expected matrix size
    expected_count = sum(len(seeds) * len(TASKS) for _, _, _, seeds in PLAN)
    assert len(written) == expected_count, (
        f"generated {len(written)} files but PLAN implies {expected_count}"
    )

    submit = Path("jobs/submit_all.sh")
    submit.write_text(_generate_submit_script(written, log_dirs, PLAN, TASKS))
    submit.chmod(0o755)
    print(f"wrote {len(written)} job files and {submit}")


def _generate_submit_script(
    written: list[tuple[Path, str, str, str, str, int]], log_dirs: set[str],
    plan: list[tuple[str, str, str, tuple[int, ...]]], tasks: tuple[str, ...]
) -> str:
    """Generate submit_all.sh with prior file checks and log directory creation."""
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Change to repo root so relative paths work",
        'cd "$(dirname "$0")/.."',
        "",
        "# Create all required log directories",
    ]

    for log_dir in sorted(log_dirs):
        lines.append(f'mkdir -p "{log_dir}"')

    lines.extend([
        "",
        "submitted=0",
        "skipped=0",
        "skipped_jobs=",
        "",
    ])

    # Generate sbatch commands with prior checks
    for path, row, task, objective, prior, seed in written:
        if prior == "estimated":
            prior_file = f"resources/prior/alpha_hat_{task}_rPU{RPU}_train.json"
            lines.append(f'if [[ -f "{prior_file}" ]]; then')
            lines.append(f'    sbatch "{path}"')
            lines.append(f'    ((submitted++))')
            lines.append(f'else')
            lines.append(
                f'    echo "SKIP {path}: missing prior file {prior_file}"'
            )
            lines.append(
                f'    echo "      produce it with: scripts/estimate_prior.sh {task} {RPU}"'
            )
            lines.append(f'    ((skipped++))')
            lines.append(
                f'    skipped_jobs="${{skipped_jobs}} {path}"'
            )
            lines.append(f'fi')
        else:
            lines.append(f'sbatch "{path}"')
            lines.append(f'((submitted++))')

    lines.extend([
        "",
        'echo ""',
        'echo "Summary: $submitted submitted, $skipped skipped"',
        'if [[ $skipped -gt 0 ]]; then',
        '    echo "Skipped jobs:$skipped_jobs"',
        'fi',
    ])

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

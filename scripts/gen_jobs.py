#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

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

source /nfs/home/dasaro/research/innonext-BERT_FP/.venv/bin/activate
scripts/finetune.sh {task} {objective} {prior} {seed} 0.8
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
    for row, objective, prior, seeds in PLAN:
        for task in TASKS:
            for seed in seeds:
                path = Path("jobs") / row / f"{task}_seed{seed}.job"
                path.parent.mkdir(parents=True, exist_ok=True)
                Path(f"logs/{row}").mkdir(parents=True, exist_ok=True)
                path.write_text(TEMPLATE.format(task=task, row=row, objective=objective,
                                                prior=prior, seed=seed))
                written.append(path)

    submit = Path("jobs/submit_all.sh")
    submit.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n"
                      + "".join(f"sbatch {p}\n" for p in written))
    submit.chmod(0o755)
    print(f"wrote {len(written)} job files and {submit}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys

ROW_TAGS = {"zero": "pu_zero", "exact": "pu_exact", "estimated": "pu_estimated"}


POST_TRAIN_TAGS = {"original": "fs_pt_ft", "zero_shot": "zero_shot_ft"}


def row_tag(objective: str, prior: str, post_train: str) -> str:
    """Table row a run belongs to; the checkpoint overrides the objective for non-ours post-training."""
    if post_train in POST_TRAIN_TAGS:
        return POST_TRAIN_TAGS[post_train]
    return "supervised_ft" if objective == "supervised" else ROW_TAGS[prior]


def run_paths(task: str, row: str, seed: int, rPU: float | None) -> tuple[str, str, str]:
    """Returns (save_path, score_file_path, metrics_path). Pure: callers create the directories."""
    base = f"./FineTuning/FT_checkpoint/{task}/{row}/seed{seed}"
    return (
        f"{base}/bert_rPU{rPU}.pt",
        f"{base}/scorefile_rPU{rPU}.txt",
        f"resources/metrics/{task}_{row}_rPU{rPU}_seed{seed}.json",
    )


def validate_args(objective: str, post_train: str, rPU: float | None) -> None:
    if rPU is None and post_train == "positive_only":
        sys.exit(
            "--rPU is required with --post-train positive_only: it names the checkpoint "
            "(bert_rPU<r>.pt) as well as the PU label column"
        )
    if rPU is None and objective == "pu":
        sys.exit("--rPU is required with --objective pu: it selects the y_pu_<rPU> label column the PU loss reads")


def write_metrics(path: str, task: str, row: str, rPU: float | None, seed: int, alpha: float | None,
                  result: tuple[float, ...], source: str, commit: str | None, wall_minutes: float | None) -> None:
    keys = ("MAP", "MRR", "P@1", "R10@1", "R10@2", "R10@5")
    payload = {
        "task": task, "row": row, "rPU": rPU, "seed": seed, "alpha": alpha,
        **dict(zip(keys, result)),
        "source": source, "commit": commit, "wall_minutes": wall_minutes,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

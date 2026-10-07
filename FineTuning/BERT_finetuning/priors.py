from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

PRIOR_DIR = Path("resources/prior")


def exact_prior(y: Sequence[int], rPU: float) -> float:
    n_pos = int(sum(y))
    n_labeled = round((1 - rPU) * n_pos)
    return (n_pos - n_labeled) / (len(y) - n_labeled)


def estimated_prior_path(task: str, rPU: float, split: str = "train") -> Path:
    return PRIOR_DIR / f"alpha_hat_{task}_rPU{rPU}_{split}.json"


def load_estimated_prior(task: str, rPU: float) -> float:
    path = estimated_prior_path(task, rPU)
    if not path.exists():
        raise FileNotFoundError(
            f"no estimated prior at {path}; produce it with:\n"
            f"    scripts/estimate_prior.sh {task} {rPU}"
        )

    payload = json.loads(path.read_text())
    if payload.get("split") != "train":
        raise ValueError(
            f"{path} holds a '{payload.get('split')}' split estimate; "
            "only the train split may be used to train"
        )
    return float(payload["alpha_hat"])


def resolve_prior(objective: str, prior: str, task: str, rPU: float, y: Sequence[int]) -> float | None:
    if objective == "supervised":
        return None

    sources = {
        "zero": lambda: 0.0,
        "exact": lambda: exact_prior(y, rPU),
        "estimated": lambda: load_estimated_prior(task, rPU),
    }
    return sources[prior]()

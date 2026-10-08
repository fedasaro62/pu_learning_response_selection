# Window ablation report

## tools/ablate_window.py
```python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from FineTuning.BERT_finetuning.priors import PRIOR_DIR, estimated_prior_path  # noqa: E402
from tools.estimate_alpha import STABLE_TAU_WINDOW, select_stable_tau  # noqa: E402

DEFAULT_WINDOWS = (3, 5, 7, 9, 11)
TASKS = ("ubuntu", "e_commerce")
TASK_LABELS = {"ubuntu": "Ubuntu", "e_commerce": "E-Comm"}
COLUMNS = ("tau_star", "alpha_hat", "delta")
MD_COLUMN_LABELS = {"tau_star": "τ*", "alpha_hat": "α̂", "delta": r"\|Δ\|"}
TEX_COLUMN_LABELS = {"tau_star": r"$\tau^{*}$", "alpha_hat": r"$\hat{\alpha}$", "delta": r"$|\Delta|$"}
FORMATS = {"tau_star": "{:.2f}", "alpha_hat": "{:.3f}", "delta": "{:.3f}"}

# one row per window: {task: {"tau_star": ..., "alpha_hat": ..., "delta": ...}}
Ablation = dict[int, dict[str, dict[str, float]]]


def load_curve(prior_dir: Path, task: str, rpu: float, split: str) -> dict:
    path = prior_dir / estimated_prior_path(task, rpu, split).name
    if not path.exists():
        raise FileNotFoundError(
            f"no estimated prior at {path}; produce it with:\n"
            f"    scripts/estimate_prior.sh {task} {rpu}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def sweep_windows(curve: dict, windows: tuple[int, ...]) -> dict[int, dict[str, float]]:
    """Re-select tau* on a persisted curve for each window, using the method's own selector."""
    taus, alphas = np.asarray(curve["taus"], dtype=float), np.asarray(curve["alpha_of_tau"], dtype=float)
    out = {}
    for window in windows:
        try:
            tau_star, alpha_hat = select_stable_tau(taus, alphas, window=window)
        except ValueError as exc:
            raise ValueError(f"invalid window {window} for a curve of {len(taus)} grid points: {exc}") from exc
        out[window] = {
            "tau_star": tau_star, "alpha_hat": alpha_hat, "delta": abs(alpha_hat - float(curve["alpha_exact"])),
        }
    return out


def ablate(prior_dir: Path, rpu: float, split: str, windows: tuple[int, ...], tasks: tuple[str, ...] = TASKS) -> Ablation:
    curves = {task: load_curve(prior_dir, task, rpu, split) for task in tasks}
    per_task = {task: sweep_windows(curve, windows) for task, curve in curves.items()}
    return {w: {task: per_task[task][w] for task in tasks} for w in windows}


def _cells(row: dict[str, dict[str, float]]) -> list[str]:
    return [FORMATS[c].format(values[c]) for values in row.values() for c in COLUMNS]


def render_markdown(ablation: Ablation) -> str:
    tasks = list(next(iter(ablation.values())))
    header = "| w | " + " | ".join(f"{TASK_LABELS[t]} {MD_COLUMN_LABELS[c]}" for t in tasks for c in COLUMNS) + " |"
    lines = [header, "| --- " * (1 + len(tasks) * len(COLUMNS)) + "|"]
    for window, row in ablation.items():
        label = f"{window} (paper)" if window == STABLE_TAU_WINDOW else str(window)
        lines.append(f"| {label} | " + " | ".join(_cells(row)) + " |")
    return "\n".join(lines)


def render_latex(ablation: Ablation) -> str:
    tasks = list(next(iter(ablation.values())))
    n = len(COLUMNS)
    spans = [(2 + i * n, 1 + (i + 1) * n) for i in range(len(tasks))]
    lines = [r"\begin{tabular}{l" + "c" * (len(tasks) * n) + "}", r"\toprule"]
    lines.append(" & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{TASK_LABELS[t]}}}" for t in tasks) + r" \\")
    lines.append(" ".join(rf"\cmidrule(lr){{{a}-{b}}}" for a, b in spans))
    lines.append("$w$ & " + " & ".join(TEX_COLUMN_LABELS[c] for _ in tasks for c in COLUMNS) + r" \\")
    lines.append(r"\midrule")
    for window, row in ablation.items():
        cells = _cells(row)
        if window == STABLE_TAU_WINDOW:
            cells = [rf"\textbf{{{c}}}" for c in [str(window)] + cells]
        else:
            cells = [str(window)] + cells
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity of the selected tau* to the stability window.")
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--rpu", type=float, default=0.8)
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--prior-dir", type=Path, default=PRIOR_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("resources"))
    args = parser.parse_args()

    ablation = ablate(args.prior_dir, args.rpu, args.split, tuple(args.windows))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "window_ablation.md").write_text(render_markdown(ablation) + "\n", encoding="utf-8")
    (args.out_dir / "window_ablation.tex").write_text(render_latex(ablation) + "\n", encoding="utf-8")
    print(render_markdown(ablation))


if __name__ == "__main__":
    main()
```

## Tests
```
.......                                                                  [100%]
7 passed in 1.70s
```

## Sample (synthetic: ubuntu = flat-bottomed bowl, e_commerce = monotone decay)
| w | Ubuntu τ* | Ubuntu α̂ | Ubuntu \|Δ\| | E-Comm τ* | E-Comm α̂ | E-Comm \|Δ\| |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.70 | 0.400 | 0.030 | 0.89 | 0.096 | 0.334 |
| 5 (paper) | 0.70 | 0.400 | 0.030 | 0.88 | 0.102 | 0.328 |
| 7 | 0.70 | 0.400 | 0.030 | 0.87 | 0.109 | 0.321 |
| 9 | 0.70 | 0.400 | 0.030 | 0.86 | 0.115 | 0.315 |
| 11 | 0.70 | 0.400 | 0.030 | 0.85 | 0.122 | 0.308 |

```latex
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{Ubuntu} & \multicolumn{3}{c}{E-Comm} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
$w$ & $\tau^{*}$ & $\hat{\alpha}$ & $|\Delta|$ & $\tau^{*}$ & $\hat{\alpha}$ & $|\Delta|$ \\
\midrule
3 & 0.70 & 0.400 & 0.030 & 0.89 & 0.096 & 0.334 \\
\textbf{5} & \textbf{0.70} & \textbf{0.400} & \textbf{0.030} & \textbf{0.88} & \textbf{0.102} & \textbf{0.328} \\
7 & 0.70 & 0.400 & 0.030 & 0.87 & 0.109 & 0.321 \\
9 & 0.70 & 0.400 & 0.030 & 0.86 & 0.115 & 0.315 \\
11 & 0.70 & 0.400 & 0.030 & 0.85 & 0.122 & 0.308 \\
\bottomrule
\end{tabular}
```

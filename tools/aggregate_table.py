from __future__ import annotations

import argparse
import json
import statistics
import warnings
from collections import defaultdict
from pathlib import Path

METRICS = ("R10@1", "R10@2", "R10@5")
TASKS = ("ubuntu", "e_commerce")
ROW_ORDER = ("fs_pt_ft", "supervised_ft", "pu_zero", "pu_exact", "pu_estimated")
ROW_LABELS = {
    "fs_pt_ft": r"FS$_{\mathrm{PT+FT}}$ \cite{han2021fine}",
    "supervised_ft": r"PU$_{\mathrm{PT}}$ + FS$_{\mathrm{FT}}$",
    "pu_zero": r"PU$_{\mathrm{PT+FT}}$ ($\alpha$=0)",
    "pu_exact": r"PU$_{\mathrm{PT+FT}}$ ($\alpha$=$\alpha^{\star}$)",
    "pu_estimated": r"PU$_{\mathrm{PT+FT}}$ ($\alpha$=$\hat{\alpha}$)",
}
TASK_LABELS = {"ubuntu": "Ubuntu", "e_commerce": "E-Commerce"}
EXPECTED_SEEDS = 2

Aggregate = dict[tuple[str, str], dict[str, tuple[float, float | None, int | None]]]


def load_metrics(metrics_dir: Path) -> list[dict]:
    records = []
    for path in sorted(Path(metrics_dir).glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} is not valid JSON, likely a killed run: {exc}") from exc
    if not records:
        raise ValueError(f"no metrics records found in {metrics_dir}")
    return records


def aggregate(records: list[dict], allow_partial: bool = False) -> Aggregate:
    """Group records by (task, row) and compute mean and sample std (ddof=1) per metric.

    Args:
        records: per-run metrics records.
        allow_partial: render run groups whose size is not EXPECTED_SEEDS instead of raising.

    Raises:
        ValueError: on duplicate seeds, mixed rPU or alpha, mixed cited and run records, a run-sourced
            fs_pt_ft group, or a non-cited group whose size is not EXPECTED_SEEDS (unless allow_partial).
    """
    grouped = defaultdict(list)
    for record in records:
        if record["task"] not in TASKS or record["row"] not in ROW_ORDER:
            raise ValueError(
                f"unknown (task, row) = ({record['task']!r}, {record['row']!r}); "
                f"valid tasks: {TASKS}, valid rows: {ROW_ORDER}"
            )
        grouped[(record["task"], record["row"])].append(record)

    out = {}
    for key, group in grouped.items():
        seeds = [r["seed"] for r in group]
        if (duplicate := next((s for s in seeds if seeds.count(s) > 1), None)) is not None:
            raise ValueError(f"duplicate seed {duplicate} for {key[0]}/{key[1]}")

        if len({r["rPU"] for r in group if r.get("rPU") is not None}) > 1:
            raise ValueError(f"mixed rPU values for {key[0]}/{key[1]}")
        is_cite = [str(r.get("source", "")).startswith("cite:") for r in group]
        if any(is_cite) and not all(is_cite):
            raise ValueError(f"mixed cited and run records for {key[0]}/{key[1]}")
        if key[1] == "fs_pt_ft" and not all(is_cite):
            raise ValueError(
                f"{key[0]}/fs_pt_ft must be cite:-sourced; a local reproduction must not be published "
                "under the Han et al. citation"
            )
        if len({r["alpha"] for r in group if r.get("alpha") is not None}) > 1:
            raise ValueError(f"mixed alpha values for {key[0]}/{key[1]}")
        if len({r.get("commit") for r in group}) > 1:
            warnings.warn(f"{key[0]}/{key[1]}: records come from different commits")
        if not all(is_cite) and len(group) != EXPECTED_SEEDS:
            message = f"{key[0]}/{key[1]}: {len(group)} of {EXPECTED_SEEDS} seeds present"
            if not allow_partial:
                raise ValueError(f"{message}; pass --allow-partial to render it annotated")
            warnings.warn(message)

        out[key] = {
            metric: (statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else None,
                     None if all(is_cite) else len(values))
            for metric in METRICS
            if (values := [r[metric] for r in group])
        }
    return out


def _cell(mean: float, std: float | None, n: int | None = None, annotate: bool = False) -> str:
    text = f"{mean:.3f}" if std is None else f"{mean:.3f} ± {std:.3f}"
    return f"{text} (n={n})" if annotate and n is not None else text


def _partial(agg: Aggregate) -> set[tuple[str, str]]:
    return {key for key, metrics in agg.items() if any(m[2] not in (None, EXPECTED_SEEDS) for m in metrics.values())}


def render_markdown(agg: Aggregate) -> str:
    header = "| Setting | " + " | ".join(f"{task} {m}" for task in TASKS for m in METRICS) + " |"
    rule = "| --- " * (1 + len(TASKS) * len(METRICS)) + "|"
    lines = [header, rule]
    for row in ROW_ORDER:
        if not any((task, row) in agg for task in TASKS):
            continue
        cells = [
            _cell(*agg[(task, row)][m], annotate=True) if (task, row) in agg else "—"
            for task in TASKS
            for m in METRICS
        ]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_latex(agg: Aggregate) -> str:
    partial = _partial(agg)
    n = len(METRICS)
    spans = [(2 + i * n, 1 + (i + 1) * n) for i in range(len(TASKS))]
    lines = [r"\begin{tabular}{l" + "c" * (len(TASKS) * n) + "}", r"\toprule"]
    # plain empty cell above "Setting" instead of \multirow, so no extra package is required
    lines.append(" & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{TASK_LABELS[t]}}}" for t in TASKS) + r" \\")
    lines.append(" ".join(rf"\cmidrule(lr){{{a}-{b}}}" for a, b in spans))
    lines.append("Setting & " + " & ".join(rf"$R_{{10}}@{m[-1]}$" for _ in TASKS for m in METRICS) + r" \\")
    lines.append(r"\midrule")
    for row in ROW_ORDER:
        if not any((task, row) in agg for task in TASKS):
            continue
        cells = [
            _cell(*agg[(task, row)][m], annotate=(task, row) in partial).replace("±", r"$\pm$")
            if (task, row) in agg else "--"
            for task in TASKS
            for m in METRICS
        ]
        lines.append(f"{ROW_LABELS[row]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Table 1 across seeds.")
    parser.add_argument("--metrics-dir", default="resources/metrics", type=Path)
    parser.add_argument("--out-dir", default="results", type=Path)
    parser.add_argument("--allow-partial", action="store_true",
                        help="render groups without exactly 2 seeds, annotated with their count")
    args = parser.parse_args()

    agg = aggregate(load_metrics(args.metrics_dir), allow_partial=args.allow_partial)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "table1.md").write_text(render_markdown(agg) + "\n", encoding="utf-8")
    (args.out_dir / "table1.tex").write_text(render_latex(agg) + "\n", encoding="utf-8")
    print(render_markdown(agg))


if __name__ == "__main__":
    main()

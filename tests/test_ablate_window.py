from __future__ import annotations

import numpy as np
import pytest

from tools.ablate_window import DEFAULT_WINDOWS, ablate, render_latex, render_markdown
from tools.estimate_alpha import STABLE_TAU_WINDOW

GRID = np.arange(0.5, 0.91, 0.01)


@pytest.fixture
def bowl() -> list[float]:
    # steep on both sides, flat bottom centred at tau = 0.70
    return list(0.4 + 20 * (GRID - 0.70) ** 2)


@pytest.fixture
def decreasing() -> list[float]:
    return list(np.exp(-6 * (GRID - 0.5)))  # convex, strictly decreasing, no flat region


def run(tmp_path, write_curve, alphas, alpha_exact=0.5, tasks=("ubuntu", "e_commerce")):
    for task in tasks:
        write_curve(tmp_path, task, alphas, alpha_exact)
    return ablate(tmp_path, 0.8, "train", DEFAULT_WINDOWS)


def test_plateau_gives_identical_tau_across_windows(tmp_path, write_curve, bowl):
    ablation = run(tmp_path, write_curve, bowl)

    taus = {round(row["ubuntu"]["tau_star"], 2) for row in ablation.values()}

    assert taus == {0.70}


def test_no_plateau_gives_tau_that_varies_with_window(tmp_path, write_curve, decreasing):
    ablation = run(tmp_path, write_curve, decreasing)

    taus = [row["ubuntu"]["tau_star"] for row in ablation.values()]

    assert len(set(taus)) == len(DEFAULT_WINDOWS)
    assert taus == sorted(taus, reverse=True)


def test_missing_prior_file_names_file_and_command(tmp_path, write_curve, bowl):
    write_curve(tmp_path, "ubuntu", bowl)

    with pytest.raises(FileNotFoundError) as exc:
        ablate(tmp_path, 0.8, "train", DEFAULT_WINDOWS)

    assert "alpha_hat_e_commerce_rPU0.8_train.json" in str(exc.value)
    assert "scripts/estimate_prior.sh e_commerce 0.8" in str(exc.value)


def test_markdown_has_one_row_per_window_and_marks_paper_window(tmp_path, write_curve, bowl):
    md = render_markdown(run(tmp_path, write_curve, bowl))
    rows = md.splitlines()[2:]

    assert len(rows) == len(DEFAULT_WINDOWS)
    assert [r for r in rows if "(paper)" in r] == [r for r in rows if r.startswith(f"| {STABLE_TAU_WINDOW} (paper)")]
    assert sum("(paper)" in r for r in rows) == 1


def test_latex_follows_aggregate_table_style(tmp_path, write_curve, bowl):
    tex = render_latex(run(tmp_path, write_curve, bowl))

    assert r"\multicolumn{3}{c}{Ubuntu}" in tex
    assert r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}" in tex
    assert tex.count(r"\textbf{") == 7


def test_delta_uses_alpha_exact_from_file(tmp_path, write_curve, bowl):
    # alpha_exact deliberately unrelated to the curve, so recomputing it would give a different answer
    ablation = run(tmp_path, write_curve, bowl, alpha_exact=0.123)

    row = ablation[5]["ubuntu"]

    assert row["delta"] == pytest.approx(abs(row["alpha_hat"] - 0.123))


def test_invalid_window_error_names_the_value(tmp_path, write_curve, bowl):
    write_curve(tmp_path, "ubuntu", bowl)
    write_curve(tmp_path, "e_commerce", bowl)

    with pytest.raises(ValueError, match="invalid window 4"):
        ablate(tmp_path, 0.8, "train", (3, 4))

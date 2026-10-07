from __future__ import annotations

import json

import pytest

from tools.aggregate_table import aggregate, load_metrics, render_latex, render_markdown


def write(dirpath, task, row, seed, r1, source="run"):
    payload = {"task": task, "row": row, "rPU": 0.8, "seed": seed, "alpha": 0.44,
               "MAP": 0.9, "MRR": 0.9, "P@1": r1, "R10@1": r1, "R10@2": 0.92,
               "R10@5": 0.98, "source": source, "commit": None, "wall_minutes": None}
    (dirpath / f"{task}_{row}_rPU0.8_seed{seed}.json").write_text(json.dumps(payload))


def test_aggregate_reports_mean_and_sample_std(tmp_path):
    for seed, r1 in enumerate([0.820, 0.824, 0.828]):
        write(tmp_path, "ubuntu", "pu_exact", seed, r1)

    agg = aggregate(load_metrics(tmp_path))
    mean, std = agg[("ubuntu", "pu_exact")]["R10@1"]

    assert mean == pytest.approx(0.824)
    assert std == pytest.approx(0.004, abs=1e-6)


def test_cited_single_seed_row_has_no_std(tmp_path):
    write(tmp_path, "ubuntu", "fs_pt_ft", 0, 0.911, source="cite:han2021")

    agg = aggregate(load_metrics(tmp_path))

    assert agg[("ubuntu", "fs_pt_ft")]["R10@1"] == (pytest.approx(0.911), None)


def test_cited_row_does_not_warn(tmp_path, recwarn):
    write(tmp_path, "ubuntu", "fs_pt_ft", 0, 0.911, source="cite:han2021")

    aggregate(load_metrics(tmp_path))

    assert not any("seeds" in str(w.message) for w in recwarn)


def test_fewer_than_three_seeds_warns(tmp_path, recwarn):
    write(tmp_path, "ubuntu", "pu_exact", 0, 0.820)
    write(tmp_path, "ubuntu", "pu_exact", 1, 0.824)

    aggregate(load_metrics(tmp_path))

    assert any("2 of 3 seeds" in str(w.message) for w in recwarn)


def test_duplicate_seed_is_an_error(tmp_path):
    # review focus 3: averaging one run twice would understate std
    write(tmp_path, "ubuntu", "pu_exact", 0, 0.820)
    (tmp_path / "ubuntu_pu_exact_rPU0.8_seed0_copy.json").write_text(
        (tmp_path / "ubuntu_pu_exact_rPU0.8_seed0.json").read_text()
    )

    with pytest.raises(ValueError, match="duplicate seed 0"):
        aggregate(load_metrics(tmp_path))


def test_truncated_file_names_itself(tmp_path):
    # review focus 4: a killed job leaves half a file behind
    (tmp_path / "ubuntu_pu_exact_rPU0.8_seed0.json").write_text('{"task": "ubuntu"')

    with pytest.raises(ValueError, match="ubuntu_pu_exact_rPU0.8_seed0.json"):
        load_metrics(tmp_path)


def test_markdown_formats_mean_plus_minus_std(tmp_path):
    for seed, r1 in enumerate([0.820, 0.824, 0.828]):
        write(tmp_path, "ubuntu", "pu_exact", seed, r1)

    table = render_markdown(aggregate(load_metrics(tmp_path)))

    assert "0.824 ± 0.004" in table


def test_rows_follow_paper_order_and_key_off_contents(tmp_path):
    for row in ("pu_estimated", "fs_pt_ft", "pu_zero"):
        write(tmp_path, "ubuntu", row, 0, 0.8, source="cite:x")
    (tmp_path / "odd_name.json").write_text(
        json.dumps({**json.loads((tmp_path / "ubuntu_pu_zero_rPU0.8_seed0.json").read_text()),
                    "row": "supervised_ft", "rPU": None})
    )

    lines = render_markdown(aggregate(load_metrics(tmp_path))).splitlines()[2:]

    assert [line.split("|")[1].strip() for line in lines] == [
        "fs_pt_ft", "supervised_ft", "pu_zero", "pu_estimated"]


def test_latex_uses_pm_and_missing_corpus_dash(tmp_path):
    for seed, r1 in enumerate([0.820, 0.824, 0.828]):
        write(tmp_path, "ubuntu", "pu_exact", seed, r1)

    table = render_latex(aggregate(load_metrics(tmp_path)))

    assert r"0.824 $\pm$ 0.004" in table and "--" in table and r"\end{tabular}" in table

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
    mean, std, n = agg[("ubuntu", "pu_exact")]["R10@1"]

    assert mean == pytest.approx(0.824)
    assert std == pytest.approx(0.004, abs=1e-6)
    assert n == 3


def test_cited_single_seed_row_has_no_std(tmp_path):
    write(tmp_path, "ubuntu", "fs_pt_ft", 0, 0.911, source="cite:han2021")

    agg = aggregate(load_metrics(tmp_path))

    assert agg[("ubuntu", "fs_pt_ft")]["R10@1"] == (pytest.approx(0.911), None, None)


def test_cited_row_does_not_warn(tmp_path, recwarn):
    write(tmp_path, "ubuntu", "fs_pt_ft", 0, 0.911, source="cite:han2021")

    aggregate(load_metrics(tmp_path))

    assert not any("seeds" in str(w.message) for w in recwarn)


def test_fewer_than_three_seeds_is_an_error(tmp_path):
    write(tmp_path, "ubuntu", "pu_exact", 0, 0.820)
    write(tmp_path, "ubuntu", "pu_exact", 1, 0.824)

    with pytest.raises(ValueError, match="2 of 3 seeds"):
        aggregate(load_metrics(tmp_path))


def test_allow_partial_annotates_the_seed_count_in_both_renderings(tmp_path):
    write(tmp_path, "ubuntu", "pu_exact", 0, 0.820)
    write(tmp_path, "ubuntu", "pu_exact", 1, 0.824)
    for seed in range(3):
        write(tmp_path, "ubuntu", "pu_zero", seed, 0.8)

    with pytest.warns(UserWarning, match="2 of 3 seeds"):
        agg = aggregate(load_metrics(tmp_path), allow_partial=True)

    markdown, latex = render_markdown(agg), render_latex(agg)
    exact_md = next(line for line in markdown.splitlines() if line.startswith("| pu_exact"))
    assert "(n=2)" in exact_md and "(n=3)" in next(l for l in markdown.splitlines() if l.startswith("| pu_zero"))
    assert latex.count("(n=2)") == 3 and "(n=3)" not in latex


def test_cited_row_in_markdown_has_no_seed_count(tmp_path):
    write(tmp_path, "ubuntu", "fs_pt_ft", 0, 0.911, source="cite:han2021")

    assert "n=" not in render_markdown(aggregate(load_metrics(tmp_path)))


def test_fs_pt_ft_must_be_cited(tmp_path):
    for seed in range(3):
        write(tmp_path, "ubuntu", "fs_pt_ft", seed, 0.9)

    with pytest.raises(ValueError, match="must not be published"):
        aggregate(load_metrics(tmp_path))


def test_differing_alpha_in_one_group_is_an_error(tmp_path):
    for seed in range(3):
        write(tmp_path, "ubuntu", "pu_estimated", seed, 0.82)
    path = tmp_path / "ubuntu_pu_estimated_rPU0.8_seed2.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "alpha": 0.40}))

    with pytest.raises(ValueError, match="mixed alpha"):
        aggregate(load_metrics(tmp_path))


def test_differing_commit_warns(tmp_path):
    for seed in range(3):
        write(tmp_path, "ubuntu", "pu_exact", seed, 0.82)
    path = tmp_path / "ubuntu_pu_exact_rPU0.8_seed2.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "commit": "abc123"}))

    with pytest.warns(UserWarning, match="different commits"):
        aggregate(load_metrics(tmp_path))


def test_empty_metrics_dir_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no metrics records"):
        load_metrics(tmp_path)


def test_missing_metrics_dir_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no metrics records"):
        load_metrics(tmp_path / "absent")


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


def test_latex_header_groups_metrics_under_corpora(tmp_path):
    for seed in range(3):
        write(tmp_path, "ubuntu", "pu_exact", seed, 0.82)

    table = render_latex(aggregate(load_metrics(tmp_path)))

    assert r"\multicolumn{3}{c}{Ubuntu}" in table and r"\multicolumn{3}{c}{E-Commerce}" in table
    assert r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}" in table


@pytest.mark.parametrize("task, row", [("ubuntu", "pu_estimate"), ("mnist", "pu_exact")])
def test_unknown_task_or_row_is_an_error(tmp_path, task, row):
    write(tmp_path, task, row, 0, 0.8)

    with pytest.raises(ValueError, match="unknown"):
        aggregate(load_metrics(tmp_path))


def test_mixed_rpu_in_one_group_is_an_error(tmp_path):
    write(tmp_path, "ubuntu", "pu_exact", 0, 0.82)
    write(tmp_path, "ubuntu", "pu_exact", 1, 0.82)
    path = tmp_path / "ubuntu_pu_exact_rPU0.8_seed1.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "rPU": 0.5}))

    with pytest.raises(ValueError, match="mixed rPU"):
        aggregate(load_metrics(tmp_path))


def test_mixed_cited_and_run_is_an_error(tmp_path):
    write(tmp_path, "ubuntu", "fs_pt_ft", 0, 0.9, source="cite:han2021")
    write(tmp_path, "ubuntu", "fs_pt_ft", 1, 0.8)

    with pytest.raises(ValueError, match="mixed cited"):
        aggregate(load_metrics(tmp_path))


def test_more_than_three_seeds_is_an_error(tmp_path):
    for seed in range(4):
        write(tmp_path, "ubuntu", "pu_exact", seed, 0.82)

    with pytest.raises(ValueError, match="4 of 3 seeds"):
        aggregate(load_metrics(tmp_path))

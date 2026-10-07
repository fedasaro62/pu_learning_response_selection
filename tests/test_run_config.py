import json
import os

import pytest

from FineTuning.run_config import row_tag, run_paths, validate_args, write_metrics


def test_row_tag_for_each_table_row():
    assert row_tag("supervised", "exact") == "supervised_ft"
    assert row_tag("pu", "zero") == "pu_zero"
    assert row_tag("pu", "exact") == "pu_exact"
    assert row_tag("pu", "estimated") == "pu_estimated"


def test_run_paths_carry_the_seed():
    save, score, metrics = run_paths("ubuntu", "pu_exact", 2, 0.8)

    assert save == "./FineTuning/FT_checkpoint/ubuntu/pu_exact/seed2/bert_rPU0.8.pt"
    assert score == "./FineTuning/FT_checkpoint/ubuntu/pu_exact/seed2/scorefile_rPU0.8.txt"
    assert metrics == "resources/metrics/ubuntu_pu_exact_rPU0.8_seed2.json"


def test_run_paths_does_not_touch_the_filesystem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    run_paths("ubuntu", "pu_exact", 2, 0.8)

    assert os.listdir(tmp_path) == []


def test_validate_args_rejects_supervised_without_rpu():
    # review focus 1: otherwise the path becomes bert_rPUNone.pt and torch.load dies obscurely
    with pytest.raises(SystemExit):
        validate_args(objective="supervised", post_train="positive_only", rPU=None)


def test_validate_args_allows_zero_shot_without_rpu():
    validate_args(objective="pu", post_train="zero_shot", rPU=None)


def test_validate_args_allows_original_without_rpu():
    validate_args(objective="pu", post_train="original", rPU=None)


def test_write_metrics_round_trips(tmp_path):
    path = tmp_path / "m.json"

    write_metrics(str(path), task="ubuntu", row="pu_exact", rPU=0.8, seed=0, alpha=0.4444,
                  result=(0.89, 0.89, 0.824, 0.824, 0.918, 0.981), source="run",
                  commit="abc123", wall_minutes=874.4)

    payload = json.loads(path.read_text())
    assert payload["R10@1"] == 0.824
    assert payload["R10@5"] == 0.981
    assert payload["commit"] == "abc123"

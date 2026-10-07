import json

import pytest

from FineTuning.BERT_finetuning import priors


def test_exact_prior_matches_the_published_value():
    # 1M pairs at 1:1, rPU=0.8 -> 100K labeled positives, alpha* = 0.444
    y = [1] * 500_000 + [0] * 500_000

    assert priors.exact_prior(y, 0.8) == pytest.approx(0.4444, abs=1e-4)


def test_exact_prior_is_one_when_nothing_is_hidden():
    y = [1] * 10 + [0] * 0

    assert priors.exact_prior(y, 1.0) == pytest.approx(1.0)


def test_resolve_prior_zero_ignores_the_labels():
    assert priors.resolve_prior("pu", "zero", "ubuntu", 0.8, y=[1, 0]) == 0.0


def test_resolve_prior_supervised_returns_none():
    assert priors.resolve_prior("supervised", "exact", "ubuntu", 0.8, y=[1, 0]) is None


def test_load_estimated_prior_reads_alpha_hat(tmp_path, monkeypatch):
    monkeypatch.setattr(priors, "PRIOR_DIR", tmp_path)
    path = tmp_path / "alpha_hat_ubuntu_rPU0.8_train.json"
    path.write_text(json.dumps({"alpha_hat": 0.4351, "split": "train"}))

    assert priors.load_estimated_prior("ubuntu", 0.8) == pytest.approx(0.4351)


def test_load_estimated_prior_names_the_command_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(priors, "PRIOR_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match=r"scripts/estimate_prior\.sh ubuntu 0\.8"):
        priors.load_estimated_prior("ubuntu", 0.8)


def test_load_estimated_prior_rejects_a_dev_split_estimate(tmp_path, monkeypatch):
    # review focus 2: a dev estimate must never be used to train
    monkeypatch.setattr(priors, "PRIOR_DIR", tmp_path)
    path = tmp_path / "alpha_hat_ubuntu_rPU0.8_train.json"
    path.write_text(json.dumps({"alpha_hat": 0.51, "split": "dev"}))

    with pytest.raises(ValueError, match="split"):
        priors.load_estimated_prior("ubuntu", 0.8)


def test_prior_map_path_always_carries_the_split_infix():
    assert priors.prior_map_path("d/", 0.8, "train").name == "prior_map_cosine_rPU0.8_train.json"
    assert priors.prior_map_path("d", 0.8, "dev").name == "prior_map_cosine_rPU0.8_dev.json"


def test_prior_map_path_without_split_keeps_the_legacy_name():
    assert priors.prior_map_path("d", 0.8).name == "prior_map_cosine_rPU0.8.json"

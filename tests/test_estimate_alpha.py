import json

import numpy as np

from tools.estimate_alpha import select_stable_tau, write_estimate


def test_select_stable_tau_finds_the_plateau():
    taus = np.arange(0.5, 0.91, 0.01)
    # falls steeply, then sits flat at 0.44, then goes noisy
    pi_hats = np.concatenate([
        np.linspace(1.0, 0.45, 20), np.full(11, 0.44), np.linspace(0.44, 0.05, 10),
    ])

    tau, pi = select_stable_tau(taus, pi_hats, window=5)

    assert pi == 0.44
    assert 0.69 < tau < 0.81


def test_write_estimate_round_trips_through_the_loader(tmp_path, monkeypatch):
    from FineTuning.BERT_finetuning import priors
    monkeypatch.setattr(priors, "PRIOR_DIR", tmp_path)

    write_estimate(tmp_path, task="ubuntu", rpu=0.8, split="train", tau_star=0.80,
                   alpha_hat=0.4351, alpha_exact=0.4444, window=5, score_transform="cosine",
                   taus=np.array([0.5, 0.51]), pi_hats=np.array([1.0, 0.99]))

    path = tmp_path / "alpha_hat_ubuntu_rPU0.8_train.json"
    payload = json.loads(path.read_text())
    assert payload["alpha_hat"] == 0.4351
    assert payload["taus"] == [0.5, 0.51]
    assert priors.load_estimated_prior("ubuntu", 0.8) == 0.4351

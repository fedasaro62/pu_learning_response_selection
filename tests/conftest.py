import pytest
import torch


@pytest.fixture
def logits() -> torch.Tensor:
    return torch.tensor([0.9, 0.8, 0.3, 0.1, 0.6])


@pytest.fixture
def pu_labels() -> torch.Tensor:
    # two labeled positives, three unlabeled
    return torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0])


@pytest.fixture
def write_curve():
    import json

    def _write(prior_dir, task, alphas, alpha_exact=0.5, rpu=0.8, split="train"):
        taus = [round(0.5 + 0.01 * i, 2) for i in range(len(alphas))]
        path = prior_dir / f"alpha_hat_{task}_rPU{rpu}_{split}.json"
        path.write_text(json.dumps({"taus": taus, "alpha_of_tau": list(alphas), "alpha_exact": alpha_exact}),
                        encoding="utf-8")
        return path

    return _write

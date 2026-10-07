import pytest
import torch


@pytest.fixture
def logits() -> torch.Tensor:
    return torch.tensor([0.9, 0.8, 0.3, 0.1, 0.6])


@pytest.fixture
def pu_labels() -> torch.Tensor:
    # two labeled positives, three unlabeled
    return torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0])

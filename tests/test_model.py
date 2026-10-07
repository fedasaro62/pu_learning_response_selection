import numpy as np
import pytest
import torch

from FineTuning.BERT_finetuning.model import CHECKPOINTS, post_trained_checkpoint, seed_everything


def test_post_trained_checkpoint_for_pu_post_training():
    path = post_trained_checkpoint("positive_only", "ubuntu", 25, 0.8)

    assert path == "./FPT/PT_checkpoint/positive_only/ubuntu/checkpoint25/bert_rPU0.8.pt"


def test_post_trained_checkpoint_for_positive_and_unlabeled_post_training():
    path = post_trained_checkpoint("positive_and_unlabeled", "ubuntu", 25, 0.8)

    assert path == "./FPT/PT_checkpoint/positive_and_unlabeled/ubuntu/checkpoint25/bert_rPU0.8.pt"


def test_post_trained_checkpoint_for_the_released_han_checkpoint():
    path = post_trained_checkpoint("original", "e_commerce", 34, 0.8)

    assert path == "./FPT/PT_checkpoint/original/e_commerce34/bert.pt"


def test_post_trained_checkpoint_is_none_for_zero_shot():
    assert post_trained_checkpoint("zero_shot", "ubuntu", 25, 0.8) is None


def test_checkpoint_defaults_match_the_bert_fp_schedule():
    assert CHECKPOINTS == {"ubuntu": 25, "douban": 27, "e_commerce": 34}


def test_seed_everything_makes_torch_draws_repeatable():
    seed_everything(1)
    first = torch.randn(4)
    seed_everything(1)

    assert torch.equal(first, torch.randn(4))


def test_seed_everything_makes_numpy_draws_repeatable():
    # review focus 5: Metrics.evaluation_one_session shuffles with numpy before sorting,
    # so evaluation is only reproducible if numpy is seeded too
    seed_everything(1)
    first = np.random.rand(4)
    seed_everything(1)

    assert np.array_equal(first, np.random.rand(4))


def test_different_seeds_give_different_draws():
    seed_everything(1)
    first = torch.randn(4)
    seed_everything(2)

    assert not torch.equal(first, torch.randn(4))

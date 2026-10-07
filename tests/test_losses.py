import torch
import torch.nn as nn

from FineTuning.BERT_finetuning.losses import pu_loss, supervised_loss

bce = nn.BCELoss(reduction="none")


def test_pu_loss_at_alpha_zero_matches_pos_neg_objective(logits, pu_labels):
    # the old pos_neg_finetune loss: positives to 1, every unlabeled to 0
    pos, unlab = logits[pu_labels == 1], logits[pu_labels == -1]
    expected = bce(pos, torch.ones_like(pos)).mean() + bce(unlab, torch.zeros_like(unlab)).mean()

    assert torch.allclose(pu_loss(logits, pu_labels, 0.0), expected)


def test_pu_loss_at_alpha_one_treats_unlabeled_as_positive(logits, pu_labels):
    pos, unlab = logits[pu_labels == 1], logits[pu_labels == -1]
    expected = bce(pos, torch.ones_like(pos)).mean() + bce(unlab, torch.ones_like(unlab)).mean()

    assert torch.allclose(pu_loss(logits, pu_labels, 1.0), expected)


def test_pu_loss_interpolates_between_the_two_extremes(logits, pu_labels):
    at_zero = pu_loss(logits, pu_labels, 0.0)
    at_one = pu_loss(logits, pu_labels, 1.0)
    at_half = pu_loss(logits, pu_labels, 0.5)

    assert torch.allclose(at_half, 0.5 * (at_zero + at_one))


def test_pu_loss_handles_batch_with_no_unlabeled(logits):
    labels = torch.ones_like(logits)
    expected = bce(logits, torch.ones_like(logits)).mean()

    assert torch.allclose(pu_loss(logits, labels, 0.4), expected)


def test_pu_loss_handles_batch_with_no_positives(logits):
    labels = torch.full_like(logits, -1.0)
    expected = (0.4 * bce(logits, torch.ones_like(logits))
                + 0.6 * bce(logits, torch.zeros_like(logits))).mean()

    assert torch.allclose(pu_loss(logits, labels, 0.4), expected)


def test_supervised_loss_is_mean_bce(logits):
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])

    assert torch.allclose(supervised_loss(logits, labels), bce(logits, labels).mean())

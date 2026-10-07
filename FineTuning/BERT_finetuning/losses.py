from __future__ import annotations

import torch
import torch.nn as nn

_bce = nn.BCELoss(reduction="none")


def supervised_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return _bce(logits, labels).mean()


def pu_loss(logits: torch.Tensor, pu_labels: torch.Tensor, alpha: float) -> torch.Tensor:
    positive_mask = pu_labels == 1
    unlabeled_mask = pu_labels == -1

    loss = logits.new_zeros(())

    if positive_mask.any():
        pos_logits = logits[positive_mask]
        loss = loss + _bce(pos_logits, torch.ones_like(pos_logits)).mean()

    if unlabeled_mask.any():
        unlab_logits = logits[unlabeled_mask]
        loss_y1 = _bce(unlab_logits, torch.ones_like(unlab_logits))
        loss_y0 = _bce(unlab_logits, torch.zeros_like(unlab_logits))
        loss = loss + (alpha * loss_y1 + (1 - alpha) * loss_y0).mean()

    return loss

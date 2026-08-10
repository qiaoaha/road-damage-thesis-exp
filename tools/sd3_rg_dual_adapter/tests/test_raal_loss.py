from __future__ import annotations

import torch

from sd3_rgda.raal import RAALConfig, compute_raal_loss_from_logits


def _loss(focus_token: int) -> torch.Tensor:
    logits = torch.zeros(1, 1, 4, 6)
    logits[:, :, focus_token, 2:4] = 8.0
    token_mask = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool)
    text_mask = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.bool)
    return compute_raal_loss_from_logits(logits, token_mask, text_mask, RAALConfig())


def test_raal_perfect_alignment_lower_loss() -> None:
    assert _loss(0) < _loss(3)


def test_raal_outside_alignment_higher_loss() -> None:
    assert float(_loss(3)) > 5.0


def test_raal_finite() -> None:
    assert torch.isfinite(_loss(0))

from __future__ import annotations

import torch

from sd3_rgda.raal import RAALConfig, compute_raal_loss_from_logits


def test_raal_negative_loss_exact_zero() -> None:
    logits = torch.randn(1, 2, 4, 8, requires_grad=True)
    token_mask = torch.zeros(1, 4, dtype=torch.bool)
    text_mask = torch.zeros(1, 8, dtype=torch.bool)
    loss = compute_raal_loss_from_logits(logits, token_mask, text_mask, RAALConfig())
    assert float(loss.detach()) == 0.0

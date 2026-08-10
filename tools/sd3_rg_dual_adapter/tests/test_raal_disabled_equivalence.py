from __future__ import annotations

import torch

from sd3_rgda.raal import RAALConfig, compute_raal_loss_from_logits


def test_raal_disabled_equivalence() -> None:
    flow_loss = torch.tensor(3.0)
    logits = torch.randn(1, 2, 4, 8, requires_grad=True)
    token_mask = torch.ones(1, 4, dtype=torch.bool)
    text_mask = torch.ones(1, 8, dtype=torch.bool)
    raal_loss = compute_raal_loss_from_logits(logits, token_mask, text_mask, RAALConfig(enabled=False))
    assert float((flow_loss + 0.02 * raal_loss).detach()) == float(flow_loss)

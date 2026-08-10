from __future__ import annotations

import torch
from raal_fakes import FakeCheckpointTransformer, FakeTransformer

from sd3_rgda.raal import RAALAttentionCollector, RAALConfig, raal_parameter_count


def test_raal_gradient_reaches_rgda() -> None:
    model = FakeTransformer()
    alpha = torch.nn.Parameter(torch.tensor(1.0))
    base_hidden = torch.randn(1, 4, 4)
    rgda_residual = torch.randn(1, 4, 4)
    hidden = base_hidden + alpha * rgda_residual
    text = torch.randn(1, 12, 4)
    token_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    text_mask = torch.zeros(1, 12, dtype=torch.bool)
    text_mask[:, 2:5] = True
    with RAALAttentionCollector(model, RAALConfig(layer_indices=(5,)), token_mask, text_mask) as collector:
        model(hidden, text)
    assert collector.stats.loss is not None
    collector.stats.loss.backward()
    assert alpha.grad is not None
    assert float(alpha.grad.abs()) > 0
    assert raal_parameter_count() == 0


def test_raal_base_grad_none() -> None:
    model = FakeTransformer()
    hidden = torch.randn(1, 4, 4, requires_grad=True)
    text = torch.randn(1, 12, 4)
    token_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    text_mask = torch.zeros(1, 12, dtype=torch.bool)
    text_mask[:, 2:5] = True
    with RAALAttentionCollector(model, RAALConfig(layer_indices=(5,)), token_mask, text_mask) as collector:
        model(hidden, text)
    assert collector.stats.loss is not None
    collector.stats.loss.backward()
    assert model.transformer_blocks[5].attn.base_weight.grad is None
    assert model.transformer_blocks[5].attn.to_q.weight.grad is None
    assert model.transformer_blocks[5].attn.add_k_proj.weight.grad is None


def test_text_path_is_not_gradient_proof() -> None:
    text = torch.randn(1, 12, 4)
    assert text.requires_grad is False


def test_nonreentrant_checkpoint_raal_grad() -> None:
    model = FakeCheckpointTransformer()
    alpha = torch.nn.Parameter(torch.tensor(1.0))
    hidden = torch.randn(1, 4, 4) + alpha * torch.randn(1, 4, 4)
    text = torch.randn(1, 12, 4)
    token_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    text_mask = torch.zeros(1, 12, dtype=torch.bool)
    text_mask[:, 2:5] = True
    with RAALAttentionCollector(model, RAALConfig(layer_indices=(5,)), token_mask, text_mask) as collector:
        model(hidden, text)
        assert collector.stats.loss is not None
        assert collector.stats.loss.requires_grad is True
        collector.stats.loss.backward()
    assert len(model.transformer_blocks[5].attn._forward_pre_hooks) == 0
    assert alpha.grad is not None
    assert float(alpha.grad.abs()) > 0

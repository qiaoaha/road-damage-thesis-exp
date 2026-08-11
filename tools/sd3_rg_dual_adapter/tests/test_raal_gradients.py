from __future__ import annotations

import pytest
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


def test_checkpoint_hooks_retained_through_backward() -> None:
    model = FakeCheckpointTransformer()
    alpha = torch.nn.Parameter(torch.tensor(1.0))
    hidden = torch.randn(1, 4, 4) + alpha * torch.randn(1, 4, 4)
    text = torch.randn(1, 12, 4)
    token_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    text_mask = torch.zeros(1, 12, dtype=torch.bool)
    text_mask[:, 2:5] = True
    collector = RAALAttentionCollector(model, RAALConfig(layer_indices=(5,)), token_mask, text_mask)
    collector.__enter__()
    try:
        model(hidden, text)
        primary = collector.snapshot()
        assert primary.hook_call_count == 1
        assert primary.loss is not None
        primary.loss.backward()
        assert len(model.transformer_blocks[5].attn._forward_pre_hooks) == 1
    finally:
        collector.close()
    assert len(model.transformer_blocks[5].attn._forward_pre_hooks) == 0
    assert alpha.grad is not None
    assert float(alpha.grad.abs()) > 0


def test_checkpoint_recompute_requires_hook_lifetime() -> None:
    model = FakeCheckpointTransformer()
    hidden = torch.randn(1, 4, 4, requires_grad=True)
    text = torch.randn(1, 12, 4)
    token_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    text_mask = torch.zeros(1, 12, dtype=torch.bool)
    text_mask[:, 2:5] = True
    collector = RAALAttentionCollector(model, RAALConfig(layer_indices=(5,)), token_mask, text_mask)
    collector.__enter__()
    model(hidden, text)
    primary = collector.snapshot()
    collector.close()
    assert primary.loss is not None
    with pytest.raises(torch.utils.checkpoint.CheckpointError):
        primary.loss.backward()
    assert len(model.transformer_blocks[5].attn._forward_pre_hooks) == 0


def test_checkpoint_primary_metrics_not_duplicated() -> None:
    model = FakeCheckpointTransformer()
    hidden = torch.randn(1, 4, 4, requires_grad=True)
    text = torch.randn(1, 12, 4)
    token_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    text_mask = torch.zeros(1, 12, dtype=torch.bool)
    text_mask[:, 2:5] = True
    collector = RAALAttentionCollector(model, RAALConfig(layer_indices=(5,)), token_mask, text_mask)
    collector.__enter__()
    try:
        model(hidden, text)
        primary = collector.snapshot()
        assert primary.hook_call_count == 1
        assert primary.layer_call_counts[5] == 1
        assert primary.loss is not None
        primary.loss.backward()
        assert collector.stats.hook_call_count >= primary.hook_call_count
        assert primary.hook_call_count == 1
    finally:
        collector.close()


def test_checkpoint_hooks_removed_after_exception() -> None:
    model = FakeCheckpointTransformer()
    hidden = torch.randn(1, 4, 4, requires_grad=True)
    text = torch.randn(1, 12, 4)
    token_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    text_mask = torch.zeros(1, 12, dtype=torch.bool)
    text_mask[:, 2:5] = True
    collector = RAALAttentionCollector(model, RAALConfig(layer_indices=(5,)), token_mask, text_mask)
    collector.__enter__()
    try:
        model(hidden, text)
        primary = collector.snapshot()
        assert primary.loss is not None
        raise RuntimeError("synthetic backward failure")
    except RuntimeError:
        collector.close()
    assert len(model.transformer_blocks[5].attn._forward_pre_hooks) == 0

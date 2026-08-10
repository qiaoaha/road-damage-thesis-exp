from __future__ import annotations

import pytest
import torch
from raal_fakes import FakeTransformer

from sd3_rgda.raal import RAALAttentionCollector, RAALConfig


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.randn(1, 4, 4)
    text = torch.randn(1, 12, 4)
    token_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    text_mask = torch.zeros(1, 12, dtype=torch.bool)
    text_mask[:, 2:5] = True
    return hidden, text, token_mask, text_mask


def test_raal_attention_shape() -> None:
    model = FakeTransformer()
    hidden, text, token_mask, text_mask = _inputs()
    with RAALAttentionCollector(model, RAALConfig(layer_indices=(5, 11, 17)), token_mask, text_mask) as collector:
        model(hidden, text)
    assert collector.stats.hook_call_count == 3
    assert collector.stats.loss is not None


def test_raal_attention_does_not_change_output() -> None:
    model = FakeTransformer()
    hidden, text, token_mask, text_mask = _inputs()
    expected = model(hidden, text)
    with RAALAttentionCollector(model, RAALConfig(layer_indices=(5, 11, 17)), token_mask, text_mask):
        actual = model(hidden, text)
    assert torch.equal(actual, expected)


def test_raal_selected_layers_only() -> None:
    model = FakeTransformer()
    hidden, text, token_mask, text_mask = _inputs()
    with RAALAttentionCollector(model, RAALConfig(layer_indices=(2, 4)), token_mask, text_mask) as collector:
        model(hidden, text)
    assert [item.layer_index for item in collector.stats.layer_stats] == [2, 4]


def test_raal_hook_cleanup_after_exception() -> None:
    model = FakeTransformer()
    _, _, token_mask, text_mask = _inputs()
    with pytest.raises(RuntimeError), RAALAttentionCollector(model, RAALConfig(layer_indices=(2,)), token_mask, text_mask):
        raise RuntimeError("boom")
    assert len(model.transformer_blocks[2].attn._forward_pre_hooks) == 0

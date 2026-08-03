from __future__ import annotations

import torch

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig


def _unzero_final_layers(block: DualAdapterBlock) -> None:
    for adapter in {block.normal_adapter, block.defect_adapter}:
        final = adapter.net[-1]
        torch.nn.init.normal_(final.weight, std=0.01)
        torch.nn.init.zeros_(final.bias)


def test_positive_sample_both_adapters_have_gradients() -> None:
    block = DualAdapterBlock(DualAdapterConfig(token_dim=8))
    _unzero_final_layers(block)
    h0 = torch.randn(2, 3, 8)
    out = block(h0, h0, h0, token_mask=torch.ones(2, 3, 1))
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.normal_adapter.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.defect_adapter.parameters())


def test_negative_sample_masks_defect_response() -> None:
    block = DualAdapterBlock(DualAdapterConfig(token_dim=8))
    _unzero_final_layers(block)
    h0 = torch.randn(2, 3, 8)
    out = block(h0, h0, h0, token_mask=torch.zeros(2, 3, 1))
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.normal_adapter.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in block.defect_adapter.parameters())

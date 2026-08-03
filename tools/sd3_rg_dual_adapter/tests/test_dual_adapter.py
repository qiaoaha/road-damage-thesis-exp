from __future__ import annotations

import torch

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig


def test_dual_adapter_shapes_and_unshared() -> None:
    block = DualAdapterBlock(DualAdapterConfig(token_dim=16))
    h0 = torch.randn(2, 5, 16)
    out = block(h0, h0, h0)
    assert tuple(out.shape) == tuple(h0.shape)
    assert block.normal_adapter is not block.defect_adapter


def test_shared_adapter_and_switches() -> None:
    shared = DualAdapterBlock(DualAdapterConfig(token_dim=8, share_adapter_weights=True))
    assert shared.normal_adapter is shared.defect_adapter
    disabled = DualAdapterBlock(DualAdapterConfig(token_dim=8, use_defect_adapter=False))
    x = torch.randn(1, 2, 8)
    torch.testing.assert_close(disabled(x, x, x), x, atol=1e-6, rtol=0)

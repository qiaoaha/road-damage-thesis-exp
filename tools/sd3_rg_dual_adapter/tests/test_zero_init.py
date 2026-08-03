from __future__ import annotations

import torch
from torch import nn

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig
from sd3_rgda.sd3_rg_transformer import RGDAInputs, SD3RGTransformerWrapper


def test_zero_init_equivalence() -> None:
    block = DualAdapterBlock(DualAdapterConfig(token_dim=8))
    h0 = torch.randn(2, 3, 8)
    out = block(h0, torch.randn(2, 3, 8), torch.randn(2, 3, 8))
    torch.testing.assert_close(out, h0, atol=1e-6, rtol=0)


def test_wrapper_matches_mock_base_at_init() -> None:
    base = nn.Linear(8, 8)
    block = DualAdapterBlock(DualAdapterConfig(token_dim=8))
    wrapper = SD3RGTransformerWrapper(base, block)
    x = torch.randn(2, 3, 8)
    inputs = RGDAInputs(normal_tokens=x, defect_tokens=x)
    torch.testing.assert_close(wrapper(x, inputs), base(x), atol=1e-6, rtol=0)

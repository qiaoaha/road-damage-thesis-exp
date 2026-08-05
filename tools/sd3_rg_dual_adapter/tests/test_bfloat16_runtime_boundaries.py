from __future__ import annotations

import torch

from sd3_rgda.injector import RGDAConditionBatch, RGDAInjector


def test_bfloat16_inputs_use_fp32_rgda_and_return_bfloat16() -> None:
    injector = RGDAInjector(token_dim=8, latent_channels=4, patch_size=1)
    h0 = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    condition = RGDAConditionBatch(
        pseudo_clean_latents=torch.randn(2, 4, 2, 2, dtype=torch.bfloat16),
        rg_maps=torch.randn(2, 7, 2, 2, dtype=torch.bfloat16),
        token_mask=torch.ones(2, 4, 1, dtype=torch.bfloat16),
        timesteps=torch.ones(2),
    )
    output = injector(h0, condition)
    assert output.dtype == torch.bfloat16
    assert all(parameter.dtype == torch.float32 for parameter in injector.parameters())
    assert injector.last_residual is not None
    assert injector.last_residual.dtype == torch.float32
    assert torch.isfinite(injector.last_residual).all()
    output.float().sum().backward()
    assert any(parameter.grad is not None for parameter in injector.parameters())

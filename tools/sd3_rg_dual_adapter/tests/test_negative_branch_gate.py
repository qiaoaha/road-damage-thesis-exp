from __future__ import annotations

import torch

from sd3_rgda.injector import RGDAConditionBatch, RGDAInjector


def test_negative_mask_closes_defect_residual_not_normal_residual() -> None:
    injector = RGDAInjector(token_dim=8, latent_channels=4, patch_size=1)
    condition = RGDAConditionBatch(
        pseudo_clean_latents=torch.randn(1, 4, 2, 2),
        rg_maps=torch.zeros(1, 7, 2, 2),
        token_mask=torch.zeros(1, 4, 1),
        timesteps=torch.ones(1),
    )
    h0 = torch.randn(1, 4, 8)
    _ = injector(h0, condition)
    assert injector.last_defect_residual is not None
    torch.testing.assert_close(injector.last_defect_residual, torch.zeros_like(injector.last_defect_residual))
    assert injector.last_normal_residual is not None

from __future__ import annotations

import torch

from sd3_rgda.injector import RGDAConditionBatch, RGDAInjector
from sd3_rgda.real_sd3_engine import find_last_linear


def _condition(batch: int, tokens: int, channels: int, size: int) -> RGDAConditionBatch:
    return RGDAConditionBatch(
        pseudo_clean_latents=torch.randn(batch, channels, size, size),
        rg_maps=torch.randn(batch, 7, size, size),
        token_mask=torch.ones(batch, tokens, 1),
        timesteps=torch.ones(batch),
    )


def test_untrained_rgda_preserves_patch_tokens() -> None:
    injector = RGDAInjector(token_dim=4, latent_channels=2, patch_size=1)
    h0 = torch.randn(1, 4, 4)
    output = injector(h0, _condition(1, 4, 2, 2))
    assert injector.last_input_tokens is not None
    assert injector.last_output_tokens is not None
    assert float((injector.last_input_tokens - injector.last_output_tokens).abs().max()) <= 1e-6
    torch.testing.assert_close(output, h0, atol=1e-6, rtol=0)


def test_modified_adapter_changes_patch_tokens() -> None:
    injector = RGDAInjector(token_dim=4, latent_channels=2, patch_size=1)
    with torch.no_grad():
        final = find_last_linear(injector.adapter.normal_adapter)
        final.weight.fill_(0.1)
    h0 = torch.randn(1, 4, 4)
    injector(h0, _condition(1, 4, 2, 2))
    assert injector.last_input_tokens is not None
    assert injector.last_output_tokens is not None
    assert float((injector.last_input_tokens - injector.last_output_tokens).abs().max()) > 0

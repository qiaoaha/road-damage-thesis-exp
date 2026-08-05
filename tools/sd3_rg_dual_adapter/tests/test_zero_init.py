from __future__ import annotations

import torch
from torch import nn

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig
from sd3_rgda.injector import RGDAConditionBatch, RGDAInjector
from sd3_rgda.sd3_rg_transformer import SD3RGTransformerWrapper


class _Output:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class _MockPosEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(4, 8, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states).flatten(2).transpose(1, 2)


class _MockSD3Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_embed = _MockPosEmbed()
        self.proj = nn.Linear(8, 8)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
    ) -> _Output:
        del encoder_hidden_states, pooled_projections, timestep
        tokens = self.pos_embed(hidden_states)
        return _Output(self.proj(tokens))


def test_zero_init_equivalence() -> None:
    block = DualAdapterBlock(DualAdapterConfig(token_dim=8))
    h0 = torch.randn(2, 3, 8)
    out = block(h0, torch.randn(2, 3, 8), torch.randn(2, 3, 8))
    torch.testing.assert_close(out, h0, atol=1e-6, rtol=0)


def test_wrapper_matches_mock_base_at_init() -> None:
    base = _MockSD3Transformer()
    injector = RGDAInjector(token_dim=8, latent_channels=4, patch_size=1)
    wrapper = SD3RGTransformerWrapper(base, injector)
    x = torch.randn(2, 4, 2, 2)
    condition = RGDAConditionBatch(
        pseudo_clean_latents=torch.randn(2, 4, 2, 2),
        rg_maps=torch.randn(2, 7, 2, 2),
        token_mask=torch.ones(2, 4, 1),
    )
    kwargs = {
        "hidden_states": x,
        "encoder_hidden_states": torch.randn(2, 4, 8),
        "pooled_projections": torch.randn(2, 8),
        "timestep": torch.ones(2),
    }
    without_hook = base(**kwargs).sample
    with_hook = wrapper(**kwargs, rgda_condition=condition, timestep_embedding=torch.randn(2, 16)).sample
    torch.testing.assert_close(with_hook, without_hook, atol=1e-6, rtol=0)

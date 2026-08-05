"""RGDA condition encoding and one-shot SD3 patch-token injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig
from sd3_rgda.condition_encoders import NormalConditionEncoder, RGConditionEncoder
from sd3_rgda.timestep_gate import TimestepGate


@dataclass
class RGDAConditionBatch:
    pseudo_clean_latents: torch.Tensor
    rg_maps: torch.Tensor
    token_mask: torch.Tensor


class RGDAInjector(nn.Module):
    def __init__(
        self,
        token_dim: int,
        latent_channels: int,
        patch_size: int,
        timestep_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.normal_encoder = NormalConditionEncoder(latent_channels, token_dim, patch_size)
        self.rg_encoder = RGConditionEncoder(token_dim, patch_size)
        self.adapter = DualAdapterBlock(DualAdapterConfig(token_dim=token_dim))
        self.timestep_gate = TimestepGate(embed_dim=timestep_embed_dim)
        self.last_residual: torch.Tensor | None = None

    def forward(self, h0: torch.Tensor, condition: RGDAConditionBatch, timestep_embedding: torch.Tensor) -> torch.Tensor:
        normal_tokens = self.normal_encoder(condition.pseudo_clean_latents)
        defect_tokens = self.rg_encoder(condition.rg_maps)
        token_mask = condition.token_mask
        if token_mask.ndim == 2:
            token_mask = token_mask.unsqueeze(-1)
        self._assert_token_shapes(h0, normal_tokens, defect_tokens, token_mask)
        gate_normal, gate_defect = self.timestep_gate(timestep_embedding)
        injected = self.adapter(h0, normal_tokens, defect_tokens, gate_normal, gate_defect, token_mask)
        self.last_residual = injected - h0
        return cast(torch.Tensor, injected)

    @staticmethod
    def _assert_token_shapes(
        h0: torch.Tensor,
        normal_tokens: torch.Tensor,
        defect_tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> None:
        if normal_tokens.shape != h0.shape:
            raise ValueError(f"normal_tokens shape {normal_tokens.shape} does not match H0 {h0.shape}")
        if defect_tokens.shape != h0.shape:
            raise ValueError(f"defect_tokens shape {defect_tokens.shape} does not match H0 {h0.shape}")
        expected_mask = h0.shape[:2] + (1,)
        if token_mask.shape != expected_mask:
            raise ValueError(f"token_mask shape {token_mask.shape} does not match {expected_mask}")

    def trainable_modules(self) -> dict[str, nn.Module]:
        return {
            "normal_encoder": self.normal_encoder,
            "rg_encoder": self.rg_encoder,
            "normal_adapter": self.adapter.normal_adapter,
            "defect_adapter": self.adapter.defect_adapter,
            "timestep_gate": self.timestep_gate,
        }


class RGDAPatchHook:
    def __init__(self, injector: RGDAInjector, condition: RGDAConditionBatch, timestep_embedding: torch.Tensor) -> None:
        self.injector = injector
        self.condition = condition
        self.timestep_embedding = timestep_embedding
        self.calls = 0

    def __call__(self, _module: nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("RGDA patch hook was called more than once")
        return cast(torch.Tensor, self.injector(output, self.condition, self.timestep_embedding))

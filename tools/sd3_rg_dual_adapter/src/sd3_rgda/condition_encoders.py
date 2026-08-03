"""Condition encoders producing SD3 image-token-aligned tensors."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class NormalConditionEncoder(nn.Module):
    def __init__(self, in_channels: int, token_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch = nn.Conv2d(in_channels, token_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(latent).flatten(2).transpose(1, 2)
        return cast(torch.Tensor, self.norm(tokens))


class RGConditionEncoder(nn.Module):
    def __init__(self, token_dim: int, patch_size: int, channels: int = 7) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(channels, token_dim // 2 if token_dim > 1 else 1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(token_dim // 2 if token_dim > 1 else 1, token_dim, kernel_size=patch_size, stride=patch_size),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, rg_map: torch.Tensor) -> torch.Tensor:
        tokens = self.net(rg_map).flatten(2).transpose(1, 2)
        return cast(torch.Tensor, self.norm(tokens))

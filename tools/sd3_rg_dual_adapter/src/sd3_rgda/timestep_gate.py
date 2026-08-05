"""Timestep-dependent residual gates."""

from __future__ import annotations

import math
from collections import deque

import torch
from torch import nn


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        if embed_dim <= 0 or embed_dim % 2 != 0:
            raise ValueError("embed_dim must be a positive even integer")
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        values = timesteps.reshape(-1).float()
        half = self.embed_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=values.device, dtype=values.dtype)
            / max(half - 1, 1)
        )
        args = values[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepGate(nn.Module):
    """Produces independent gates gn(t) and gd(t), both initialized to one."""

    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int | None = None,
        enabled: bool = True,
        record_history: bool = False,
        max_history: int = 128,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or embed_dim
        self.enabled = enabled
        self.record_history = record_history
        self.max_history = max_history
        self.embedding = SinusoidalTimestepEmbedding(embed_dim)
        self.normal = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.defect = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        for branch in (self.normal, self.defect):
            final = branch[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("Gate final layer must be Linear")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.history: deque[tuple[float, float]] = deque(maxlen=max_history)

    def forward(self, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_embedding = self.embedding(timesteps)
        if not self.enabled:
            one = torch.ones((*timestep_embedding.shape[:-1], 1), device=timestep_embedding.device)
            return one, one
        gn = 1.0 + torch.tanh(self.normal(timestep_embedding))
        gd = 1.0 + torch.tanh(self.defect(timestep_embedding))
        if self.record_history:
            values = torch.stack([gn.detach().flatten(), gd.detach().flatten()], dim=-1).float().cpu().tolist()
            self.history.extend((float(normal), float(defect)) for normal, defect in values)
        return gn, gd

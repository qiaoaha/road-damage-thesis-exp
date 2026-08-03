"""Timestep-dependent residual gates."""

from __future__ import annotations

import torch
from torch import nn


class TimestepGate(nn.Module):
    """Produces independent gates gn(t) and gd(t), both initialized to one."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None, enabled: bool = True) -> None:
        super().__init__()
        hidden = hidden_dim or embed_dim
        self.enabled = enabled
        self.normal = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.defect = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        for branch in (self.normal, self.defect):
            final = branch[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("Gate final layer must be Linear")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.history: list[tuple[float, float]] = []

    def forward(self, timestep_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            one = torch.ones((*timestep_embedding.shape[:-1], 1), device=timestep_embedding.device)
            return one, one
        gn = 1.0 + torch.tanh(self.normal(timestep_embedding))
        gd = 1.0 + torch.tanh(self.defect(timestep_embedding))
        self.history.extend(
            (float(n.detach().cpu()), float(d.detach().cpu()))
            for n, d in zip(gn.flatten(), gd.flatten(), strict=False)
        )
        return gn, gd

"""SD3-RGDA dual residual adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True)
class DualAdapterConfig:
    token_dim: int
    bottleneck: int | None = None
    dropout: float = 0.0
    use_normal_adapter: bool = True
    use_defect_adapter: bool = True
    share_adapter_weights: bool = False
    use_region_token_mask: bool = True


class ResidualAdapter(nn.Module):
    def __init__(self, token_dim: int, bottleneck: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = bottleneck or max(token_dim // 8, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, token_dim),
        )
        final = self.net[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Final adapter layer must be Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(tokens))


class DualAdapterBlock(nn.Module):
    """Applies normal and defect residuals to SD3 image tokens."""

    def __init__(self, config: DualAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.normal_adapter = ResidualAdapter(config.token_dim, config.bottleneck, config.dropout)
        self.defect_adapter = (
            self.normal_adapter
            if config.share_adapter_weights
            else ResidualAdapter(config.token_dim, config.bottleneck, config.dropout)
        )

    def forward(
        self,
        h0: torch.Tensor,
        normal_tokens: torch.Tensor,
        defect_tokens: torch.Tensor,
        gate_normal: torch.Tensor | float = 1.0,
        gate_defect: torch.Tensor | float = 1.0,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = h0
        if self.config.use_normal_adapter:
            out = out + self._scale(self.normal_adapter(normal_tokens), gate_normal)
        if self.config.use_defect_adapter:
            defect = self.defect_adapter(defect_tokens)
            if self.config.use_region_token_mask and token_mask is not None:
                defect = defect * token_mask
            out = out + self._scale(defect, gate_defect)
        return out

    def compute_residuals(
        self,
        normal_tokens: torch.Tensor,
        defect_tokens: torch.Tensor,
        gate_normal: torch.Tensor | float = 1.0,
        gate_defect: torch.Tensor | float = 1.0,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normal = torch.zeros_like(normal_tokens)
        defect = torch.zeros_like(defect_tokens)
        if self.config.use_normal_adapter:
            normal = self._scale(self.normal_adapter(normal_tokens), gate_normal)
        if self.config.use_defect_adapter:
            defect = self.defect_adapter(defect_tokens)
            if self.config.use_region_token_mask and token_mask is not None:
                defect = defect * token_mask
            defect = self._scale(defect, gate_defect)
        return normal, defect, normal + defect

    @staticmethod
    def _scale(tokens: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
        if isinstance(scale, torch.Tensor):
            while scale.ndim < tokens.ndim:
                scale = scale.unsqueeze(-1)
        return tokens * scale

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

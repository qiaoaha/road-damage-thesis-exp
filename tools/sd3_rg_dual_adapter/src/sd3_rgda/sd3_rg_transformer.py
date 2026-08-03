"""Wrapper that injects SD3-RGDA residuals after image patch embedding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from sd3_rgda.adapters import DualAdapterBlock


@dataclass
class RGDAInputs:
    normal_tokens: torch.Tensor
    defect_tokens: torch.Tensor
    timestep_embedding: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    gate_normal: torch.Tensor | float = 1.0
    gate_defect: torch.Tensor | float = 1.0


class SD3RGTransformerWrapper(nn.Module):
    """Small wrapper for tests and future SD3 integration."""

    def __init__(self, base: nn.Module, adapter: DualAdapterBlock) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, image_tokens: torch.Tensor, rgda_inputs: RGDAInputs | None = None) -> torch.Tensor:
        base_tokens = cast(torch.Tensor, self.base(image_tokens))
        if rgda_inputs is None:
            return base_tokens
        return cast(
            torch.Tensor,
            self.adapter(
                base_tokens,
                rgda_inputs.normal_tokens,
                rgda_inputs.defect_tokens,
                rgda_inputs.gate_normal,
                rgda_inputs.gate_defect,
                rgda_inputs.token_mask,
            ),
        )

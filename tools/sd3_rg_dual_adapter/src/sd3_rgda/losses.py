"""Loss placeholders for adapter-only SD3 flow matching."""

from __future__ import annotations

import torch


def flow_matching_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    return torch.mean((prediction - target) ** 2)

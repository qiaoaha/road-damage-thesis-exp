"""Clean-room RoadFusion-inspired reference modules.

These classes implement the paper-described dual-path idea for testing and ablation design.
They are not copied from official author code.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class _FeatureAdapter(nn.Module):
    def __init__(self, feature_dim: int = 1536, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        original_shape = features.shape
        flat = features.reshape(-1, original_shape[-1])
        out = self.net(flat)
        return cast(torch.Tensor, out.reshape(original_shape))


class NormalFeatureAdapter(_FeatureAdapter):
    """Normal-path feature adapter."""


class AnomalyFeatureAdapter(_FeatureAdapter):
    """Anomaly-path feature adapter."""


class PatchDiscriminator(nn.Module):
    """Two-layer MLP discriminator producing one patch normality score."""

    def __init__(self, feature_dim: int = 1536, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or max(feature_dim // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        original_shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        score = self.net(flat)
        return cast(torch.Tensor, score.reshape(*original_shape, 1))


class RoadFusionReferenceHead(nn.Module):
    """Dual independent adapters plus patch discriminator."""

    def __init__(self, feature_dim: int = 1536) -> None:
        super().__init__()
        self.normal_adapter = NormalFeatureAdapter(feature_dim)
        self.anomaly_adapter = AnomalyFeatureAdapter(feature_dim)
        self.discriminator = PatchDiscriminator(feature_dim)

    def forward(self, normal: torch.Tensor, anomaly: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normal_features = self.normal_adapter(normal)
        anomaly_features = self.anomaly_adapter(anomaly)
        return normal_features, anomaly_features, self.discriminator(anomaly_features)

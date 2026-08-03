"""Configuration helpers for SD3-RGDA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SD3ShapeConfig:
    """Subset of SD3 transformer config required by the adapter."""

    in_channels: int
    out_channels: int
    patch_size: int
    num_attention_heads: int
    attention_head_dim: int
    joint_attention_dim: int
    caption_projection_dim: int

    @property
    def token_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping YAML at {path}")
    return data


def sd3_config_from_object(config: object) -> SD3ShapeConfig:
    values: dict[str, int] = {}
    required = (
        "in_channels",
        "out_channels",
        "patch_size",
        "num_attention_heads",
        "attention_head_dim",
        "joint_attention_dim",
        "caption_projection_dim",
    )
    for name in required:
        if not hasattr(config, name):
            raise ValueError(f"SD3 transformer config missing required field: {name}")
        value = getattr(config, name)
        if not isinstance(value, int):
            raise TypeError(f"SD3 transformer config field {name} must be int, got {type(value)!r}")
        values[name] = value
    return SD3ShapeConfig(**values)

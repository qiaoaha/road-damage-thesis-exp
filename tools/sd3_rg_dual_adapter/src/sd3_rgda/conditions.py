"""Region-guided condition maps for RDD2022 labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Box:
    """Pixel-space bounding box with RDD2022 class id."""

    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int


@dataclass(frozen=True)
class GenerationCondition:
    source_image: Path
    target_image: Path
    rg_map: torch.Tensor
    boxes: tuple[Box, ...]
    class_ids: tuple[int, ...]
    prompt: str
    negative_prompt: str = ""
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_box(box: Box, width: int, height: int) -> None:
    if box.class_id not in (0, 1, 2, 3):
        raise ValueError(f"Invalid RDD2022 class id {box.class_id}; expected 0..3")
    if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
        raise ValueError(f"Out-of-bounds or empty bbox {box} for image size {(width, height)}")


def build_rg_map(boxes: list[Box] | tuple[Box, ...], image_size: tuple[int, int]) -> torch.Tensor:
    """Build a seven-channel RG map at image resolution.

    Channels: union, border, normalized box distance, D00, D10, D20, D40.
    """

    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive (height, width), got {image_size}")
    rg = torch.zeros(7, height, width, dtype=torch.float32)
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    for box in boxes:
        _validate_box(box, width, height)
        xs = slice(box.x1, box.x2)
        ys = slice(box.y1, box.y2)
        rg[0, ys, xs] = 1.0
        rg[1, box.y1, box.x1 : box.x2] = 1.0
        rg[1, box.y2 - 1, box.x1 : box.x2] = 1.0
        rg[1, box.y1 : box.y2, box.x1] = 1.0
        rg[1, box.y1 : box.y2, box.x2 - 1] = 1.0
        cx = (box.x1 + box.x2 - 1) / 2.0
        cy = (box.y1 + box.y2 - 1) / 2.0
        half_w = max((box.x2 - box.x1) / 2.0, 1.0)
        half_h = max((box.y2 - box.y1) / 2.0, 1.0)
        dist = 1.0 - torch.maximum((xx - cx).abs() / half_w, (yy - cy).abs() / half_h)
        rg[2] = torch.maximum(rg[2], dist.clamp(0.0, 1.0))
        rg[3 + box.class_id, ys, xs] = 1.0
    return rg.clamp(0.0, 1.0)


def downsample_rg_map(rg_map: torch.Tensor, latent_size: tuple[int, int]) -> torch.Tensor:
    if rg_map.ndim != 3 or rg_map.shape[0] != 7:
        raise ValueError(f"Expected RG map [7,H,W], got {tuple(rg_map.shape)}")
    return F.interpolate(
        rg_map.unsqueeze(0), size=latent_size, mode="bilinear", align_corners=False
    ).squeeze(0).clamp(0.0, 1.0)


def token_region_mask(rg_map: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    union = rg_map[0:1].unsqueeze(0)
    pooled = F.avg_pool2d(union, kernel_size=patch_size, stride=patch_size)
    return (pooled.flatten(2).transpose(1, 2) > 0).to(dtype=rg_map.dtype)

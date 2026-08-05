"""Cache contract for real Czech SD3-RGDA validation inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachedInputSpec:
    image_path: Path
    label_path: Path
    target_latent_path: Path
    pseudo_clean_latent_path: Path
    prompt_embeds_path: Path
    pooled_prompt_embeds_path: Path
    rg_map_path: Path
    token_mask_path: Path


REQUIRED_CACHE_COLUMNS = tuple(field for field in CachedInputSpec.__dataclass_fields__)


def validate_cache_manifest_header(columns: set[str]) -> None:
    missing = sorted(set(REQUIRED_CACHE_COLUMNS) - columns)
    if missing:
        raise ValueError(f"Cache manifest is missing required columns: {missing}")

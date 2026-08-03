"""Clean-room bridge for DIAG-style prompt and mask metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import SupportsInt, cast

import torch

from sd3_rgda.conditions import Box, GenerationCondition, build_rg_map


class DIAGBridge:
    """Converts external DIAG-style metadata into local GenerationCondition rows."""

    def __init__(self, diag_root: str | Path | None = None) -> None:
        self.diag_root = Path(diag_root) if diag_root is not None else None

    def from_record(self, record: dict[str, object], image_size: tuple[int, int]) -> GenerationCondition:
        boxes = tuple(
            Box(
                _to_int(item["x1"], "x1"),
                _to_int(item["y1"], "y1"),
                _to_int(item["x2"], "x2"),
                _to_int(item["y2"], "y2"),
                _to_int(item["class_id"], "class_id"),
            )
            for item in self._expect_boxes(record.get("boxes", []))
        )
        prompt = str(record.get("prompt", "road damage"))
        source = Path(str(record["source_image"]))
        target = Path(str(record.get("target_image", source)))
        rg_map = build_rg_map(boxes, image_size)
        return GenerationCondition(
            source_image=source,
            target_image=target,
            rg_map=rg_map,
            boxes=boxes,
            class_ids=tuple(box.class_id for box in boxes),
            prompt=prompt,
            negative_prompt=str(record.get("negative_prompt", "")),
            seed=_to_int(record.get("seed", 0), "seed"),
            metadata={"diag_bridge": "clean-room", **{k: str(v) for k, v in record.items() if k != "boxes"}},
        )

    @staticmethod
    def _expect_boxes(value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            raise TypeError("boxes must be a list of mappings")
        for item in value:
            if not isinstance(item, dict):
                raise TypeError("each box must be a mapping")
        return value

    def save_manifest(self, conditions: list[GenerationCondition], path: str | Path) -> None:
        rows = []
        for condition in conditions:
            rows.append(
                {
                    "source_image": str(condition.source_image),
                    "target_image": str(condition.target_image),
                    "prompt": condition.prompt,
                    "negative_prompt": condition.negative_prompt,
                    "seed": condition.seed,
                    "boxes": [box.__dict__ for box in condition.boxes],
                    "class_ids": list(condition.class_ids),
                    "rg_map_shape": list(condition.rg_map.shape),
                    "metadata": condition.metadata,
                }
            )
        Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def rg_map_preview_tensor(rg_map: torch.Tensor) -> torch.Tensor:
    if rg_map.ndim != 3 or rg_map.shape[0] != 7:
        raise ValueError("Expected [7,H,W] RG map")
    return rg_map[:3].clamp(0.0, 1.0)


def _to_int(value: object, field_name: str) -> int:
    try:
        return int(cast(SupportsInt, value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be integer-like, got {value!r}") from exc

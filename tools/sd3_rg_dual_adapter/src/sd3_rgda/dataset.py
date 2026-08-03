"""Dataset split guards for Czech/RDD2022 manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRow:
    image: str
    label: str
    split: str


def read_manifest(path: str | Path, train_only: bool = True) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            split = raw.get("split", "")
            if train_only and split != "train":
                continue
            rows.append(ManifestRow(raw.get("image", ""), raw.get("label", ""), split))
    return rows


def assert_no_val_test_training_rows(rows: list[ManifestRow]) -> None:
    leaked = [row for row in rows if row.split in {"val", "test"}]
    if leaked:
        raise ValueError(f"Manifest contains {len(leaked)} val/test rows in training input")

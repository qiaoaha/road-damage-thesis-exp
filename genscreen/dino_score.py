from __future__ import annotations

import hashlib
from pathlib import Path

from .io_utils import write_csv


def deterministic_score(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return round(int(digest[:8], 16) / 0xFFFFFFFF, 6)


def dummy_dino_scores(generated_rows: list[dict], output_path: Path) -> list[dict]:
    rows = [{"image_path": r["image_path"], "class_id": r["class_id"], "S_DINO": deterministic_score(r["image_path"], "dino")} for r in generated_rows]
    write_csv(output_path, rows, ["image_path", "class_id", "S_DINO"])
    return rows

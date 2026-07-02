from __future__ import annotations

from pathlib import Path

from .dino_score import deterministic_score
from .io_utils import write_csv


def dummy_pcs_scores(generated_rows: list[dict], output_path: Path) -> list[dict]:
    rows = [{"image_path": r["image_path"], "class_id": r["class_id"], "S_PCS": deterministic_score(r["image_path"], "pcs")} for r in generated_rows]
    write_csv(output_path, rows, ["image_path", "class_id", "S_PCS"])
    return rows

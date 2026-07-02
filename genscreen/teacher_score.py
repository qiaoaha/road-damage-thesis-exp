from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .dino_score import deterministic_score
from .io_utils import read_csv, write_csv


def score_from_predictions(prediction_csv: Path, generated_rows: list[dict], output_path: Path, cfg: dict) -> list[dict]:
    preds = read_csv(prediction_csv)
    by_image: dict[str, list[dict]] = defaultdict(list)
    for pred in preds:
        by_image[pred.get("image_path", "")].append(pred)
    rows = []
    for item in generated_rows:
        target = str(item["class_id"])
        boxes = by_image.get(item["image_path"], [])
        same = [float(b.get("conf", 0) or 0) for b in boxes if str(b.get("class_id", "")) == target]
        wrong = [float(b.get("conf", 0) or 0) for b in boxes if str(b.get("class_id", "")) != target]
        if same:
            p_max = max(same)
            p_mean = sum(same) / len(same)
            raw = 0.7 * p_max + 0.3 * p_mean
            penalty = len(wrong) / (len(boxes) + 1.0e-8)
            conf = raw * (1 - 0.5 * penalty)
            margin = max(0.0, min(2.0, p_max - (max(wrong) if wrong else 0.0) + 1.0)) / 2.0
            source = "prediction_csv_fallback_margin"
        else:
            conf = 0.0
            margin = 0.0
            source = "prediction_csv_no_same_class"
        rows.append({"image_path": item["image_path"], "class_id": target, "S_conf": round(conf, 6), "S_margin": round(margin, 6), "margin_source": source})
    write_csv(output_path, rows, ["image_path", "class_id", "S_conf", "S_margin", "margin_source"])
    return rows


def dummy_teacher_scores(generated_rows: list[dict], output_path: Path) -> list[dict]:
    rows = []
    for item in generated_rows:
        conf = deterministic_score(item["image_path"], "teacher")
        margin = deterministic_score(item["image_path"], "margin")
        rows.append(
            {
                "image_path": item["image_path"],
                "class_id": item["class_id"],
                "S_conf": conf,
                "S_margin": margin,
                "margin_source": "dry_run_dummy",
            }
        )
    write_csv(output_path, rows, ["image_path", "class_id", "S_conf", "S_margin", "margin_source"])
    return rows

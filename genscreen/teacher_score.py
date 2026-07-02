from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import json

from .config import as_path, cfg_path
from .dino_score import deterministic_score
from .io_utils import read_csv, write_csv


def score_from_predictions(prediction_csv: Path, generated_rows: list[dict], output_path: Path, cfg: dict) -> list[dict]:
    preds = read_csv(prediction_csv)
    by_image: dict[str, list[dict]] = defaultdict(list)
    for pred in preds:
        by_image[pred.get("image_path", "")].append(pred)
    rows = []
    eps = float(cfg_path(cfg, "quality.eps", 1.0e-8))
    for item in generated_rows:
        target = str(item["class_id"])
        boxes = by_image.get(item["image_path"], [])
        same = [float(b.get("conf", 0) or 0) for b in boxes if str(b.get("class_id", "")) == target]
        wrong = [float(b.get("conf", 0) or 0) for b in boxes if str(b.get("class_id", "")) != target]
        p_max = max(same) if same else 0.0
        p_mean = sum(same) / len(same) if same else 0.0
        p_max_wrong = max(wrong) if wrong else 0.0
        if same:
            raw = 0.7 * p_max + 0.3 * p_mean
            penalty = len(wrong) / (len(boxes) + eps)
            conf = raw * (1 - 0.5 * penalty)
        else:
            conf = 0.0
        prob_margins = []
        for box in boxes:
            probs = _class_probs(box)
            if probs and target.isdigit() and int(target) in probs:
                target_prob = probs[int(target)]
                other = [v for k, v in probs.items() if k != int(target)]
                prob_margins.append(target_prob - (max(other) if other else 0.0))
        if prob_margins:
            margin = 0.5 * (1.0 + sum(prob_margins) / len(prob_margins))
            margin = max(0.0, min(1.0, margin))
            source = "class_probs"
        elif same:
            margin = max(0.0, min(2.0, p_max - p_max_wrong + 1.0)) / 2.0
            source = "fallback_confidence"
        else:
            margin = 0.0
            source = "prediction_csv_no_same_class"
        rows.append(
            {
                "image_path": item["image_path"],
                "class_id": target,
                "S_conf": round(conf, 6),
                "S_margin": round(margin, 6),
                "p_max_same": round(p_max, 6),
                "p_mean_same": round(p_mean, 6),
                "p_max_wrong": round(p_max_wrong, 6),
                "num_same": len(same),
                "num_wrong": len(wrong),
                "num_all": len(boxes),
                "margin_source": source,
            }
        )
    write_csv(output_path, rows, TEACHER_SCORE_FIELDS)
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
                "p_max_same": conf,
                "p_mean_same": conf,
                "p_max_wrong": 0.0,
                "num_same": 0,
                "num_wrong": 0,
                "num_all": 0,
                "margin_source": "dry_run_dummy",
            }
        )
    write_csv(output_path, rows, TEACHER_SCORE_FIELDS)
    return rows


def _class_probs(row: dict) -> dict[int, float]:
    probs: dict[int, float] = {}
    for key, value in row.items():
        if not key.startswith("prob_") or value in ("", None):
            continue
        try:
            probs[int(key.split("_", 1)[1])] = float(value)
        except ValueError:
            continue
    raw = row.get("optional_class_probs") or row.get("class_probs")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    probs[int(key)] = float(value)
            elif isinstance(parsed, list):
                for idx, value in enumerate(parsed):
                    probs[idx] = float(value)
        except Exception:
            pass
    return probs


def run_yolo_predictions(cfg: dict, generated_rows: list[dict], output_dir: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("YOLO teacher inference requires ultralytics. Install it or set models.teacher.existing_predictions.") from exc
    weights = as_path(cfg_path(cfg, "models.teacher.weights"))
    if not weights or not weights.exists():
        raise RuntimeError(f"YOLO teacher weights are missing or do not exist: {weights}")
    from .features import resolve_device

    device = resolve_device(cfg_path(cfg, "models.teacher.device", "auto"))
    conf = float(cfg_path(cfg, "models.teacher.conf_threshold", 0.25))
    model = YOLO(str(weights))
    image_paths = [row["image_path"] for row in generated_rows]
    pred_rows: list[dict] = []
    if image_paths:
        results = model.predict(source=image_paths, conf=conf, device=device, verbose=False)
        for image_path, result in zip(image_paths, results):
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            for coords, cid, score in zip(xyxy, cls, confs):
                pred_rows.append(
                    {
                        "image_path": image_path,
                        "class_id": int(cid),
                        "conf": round(float(score), 6),
                        "x1": round(float(coords[0]), 3),
                        "y1": round(float(coords[1]), 3),
                        "x2": round(float(coords[2]), 3),
                        "y2": round(float(coords[3]), 3),
                    }
                )
    pred_path = output_dir / "cache" / "teacher_predictions.csv"
    write_csv(pred_path, pred_rows, ["image_path", "class_id", "conf", "x1", "y1", "x2", "y2"])
    return pred_path


def compute_teacher_scores(cfg: dict, generated_rows: list[dict], output_dir: Path) -> list[dict]:
    existing = as_path(cfg_path(cfg, "models.teacher.existing_predictions"))
    if existing and existing.exists():
        return score_from_predictions(existing, generated_rows, output_dir / "scores" / "teacher_scores.csv", cfg)
    teacher_type = str(cfg_path(cfg, "models.teacher.type", "") or "").lower()
    if teacher_type == "yolo":
        pred_path = run_yolo_predictions(cfg, generated_rows, output_dir)
        return score_from_predictions(pred_path, generated_rows, output_dir / "scores" / "teacher_scores.csv", cfg)
    if teacher_type == "dfine":
        raise RuntimeError("D-FINE inference adapter is not implemented. Set models.teacher.existing_predictions to a CSV exported by D-FINE.")
    raise RuntimeError(f"Unsupported teacher type '{teacher_type}'. Use type=yolo or set models.teacher.existing_predictions.")


TEACHER_SCORE_FIELDS = [
    "image_path",
    "class_id",
    "S_conf",
    "S_margin",
    "p_max_same",
    "p_mean_same",
    "p_max_wrong",
    "num_same",
    "num_wrong",
    "num_all",
    "margin_source",
]

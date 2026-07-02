from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .config import cfg_path
from .io_utils import read_csv, write_csv


def by_image(rows: list[dict], key: str) -> dict[str, dict]:
    return {r[key]: r for r in rows if r.get(key)}


def fval(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None, "NA"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_by_class(rows: list[dict], raw_key: str, out_key: str, eps: float, constant_value: float = 0.5) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if fval(row, raw_key) is not None:
            groups[str(row.get("class_id", "unknown"))].append(row)
    for group in groups.values():
        vals = [fval(r, raw_key) for r in group]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        for row in group:
            value = fval(row, raw_key)
            if value is None:
                row[out_key] = ""
            elif abs(hi - lo) <= eps:
                row[out_key] = constant_value
            else:
                row[out_key] = (value - lo) / (hi - lo + eps)


def active_weights(cfg: dict, score_keys: list[str]) -> dict[str, float]:
    raw = cfg_path(cfg, "quality.weights", {}) or {}
    enabled = {
        "dino": cfg_path(cfg, "scores.dino.enabled", True),
        "pcs": cfg_path(cfg, "scores.pcs.enabled", True),
        "teacher_conf": cfg_path(cfg, "scores.teacher_conf.enabled", True),
        "margin": cfg_path(cfg, "scores.margin.enabled", True),
    }
    weights = {k: float(raw.get(k, 0.0)) for k in score_keys if enabled.get(k, True)}
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights} if weights else {}
    return {k: v / total for k, v in weights.items()}


def build_quality_scores(cfg: dict, index_rows: list[dict], scores_dir: Path, output_path: Path) -> tuple[list[dict], dict[str, float]]:
    dino = by_image(read_csv(scores_dir / "dino_scores.csv"), "image_path")
    pcs = by_image(read_csv(scores_dir / "pcs_scores.csv"), "image_path")
    teacher = by_image(read_csv(scores_dir / "teacher_scores.csv"), "image_path")
    rows = []
    for idx in index_rows:
        image = idx["image_path"]
        row = dict(idx)
        row.update(
            {
                "S_DINO": dino.get(image, {}).get("S_DINO", ""),
                "S_PCS": pcs.get(image, {}).get("S_PCS", ""),
                "S_conf": teacher.get(image, {}).get("S_conf", ""),
                "S_margin": teacher.get(image, {}).get("S_margin", ""),
                "margin_source": teacher.get(image, {}).get("margin_source", ""),
                "is_valid": "true" if idx["class_id"] != "unknown" else "false",
                "invalid_reason": "" if idx["class_id"] != "unknown" else "unknown_class",
            }
        )
        rows.append(row)
    eps = float(cfg_path(cfg, "quality.eps", 1.0e-8))
    normalize_by_class(rows, "S_DINO", "norm_DINO", eps)
    normalize_by_class(rows, "S_PCS", "norm_PCS", eps)
    normalize_by_class(rows, "S_conf", "norm_conf", eps)
    normalize_by_class(rows, "S_margin", "norm_margin", eps)
    weights = active_weights(cfg, ["dino", "pcs", "teacher_conf", "margin"])
    mapping = {"dino": "norm_DINO", "pcs": "norm_PCS", "teacher_conf": "norm_conf", "margin": "norm_margin"}
    for row in rows:
        q = 0.0
        valid_weight = 0.0
        for weight_key, weight in weights.items():
            value = fval(row, mapping[weight_key])
            if value is not None:
                q += weight * value
                valid_weight += weight
        row["Q"] = round(q / valid_weight, 6) if valid_weight > 0 and row["is_valid"] == "true" else 0.0
    fields = [
        "image_path",
        "label_path",
        "class_id",
        "class_name",
        "S_DINO",
        "S_PCS",
        "S_conf",
        "S_margin",
        "norm_DINO",
        "norm_PCS",
        "norm_conf",
        "norm_margin",
        "Q",
        "is_valid",
        "invalid_reason",
        "margin_source",
    ]
    write_csv(output_path, rows, fields)
    return rows, weights

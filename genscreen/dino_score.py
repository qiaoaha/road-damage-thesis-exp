from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import as_path, cfg_path
from .features import encode_dinov2_images, force_enabled, read_feature_cache, write_feature_cache
from .io_utils import write_csv


def deterministic_score(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return round(int(digest[:8], 16) / 0xFFFFFFFF, 6)


def dummy_dino_scores(generated_rows: list[dict], output_path: Path) -> list[dict]:
    rows = [
        {
            "image_path": r["image_path"],
            "class_id": r["class_id"],
            "S_DINO": deterministic_score(r["image_path"], "dino"),
            "dino_topk_mean": deterministic_score(r["image_path"], "dino"),
            "dino_topk_used": 0,
            "invalid_reason": "dry_run_dummy",
        }
        for r in generated_rows
    ]
    write_csv(output_path, rows, ["image_path", "class_id", "S_DINO", "dino_topk_mean", "dino_topk_used", "invalid_reason"])
    return rows


def load_or_extract_dino_features(cfg: dict, rows: list[dict], output_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    cache_path = output_dir / "cache" / f"{split_name}_dino_features.npz"
    if cache_path.exists() and not force_enabled(cfg):
        return read_feature_cache(cache_path)
    model_dir = as_path(cfg_path(cfg, "models.dinov2.model_dir"))
    if not model_dir or not model_dir.exists():
        raise RuntimeError(f"DINOv2 model_dir is missing or does not exist: {model_dir}")
    device = cfg_path(cfg, "models.dinov2.device", "auto")
    from .features import resolve_device

    device = resolve_device(device)
    batch_size = int(cfg_path(cfg, "models.dinov2.batch_size", 32) or 32)
    features = encode_dinov2_images(rows, model_dir, device, batch_size)
    image_paths = [row["image_path"] for row in rows]
    class_ids = [str(row["class_id"]) for row in rows]
    write_feature_cache(cache_path, image_paths, class_ids, features)
    return {"image_path": np.array(image_paths, dtype=object), "class_id": np.array(class_ids, dtype=object), "feature": features}


def compute_dino_scores(cfg: dict, real_rows: list[dict], generated_rows: list[dict], output_dir: Path) -> list[dict]:
    topk = int(cfg_path(cfg, "scores.dino.topk", 5) or 5)
    real = load_or_extract_dino_features(cfg, real_rows, output_dir, "real")
    generated = load_or_extract_dino_features(cfg, generated_rows, output_dir, "generated")
    real_by_class: dict[str, np.ndarray] = {}
    for cls in sorted({str(c) for c in real["class_id"]}):
        mask = np.array([str(c) == cls for c in real["class_id"]])
        real_by_class[cls] = real["feature"][mask]
    rows: list[dict] = []
    for idx, image_path in enumerate(generated["image_path"]):
        cls = str(generated["class_id"][idx])
        candidates = real_by_class.get(cls)
        if candidates is None or len(candidates) == 0:
            score = 0.0
            topk_used = 0
            invalid = "no_real_same_class"
        else:
            sims = (candidates @ generated["feature"][idx].reshape(-1, 1)).reshape(-1)
            sims01 = (sims + 1.0) / 2.0
            top = np.sort(sims01)[-min(topk, len(sims01)) :]
            score = float(np.mean(top))
            topk_used = int(len(top))
            invalid = ""
        rows.append(
            {
                "image_path": str(image_path),
                "class_id": cls,
                "S_DINO": round(score, 6),
                "dino_topk_mean": round(score, 6),
                "dino_topk_used": topk_used,
                "invalid_reason": invalid,
            }
        )
    write_csv(output_dir / "scores" / "dino_scores.csv", rows, ["image_path", "class_id", "S_DINO", "dino_topk_mean", "dino_topk_used", "invalid_reason"])
    return rows

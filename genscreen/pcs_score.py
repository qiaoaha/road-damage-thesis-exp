from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import as_path, cfg_path
from .dino_score import deterministic_score
from .features import (
    encode_clip_images,
    encode_clip_texts,
    force_enabled,
    load_clip_encoder,
    pil_load,
    read_feature_cache,
    resolve_device,
    weak_augmentations,
)
from .io_utils import write_csv


def dummy_pcs_scores(generated_rows: list[dict], output_path: Path) -> list[dict]:
    rows = [
        {
            "image_path": r["image_path"],
            "class_id": r["class_id"],
            "prompt": "",
            "S_IT": deterministic_score(r["image_path"], "pcs_it"),
            "S_stable": deterministic_score(r["image_path"], "pcs_stable"),
            "S_PCS": deterministic_score(r["image_path"], "pcs"),
            "num_augmentations": 0,
        }
        for r in generated_rows
    ]
    write_csv(output_path, rows, ["image_path", "class_id", "prompt", "S_IT", "S_stable", "S_PCS", "num_augmentations"])
    return rows


def _prompt_for(cfg: dict, class_id: str) -> str:
    prompts = cfg_path(cfg, "classes.prompts", {}) or {}
    return str(prompts.get(class_id, prompts.get(int(class_id), "")) if class_id != "unknown" else "")


def compute_pcs_scores(cfg: dict, generated_rows: list[dict], output_dir: Path) -> list[dict]:
    cache_path = output_dir / "cache" / "pcs_features.npz"
    if cache_path.exists() and not force_enabled(cfg):
        cache = read_feature_cache(cache_path)
        rows = []
        for idx, image_path in enumerate(cache["image_path"]):
            rows.append(
                {
                    "image_path": str(image_path),
                    "class_id": str(cache["class_id"][idx]),
                    "prompt": str(cache["prompt"][idx]),
                    "S_IT": round(float(cache["S_IT"][idx]), 6),
                    "S_stable": round(float(cache["S_stable"][idx]), 6),
                    "S_PCS": round(float(cache["S_PCS"][idx]), 6),
                    "num_augmentations": int(cache["num_augmentations"][idx]),
                }
            )
        write_csv(output_dir / "scores" / "pcs_scores.csv", rows, ["image_path", "class_id", "prompt", "S_IT", "S_stable", "S_PCS", "num_augmentations"])
        return rows

    model_dir = as_path(cfg_path(cfg, "models.clip.model_dir"))
    if not model_dir or not model_dir.exists():
        raise RuntimeError(f"CLIP model_dir is missing or does not exist: {model_dir}")
    device = resolve_device(cfg_path(cfg, "models.clip.device", "auto"))
    batch_size = int(cfg_path(cfg, "models.clip.batch_size", 32) or 32)
    num_aug = int(cfg_path(cfg, "scores.pcs.num_augmentations", 4) or 4)
    image_text_weight = float(cfg_path(cfg, "scores.pcs.image_text_weight", 0.6))
    stable_weight = float(cfg_path(cfg, "scores.pcs.stable_weight", 0.4))
    processor, model, torch = load_clip_encoder(model_dir, device)
    rows: list[dict] = []
    image_features: list[np.ndarray] = []
    text_features: list[np.ndarray] = []
    for row in generated_rows:
        prompt = _prompt_for(cfg, str(row["class_id"]))
        image = pil_load(row["image_path"])
        aug_images = weak_augmentations(image, num_aug)
        original_feat = encode_clip_images([image], processor, model, torch, device, batch_size)[0]
        aug_feat = encode_clip_images(aug_images, processor, model, torch, device, batch_size) if aug_images else np.zeros((0, original_feat.shape[0]), dtype="float32")
        text_feat = encode_clip_texts([prompt], processor, model, torch, device, batch_size)[0]
        sims_it = [float((original_feat @ text_feat + 1.0) / 2.0)]
        sims_it.extend(float((feat @ text_feat + 1.0) / 2.0) for feat in aug_feat)
        sims_stable = [float((original_feat @ feat + 1.0) / 2.0) for feat in aug_feat]
        s_it = float(np.mean(sims_it)) if sims_it else 0.0
        s_stable = float(np.mean(sims_stable)) if sims_stable else 0.0
        s_pcs = image_text_weight * s_it + stable_weight * s_stable
        rows.append(
            {
                "image_path": row["image_path"],
                "class_id": str(row["class_id"]),
                "prompt": prompt,
                "S_IT": round(s_it, 6),
                "S_stable": round(s_stable, 6),
                "S_PCS": round(s_pcs, 6),
                "num_augmentations": num_aug,
            }
        )
        image_features.append(original_feat)
        text_features.append(text_feat)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        image_path=np.array([r["image_path"] for r in rows], dtype=object),
        class_id=np.array([r["class_id"] for r in rows], dtype=object),
        prompt=np.array([r["prompt"] for r in rows], dtype=object),
        image_feature=np.array(image_features, dtype="float32") if image_features else np.zeros((0, 0), dtype="float32"),
        text_feature=np.array(text_features, dtype="float32") if text_features else np.zeros((0, 0), dtype="float32"),
        S_IT=np.array([r["S_IT"] for r in rows], dtype="float32"),
        S_stable=np.array([r["S_stable"] for r in rows], dtype="float32"),
        S_PCS=np.array([r["S_PCS"] for r in rows], dtype="float32"),
        num_augmentations=np.array([r["num_augmentations"] for r in rows], dtype="int32"),
    )
    write_csv(output_dir / "scores" / "pcs_scores.csv", rows, ["image_path", "class_id", "prompt", "S_IT", "S_stable", "S_PCS", "num_augmentations"])
    return rows

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .config import as_path, cfg_path


def require_gpu_step(step_name: str, cfg: dict) -> None:
    cpu_only = bool(cfg.get("runtime", {}).get("cpu_only_check", False))
    if cpu_only:
        raise RuntimeError(f"{step_name} requires model inference and is disabled by cpu_only_check=true")


def cache_exists(path: Path | None) -> bool:
    return bool(path and path.exists())


def force_enabled(cfg: dict) -> bool:
    return bool(cfg_path(cfg, "cache.force", False))


def resolve_device(requested: str | None = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if requested == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        except ImportError as exc:
            raise RuntimeError("CUDA was requested but torch is not installed") from exc
    return requested


def normalize_features(features: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    features = features.astype("float32", copy=False)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def read_feature_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def write_feature_cache(path: Path, image_paths: list[str], class_ids: list[str], features: np.ndarray, **extra: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": np.array(image_paths, dtype=object),
        "class_id": np.array(class_ids, dtype=object),
        "feature": features.astype("float32", copy=False),
    }
    payload.update(extra)
    np.savez_compressed(path, **payload)


def pil_load(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def batched(items: list, batch_size: int) -> Iterable[list]:
    batch_size = max(int(batch_size or 1), 1)
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def load_dinov2_encoder(model_dir: str | Path, device: str):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError("DINOv2 extraction requires torch and transformers") from exc
    model_dir = str(model_dir)
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()
    return processor, model, torch


def encode_dinov2_images(rows: list[dict], model_dir: Path, device: str, batch_size: int) -> np.ndarray:
    processor, model, torch = load_dinov2_encoder(model_dir, device)
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batched(rows, batch_size):
            images = [pil_load(row["image_path"]) for row in batch]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            else:
                feat = outputs.last_hidden_state[:, 0]
            feats.append(feat.detach().cpu().numpy())
    return normalize_features(np.concatenate(feats, axis=0)) if feats else np.zeros((0, 0), dtype="float32")


def load_clip_encoder(model_dir: str | Path, device: str):
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("PCS extraction requires torch and transformers with CLIP support") from exc
    model_dir = str(model_dir)
    processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()
    return processor, model, torch


def encode_clip_images(images: list[Image.Image], processor, model, torch, device: str, batch_size: int) -> np.ndarray:
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batched(images, batch_size):
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            feat = model.get_image_features(**inputs)
            feats.append(feat.detach().cpu().numpy())
    return normalize_features(np.concatenate(feats, axis=0)) if feats else np.zeros((0, 0), dtype="float32")


def encode_clip_texts(texts: list[str], processor, model, torch, device: str, batch_size: int) -> np.ndarray:
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batched(texts, batch_size):
            inputs = processor(text=batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            feat = model.get_text_features(**inputs)
            feats.append(feat.detach().cpu().numpy())
    return normalize_features(np.concatenate(feats, axis=0)) if feats else np.zeros((0, 0), dtype="float32")


def weak_augmentations(image: Image.Image, count: int) -> list[Image.Image]:
    variants: list[Image.Image] = []
    width, height = image.size
    recipes = [
        lambda im: ImageEnhance.Brightness(im).enhance(1.08),
        lambda im: ImageEnhance.Contrast(im).enhance(1.08),
        lambda im: im.resize((max(1, int(width * 1.04)), max(1, int(height * 1.04)))).crop((0, 0, width, height)),
        lambda im: im.filter(ImageFilter.GaussianBlur(radius=0.35)),
    ]
    for idx in range(max(count, 0)):
        variants.append(recipes[idx % len(recipes)](image.copy()))
    return variants


def cosine01(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T + 1.0) / 2.0

from __future__ import annotations

from pathlib import Path
import sys
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
    import torch

    model_path = Path(model_dir)
    if model_path.suffix.lower() in {".pth", ".pt"}:
        return load_dinov2_torchhub(model_path, device, torch)
    try:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device)
        model.eval()
        return {"kind": "hf", "processor": processor}, model, torch
    except Exception:
        pths = sorted(model_path.glob("*.pth")) if model_path.is_dir() else []
        if pths:
            return load_dinov2_torchhub(pths[0], device, torch)
        raise RuntimeError(f"Could not load DINOv2 as a HuggingFace model or torchhub checkpoint: {model_path}")


def load_dinov2_torchhub(weight_path: Path, device: str, torch):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("DINOv2 torchhub checkpoints require torchvision") from exc
    repo = Path("/root/autodl-tmp/cache/torch/hub/facebookresearch_dinov2_main")
    if not repo.exists():
        repo = Path("/root/autodl-tmp/road_damage_exp/third_party/dinov2")
    if not repo.exists():
        raise RuntimeError("DINOv2 torchhub repository was not found on this machine")
    name = "dinov2_vitb14" if "vitb" in weight_path.name.lower() else "dinov2_vits14"
    model = torch.hub.load(str(repo), name, source="local", pretrained=False)
    state = torch.load(str(weight_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return {"kind": "torchhub", "transform": transform}, model, torch


def encode_dinov2_images(rows: list[dict], model_dir: Path, device: str, batch_size: int) -> np.ndarray:
    processor, model, torch = load_dinov2_encoder(model_dir, device)
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batched(rows, batch_size):
            images = [pil_load(row["image_path"]) for row in batch]
            if processor["kind"] == "hf":
                inputs = processor["processor"](images=images, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    feat = outputs.pooler_output
                else:
                    feat = outputs.last_hidden_state[:, 0]
            else:
                tensor = torch.stack([processor["transform"](image) for image in images]).to(device)
                feat = model(tensor)
            feats.append(feat.detach().cpu().numpy())
    return normalize_features(np.concatenate(feats, axis=0)) if feats else np.zeros((0, 0), dtype="float32")


def load_clip_encoder(model_dir: str | Path, device: str):
    import torch

    model_path = Path(model_dir)
    if model_path.suffix.lower() in {".pt", ".pth"}:
        return load_openai_clip(model_path, device, torch)
    try:
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(str(model_path), local_files_only=True)
        model = CLIPModel.from_pretrained(str(model_path), local_files_only=True).to(device)
        model.eval()
        return {"kind": "hf", "processor": processor}, model, torch
    except Exception:
        pts = sorted(model_path.glob("*.pt")) if model_path.is_dir() else []
        if pts:
            return load_openai_clip(pts[0], device, torch)
        raise RuntimeError(f"Could not load CLIP as a HuggingFace model or OpenAI CLIP checkpoint: {model_path}")


def load_openai_clip(weight_path: Path, device: str, torch):
    for package_dir in ["/root/autodl-tmp/road_damage_exp/third_party/SDS", "/root/autodl-tmp/road_damage_exp/third_party/CLIP"]:
        if Path(package_dir).exists() and package_dir not in sys.path:
            sys.path.insert(0, package_dir)
    try:
        import clip
    except ImportError as exc:
        raise RuntimeError("OpenAI CLIP checkpoint requires the local clip package") from exc
    model, preprocess = clip.load(str(weight_path), device=device)
    model.eval()
    return {"kind": "openai", "preprocess": preprocess, "clip": clip}, model, torch


def encode_clip_images(images: list[Image.Image], processor, model, torch, device: str, batch_size: int) -> np.ndarray:
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batched(images, batch_size):
            if processor["kind"] == "hf":
                inputs = processor["processor"](images=batch, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                feat = model.get_image_features(**inputs)
            else:
                inputs = torch.stack([processor["preprocess"](image) for image in batch]).to(device)
                feat = model.encode_image(inputs)
            feats.append(feat.detach().cpu().numpy())
    return normalize_features(np.concatenate(feats, axis=0)) if feats else np.zeros((0, 0), dtype="float32")


def encode_clip_texts(texts: list[str], processor, model, torch, device: str, batch_size: int) -> np.ndarray:
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batched(texts, batch_size):
            if processor["kind"] == "hf":
                inputs = processor["processor"](text=batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                feat = model.get_text_features(**inputs)
            else:
                inputs = processor["clip"].tokenize(batch, truncate=True).to(device)
                feat = model.encode_text(inputs)
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

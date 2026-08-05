"""Real SD3 cache creation and loading for Czech road-damage samples."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from sd3_rgda.conditions import Box, build_rg_map, downsample_rg_map, token_region_mask


@dataclass
class CachedSD3Sample:
    image_path: str
    label_path: str
    target_latent: torch.Tensor
    pseudo_clean_latent: torch.Tensor
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    rg_map_latent: torch.Tensor
    token_mask: torch.Tensor
    class_ids: tuple[int, ...]
    is_negative: bool


CACHE_MANIFEST_FIELDS = [
    "sample_id",
    "image_path",
    "label_path",
    "cache_path",
    "class_ids",
    "is_negative",
    "split",
]


def read_yolo_boxes(label_path: str | Path, image_size: tuple[int, int]) -> tuple[Box, ...]:
    path = Path(label_path)
    if not path.exists() or path.stat().st_size == 0:
        return ()
    width, height = image_size
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = (float(value) for value in parts[1:5])
        x1 = max(0, round((cx - bw / 2.0) * width))
        y1 = max(0, round((cy - bh / 2.0) * height))
        x2 = min(width, round((cx + bw / 2.0) * width))
        y2 = min(height, round((cy + bh / 2.0) * height))
        if x2 > x1 and y2 > y1:
            boxes.append(Box(x1, y1, x2, y2, class_id))
    return tuple(boxes)


def image_to_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
    data = torch.tensor(list(resized.getdata()), dtype=torch.float32)
    tensor = data.reshape(resolution, resolution, 3).permute(2, 0, 1) / 127.5 - 1.0
    return tensor.unsqueeze(0)


def build_pseudo_clean_image(image: Image.Image, boxes: tuple[Box, ...]) -> Image.Image:
    clean = image.convert("RGB").copy()
    draw = ImageDraw.Draw(clean)
    for box in boxes:
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), fill=(127, 127, 127))
    return clean


def encode_latent(vae: Any, image_tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    image_tensor = image_tensor.to(device=vae.device, dtype=dtype)
    encoded = vae.encode(image_tensor).latent_dist.sample()
    scale = getattr(getattr(vae, "config", object()), "scaling_factor", 1.0)
    return torch.as_tensor(encoded * scale).detach().cpu()


def encode_prompt(pipe: Any, prompt: str, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds, _negative_embeds, pooled_prompt_embeds, _negative_pooled = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return prompt_embeds.to(dtype=dtype).detach().cpu(), pooled_prompt_embeds.to(dtype=dtype).detach().cpu()


def cache_manifest_rows(
    pipe: Any,
    manifest: str | Path,
    out_dir: str | Path,
    resolution: int,
    dtype: torch.dtype,
    patch_size: int,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest)
    cache_manifest = out / "cache_manifest.csv"
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    with cache_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_MANIFEST_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows):
            if row.get("split") != "train":
                raise ValueError(f"NO_VAL_TEST_LEAKAGE failed for row {index}: {row.get('split')}")
            image_path = Path(row["image_path"])
            label_path = Path(row["label_path"])
            image = Image.open(image_path)
            boxes = read_yolo_boxes(label_path, image.size)
            resized_boxes = _resize_boxes(boxes, image.size, (resolution, resolution))
            target_tensor = image_to_tensor(image, resolution)
            clean_tensor = image_to_tensor(build_pseudo_clean_image(image, boxes), resolution)
            target_latent = encode_latent(pipe.vae, target_tensor, dtype)
            pseudo_clean_latent = encode_latent(pipe.vae, clean_tensor, dtype)
            latent_size = (int(target_latent.shape[-2]), int(target_latent.shape[-1]))
            rg_map = build_rg_map(resized_boxes, (resolution, resolution))
            rg_map_latent = downsample_rg_map(rg_map, latent_size).unsqueeze(0)
            token_mask = token_region_mask(rg_map_latent.squeeze(0), patch_size)
            class_ids = tuple(sorted({box.class_id for box in resized_boxes}))
            prompt = _prompt_for_classes(class_ids)
            prompt_embeds, pooled_prompt_embeds = encode_prompt(pipe, prompt, target_latent.device, dtype)
            sample = CachedSD3Sample(
                image_path=str(image_path),
                label_path=str(label_path),
                target_latent=target_latent,
                pseudo_clean_latent=pseudo_clean_latent,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                rg_map_latent=rg_map_latent,
                token_mask=token_mask,
                class_ids=class_ids,
                is_negative=len(class_ids) == 0,
            )
            _assert_sample(sample)
            cache_path = out / f"sample_{index:04d}.pt"
            torch.save(sample.__dict__, cache_path)
            writer.writerow(
                {
                    "sample_id": index,
                    "image_path": sample.image_path,
                    "label_path": sample.label_path,
                    "cache_path": str(cache_path),
                    "class_ids": " ".join(str(item) for item in class_ids),
                    "is_negative": str(sample.is_negative).lower(),
                    "split": "train",
                }
            )
    return cache_manifest


def load_cached_sample(path: str | Path, device: torch.device, dtype: torch.dtype) -> CachedSD3Sample:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"Invalid cache payload: {path}")
    sample = CachedSD3Sample(**raw)
    return CachedSD3Sample(
        image_path=sample.image_path,
        label_path=sample.label_path,
        target_latent=sample.target_latent.to(device=device, dtype=dtype),
        pseudo_clean_latent=sample.pseudo_clean_latent.to(device=device, dtype=dtype),
        prompt_embeds=sample.prompt_embeds.to(device=device, dtype=dtype),
        pooled_prompt_embeds=sample.pooled_prompt_embeds.to(device=device, dtype=dtype),
        rg_map_latent=sample.rg_map_latent.to(device=device, dtype=dtype),
        token_mask=sample.token_mask.to(device=device, dtype=dtype),
        class_ids=tuple(sample.class_ids),
        is_negative=bool(sample.is_negative),
    )


def validate_cache_manifest(path: str | Path, expected_rows: int | None = None) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"CACHE_ROWS {len(rows)} != manifest rows {expected_rows}")
    for row in rows:
        cache_path = Path(row["cache_path"])
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
    return rows


def _resize_boxes(boxes: tuple[Box, ...], source: tuple[int, int], target: tuple[int, int]) -> tuple[Box, ...]:
    src_w, src_h = source
    dst_w, dst_h = target
    return tuple(
        Box(
            round(box.x1 * dst_w / src_w),
            round(box.y1 * dst_h / src_h),
            round(box.x2 * dst_w / src_w),
            round(box.y2 * dst_h / src_h),
            box.class_id,
        )
        for box in boxes
    )


def _prompt_for_classes(class_ids: tuple[int, ...]) -> str:
    if not class_ids:
        return "a clean road surface without visible damage"
    names = {0: "longitudinal crack", 1: "transverse crack", 2: "alligator crack", 3: "pothole"}
    return "road damage: " + ", ".join(names.get(class_id, f"class {class_id}") for class_id in class_ids)


def _assert_sample(sample: CachedSD3Sample) -> None:
    tensors = [
        sample.target_latent,
        sample.pseudo_clean_latent,
        sample.prompt_embeds,
        sample.pooled_prompt_embeds,
        sample.rg_map_latent,
        sample.token_mask,
    ]
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("ALL_TENSORS_FINITE failed")
    if sample.is_negative and (
        float(sample.rg_map_latent.abs().sum()) != 0.0 or float(sample.token_mask.abs().sum()) != 0.0
    ):
        raise ValueError("Negative cache sample must have zero RG map and token mask")

"""Real SD3 cache creation and loading for Czech road-damage samples."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from sd3_rgda.clean_proxy import telea_inpaint_proxy
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
    sample_id: str = ""
    anchor_class: str = ""
    source_image_sha256: str = ""
    label_sha256: str = ""
    clean_proxy_sha256: str = ""
    split: str = "train"
    source_split: str = "train"
    pilot_split: str = "train"


@dataclass
class CacheRequest:
    sample_id: int
    image_path: str
    label_path: str
    boxes: tuple[Box, ...]
    class_ids: tuple[int, ...]
    prompt: str
    target_pixels: torch.Tensor
    pseudo_clean_pixels: torch.Tensor
    rg_map: torch.Tensor
    split: str
    source_sample_id: str = ""
    anchor_class: str = ""
    source_image_sha256: str = ""
    label_sha256: str = ""
    clean_proxy_sha256: str = ""
    source_split: str = "train"
    pilot_split: str = "train"


@dataclass
class CacheBuildReport:
    cache_manifest: Path
    cache_rows: int
    unique_prompt_count: int
    vae_device_during_encoding: str
    text_encoder_device_during_encoding: str
    all_cache_files_exist: bool
    all_tensors_finite: bool
    no_val_test_leakage: bool
    negative_rg_map_zero: bool
    negative_token_mask_zero: bool
    vae_shift_factor: float
    vae_scaling_factor: float
    latent_dtype: str
    latent_shape: tuple[int, ...]
    vae_parameter_dtype: str = "unknown"
    vae_input_dtype: str = "unknown"
    cached_latent_dtype: str = "unknown"


CACHE_MANIFEST_FIELDS = [
    "sample_id",
    "source_sample_id",
    "image_path",
    "label_path",
    "cache_path",
    "class_ids",
    "anchor_class",
    "is_negative",
    "split",
    "source_image_sha256",
    "label_sha256",
    "clean_proxy_sha256",
    "source_split",
    "pilot_split",
]


def collect_cache_requests(manifest: str | Path, resolution: int) -> list[CacheRequest]:
    with Path(manifest).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    requests: list[CacheRequest] = []
    for index, row in enumerate(rows):
        if row.get("split") != "train":
            raise ValueError(f"NO_VAL_TEST_LEAKAGE failed for row {index}: {row.get('split')}")
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        image = Image.open(image_path)
        boxes = read_yolo_boxes(label_path, image.size)
        resized_boxes = _resize_boxes(boxes, image.size, (resolution, resolution))
        class_ids = tuple(sorted({box.class_id for box in resized_boxes}))
        prompt = _prompt_for_classes(class_ids)
        target_pixels = image_to_tensor(image, resolution)
        pseudo_clean_pixels = image_to_tensor(build_pseudo_clean_image(image, boxes), resolution)
        rg_map = build_rg_map(resized_boxes, (resolution, resolution))
        requests.append(
            CacheRequest(
                sample_id=index,
                image_path=str(image_path),
                label_path=str(label_path),
                boxes=resized_boxes,
                class_ids=class_ids,
                prompt=prompt,
                target_pixels=target_pixels,
                pseudo_clean_pixels=pseudo_clean_pixels,
                rg_map=rg_map,
                split="train",
            )
        )
    return requests


def collect_pilot_cache_requests(clean_proxy_manifest: str | Path, resolution: int) -> list[CacheRequest]:
    with Path(clean_proxy_manifest).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    requests: list[CacheRequest] = []
    for index, row in enumerate(rows):
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        clean_proxy_path = Path(row["clean_proxy_path"])
        actual_source_sha = _sha256_file(image_path)
        actual_label_sha = _sha256_file(label_path)
        actual_proxy_sha = _sha256_file(clean_proxy_path)
        if row.get("image_sha256", "") != actual_source_sha:
            raise ValueError("SOURCE_IMAGE_SHA256_MISMATCH")
        if row.get("label_sha256", "") != actual_label_sha:
            raise ValueError("LABEL_SHA256_MISMATCH")
        if row.get("clean_proxy_sha256", "") != actual_proxy_sha:
            raise ValueError("CLEAN_PROXY_SHA256_MISMATCH")
        image = Image.open(image_path)
        clean = Image.open(clean_proxy_path)
        boxes = read_yolo_boxes(label_path, image.size)
        resized_boxes = _resize_boxes(boxes, image.size, (resolution, resolution))
        class_ids = tuple(sorted({box.class_id for box in resized_boxes}))
        requests.append(
            CacheRequest(
                sample_id=index,
                image_path=str(image_path),
                label_path=str(label_path),
                boxes=resized_boxes,
                class_ids=class_ids,
                prompt=_prompt_for_classes(class_ids),
                target_pixels=image_to_tensor(image, resolution),
                pseudo_clean_pixels=image_to_tensor(clean, resolution),
                rg_map=build_rg_map(resized_boxes, (resolution, resolution)),
                split="train",
                source_sample_id=row.get("sample_id", ""),
                anchor_class=row.get("anchor_class", ""),
                source_image_sha256=actual_source_sha,
                label_sha256=actual_label_sha,
                clean_proxy_sha256=actual_proxy_sha,
                source_split="train",
                pilot_split=row.get("split", ""),
            )
        )
    return requests


def encode_all_latents(
    vae: Any,
    requests: list[CacheRequest],
    device: torch.device,
    cache_dtype: torch.dtype,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], str, float, float, str, str, str]:
    vae.to(device=device, dtype=torch.float32)
    vae.eval()
    actual_device = _module_device(vae)
    vae_parameter_dtype = _module_parameter_dtype(vae)
    latents: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    vae_input_dtype = "unknown"
    with torch.no_grad():
        for request in requests:
            target, target_input_dtype = encode_latent(vae, request.target_pixels, cache_dtype)
            clean, clean_input_dtype = encode_latent(vae, request.pseudo_clean_pixels, cache_dtype)
            if target_input_dtype != clean_input_dtype:
                raise TypeError("VAE input dtype changed within cache build")
            vae_input_dtype = target_input_dtype
            latents[request.sample_id] = (target, clean)
    actual_cached_dtype = str(next(iter(latents.values()))[0].dtype) if latents else "unknown"
    vae.to("cpu")
    torch.cuda.empty_cache()
    config = getattr(vae, "config", object())
    return (
        latents,
        str(actual_device),
        float(getattr(config, "shift_factor", 0.0)),
        float(getattr(config, "scaling_factor", 1.0)),
        vae_parameter_dtype,
        vae_input_dtype,
        actual_cached_dtype,
    )


def encode_unique_prompts(
    pipe: Any,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], str]:
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        encoder = getattr(pipe, name)
        encoder.to(device)
        encoder.eval()
    actual_device = _module_device(pipe.text_encoder)
    embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for prompt in sorted(set(prompts)):
            embeddings[prompt] = encode_prompt(pipe, prompt, device, dtype)
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        getattr(pipe, name).to("cpu")
    torch.cuda.empty_cache()
    return embeddings, str(actual_device)


def write_cache_samples(
    requests: list[CacheRequest],
    latents: dict[int, tuple[torch.Tensor, torch.Tensor]],
    prompt_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    out_dir: str | Path,
    patch_size: int,
) -> tuple[Path, list[CachedSD3Sample]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_manifest = out / "cache_manifest.csv"
    samples: list[CachedSD3Sample] = []
    with cache_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_MANIFEST_FIELDS)
        writer.writeheader()
        for request in requests:
            target_latent, pseudo_clean_latent = latents[request.sample_id]
            prompt_embeds, pooled_prompt_embeds = prompt_embeddings[request.prompt]
            latent_size = (int(target_latent.shape[-2]), int(target_latent.shape[-1]))
            rg_map_latent = downsample_rg_map(request.rg_map, latent_size).unsqueeze(0)
            token_mask = token_region_mask(rg_map_latent.squeeze(0), patch_size)
            sample = CachedSD3Sample(
                image_path=request.image_path,
                label_path=request.label_path,
                target_latent=target_latent,
                pseudo_clean_latent=pseudo_clean_latent,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                rg_map_latent=rg_map_latent,
                token_mask=token_mask,
                class_ids=request.class_ids,
                is_negative=len(request.class_ids) == 0,
                sample_id=request.source_sample_id or str(request.sample_id),
                anchor_class=request.anchor_class,
                source_image_sha256=request.source_image_sha256,
                label_sha256=request.label_sha256,
                clean_proxy_sha256=request.clean_proxy_sha256,
                split=request.split,
                source_split=request.source_split,
                pilot_split=request.pilot_split,
            )
            _assert_sample(sample)
            cache_path = out / f"sample_{request.sample_id:04d}.pt"
            torch.save(sample.__dict__, cache_path)
            samples.append(sample)
            writer.writerow(
                {
                    "sample_id": request.sample_id,
                    "source_sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "label_path": sample.label_path,
                    "cache_path": str(cache_path),
                    "class_ids": " ".join(str(item) for item in sample.class_ids),
                    "anchor_class": sample.anchor_class,
                    "is_negative": str(sample.is_negative).lower(),
                    "split": request.split,
                    "source_image_sha256": sample.source_image_sha256,
                    "label_sha256": sample.label_sha256,
                    "clean_proxy_sha256": sample.clean_proxy_sha256,
                    "source_split": sample.source_split,
                    "pilot_split": sample.pilot_split,
                }
            )
    return cache_manifest, samples


def cache_manifest_rows(
    pipe: Any,
    manifest: str | Path,
    out_dir: str | Path,
    resolution: int,
    dtype: torch.dtype,
    patch_size: int,
) -> CacheBuildReport:
    device = torch.device("cuda")
    requests = collect_cache_requests(manifest, resolution)
    latents, vae_device, shift_factor, scaling_factor, vae_parameter_dtype, vae_input_dtype, cached_latent_dtype = (
        encode_all_latents(pipe.vae, requests, device, dtype)
    )
    prompt_embeddings, text_device = encode_unique_prompts(pipe, [request.prompt for request in requests], device, dtype)
    cache_manifest, samples = write_cache_samples(requests, latents, prompt_embeddings, out_dir, patch_size)
    manifest_rows = validate_cache_manifest(cache_manifest, expected_rows=len(requests))
    negative_samples = [sample for sample in samples if sample.is_negative]
    first_latent = samples[0].target_latent if samples else torch.empty(0)
    return CacheBuildReport(
        cache_manifest=cache_manifest,
        cache_rows=len(manifest_rows),
        unique_prompt_count=len(prompt_embeddings),
        vae_device_during_encoding=vae_device,
        text_encoder_device_during_encoding=text_device,
        all_cache_files_exist=all(Path(row["cache_path"]).exists() for row in manifest_rows),
        all_tensors_finite=all(_sample_tensors_finite(sample) for sample in samples),
        no_val_test_leakage=all(row["split"] == "train" for row in manifest_rows),
        negative_rg_map_zero=all(float(sample.rg_map_latent.abs().sum()) == 0.0 for sample in negative_samples),
        negative_token_mask_zero=all(float(sample.token_mask.abs().sum()) == 0.0 for sample in negative_samples),
        vae_shift_factor=shift_factor,
        vae_scaling_factor=scaling_factor,
        latent_dtype=str(first_latent.dtype),
        latent_shape=tuple(int(item) for item in first_latent.shape),
        vae_parameter_dtype=vae_parameter_dtype,
        vae_input_dtype=vae_input_dtype,
        cached_latent_dtype=cached_latent_dtype,
    )


def cache_pilot_manifest_rows(
    pipe: Any,
    clean_proxy_manifest: str | Path,
    out_dir: str | Path,
    resolution: int,
    dtype: torch.dtype,
    patch_size: int,
) -> CacheBuildReport:
    device = torch.device("cuda")
    requests = collect_pilot_cache_requests(clean_proxy_manifest, resolution)
    latents, vae_device, shift_factor, scaling_factor, vae_parameter_dtype, vae_input_dtype, cached_latent_dtype = (
        encode_all_latents(pipe.vae, requests, device, dtype)
    )
    prompt_embeddings, text_device = encode_unique_prompts(pipe, [request.prompt for request in requests], device, dtype)
    cache_manifest, samples = write_cache_samples(requests, latents, prompt_embeddings, out_dir, patch_size)
    manifest_rows = validate_cache_manifest(cache_manifest, expected_rows=len(requests))
    negative_samples = [sample for sample in samples if sample.is_negative]
    first_latent = samples[0].target_latent if samples else torch.empty(0)
    return CacheBuildReport(
        cache_manifest=cache_manifest,
        cache_rows=len(manifest_rows),
        unique_prompt_count=len(prompt_embeddings),
        vae_device_during_encoding=vae_device,
        text_encoder_device_during_encoding=text_device,
        all_cache_files_exist=all(Path(row["cache_path"]).exists() for row in manifest_rows),
        all_tensors_finite=all(_sample_tensors_finite(sample) for sample in samples),
        no_val_test_leakage=all(row["split"] == "train" for row in manifest_rows),
        negative_rg_map_zero=all(float(sample.rg_map_latent.abs().sum()) == 0.0 for sample in negative_samples),
        negative_token_mask_zero=all(float(sample.token_mask.abs().sum()) == 0.0 for sample in negative_samples),
        vae_shift_factor=shift_factor,
        vae_scaling_factor=scaling_factor,
        latent_dtype=str(first_latent.dtype),
        latent_shape=tuple(int(item) for item in first_latent.shape),
        vae_parameter_dtype=vae_parameter_dtype,
        vae_input_dtype=vae_input_dtype,
        cached_latent_dtype=cached_latent_dtype,
    )


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
    if not boxes:
        return image.convert("RGB").copy()
    filename = getattr(image, "filename", "")
    if not isinstance(filename, str) or not filename:
        raise ValueError("Telea clean proxy requires an image with a source filename")
    clean, _mask = telea_inpaint_proxy(filename, boxes)
    return clean


def encode_latent(vae: Any, image_tensor: torch.Tensor, cache_dtype: torch.dtype) -> tuple[torch.Tensor, str]:
    image_tensor = image_tensor.to(device=_module_device(vae), dtype=torch.float32)
    actual_input_dtype = str(image_tensor.dtype)
    if image_tensor.dtype != torch.float32:
        raise TypeError("VAE input must be float32")
    latent = vae.encode(image_tensor).latent_dist.sample()
    config = getattr(vae, "config", object())
    shift_factor = float(getattr(config, "shift_factor", 0.0))
    scaling_factor = float(getattr(config, "scaling_factor", 1.0))
    latent = (torch.as_tensor(latent).float() - shift_factor) * scaling_factor
    return latent.to(dtype=cache_dtype).detach().cpu(), actual_input_dtype


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
        sample_id=sample.sample_id,
        anchor_class=sample.anchor_class,
        source_image_sha256=sample.source_image_sha256,
        label_sha256=sample.label_sha256,
        clean_proxy_sha256=sample.clean_proxy_sha256,
        split=sample.split,
        source_split=sample.source_split,
        pilot_split=sample.pilot_split,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _module_device(module: Any) -> torch.device:
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        first = next(parameters(), None)
        if first is not None:
            return torch.device(first.device)
    return torch.device("cpu")


def _module_parameter_dtype(module: Any) -> str:
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        first = next(parameters(), None)
        if first is not None:
            return str(first.dtype)
    return "unknown"


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


def _sample_tensors_finite(sample: CachedSD3Sample) -> bool:
    return all(
        torch.isfinite(tensor).all()
        for tensor in [
            sample.target_latent,
            sample.pseudo_clean_latent,
            sample.prompt_embeds,
            sample.pooled_prompt_embeds,
            sample.rg_map_latent,
            sample.token_mask,
        ]
    )


def _assert_sample(sample: CachedSD3Sample) -> None:
    if not _sample_tensors_finite(sample):
        raise ValueError("ALL_TENSORS_FINITE failed")
    if sample.is_negative and (
        float(sample.rg_map_latent.abs().sum()) != 0.0 or float(sample.token_mask.abs().sum()) != 0.0
    ):
        raise ValueError("Negative cache sample must have zero RG map and token mask")

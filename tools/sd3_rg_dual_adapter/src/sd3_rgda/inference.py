"""No-GPU-testable SD3-RGDA paired generation helpers."""

from __future__ import annotations

import csv
import os
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch import nn

from sd3_rgda.checkpoint import inspect_adapter_checkpoint, load_adapter_checkpoint
from sd3_rgda.generation_manifest import derive_generation_seed, sha256_file
from sd3_rgda.injector import RGDAConditionBatch, RGDAInjector, RGDAPatchHook
from sd3_rgda.sd3_hook import resolve_patch_embedding, temporary_forward_hook

Mode = str

GENERATION_RESULT_FIELDS = [
    "generation_index",
    "mode",
    "source_sample_id",
    "seed",
    "prompt",
    "source_image_sha256",
    "source_label_sha256",
    "rgda_checkpoint_sha256",
    "output_image_path",
    "output_image_sha256",
    "output_label_path",
    "output_label_sha256",
    "width",
    "height",
    "inference_steps",
    "guidance_scale",
    "scheduler",
    "generation_seconds",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "transformer_forward_count",
    "rgda_hook_call_count",
    "status",
    "error_type",
    "error_message",
]


@dataclass(frozen=True)
class FormalCacheRecord:
    source_image_sha256: str
    cache_path: Path


@dataclass(frozen=True)
class GenerationRunSummary:
    result_rows: int
    base_success: int
    rgda_success: int
    failures: int
    cache_matched: int
    cache_missing: int
    cache_duplicate: int


@dataclass
class RGDAInferenceStats:
    transformer_forward_count: int = 0
    rgda_hook_call_count: int = 0


def verify_checkpoint_sha256(path: str | Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f"checkpoint SHA mismatch: expected {expected_sha256}, got {actual}")
    info = inspect_adapter_checkpoint(path)
    if not info["valid_adapter_only"]:
        raise ValueError(f"checkpoint is not adapter-only: {info}")
    return actual


def prepare_injector_from_checkpoint(
    checkpoint_path: str | Path,
    expected_sha256: str,
    *,
    token_dim: int,
    latent_channels: int,
    patch_size: int,
) -> RGDAInjector:
    verify_checkpoint_sha256(checkpoint_path, expected_sha256)
    injector = RGDAInjector(token_dim=token_dim, latent_channels=latent_channels, patch_size=patch_size)
    load_adapter_checkpoint(checkpoint_path, injector.trainable_modules())
    for module in injector.trainable_modules().values():
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    injector.eval()
    return injector


@contextmanager
def rgda_inference_context(
    transformer: nn.Module,
    injector: RGDAInjector,
    pseudo_clean_latent: torch.Tensor,
    rg_map_latent: torch.Tensor,
    token_mask: torch.Tensor,
) -> Iterator[RGDAInferenceStats]:
    original_forward = transformer.forward
    stats = RGDAInferenceStats()

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        hidden_states = _extract_hidden_states(args, kwargs)
        target_batch = int(hidden_states.shape[0])
        timestep = _expand_timestep_to_batch(_extract_timestep(args, kwargs), target_batch).to(
            device=pseudo_clean_latent.device
        )
        pseudo_batch, rg_batch, mask_batch = expand_rgda_condition_to_batch(
            pseudo_clean_latent,
            rg_map_latent,
            token_mask,
            target_batch,
        )
        condition = RGDAConditionBatch(
            pseudo_clean_latents=pseudo_batch,
            rg_maps=rg_batch,
            token_mask=mask_batch,
            timesteps=timestep,
        )
        _name, patch_embed = resolve_patch_embedding(transformer)
        patch_hook = RGDAPatchHook(injector, condition)
        stats.transformer_forward_count += 1
        with temporary_forward_hook(patch_embed, patch_hook):
            result = original_forward(*args, **kwargs)
        stats.rgda_hook_call_count += patch_hook.calls
        if patch_hook.calls != 1:
            raise RuntimeError(f"Expected exactly one RGDA patch hook call, got {patch_hook.calls}")
        return result

    transformer.forward = patched_forward
    try:
        yield stats
    finally:
        transformer.forward = original_forward


@contextmanager
def transformer_forward_counter(transformer: nn.Module) -> Iterator[RGDAInferenceStats]:
    original_forward = transformer.forward
    stats = RGDAInferenceStats()

    def counted_forward(*args: Any, **kwargs: Any) -> Any:
        stats.transformer_forward_count += 1
        return original_forward(*args, **kwargs)

    transformer.forward = counted_forward
    try:
        yield stats
    finally:
        transformer.forward = original_forward


def expand_rgda_condition_to_batch(
    pseudo_clean_latent: torch.Tensor,
    rg_map_latent: torch.Tensor,
    token_mask: torch.Tensor,
    target_batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _expand_first_dim(pseudo_clean_latent, target_batch, "pseudo_clean_latent"),
        _expand_first_dim(rg_map_latent, target_batch, "rg_map_latent"),
        _expand_first_dim(token_mask, target_batch, "token_mask"),
    )


def run_with_optional_rgda(
    pipe: Any,
    *,
    mode: Mode,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    injector: RGDAInjector | None = None,
    condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    generator_factory: Callable[[int], Any] | None = None,
) -> tuple[Any, RGDAInferenceStats]:
    generator = generator_factory(seed) if generator_factory is not None else torch.Generator(device="cpu").manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    if mode == "base":
        with transformer_forward_counter(pipe.transformer) as stats:
            output = pipe(**kwargs)
        return output, stats
    if mode != "rgda":
        raise ValueError(f"Unsupported mode {mode}")
    if injector is None or condition is None:
        raise ValueError("RGDA mode requires injector and condition tensors")
    pseudo_clean, rg_map, token_mask = condition
    with torch.inference_mode(), rgda_inference_context(pipe.transformer, injector, pseudo_clean, rg_map, token_mask) as stats:
        output = pipe(**kwargs)
    return output, stats


def copy_label_with_sha(source_label: str | Path, output_label: str | Path) -> str:
    dst = Path(output_label)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_label, dst)
    source_sha = sha256_file(source_label)
    label_sha = sha256_file(dst)
    if source_sha != label_sha:
        raise RuntimeError("Copied label SHA does not match source label SHA")
    return label_sha


def completed_result_is_valid(
    row: dict[str, str],
    result: dict[str, str],
    *,
    mode: str,
    checkpoint_sha256: str,
) -> bool:
    image_path = Path(result.get("output_image_path", ""))
    label_path = Path(result.get("output_label_path", ""))
    if result.get("status") != "PASS" or result.get("mode") != mode:
        return False
    expected_checkpoint = "NONE" if mode == "base" else checkpoint_sha256
    if result.get("seed") != row["seed"] or result.get("rgda_checkpoint_sha256") != expected_checkpoint:
        return False
    for key, row_key in [
        ("generation_index", "generation_index"),
        ("source_image_sha256", "source_image_sha256"),
        ("prompt", "prompt"),
        ("width", "width"),
        ("height", "height"),
        ("inference_steps", "num_inference_steps"),
        ("guidance_scale", "guidance_scale"),
        ("scheduler", "scheduler"),
    ]:
        if result.get(key) != row[row_key]:
            return False
    if not image_path.exists() or not label_path.exists():
        return False
    try:
        with Image.open(image_path) as image:
            if image.size != (int(row["width"]), int(row["height"])):
                return False
            if image.mode != "RGB":
                return False
    except OSError:
        return False
    return sha256_file(image_path) == result.get("output_image_sha256") and sha256_file(label_path) == result.get(
        "output_label_sha256"
    )


def upsert_generation_result(path: str | Path, row: dict[str, str], result: dict[str, str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if out.exists():
        with out.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                existing[(item["mode"], item["generation_index"])] = item
    merged = {
        "generation_index": row["generation_index"],
        "source_sample_id": row["source_sample_id"],
        "seed": row["seed"],
        "prompt": row["prompt"],
        "source_image_sha256": row["source_image_sha256"],
        "source_label_sha256": row["source_label_sha256"],
        "rgda_checkpoint_sha256": row["rgda_checkpoint_sha256"],
        "width": row["width"],
        "height": row["height"],
        "inference_steps": row["num_inference_steps"],
        "guidance_scale": row["guidance_scale"],
        "scheduler": row["scheduler"],
        **result,
    }
    existing[(str(merged["mode"]), str(merged["generation_index"]))] = {
        key: str(merged.get(key, "")) for key in GENERATION_RESULT_FIELDS
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATION_RESULT_FIELDS)
        writer.writeheader()
        for key in sorted(existing, key=lambda item: (int(item[1]), item[0])):
            writer.writerow(existing[key])
    os.replace(tmp, out)


def append_generation_result(path: str | Path, row: dict[str, str], result: dict[str, str]) -> None:
    upsert_generation_result(path, row, result)


def read_generation_results(path: str | Path) -> dict[tuple[str, str], dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8") as handle:
        return {(row["mode"], row["generation_index"]): row for row in csv.DictReader(handle)}


def build_formal_generation_cache_index(cache_manifest: str | Path) -> tuple[dict[str, FormalCacheRecord], int]:
    index: dict[str, FormalCacheRecord] = {}
    duplicates = 0
    with Path(cache_manifest).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sha = row.get("source_image_sha256") or row.get("image_sha256")
            cache_path = row.get("cache_path") or row.get("pt_path") or row.get("path")
            if not sha or not cache_path:
                continue
            if sha in index:
                duplicates += 1
                continue
            index[sha] = FormalCacheRecord(source_image_sha256=sha, cache_path=Path(cache_path))
    return index, duplicates


def validate_generation_cache_join(rows: list[dict[str, str]], cache_manifest: str | Path) -> tuple[int, int, int]:
    index, duplicates = build_formal_generation_cache_index(cache_manifest)
    matched = sum(row["source_image_sha256"] in index for row in rows)
    missing = len(rows) - matched
    if missing or duplicates:
        raise ValueError(f"GENERATION_CACHE_JOIN=FAIL CACHE_MATCHED={matched} CACHE_MISSING={missing} CACHE_DUPLICATE={duplicates}")
    return matched, missing, duplicates


def run_paired_generation(
    rows: list[dict[str, str]],
    *,
    pipe: Any,
    output_root: str | Path,
    results_csv: str | Path,
    checkpoint_sha256: str,
    injector: RGDAInjector | None,
    condition_loader: Callable[[dict[str, str]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    resume: bool = False,
    modes: tuple[str, ...] = ("base", "rgda"),
) -> GenerationRunSummary:
    output = Path(output_root)
    results = read_generation_results(results_csv)
    base_success = rgda_success = failures = 0
    for row in rows:
        for mode in modes:
            expected_sha = "NONE" if mode == "base" else checkpoint_sha256
            existing = results.get((mode, row["generation_index"]))
            if resume and existing and completed_result_is_valid(row, existing, mode=mode, checkpoint_sha256=checkpoint_sha256):
                if mode == "base":
                    base_success += 1
                else:
                    rgda_success += 1
                continue
            try:
                result = _generate_one(pipe, row, mode, output, expected_sha, injector, condition_loader)
                if result["status"] == "PASS":
                    if mode == "base":
                        base_success += 1
                    else:
                        rgda_success += 1
                else:
                    failures += 1
            except (RuntimeError, ValueError, OSError) as exc:
                result = _failure_result(row, mode, expected_sha, exc)
                failures += 1
                upsert_generation_result(results_csv, row, result)
                raise
            upsert_generation_result(results_csv, row, result)
    final = read_generation_results(results_csv)
    return GenerationRunSummary(
        result_rows=len(final),
        base_success=base_success,
        rgda_success=rgda_success,
        failures=failures,
        cache_matched=0,
        cache_missing=0,
        cache_duplicate=0,
    )


def _generate_one(
    pipe: Any,
    row: dict[str, str],
    mode: str,
    output_root: Path,
    checkpoint_sha: str,
    injector: RGDAInjector | None,
    condition_loader: Callable[[dict[str, str]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict[str, str]:
    image_name = row["base_output_filename"] if mode == "base" else row["rgda_output_filename"]
    out_image = output_root / mode / "images" / image_name
    out_label = output_root / mode / "labels" / f"{Path(image_name).stem}.txt"
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    condition = None if mode == "base" else condition_loader(row)
    generated, seconds = timed_generation(
        lambda: run_with_optional_rgda(
            pipe,
            mode=mode,
            prompt=row["prompt"],
            seed=int(row["seed"]),
            height=int(row["height"]),
            width=int(row["width"]),
            num_inference_steps=int(row["num_inference_steps"]),
            guidance_scale=float(row["guidance_scale"]),
            injector=injector,
            condition=condition,
        )
    )
    output_obj, stats = generated
    image = _extract_image(output_obj)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_image)
    with Image.open(out_image) as reopened:
        if reopened.mode != "RGB" or reopened.size != (int(row["width"]), int(row["height"])):
            raise RuntimeError("generated image failed RGB/shape validation")
    label_sha = copy_label_with_sha(row["source_label_path"], out_label)
    allocated = reserved = "0"
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        allocated = str(torch.cuda.max_memory_allocated() / 1024 / 1024)
        reserved = str(torch.cuda.max_memory_reserved() / 1024 / 1024)
    if stats.transformer_forward_count <= 0:
        raise RuntimeError("TRANSFORMER_FORWARD_COUNT_ZERO")
    if mode == "base" and stats.rgda_hook_call_count != 0:
        raise RuntimeError("BASE_RGDA_HOOK_CALLS_NONZERO")
    if mode == "rgda" and stats.rgda_hook_call_count != stats.transformer_forward_count:
        raise RuntimeError("RGDA_HOOK_FORWARD_COUNT_MISMATCH")
    return {
        "mode": mode,
        "rgda_checkpoint_sha256": checkpoint_sha,
        "output_image_path": str(out_image),
        "output_image_sha256": sha256_file(out_image),
        "output_label_path": str(out_label),
        "output_label_sha256": label_sha,
        "generation_seconds": str(seconds),
        "peak_allocated_mib": allocated,
        "peak_reserved_mib": reserved,
        "transformer_forward_count": str(stats.transformer_forward_count),
        "rgda_hook_call_count": str(stats.rgda_hook_call_count),
        "status": "PASS",
        "error_type": "",
        "error_message": "",
    }


def _extract_image(output: Any) -> Image.Image:
    if isinstance(output, Image.Image):
        return output.convert("RGB")
    images = getattr(output, "images", None)
    if images:
        return cast(Image.Image, images[0].convert("RGB"))
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], Image.Image):
        return output[0].convert("RGB")
    raise RuntimeError("pipeline output did not contain a PIL image")


def _failure_result(row: dict[str, str], mode: str, checkpoint_sha: str, exc: BaseException) -> dict[str, str]:
    return {
        "mode": mode,
        "rgda_checkpoint_sha256": checkpoint_sha,
        "output_image_path": "",
        "output_image_sha256": "",
        "output_label_path": "",
        "output_label_sha256": "",
        "generation_seconds": "0",
        "peak_allocated_mib": "0",
        "peak_reserved_mib": "0",
        "transformer_forward_count": "0",
        "rgda_hook_call_count": "0",
        "status": "FAIL",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def timed_generation(call: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    output = call()
    return output, time.perf_counter() - start


def _extract_timestep(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    value = kwargs.get("timestep", kwargs.get("timesteps", None))
    if value is None and len(args) >= 2:
        value = args[1]
    if value is None:
        raise RuntimeError("Could not extract current SD3 timestep for RGDA inference")
    if isinstance(value, torch.Tensor):
        return value.flatten().float()
    return torch.as_tensor([float(value)], dtype=torch.float32)


def _extract_hidden_states(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    value = kwargs.get("hidden_states", None)
    if value is None and args:
        value = args[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError("Could not extract SD3 hidden_states for RGDA batch alignment")
    return value


def _expand_first_dim(tensor: torch.Tensor, target_batch: int, name: str) -> torch.Tensor:
    current = int(tensor.shape[0])
    if current == target_batch:
        return tensor
    if current == 1 and target_batch > 1:
        repeats = [target_batch] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repeats)
    raise ValueError(f"RGDA_CONDITION_BATCH_MISMATCH {name} batch={current} target={target_batch}")


def _expand_timestep_to_batch(timestep: torch.Tensor, target_batch: int) -> torch.Tensor:
    flat = timestep.flatten().float()
    if flat.numel() == target_batch:
        return flat
    if flat.numel() == 1:
        return flat.repeat(target_batch)
    raise ValueError(f"RGDA_CONDITION_BATCH_MISMATCH timestep batch={flat.numel()} target={target_batch}")


def paired_seed_for_row(row: dict[str, str]) -> int:
    return derive_generation_seed(2026, int(row["generation_index"]))

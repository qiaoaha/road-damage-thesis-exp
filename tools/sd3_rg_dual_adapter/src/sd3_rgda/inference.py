"""No-GPU-testable SD3-RGDA paired generation helpers."""

from __future__ import annotations

import csv
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from sd3_rgda.checkpoint import inspect_adapter_checkpoint, load_adapter_checkpoint
from sd3_rgda.generation_manifest import derive_generation_seed, sha256_file
from sd3_rgda.injector import RGDAConditionBatch, RGDAInjector, RGDAPatchHook
from sd3_rgda.sd3_hook import resolve_patch_embedding, temporary_forward_hook

Mode = str


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
) -> Iterator[list[int]]:
    original_forward = transformer.forward
    hook_calls: list[int] = []

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        timestep = _extract_timestep(args, kwargs)
        condition = RGDAConditionBatch(
            pseudo_clean_latents=pseudo_clean_latent,
            rg_maps=rg_map_latent,
            token_mask=token_mask,
            timesteps=timestep,
        )
        _name, patch_embed = resolve_patch_embedding(transformer)
        patch_hook = RGDAPatchHook(injector, condition)
        with temporary_forward_hook(patch_embed, patch_hook):
            result = original_forward(*args, **kwargs)
        hook_calls.append(patch_hook.calls)
        if patch_hook.calls != 1:
            raise RuntimeError(f"Expected exactly one RGDA patch hook call, got {patch_hook.calls}")
        return result

    transformer.forward = patched_forward
    try:
        yield hook_calls
    finally:
        transformer.forward = original_forward


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
) -> tuple[Any, list[int]]:
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
        return pipe(**kwargs), []
    if mode != "rgda":
        raise ValueError(f"Unsupported mode {mode}")
    if injector is None or condition is None:
        raise ValueError("RGDA mode requires injector and condition tensors")
    pseudo_clean, rg_map, token_mask = condition
    with torch.inference_mode(), rgda_inference_context(pipe.transformer, injector, pseudo_clean, rg_map, token_mask) as calls:
        output = pipe(**kwargs)
    return output, calls


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
    if result.get("seed") != row["seed"] or result.get("rgda_checkpoint_sha256") != checkpoint_sha256:
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


def append_generation_result(path: str | Path, row: dict[str, str], result: dict[str, str]) -> None:
    fields = [
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
        "status",
        "error_type",
        "error_message",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
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
    with out.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: str(merged.get(key, "")) for key in fields})


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


def paired_seed_for_row(row: dict[str, str]) -> int:
    return derive_generation_seed(2026, int(row["generation_index"]))

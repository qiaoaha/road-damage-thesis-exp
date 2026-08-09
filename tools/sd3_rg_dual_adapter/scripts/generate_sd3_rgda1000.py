#!/usr/bin/env python3
"""Formal GPU entrypoint for paired SD3 / SD3-RGDA generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from sd3_rgda.generation_manifest import read_generation_manifest
from sd3_rgda.inference import (
    build_formal_generation_cache_index,
    prepare_injector_from_checkpoint,
    run_paired_generation,
    validate_generation_cache_join,
    verify_checkpoint_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--formal-cache-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--mode", choices=["base", "rgda", "paired"], default="paired")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--indices", default="")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading SD3.")
    args = parser.parse_args()
    rows = read_generation_manifest(args.manifest)
    if args.smoke:
        rows = _select_smoke_rows(rows)
    elif args.indices:
        wanted = {item.strip() for item in args.indices.split(",") if item.strip()}
        rows = [row for row in rows if row["generation_index"] in wanted]
    elif len(rows) != 1000:
        raise ValueError(f"Expected 1000 manifest rows, got {len(rows)}")
    validate_generation_cache_join(rows, args.formal_cache_manifest)
    if args.dry_run:
        print("REAL_SD3_USED=NO")
        print("GPU_USED=NO")
        print(f"MODE={args.mode}")
        print(f"ROWS={len(rows)}")
        return 0
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    checkpoint_sha = verify_checkpoint_sha256(args.checkpoint, args.checkpoint_sha256)
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.to("cuda")
    scheduler_name = pipe.scheduler.__class__.__name__
    expected_scheduler = {row["scheduler"] for row in rows}
    if expected_scheduler != {scheduler_name}:
        raise RuntimeError(f"SCHEDULER_CLASS_MISMATCH expected={expected_scheduler} actual={scheduler_name}")
    token_dim = int(getattr(pipe.transformer.config, "caption_projection_dim", 1536))
    latent_channels = int(getattr(pipe.transformer.config, "in_channels", 16))
    patch_size = int(getattr(pipe.transformer.config, "patch_size", 2))
    injector = prepare_injector_from_checkpoint(
        args.checkpoint,
        checkpoint_sha,
        token_dim=token_dim,
        latent_channels=latent_channels,
        patch_size=patch_size,
    ).to(device="cuda")
    _assert_rgda_fp32(injector)
    print("RGDA_PARAMETER_DTYPE=torch.float32")
    print(f"TOKEN_DIM={token_dim}")
    print(f"LATENT_CHANNELS={latent_channels}")
    print(f"PATCH_SIZE={patch_size}")
    cache_index, _duplicates = build_formal_generation_cache_index(args.formal_cache_manifest)

    def condition_loader(row: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        payload = torch.load(cache_index[row["source_image_sha256"]].cache_path, map_location="cuda", weights_only=False)
        return (
            payload["pseudo_clean_latent"].to(device="cuda", dtype=torch.bfloat16),
            payload["rg_map_latent"].to(device="cuda", dtype=torch.bfloat16),
            payload["token_mask"].to(device="cuda"),
        )

    modes = ("base", "rgda") if args.mode == "paired" else (args.mode,)
    summary = run_paired_generation(
        rows,
        pipe=pipe,
        output_root=args.output_root,
        results_csv=args.output_root / "generation_results.csv",
        checkpoint_sha256=checkpoint_sha,
        injector=injector,
        condition_loader=condition_loader,
        resume=args.resume,
        modes=modes,
    )
    print(f"GENERATION_RESULTS_ROWS={summary.result_rows}")
    print(f"BASE_SUCCESS={summary.base_success}")
    print(f"RGDA_SUCCESS={summary.rgda_success}")
    return 0


def _assert_rgda_fp32(injector: torch.nn.Module) -> None:
    bad = {
        name: str(parameter.dtype)
        for name, parameter in injector.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.float32
    }
    if bad:
        raise RuntimeError(f"RGDA_PARAMETER_DTYPE_GATE=FAIL {bad}")


def _select_smoke_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for cls in ["D00", "D10", "D20", "D40"]:
        selected.extend([row for row in rows if row["anchor_class"] == cls][:2])
    selected.extend([row for row in rows if row["is_negative"] == "true"][:8])
    if len(selected) != 16:
        raise ValueError(f"SMOKE16_SOURCE_COUNT={len(selected)}")
    return selected


if __name__ == "__main__":
    raise SystemExit(main())

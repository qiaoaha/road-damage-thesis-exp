#!/usr/bin/env python3
"""Formal GPU entrypoint for paired SD3 / SD3-RGDA generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.generation_manifest import read_generation_manifest
from sd3_rgda.inference import verify_checkpoint_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--mode", choices=["base", "rgda", "paired"], default="paired")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate CLI inputs without loading SD3.")
    args = parser.parse_args()
    rows = read_generation_manifest(args.manifest)
    if len(rows) != 1000:
        raise ValueError(f"Expected 1000 manifest rows, got {len(rows)}")
    if args.dry_run:
        print("REAL_SD3_USED=NO")
        print("GPU_USED=NO")
        print(f"MODE={args.mode}")
        print(f"ROWS={len(rows)}")
        return 0
    verify_checkpoint_sha256(args.checkpoint, args.checkpoint_sha256)
    raise SystemExit(
        "Real SD3 generation is intentionally staged for the GPU phase; use this entrypoint there with local SD3 files."
    )


if __name__ == "__main__":
    raise SystemExit(main())

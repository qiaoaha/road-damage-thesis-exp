from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sd3_rgda.cache import cache_manifest_rows
from sd3_rgda.real_sd3_engine import load_sd3_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Build real SD3 Czech cache files.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run VAE and prompt encoders for cache creation")
    pipe = load_sd3_pipeline(args.model_path, torch.bfloat16)
    cache_manifest_rows(pipe, args.manifest, args.out_dir, args.resolution, torch.bfloat16, args.patch_size)
    print("REAL_CZECH_CACHE_IMPLEMENTED=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

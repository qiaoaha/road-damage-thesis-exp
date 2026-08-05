from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sd3_rgda.real_sd3_engine import run_full_real_validation, write_failure_report
from sd3_rgda.runtime_state import RuntimeState


def parse_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Executable real SD3-RGDA GPU validation.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--micro-manifest", type=Path, required=True)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    state = RuntimeState()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for executable real SD3-RGDA validation")
        if not args.dataset_root.exists():
            raise FileNotFoundError(args.dataset_root)
        run_full_real_validation(
            model_path=args.model_path,
            smoke_manifest=args.smoke_manifest,
            micro_manifest=args.micro_manifest,
            negative_manifest=args.negative_manifest,
            report_dir=args.report_dir,
            resolution=args.resolution,
            dtype=parse_dtype(args.dtype),
            seed=args.seed,
            runtime_state=state,
        )
    except torch.cuda.OutOfMemoryError as exc:
        state.oom_count += 1
        write_failure_report(args.report_dir, state.current_stage, exc, state.oom_count, state.nan_inf_count)
        raise
    except BaseException as exc:
        write_failure_report(args.report_dir, state.current_stage, exc, state.oom_count, state.nan_inf_count)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

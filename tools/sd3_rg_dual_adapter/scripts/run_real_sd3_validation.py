from __future__ import annotations

import argparse
from pathlib import Path


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Real SD3-RGDA GPU validation entrypoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--stage", choices=["smoke100", "micro500", "checkpoint"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.dry_run:
        write_report(
            args.report,
            [
                f"REAL_SD3_VALIDATION_STAGE={args.stage}",
                "REAL_SD3_FORWARD=NOT_RUN_NO_GPU",
                "REAL_CZECH_CACHE=ENTRY_CHECK_ONLY",
                f"STEPS_REQUESTED={args.steps}",
                "GPU_USED=NO",
            ],
        )
        print(f"REAL_SD3_VALIDATION_STAGE={args.stage}")
        return 0

    import torch

    from sd3_rgda.cache import validate_cache_manifest_header
    from sd3_rgda.real_sd3_engine import load_real_sd3_transformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real SD3-RGDA validation")
    if not args.cache_manifest.exists():
        raise FileNotFoundError(args.cache_manifest)
    header = args.cache_manifest.read_text(encoding="utf-8").splitlines()[0].split(",")
    validate_cache_manifest_header(set(header))
    transformer, stats = load_real_sd3_transformer(args.model_path)
    del transformer
    write_report(
        args.report,
        [
            f"REAL_SD3_VALIDATION_STAGE={args.stage}",
            "SD3_FULL_LOAD=PASS",
            f"TRANSFORMER_CLASS={stats.transformer_class}",
            f"PARAMETER_COUNT={stats.parameter_count}",
            f"DTYPE={stats.dtype}",
            f"DEVICE={stats.device}",
            f"PATCH_MODULE_NAME={stats.patch_module_name}",
            "REAL_SD3_FORWARD=PENDING_REAL_FORWARD_LOOP",
            "FINAL_VERDICT=SD3_RGDA_GPU_VALIDATION_FAIL",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

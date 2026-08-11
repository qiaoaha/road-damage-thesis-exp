from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.raal import RAALConfig
from sd3_rgda.raal_formal_engine import RAALFormal5000Runner, write_raal_formal_dry_integration


def _parse_layers(raw: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not layers:
        raise argparse.ArgumentTypeError("--raal-layers must not be empty")
    return layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SD3-RGDA RAAL Formal5000 or run no-GPU dry integration.")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--train-cache-manifest", type=Path)
    parser.add_argument("--eval128-cache-manifest", type=Path)
    parser.add_argument("--val-cache-manifest", type=Path)
    parser.add_argument("--train-pool-manifest", type=Path)
    parser.add_argument("--val-full-manifest", type=Path)
    parser.add_argument("--eval128-manifest", type=Path)
    parser.add_argument("--schedule-manifest", type=Path)
    parser.add_argument("--manifest-summary", type=Path)
    parser.add_argument("--schedule-audit", type=Path)
    parser.add_argument("--clean-proxy-audit", type=Path)
    parser.add_argument("--cache-audit", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--raal-weight", type=float, default=0.02)
    parser.add_argument("--raal-temperature", type=float, default=1.0)
    parser.add_argument("--raal-layers", type=_parse_layers, default=(5, 11, 17))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-integration", action="store_true")
    parser.add_argument("--dry-steps", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_integration:
        write_raal_formal_dry_integration(args.report_dir, steps=args.dry_steps)
        print("RAAL_FORMAL_DRY_INTEGRATION=PASS")
        print("REAL_SD3_USED=NO")
        print("GPU_USED=NO")
        return 0
    required = {
        "--model-path": args.model_path,
        "--train-cache-manifest": args.train_cache_manifest,
        "--eval128-cache-manifest": args.eval128_cache_manifest,
        "--val-cache-manifest": args.val_cache_manifest,
        "--train-pool-manifest": args.train_pool_manifest,
        "--val-full-manifest": args.val_full_manifest,
        "--eval128-manifest": args.eval128_manifest,
        "--schedule-manifest": args.schedule_manifest,
        "--manifest-summary": args.manifest_summary,
        "--schedule-audit": args.schedule_audit,
        "--clean-proxy-audit": args.clean_proxy_audit,
        "--cache-audit": args.cache_audit,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("Real RAAL Formal5000 requires: " + ", ".join(missing))
    runner = RAALFormal5000Runner(
        model_path=args.model_path,
        train_cache_manifest=args.train_cache_manifest,
        eval128_cache_manifest=args.eval128_cache_manifest,
        val_cache_manifest=args.val_cache_manifest,
        train_pool_manifest=args.train_pool_manifest,
        val_full_manifest=args.val_full_manifest,
        eval128_manifest=args.eval128_manifest,
        schedule_manifest=args.schedule_manifest,
        manifest_summary=args.manifest_summary,
        schedule_audit=args.schedule_audit,
        clean_proxy_audit=args.clean_proxy_audit,
        cache_audit=args.cache_audit,
        report_dir=args.report_dir,
        steps=args.steps,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        raal_config=RAALConfig(
            enabled=True,
            weight=args.raal_weight,
            temperature=args.raal_temperature,
            layer_indices=args.raal_layers,
        ),
    )
    runner.run(args.resume_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from sd3_rgda.pilot_engine import Pilot1000Runner, write_mock_dry_integration


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SD3-RGDA Pilot1000 or run no-GPU dry integration.")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--train-cache-manifest", type=Path)
    parser.add_argument("--eval-cache-manifest", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--dry-integration", action="store_true")
    parser.add_argument("--dry-steps", type=int, default=10)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.dry_integration:
        write_mock_dry_integration(args.report_dir, steps=args.dry_steps)
        _package(args.report_dir)
        return 0
    missing = [
        name
        for name, value in {
            "--model-path": args.model_path,
            "--train-cache-manifest": args.train_cache_manifest,
            "--eval-cache-manifest": args.eval_cache_manifest,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError("Real Pilot1000 requires: " + ", ".join(missing))
    runner = Pilot1000Runner(
        model_path=args.model_path,
        train_cache_manifest=args.train_cache_manifest,
        eval_cache_manifest=args.eval_cache_manifest,
        report_dir=args.report_dir,
        steps=args.steps,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
    )
    runner.run(args.resume_from)
    return 0


def _package(report_dir: Path) -> None:
    archive = report_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(report_dir, arcname=report_dir.name)
    print(f"ARCHIVE={archive}")


if __name__ == "__main__":
    raise SystemExit(main())

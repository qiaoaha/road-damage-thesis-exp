from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from sd3_rgda.pilot_engine import write_mock_dry_integration


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SD3-RGDA Pilot1000 or run no-GPU dry integration.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--dry-integration", action="store_true")
    parser.add_argument("--dry-steps", type=int, default=10)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.dry_integration:
        write_mock_dry_integration(args.report_dir, steps=args.dry_steps)
        _package(args.report_dir)
        return 0
    raise RuntimeError("Real Pilot1000 GPU training is intentionally gated for a later authorized run")


def _package(report_dir: Path) -> None:
    archive = report_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(report_dir, arcname=report_dir.name)
    print(f"ARCHIVE={archive}")


if __name__ == "__main__":
    raise SystemExit(main())

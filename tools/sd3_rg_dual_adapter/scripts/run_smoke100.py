from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    if len(rows) < 1:
        raise RuntimeError(f"Empty smoke manifest: {args.manifest}")
    status = "DRY_RUN" if args.dry_run else "PENDING_GPU_IMPLEMENTATION"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(f"SMOKE100={status}\nSTEPS_REQUESTED={args.steps}\nROWS={len(rows)}\n", encoding="utf-8")
    print(f"SMOKE100={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

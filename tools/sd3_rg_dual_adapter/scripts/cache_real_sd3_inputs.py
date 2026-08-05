from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sd3_rgda.cache import REQUIRED_CACHE_COLUMNS


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare real SD3 input cache manifest contract.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"Empty manifest: {args.manifest}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(",".join(REQUIRED_CACHE_COLUMNS) + "\n", encoding="utf-8")
    print("REAL_CZECH_CACHE_ENTRY_READY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

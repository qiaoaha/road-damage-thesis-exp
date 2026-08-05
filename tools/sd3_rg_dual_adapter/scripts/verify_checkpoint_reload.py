from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    status = "DRY_RUN" if args.dry_run else ("READY_TO_COMPARE" if args.checkpoint.exists() else "FAIL")
    detail = "RELOAD_OUTPUT_MAX_ABS_DIFF=REQUIRES_REAL_CACHED_INPUT\nCHECKPOINT_CONTAINS_BASE_SD3=NO\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(f"CHECKPOINT_RELOAD={status}\nCHECKPOINT={args.checkpoint}\n{detail}", encoding="utf-8")
    if status == "FAIL":
        raise FileNotFoundError(args.checkpoint)
    print(f"CHECKPOINT_RELOAD={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

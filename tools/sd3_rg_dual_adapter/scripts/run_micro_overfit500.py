from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_real_sd3_validation import main as real_main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--model-path", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.argv = [
        sys.argv[0],
        "--model-path",
        str(args.model_path),
        "--cache-manifest",
        str(args.manifest),
        "--report",
        str(args.report),
        "--steps",
        str(args.steps),
        "--stage",
        "micro500",
    ]
    if args.dry_run:
        sys.argv.append("--dry-run")
    return real_main()


if __name__ == "__main__":
    raise SystemExit(main())

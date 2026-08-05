from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_real_sd3_validation import main as real_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Thin smoke wrapper for unified real SD3-RGDA validation.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--micro-manifest", type=Path, required=True)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.argv = [
        sys.argv[0],
        "--model-path",
        str(args.model_path),
        "--dataset-root",
        str(args.dataset_root),
        "--smoke-manifest",
        str(args.smoke_manifest),
        "--micro-manifest",
        str(args.micro_manifest),
        "--negative-manifest",
        str(args.negative_manifest),
        "--report-dir",
        str(args.report_dir),
    ]
    return real_main()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapter-only SD3-RGDA training entrypoint.")
    parser.add_argument("--mode", choices=["adapter_only"], default="adapter_only")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.dry_run:
        print("TRAIN_ENTRY_READY=PASS")
        print(f"MODE={args.mode}")
        print(f"LOCAL_FILES_ONLY={args.local_files_only}")
        return 0
    raise SystemExit("Real GPU training is executed by run_sd3_rgda_gpu_validation.sh stage scripts.")


if __name__ == "__main__":
    raise SystemExit(main())

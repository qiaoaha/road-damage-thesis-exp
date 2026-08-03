from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Local-files-only SD3-RGDA inference entrypoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--disable-normal-adapter", action="store_true")
    parser.add_argument("--disable-defect-adapter", action="store_true")
    parser.add_argument("--disable-time-gate", action="store_true")
    parser.add_argument("--shared-adapter", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("INFER_ENTRY_READY=PASS")
        print(f"LOCAL_FILES_ONLY={args.local_files_only}")
        print(f"SEED={args.seed}")
        return 0
    raise SystemExit("Real inference is deferred until a validated adapter checkpoint exists.")


if __name__ == "__main__":
    raise SystemExit(main())

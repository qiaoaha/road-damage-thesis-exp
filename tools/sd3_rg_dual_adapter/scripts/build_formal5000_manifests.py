from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.formal_manifest import write_formal_manifest_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Formal5000 Czech train/val manifests and schedule.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("manifests/formal5000"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    summary, schedule = write_formal_manifest_outputs(args.dataset_root, args.out_dir, args.seed)
    print(f"TRAIN_POOL_ROWS={summary.train_pool_rows}")
    print(f"VAL_FULL_ROWS={summary.val_full_rows}")
    print(f"EVAL128_ROWS={summary.eval128_rows}")
    print(f"SCHEDULE_ROWS={schedule.schedule_rows}")
    print(f"SCHEDULE_AUDIT={schedule.schedule_audit}")
    return 0 if schedule.schedule_audit == "PASS" and summary.class_minimum_gate == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

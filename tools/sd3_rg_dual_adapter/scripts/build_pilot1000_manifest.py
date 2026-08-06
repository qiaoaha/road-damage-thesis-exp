from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.pilot_manifest import write_pilot_manifest_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Pilot1000 Czech train/eval manifests.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("manifests/pilot1000"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    summary = write_pilot_manifest_outputs(args.dataset_root, args.out_dir, args.seed)
    print(f"TRAIN_ROWS={summary.train_rows}")
    print(f"EVAL_ROWS={summary.eval_rows}")
    print(f"TRAIN_EVAL_OVERLAP={summary.train_eval_overlap}")
    print(f"CLASS_MINIMUM_GATE={summary.class_minimum_gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

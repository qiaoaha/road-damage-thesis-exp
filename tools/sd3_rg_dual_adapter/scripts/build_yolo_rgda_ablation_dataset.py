#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.generation_audit import build_yolo_ablation_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--group", choices=["real", "sd3", "rgda", "raal"], required=True)
    parser.add_argument("--synthetic-images", type=Path)
    parser.add_argument("--synthetic-labels", type=Path)
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    args = parser.parse_args()
    audit = build_yolo_ablation_dataset(
        args.real_dataset_root,
        args.output_root,
        group=args.group,
        synthetic_images=args.synthetic_images,
        synthetic_labels=args.synthetic_labels,
        link_mode=args.link_mode,
    )
    print(f"YOLO_ABLATION_DATASET_BUILDER={audit.dataset_gate}")
    print(f"SYNTHETIC_VAL_TEST_LEAKAGE={audit.synthetic_val_test_leakage}")
    return 0 if audit.dataset_gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

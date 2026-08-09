#!/usr/bin/env python3
"""YOLOv11s ablation runner wrapper; not executed in the no-GPU prep stage."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["real", "sd3", "rgda"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-root", type=Path, default=Path("results/yolo_rgda_ablation"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = args.output_root
    name = args.group
    command = [
        "yolo",
        "detect",
        "train",
        f"model={args.model}",
        f"data={args.data}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"seed={args.seed}",
        f"device={args.device}",
        f"project={project}",
        f"name={name}",
    ]
    if args.dry_run:
        print(json.dumps({"command": command, "REAL_SD3_USED": "NO", "GPU_USED": "NO"}))
        return 0
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

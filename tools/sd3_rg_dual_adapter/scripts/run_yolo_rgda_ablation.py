#!/usr/bin/env python3
"""YOLOv11s ablation runner with fixed test425 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["real", "sd3", "rgda"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/yolo_czech_ablation.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = Path(config["result_root"]) / args.group
    run_dir.mkdir(parents=True, exist_ok=True)
    train_args = _train_args(config, args.data, run_dir, args.group)
    (run_dir / "run_config.json").write_text(json.dumps(train_args, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"train_args": train_args, "split": "test", "REAL_SD3_USED": "NO", "GPU_USED": "NO"}))
        return 0
    from ultralytics import YOLO

    model = YOLO(config["model"])
    train_result = model.train(**train_args)
    best_pt = Path(getattr(train_result, "save_dir", run_dir)) / "weights" / "best.pt"
    test_model = YOLO(best_pt)
    metrics = test_model.val(data=str(args.data), split="test", imgsz=config["imgsz"], batch=config["batch"], device=config["device"])
    payload = {
        "group": args.group,
        "best_pt": str(best_pt),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "test_images": 425,
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def _train_args(config: dict[str, Any], data: Path, run_dir: Path, group: str) -> dict[str, Any]:
    keys = ["epochs", "imgsz", "batch", "optimizer", "seed", "patience", "workers", "device"]
    args = {key: config[key] for key in keys if config.get(key) is not None}
    args.update({"data": str(data), "project": str(run_dir.parent), "name": group, "exist_ok": True})
    return args


if __name__ == "__main__":
    raise SystemExit(main())

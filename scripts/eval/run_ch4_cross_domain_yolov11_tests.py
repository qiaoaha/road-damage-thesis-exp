#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run YOLOv11-only Chapter 4 cross-domain validation on Japan and Norway."""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

PYTHON = "/root/miniconda3/envs/yolo5090/bin/python"
YOLO = "/root/miniconda3/envs/yolo5090/bin/yolo"
RUN_ROOT = Path("/root/autodl-tmp/road_damage_exp/runs_ch4_cross_domain_yolov11")
LOG_DIR = RUN_ROOT / "logs"
STATUS_PATH = RUN_ROOT / "ch4_cross_domain_yolov11_status.jsonl"
DATASET_SUMMARY = Path("/root/autodl-tmp/road_damage_exp/datasets_cross_domain/cross_domain_dataset_summary.csv")
PROJECT = "road_damage_ch4_cross_domain_yolov11"
TRAIN_PROJECT = "road_damage_ch4_base80_yolo_compare"
GITHUB_COMMIT = "a7bb254b8ff9be460a7fbb1de279f65a21cb9a2e"
IMG_SIZE = 640
BATCH = 16
SEED = 42

MODEL_VARIANTS = [
    {
        "dataset_variant": "base80_only",
        "selector_method": "none",
        "best_pt_path": "/root/autodl-tmp/road_damage_exp/runs_ch4_base80_wandb/yolov11/yolov11_base80_only/weights/best.pt",
    },
    {
        "dataset_variant": "base80_plus_random_200",
        "selector_method": "random",
        "best_pt_path": "/root/autodl-tmp/road_damage_exp/runs_ch4_base80_wandb/yolov11/yolov11_base80_plus_random_200/weights/best.pt",
    },
    {
        "dataset_variant": "base80_plus_lpips_200",
        "selector_method": "lpips_diversity_only",
        "best_pt_path": "/root/autodl-tmp/road_damage_exp/runs_ch4_base80_wandb/yolov11/yolov11_base80_plus_lpips_200/weights/best.pt",
    },
    {
        "dataset_variant": "base80_plus_ours_200",
        "selector_method": "structure_domain_lpips",
        "best_pt_path": "/root/autodl-tmp/road_damage_exp/runs_ch4_base80_wandb/yolov11/yolov11_base80_plus_ours_200/weights/best.pt",
    },
]

DOMAINS = [
    {
        "target_test_domain": "Japan",
        "domain_key": "japan",
        "target_data_yaml": "/root/autodl-tmp/road_damage_exp/datasets_cross_domain/japan_yolo/data.yaml",
    },
    {
        "target_test_domain": "Norway",
        "domain_key": "norway",
        "target_data_yaml": "/root/autodl-tmp/road_damage_exp/datasets_cross_domain/norway_yolo/data.yaml",
    },
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_metrics(text: str) -> dict:
    clean = strip_ansi(text)
    best = {}
    for raw in clean.splitlines():
        line = raw.strip()
        if not line or not line.startswith("all"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            best = {
                "test_images_reported": int(float(parts[1])),
                "test_instances_reported": int(float(parts[2])),
                "precision": float(parts[3]),
                "recall": float(parts[4]),
                "mAP50": float(parts[5]),
                "mAP50_95": float(parts[6]),
            }
        except ValueError:
            continue
    return best


def load_dataset_stats() -> dict:
    stats = {}
    if DATASET_SUMMARY.exists():
        with DATASET_SUMMARY.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stats[row["domain"]] = row
    return stats


def load_completed() -> set[tuple[str, str]]:
    done = set()
    if not STATUS_PATH.exists():
        return done
    with STATUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "completed" and row.get("mAP50") not in (None, ""):
                done.add((row.get("dataset_variant"), row.get("target_test_domain")))
    return done


def append_status(row: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def log_wandb(row: dict, config: dict) -> tuple[bool, str]:
    try:
        import wandb
        run = wandb.init(project=PROJECT, name=config["run_name"], config=config, reinit=True)
        metric_keys = ["precision", "recall", "mAP50", "mAP50_95", "test_images_reported", "test_instances_reported"]
        payload = {k: row[k] for k in metric_keys if k in row and row[k] is not None}
        payload["status_completed"] = 1 if row.get("status") == "completed" else 0
        run.log(payload)
        run.finish()
        return True, "synced"
    except Exception as exc:
        return False, repr(exc)


def run_one(domain: dict, variant: dict, dataset_stats: dict) -> dict:
    domain_key = domain["domain_key"]
    run_name = f"yolov11_{variant['dataset_variant']}_{domain_key}_test"
    test_dir = RUN_ROOT / domain_key / run_name
    log_path = LOG_DIR / f"{run_name}.log"
    config = {
        "model_family": "yolov11",
        "dataset_variant": variant["dataset_variant"],
        "selector_method": variant["selector_method"],
        "source_train_domain": "China",
        "target_test_domain": domain["target_test_domain"],
        "best_pt_path": variant["best_pt_path"],
        "target_data_yaml": domain["target_data_yaml"],
        "imgsz": IMG_SIZE,
        "batch": BATCH,
        "seed": SEED,
        "github_commit": GITHUB_COMMIT,
        "train_wandb_project": TRAIN_PROJECT,
        "cross_domain_wandb_project": PROJECT,
        "run_name": run_name,
    }
    row = dict(config)
    row.update({
        "run_name": run_name,
        "test_dir": str(test_dir),
        "log_path": str(log_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })

    if not Path(variant["best_pt_path"]).exists():
        row.update({"status": "failed_missing_best_pt", "returncode": -1, "error": "best.pt not found"})
        append_status(row)
        return row
    if not Path(domain["target_data_yaml"]).exists():
        row.update({"status": "failed_missing_data_yaml", "returncode": -1, "error": "data.yaml not found"})
        append_status(row)
        return row

    cmd = [
        YOLO, "detect", "val",
        f"model={variant['best_pt_path']}",
        f"data={domain['target_data_yaml']}",
        "split=test",
        f"imgsz={IMG_SIZE}",
        f"batch={BATCH}",
        "device=0",
        "plots=True",
        f"project={RUN_ROOT / domain_key}",
        f"name={run_name}",
        "exist_ok=True",
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    with log_path.open("w", encoding="utf-8", errors="replace") as logf:
        logf.write("COMMAND: " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        logf.write(proc.stdout)
    metrics = parse_metrics(proc.stdout)
    row.update(metrics)
    row["returncode"] = proc.returncode
    row["finished_at"] = datetime.now().isoformat(timespec="seconds")
    row["status"] = "completed" if proc.returncode == 0 and metrics else "failed_metrics_missing" if proc.returncode == 0 else "failed"
    stat = dataset_stats.get(domain["target_test_domain"], {})
    for key in ["image_count", "label_count", "empty_label_count", "bbox_count", "D00_bbox", "D10_bbox", "D20_bbox", "D40_bbox"]:
        row[key] = stat.get(key)
    wandb_ok, wandb_note = log_wandb(row, config)
    row["wandb_synced"] = wandb_ok
    row["wandb_note"] = wandb_note
    append_status(row)
    return row


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    dataset_stats = load_dataset_stats()
    completed = load_completed()
    rows = []
    for domain in DOMAINS:
        for variant in MODEL_VARIANTS:
            key = (variant["dataset_variant"], domain["target_test_domain"])
            run_name = f"yolov11_{variant['dataset_variant']}_{domain['domain_key']}_test"
            if key in completed:
                row = {
                    "model_family": "yolov11",
                    "dataset_variant": variant["dataset_variant"],
                    "selector_method": variant["selector_method"],
                    "source_train_domain": "China",
                    "target_test_domain": domain["target_test_domain"],
                    "best_pt_path": variant["best_pt_path"],
                    "target_data_yaml": domain["target_data_yaml"],
                    "imgsz": IMG_SIZE,
                    "batch": BATCH,
                    "seed": SEED,
                    "github_commit": GITHUB_COMMIT,
                    "train_wandb_project": TRAIN_PROJECT,
                    "cross_domain_wandb_project": PROJECT,
                    "run_name": run_name,
                    "test_dir": str(RUN_ROOT / domain["domain_key"] / run_name),
                    "status": "skipped_existing",
                    "wandb_synced": "already_recorded",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
                append_status(row)
                print(f"SKIP {run_name}", flush=True)
                rows.append(row)
                continue
            print(f"RUN {run_name}", flush=True)
            row = run_one(domain, variant, dataset_stats)
            print(f"DONE {run_name} status={row.get('status')} mAP50={row.get('mAP50')} mAP50_95={row.get('mAP50_95')} wandb={row.get('wandb_synced')}", flush=True)
            rows.append(row)
    failed = [r for r in rows if str(r.get("status", "")).startswith("failed")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

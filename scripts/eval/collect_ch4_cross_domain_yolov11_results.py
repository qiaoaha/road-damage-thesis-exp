#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect YOLOv11-only Chapter 4 cross-domain validation results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

RUN_ROOT = Path("/root/autodl-tmp/road_damage_exp/runs_ch4_cross_domain_yolov11")
STATUS_PATH = RUN_ROOT / "ch4_cross_domain_yolov11_status.jsonl"
CSV_PATH = RUN_ROOT / "ch4_cross_domain_yolov11_results.csv"
MD_PATH = RUN_ROOT / "ch4_cross_domain_yolov11_results.md"
DATASET_SUMMARY = Path("/root/autodl-tmp/road_damage_exp/datasets_cross_domain/cross_domain_dataset_summary.csv")

FIELDS = [
    "model_family", "dataset_variant", "selector_method", "source_train_domain", "target_test_domain",
    "run_name", "best_pt_path", "target_data_yaml", "test_dir", "status", "wandb_synced",
    "precision", "recall", "mAP50", "mAP50_95", "test_images_reported", "test_instances_reported",
    "image_count", "label_count", "empty_label_count", "bbox_count", "D00_bbox", "D10_bbox", "D20_bbox", "D40_bbox",
    "imgsz", "batch", "seed", "github_commit", "train_wandb_project", "cross_domain_wandb_project", "log_path",
]

ORDER = {
    "base80_only": 0,
    "base80_plus_random_200": 1,
    "base80_plus_lpips_200": 2,
    "base80_plus_ours_200": 3,
}
DOMAIN_ORDER = {"Japan": 0, "Norway": 1}


def load_latest_rows() -> list[dict]:
    latest = {}
    if not STATUS_PATH.exists():
        return []
    with STATUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "skipped_existing":
                continue
            key = (row.get("target_test_domain"), row.get("dataset_variant"))
            latest[key] = row
    rows = list(latest.values())
    rows.sort(key=lambda r: (DOMAIN_ORDER.get(r.get("target_test_domain"), 99), ORDER.get(r.get("dataset_variant"), 99)))
    return rows


def write_csv(rows: list[dict]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(rows: list[dict]) -> None:
    lines = []
    lines.append("# Chapter 4 Cross-Domain YOLOv11 Results")
    lines.append("")
    lines.append(f"- Status source: `{STATUS_PATH}`")
    lines.append(f"- Dataset summary: `{DATASET_SUMMARY}`")
    lines.append(f"- Results CSV: `{CSV_PATH}`")
    lines.append("- Model family: YOLOv11 only")
    lines.append("- Source train domain: China")
    lines.append("- Target test domains: Japan, Norway")
    lines.append("")
    lines.append("| target_domain | dataset_variant | selector_method | precision | recall | mAP50 | mAP50-95 | status | wandb |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|---|")
    for r in rows:
        lines.append(
            f"| {r.get('target_test_domain','')} | {r.get('dataset_variant','')} | {r.get('selector_method','')} | "
            f"{r.get('precision','')} | {r.get('recall','')} | {r.get('mAP50','')} | {r.get('mAP50_95','')} | "
            f"{r.get('status','')} | {r.get('wandb_synced','')} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("Only the four YOLOv11 `best.pt` checkpoints from the China-domain Chapter 4 main experiment are evaluated here. YOLOv5 and YOLOv8 are intentionally excluded from cross-domain validation.")
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = load_latest_rows()
    write_csv(rows)
    write_md(rows)
    print(f"rows={len(rows)}")
    print(CSV_PATH)
    print(MD_PATH)
    incomplete = [r for r in rows if r.get("status") != "completed"]
    missing = len(rows) != 8
    return 1 if incomplete or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

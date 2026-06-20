#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-domain sanity checks for YOLOv11-only Japan/Norway evaluation."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = Path("/root/autodl-tmp/road_damage_exp")
RUN_ROOT = BASE / "runs_ch4_cross_domain_yolov11"
SANITY_ROOT = RUN_ROOT / "sanity_check"
GT_VIS_DIR = SANITY_ROOT / "gt_vis"
PRED_VIS_DIR = SANITY_ROOT / "pred_vis"
FULLREAL_ROOT = RUN_ROOT / "sanity_fullreal"
LABEL_SANITY_CSV = SANITY_ROOT / "cross_domain_label_sanity.csv"
PRED_DIST_CSV = SANITY_ROOT / "prediction_distribution_ours.csv"
FULLREAL_CSV = SANITY_ROOT / "fullreal_yolov11_cross_domain_results.csv"
FULLREAL_MD = SANITY_ROOT / "fullreal_yolov11_cross_domain_results.md"
REPORT_MD = SANITY_ROOT / "cross_domain_sanity_report.md"

DOMAINS = {
    "Japan": BASE / "datasets_cross_domain/japan_yolo",
    "Norway": BASE / "datasets_cross_domain/norway_yolo",
}

OURS_BEST = BASE / "runs_ch4_base80_wandb/yolov11/yolov11_base80_plus_ours_200/weights/best.pt"
FULLREAL_BEST = BASE / "runs/detect/road_damage_yolov11_aug/road_damage_base80_yolov11/weights/best.pt"
YOLO_BIN = "/root/miniconda3/envs/yolo5090/bin/yolo"

CLASS_NAMES = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole",
}
COLORS = {
    0: (230, 57, 70),
    1: (29, 140, 248),
    2: (46, 196, 111),
    3: (255, 183, 3),
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean_text(text: str) -> str:
    return ANSI_RE.sub("", text)


def safe_stats(values: list[float]) -> dict[str, float | str]:
    if not values:
        return {"min": "", "mean": "", "median": "", "max": ""}
    return {
        "min": float(min(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "max": float(max(values)),
    }


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_path_for(image_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def read_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            vals = [float(x) for x in parts[1:5]]
            boxes.append((cid, vals[0], vals[1], vals[2], vals[3]))
        except ValueError:
            continue
    return boxes


def yolo_to_xyxy(box: tuple[int, float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    _, xc, yc, bw, bh = box
    x1 = int(round((xc - bw / 2) * w))
    y1 = int(round((yc - bh / 2) * h))
    x2 = int(round((xc + bw / 2) * w))
    y2 = int(round((yc + bh / 2) * h))
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def draw_solid_box(draw: ImageDraw.ImageDraw, xyxy, color, label: str, width: int = 3) -> None:
    draw.rectangle(xyxy, outline=color, width=width)
    x1, y1, x2, y2 = xyxy
    ty = max(0, y1 - 14)
    draw.rectangle([x1, ty, min(x2, x1 + max(80, len(label) * 7)), ty + 14], fill=color)
    draw.text((x1 + 2, ty), label, fill=(255, 255, 255))


def draw_dashed_box(draw: ImageDraw.ImageDraw, xyxy, color, label: str, width: int = 3, dash: int = 10) -> None:
    x1, y1, x2, y2 = xyxy
    for x in range(x1, x2, dash * 2):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
    for y in range(y1, y2, dash * 2):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)
    ty = min(max(0, y1 + 2), max(0, y2 - 16))
    draw.rectangle([x1, ty, min(x2, x1 + max(90, len(label) * 7)), ty + 14], fill=color)
    draw.text((x1 + 2, ty), label, fill=(0, 0, 0))


def resize_for_sheet(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((cell_w, cell_h), Image.LANCZOS)
    canvas = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
    x = (cell_w - img.width) // 2
    y = (cell_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_contact_sheet(items: list[Image.Image], out_path: Path, cols: int = 5, cell_w: int = 260, cell_h: int = 220) -> None:
    if not items:
        sheet = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
        ImageDraw.Draw(sheet).text((10, 10), "No samples", fill=(0, 0, 0))
        sheet.save(out_path, quality=92)
        return
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (250, 250, 250))
    for i, img in enumerate(items):
        tile = resize_for_sheet(img, cell_w, cell_h)
        sheet.paste(tile, ((i % cols) * cell_w, (i // cols) * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def label_sanity_for_domain(domain: str, yolo_dir: Path) -> dict:
    image_dir = yolo_dir / "images/test"
    label_dir = yolo_dir / "labels/test"
    images = list_images(image_dir)
    label_files = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in label_files}
    empty_count = 0
    bbox_count = 0
    class_counts = {i: 0 for i in range(4)}
    illegal_class_rows = []
    coord_invalid_rows = []
    nonpositive_rows = []
    area_values = []
    bboxes_per_image = []
    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)
    malformed_line_count = 0
    for img in images:
        lp = label_path_for(img, label_dir)
        text = lp.read_text(encoding="utf-8", errors="replace") if lp.exists() else ""
        if not text.strip():
            empty_count += 1
        boxes = []
        for line_no, line in enumerate(text.splitlines(), 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                malformed_line_count += 1
                continue
            try:
                cid = int(float(parts[0]))
                xc, yc, bw, bh = [float(x) for x in parts[1:5]]
            except ValueError:
                malformed_line_count += 1
                continue
            boxes.append((cid, xc, yc, bw, bh))
            bbox_count += 1
            if cid in class_counts:
                class_counts[cid] += 1
            else:
                illegal_class_rows.append(f"{img.name}:{line_no}:{cid}")
            if not all(0.0 <= v <= 1.0 for v in [xc, yc, bw, bh]):
                coord_invalid_rows.append(f"{img.name}:{line_no}")
            if bw <= 0 or bh <= 0:
                nonpositive_rows.append(f"{img.name}:{line_no}")
            else:
                area_values.append(bw * bh)
        bboxes_per_image.append(len(boxes))
    area_stats = safe_stats(area_values)
    count_stats = safe_stats([float(x) for x in bboxes_per_image])
    return {
        "domain": domain,
        "yolo_dir": str(yolo_dir),
        "image_count": len(images),
        "label_count": len(label_files),
        "pair_match": len(missing_labels) == 0 and len(extra_labels) == 0,
        "missing_label_count": len(missing_labels),
        "extra_label_count": len(extra_labels),
        "empty_label_count": empty_count,
        "bbox_count": bbox_count,
        "class_0_bbox": class_counts[0],
        "class_1_bbox": class_counts[1],
        "class_2_bbox": class_counts[2],
        "class_3_bbox": class_counts[3],
        "illegal_class_count": len(illegal_class_rows),
        "bbox_coord_out_of_range_count": len(coord_invalid_rows),
        "bbox_nonpositive_wh_count": len(nonpositive_rows),
        "malformed_line_count": malformed_line_count,
        "bbox_area_min": area_stats["min"],
        "bbox_area_mean": area_stats["mean"],
        "bbox_area_median": area_stats["median"],
        "bbox_area_max": area_stats["max"],
        "bbox_per_image_min": count_stats["min"],
        "bbox_per_image_mean": count_stats["mean"],
        "bbox_per_image_median": count_stats["median"],
        "bbox_per_image_max": count_stats["max"],
        "illegal_class_examples": ";".join(illegal_class_rows[:10]),
        "coord_invalid_examples": ";".join(coord_invalid_rows[:10]),
        "nonpositive_examples": ";".join(nonpositive_rows[:10]),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_gt_contact_sheet(domain: str, yolo_dir: Path, sample_n: int) -> Path:
    random.seed(SEED + len(domain))
    image_dir = yolo_dir / "images/test"
    label_dir = yolo_dir / "labels/test"
    images = list_images(image_dir)
    sample = random.sample(images, min(sample_n, len(images)))
    panels = []
    for img_path in sample:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)
        draw.text((6, 6), img_path.name, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        for box in read_label_file(label_path_for(img_path, label_dir)):
            cid = box[0]
            color = COLORS.get(cid, (255, 255, 255))
            draw_solid_box(draw, yolo_to_xyxy(box, w, h), color, f"GT {cid} {CLASS_NAMES.get(cid, 'unknown')}")
        panels.append(img)
    out = GT_VIS_DIR / f"{domain.lower()}_gt_contact_sheet.jpg"
    make_contact_sheet(panels, out, cols=5, cell_w=280, cell_h=230)
    return out


def predict_for_images(model: YOLO, images: list[Path], conf: float) -> dict[str, list[dict]]:
    preds = {}
    if not images:
        return preds
    results = model.predict(source=[str(p) for p in images], imgsz=640, conf=conf, device=0, verbose=False, stream=False)
    for img_path, res in zip(images, results):
        rows = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            cls = res.boxes.cls.detach().cpu().numpy().astype(int)
            confs = res.boxes.conf.detach().cpu().numpy()
            for b, c, cf in zip(xyxy, cls, confs):
                rows.append({"class_id": int(c), "conf": float(cf), "xyxy": [float(x) for x in b]})
        preds[img_path.name] = rows
    return preds


def create_pred_and_compare_sheets(domain: str, yolo_dir: Path, model: YOLO, sample_n: int) -> tuple[Path, Path, list[Path]]:
    random.seed(SEED + 100 + len(domain))
    image_dir = yolo_dir / "images/test"
    label_dir = yolo_dir / "labels/test"
    images = list_images(image_dir)
    sample = random.sample(images, min(sample_n, len(images)))
    preds = predict_for_images(model, sample, conf=0.05)
    pred_panels = []
    compare_panels = []
    for img_path in sample:
        original = Image.open(img_path).convert("RGB")
        w, h = original.size
        pred_img = original.copy()
        pred_draw = ImageDraw.Draw(pred_img)
        pred_draw.text((6, 6), img_path.name, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        comp_img = original.copy()
        comp_draw = ImageDraw.Draw(comp_img)
        comp_draw.text((6, 6), img_path.name, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        for box in read_label_file(label_path_for(img_path, label_dir)):
            cid = box[0]
            draw_solid_box(comp_draw, yolo_to_xyxy(box, w, h), COLORS.get(cid, (255, 255, 255)), f"GT {cid}")
        for p in preds.get(img_path.name, []):
            cid = p["class_id"]
            label = f"P {cid} {p['conf']:.2f}"
            color = COLORS.get(cid, (255, 255, 255))
            xyxy = tuple(int(round(x)) for x in p["xyxy"])
            draw_solid_box(pred_draw, xyxy, color, label)
            draw_dashed_box(comp_draw, xyxy, color, label)
        pred_panels.append(pred_img)
        compare_panels.append(comp_img)
    pred_out = PRED_VIS_DIR / f"{domain.lower()}_ours_pred_contact_sheet.jpg"
    compare_out = PRED_VIS_DIR / f"{domain.lower()}_gt_pred_compare.jpg"
    make_contact_sheet(pred_panels, pred_out, cols=5, cell_w=280, cell_h=230)
    make_contact_sheet(compare_panels, compare_out, cols=5, cell_w=280, cell_h=230)
    return pred_out, compare_out, sample


def prediction_distribution_for_domain(domain: str, yolo_dir: Path, model: YOLO) -> dict:
    image_dir = yolo_dir / "images/test"
    images = list_images(image_dir)
    results = model.predict(source=[str(p) for p in images], imgsz=640, conf=0.001, device=0, verbose=False, stream=False)
    per_image_counts = []
    confs = []
    conf_ge_005 = 0
    conf_ge_025 = 0
    class_counts = {i: 0 for i in range(4)}
    other_class_count = 0
    for res in results:
        n = 0
        if res.boxes is not None and len(res.boxes) > 0:
            cls = res.boxes.cls.detach().cpu().numpy().astype(int)
            cf = res.boxes.conf.detach().cpu().numpy()
            n = len(cls)
            for c, score in zip(cls, cf):
                confs.append(float(score))
                if score >= 0.05:
                    conf_ge_005 += 1
                if score >= 0.25:
                    conf_ge_025 += 1
                if int(c) in class_counts:
                    class_counts[int(c)] += 1
                else:
                    other_class_count += 1
        per_image_counts.append(n)
    conf_stats = safe_stats(confs)
    count_stats = safe_stats([float(x) for x in per_image_counts])
    pred_box_count = sum(per_image_counts)
    top_class = max(class_counts, key=lambda k: class_counts[k]) if pred_box_count else ""
    top_class_count = class_counts[top_class] if pred_box_count else 0
    top_class_ratio = (top_class_count / pred_box_count) if pred_box_count else 0
    almost_no_predictions = pred_box_count < len(images) * 0.1 or conf_ge_005 < len(images) * 0.05
    severe_bias = top_class_ratio >= 0.75 if pred_box_count else False
    return {
        "domain": domain,
        "model": "yolov11_base80_plus_ours_200",
        "predict_conf_threshold": 0.001,
        "image_count": len(images),
        "pred_box_count": pred_box_count,
        "pred_box_count_conf_ge_0_05": conf_ge_005,
        "pred_box_count_conf_ge_0_25": conf_ge_025,
        "pred_per_image_min": count_stats["min"],
        "pred_per_image_mean": count_stats["mean"],
        "pred_per_image_median": count_stats["median"],
        "pred_per_image_max": count_stats["max"],
        "pred_class_0_count": class_counts[0],
        "pred_class_1_count": class_counts[1],
        "pred_class_2_count": class_counts[2],
        "pred_class_3_count": class_counts[3],
        "pred_other_class_count": other_class_count,
        "conf_min": conf_stats["min"],
        "conf_mean": conf_stats["mean"],
        "conf_median": conf_stats["median"],
        "conf_max": conf_stats["max"],
        "top_pred_class": top_class,
        "top_pred_class_ratio": top_class_ratio,
        "almost_no_predictions": almost_no_predictions,
        "severe_class_bias": severe_bias,
    }


def parse_val_metrics(log_text: str) -> tuple[dict, list[dict]]:
    overall = {}
    per_class = []
    for raw in clean_text(log_text).splitlines():
        line = raw.strip()
        if not line or line.startswith("Speed:") or line.startswith("Results saved"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        name = parts[0]
        try:
            imgs = int(float(parts[1]))
            inst = int(float(parts[2]))
            p, r, map50, map5095 = [float(x) for x in parts[3:7]]
        except ValueError:
            continue
        rec = {
            "class_name": name,
            "images": imgs,
            "instances": inst,
            "precision": p,
            "recall": r,
            "mAP50": map50,
            "mAP50_95": map5095,
        }
        if name == "all":
            overall = rec
        else:
            per_class.append(rec)
    return overall, per_class


def run_fullreal_val(domain: str, yolo_dir: Path) -> tuple[dict, list[dict], Path]:
    name = f"fullreal_yolov11_{domain.lower()}_test"
    log_path = SANITY_ROOT / f"{name}.log"
    cmd = [
        YOLO_BIN, "detect", "val",
        f"model={FULLREAL_BEST}",
        f"data={yolo_dir / 'data.yaml'}",
        "split=test",
        "imgsz=640",
        "batch=16",
        "device=0",
        f"project={FULLREAL_ROOT}",
        f"name={name}",
        "exist_ok=True",
        "plots=True",
    ]
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
    log_path.write_text("COMMAND: " + " ".join(cmd) + "\n\n" + proc.stdout, encoding="utf-8")
    overall, per_class = parse_val_metrics(proc.stdout)
    overall.update({
        "domain": domain,
        "run_name": name,
        "model_path": str(FULLREAL_BEST),
        "data_yaml": str(yolo_dir / "data.yaml"),
        "test_dir": str(FULLREAL_ROOT / name),
        "log_path": str(log_path),
        "returncode": proc.returncode,
        "status": "completed" if proc.returncode == 0 and overall else "failed",
    })
    for rec in per_class:
        rec.update({
            "domain": domain,
            "run_name": name,
            "model_path": str(FULLREAL_BEST),
            "data_yaml": str(yolo_dir / "data.yaml"),
            "test_dir": str(FULLREAL_ROOT / name),
            "log_path": str(log_path),
            "status": overall.get("status", "failed"),
        })
    return overall, per_class, log_path


def write_fullreal_outputs(overall_rows: list[dict], class_rows: list[dict]) -> None:
    fields = ["domain", "level", "class_name", "images", "instances", "precision", "recall", "mAP50", "mAP50_95", "status", "run_name", "model_path", "data_yaml", "test_dir", "log_path"]
    rows = []
    for row in overall_rows:
        r = dict(row)
        r["level"] = "overall"
        rows.append(r)
    for row in class_rows:
        r = dict(row)
        r["level"] = "class"
        rows.append(r)
    write_csv(FULLREAL_CSV, rows, fields)
    lines = ["# Full-Real YOLOv11 Cross-Domain Sanity Results", "", "| domain | level | class | P | R | mAP50 | mAP50-95 | status |", "|---|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r.get('domain','')} | {r.get('level','')} | {r.get('class_name','')} | {r.get('precision','')} | {r.get('recall','')} | {r.get('mAP50','')} | {r.get('mAP50_95','')} | {r.get('status','')} |")
    FULLREAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(label_rows: list[dict], pred_rows: list[dict], fullreal_rows: list[dict], gt_paths: dict, pred_paths: dict, compare_paths: dict) -> None:
    labels_ok = all(
        r["pair_match"] and r["illegal_class_count"] == 0 and r["bbox_coord_out_of_range_count"] == 0 and r["bbox_nonpositive_wh_count"] == 0 and r["malformed_line_count"] == 0
        for r in label_rows
    )
    bbox_ok = all(r["bbox_count"] > 0 and float(r["bbox_area_mean"]) > 0 for r in label_rows)
    mapping_suspect = any(r["illegal_class_count"] > 0 for r in label_rows)
    ours_no_pred = any(bool(r["almost_no_predictions"]) for r in pred_rows)
    fullreal_low = all(float(r.get("mAP50") or 0) < 0.1 for r in fullreal_rows if r.get("class_name") == "all")
    if not labels_ok or mapping_suspect:
        likely = "B. ????/??????"
    elif fullreal_low and ours_no_pred:
        likely = "D. ???????????? + ?????????"
    elif fullreal_low:
        likely = "A. ????????"
    else:
        likely = "C. ?????????"
    lines = [
        "# Cross-Domain Sanity Check Report",
        "",
        "## Inputs",
        "",
        f"- Japan dataset: `{DOMAINS['Japan']}`",
        f"- Norway dataset: `{DOMAINS['Norway']}`",
        f"- Ours YOLOv11 weight: `{OURS_BEST}`",
        f"- Full-real/real-only YOLOv11 weight checked: `{FULLREAL_BEST}`",
        "",
        "## Label Sanity",
        "",
        "| domain | images | labels | empty_labels | boxes | c0 | c1 | c2 | c3 | illegal_class | coord_oob | nonpositive_wh | area_mean | boxes_per_image_mean | pair_match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in label_rows:
        lines.append(f"| {r['domain']} | {r['image_count']} | {r['label_count']} | {r['empty_label_count']} | {r['bbox_count']} | {r['class_0_bbox']} | {r['class_1_bbox']} | {r['class_2_bbox']} | {r['class_3_bbox']} | {r['illegal_class_count']} | {r['bbox_coord_out_of_range_count']} | {r['bbox_nonpositive_wh_count']} | {r['bbox_area_mean']} | {r['bbox_per_image_mean']} | {r['pair_match']} |")
    lines += [
        "",
        "## Visualization Artifacts",
        "",
    ]
    for d in DOMAINS:
        lines.append(f"- {d} GT contact sheet: `{gt_paths[d]}`")
        lines.append(f"- {d} Ours prediction contact sheet: `{pred_paths[d]}`")
        lines.append(f"- {d} GT + prediction compare sheet: `{compare_paths[d]}`")
    lines += [
        "",
        "## Ours Prediction Distribution",
        "",
        "| domain | pred_boxes(conf>=0.001) | pred_boxes(conf>=0.05) | pred_boxes(conf>=0.25) | per_image_mean | conf_mean | conf_median | conf_max | c0 | c1 | c2 | c3 | top_class_ratio | almost_no_predictions | severe_class_bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in pred_rows:
        lines.append(f"| {r['domain']} | {r['pred_box_count']} | {r['pred_box_count_conf_ge_0_05']} | {r['pred_box_count_conf_ge_0_25']} | {r['pred_per_image_mean']} | {r['conf_mean']} | {r['conf_median']} | {r['conf_max']} | {r['pred_class_0_count']} | {r['pred_class_1_count']} | {r['pred_class_2_count']} | {r['pred_class_3_count']} | {r['top_pred_class_ratio']} | {r['almost_no_predictions']} | {r['severe_class_bias']} |")
    lines += [
        "",
        "## Full-Real YOLOv11 Sanity Val",
        "",
        "| domain | P | R | mAP50 | mAP50-95 | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in fullreal_rows:
        if r.get("class_name") == "all":
            lines.append(f"| {r.get('domain','')} | {r.get('precision','')} | {r.get('recall','')} | {r.get('mAP50','')} | {r.get('mAP50_95','')} | {r.get('status','')} |")
    lines += [
        "",
        "## Judgement",
        "",
        f"1. Japan / Norway labels normal: `{labels_ok}`.",
        f"2. Bbox values normal: `{bbox_ok}`.",
        f"3. Class mapping suspicious: `{mapping_suspect}`.",
        "4. GT visualization generated for manual inspection; inspect contact sheets listed above.",
        f"5. Ours almost no predictions: `{ours_no_pred}`.",
        f"6. Full-real YOLOv11 cross-domain results also very low: `{fullreal_low}`.",
        f"7. Most likely current cause: `{likely}`.",
        "",
        "Note: This report is a sanity check only. It does not retrain models and does not change the experiment conclusion.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    SANITY_ROOT.mkdir(parents=True, exist_ok=True)
    GT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_VIS_DIR.mkdir(parents=True, exist_ok=True)
    FULLREAL_ROOT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    label_rows = [label_sanity_for_domain(domain, path) for domain, path in DOMAINS.items()]
    label_fields = list(label_rows[0].keys())
    write_csv(LABEL_SANITY_CSV, label_rows, label_fields)

    gt_paths = {domain: create_gt_contact_sheet(domain, path, 50) for domain, path in DOMAINS.items()}

    model = YOLO(str(OURS_BEST))
    pred_paths = {}
    compare_paths = {}
    for domain, path in DOMAINS.items():
        pred_out, comp_out, _ = create_pred_and_compare_sheets(domain, path, model, 30)
        pred_paths[domain] = pred_out
        compare_paths[domain] = comp_out
    pred_rows = [prediction_distribution_for_domain(domain, path, model) for domain, path in DOMAINS.items()]
    pred_fields = list(pred_rows[0].keys())
    write_csv(PRED_DIST_CSV, pred_rows, pred_fields)

    fullreal_overall = []
    fullreal_classes = []
    if FULLREAL_BEST.exists():
        for domain, path in DOMAINS.items():
            overall, per_class, _ = run_fullreal_val(domain, path)
            if overall:
                fullreal_overall.append(overall)
            fullreal_classes.extend(per_class)
    else:
        for domain, path in DOMAINS.items():
            fullreal_overall.append({
                "domain": domain,
                "class_name": "all",
                "status": "missing_fullreal_weight",
                "model_path": str(FULLREAL_BEST),
                "data_yaml": str(path / "data.yaml"),
            })
    write_fullreal_outputs(fullreal_overall, fullreal_classes)
    generate_report(label_rows, pred_rows, fullreal_overall, gt_paths, pred_paths, compare_paths)

    print("LABEL_SANITY_CSV", LABEL_SANITY_CSV)
    print("PRED_DIST_CSV", PRED_DIST_CSV)
    print("FULLREAL_CSV", FULLREAL_CSV)
    print("FULLREAL_MD", FULLREAL_MD)
    print("REPORT_MD", REPORT_MD)
    for d in DOMAINS:
        print("GT_VIS", d, gt_paths[d])
        print("PRED_VIS", d, pred_paths[d])
        print("COMPARE_VIS", d, compare_paths[d])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

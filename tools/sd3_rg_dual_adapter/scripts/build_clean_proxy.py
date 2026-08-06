from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from sd3_rgda.cache import read_yolo_boxes
from sd3_rgda.clean_proxy import audit_clean_proxy, sha256_file, telea_inpaint_proxy


def main() -> int:
    parser = argparse.ArgumentParser(description="Build audited Telea pseudo-clean proxy images.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("pilot_assets/clean_proxy"))
    parser.add_argument("--preview-per-class", type=int, default=4)
    args = parser.parse_args()
    result = build_clean_proxy_assets(args.train_manifest, args.eval_manifest, args.out_dir, args.preview_per_class)
    print(f"PROXY_TOTAL={result['PROXY_TOTAL']}")
    print(f"PROXY_MISSING={result['PROXY_MISSING']}")
    print(f"CLEAN_PROXY_READY={result['CLEAN_PROXY_READY']}")
    return 0 if result["CLEAN_PROXY_READY"] == "PASS" else 2


def build_clean_proxy_assets(
    train_manifest: Path,
    eval_manifest: Path,
    out_dir: Path,
    preview_per_class: int = 4,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "clean_proxy_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "clean_proxy_manifest.csv"
    rows = [("train", row) for row in _read_rows(train_manifest)] + [("eval", row) for row in _read_rows(eval_manifest)]
    audit_rows: list[dict[str, str]] = []
    qa = {
        "PROXY_TOTAL": 0,
        "PROXY_MISSING": 0,
        "PROXY_CORRUPT": 0,
        "PROXY_SHAPE_MISMATCH": 0,
        "NEGATIVE_PROXY_EXACT_MATCH": "PASS",
        "POSITIVE_MASK_CHANGED": "PASS",
        "OUTSIDE_MASK_UNCHANGED": "PASS",
        "GRAY_RECTANGLE_METHOD_USED": "NO",
        "SAM_USED": "NO",
        "CLEAN_PROXY_READY": "PASS",
    }
    preview_counts: dict[str, int] = {"NEG": 0, "D00": 0, "D10": 0, "D20": 0, "D40": 0}
    for split_name, row in rows:
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        with Image.open(image_path) as image:
            original = image.convert("RGB")
            boxes = read_yolo_boxes(label_path, original.size)
            proxy, mask = telea_inpaint_proxy(image_path, boxes)
        audit = audit_clean_proxy(original, proxy, mask, row["is_negative"] == "true")
        subdir = out_dir / split_name
        subdir.mkdir(parents=True, exist_ok=True)
        proxy_path = subdir / f"{row['sample_id']}.png"
        mask_path = subdir / f"{row['sample_id']}_mask.png"
        proxy.save(proxy_path)
        mask.save(mask_path)
        try:
            reloaded_proxy = Image.open(proxy_path).convert("RGB")
            reloaded_mask = Image.open(mask_path).convert("L")
        except OSError:
            qa["PROXY_CORRUPT"] = int(qa["PROXY_CORRUPT"]) + 1
            reloaded_proxy = proxy
            reloaded_mask = mask
        audit = audit_clean_proxy(original, reloaded_proxy, reloaded_mask, row["is_negative"] == "true")
        if not proxy_path.exists():
            qa["PROXY_MISSING"] = int(qa["PROXY_MISSING"]) + 1
        if reloaded_proxy.size != original.size:
            qa["PROXY_SHAPE_MISMATCH"] = int(qa["PROXY_SHAPE_MISMATCH"]) + 1
        if not audit.outside_mask_unchanged:
            qa["OUTSIDE_MASK_UNCHANGED"] = "FAIL"
        if not audit.negative_exact_match:
            qa["NEGATIVE_PROXY_EXACT_MATCH"] = "FAIL"
        if not audit.positive_mask_changed:
            qa["POSITIVE_MASK_CHANGED"] = "FAIL"
        qa["PROXY_TOTAL"] = int(qa["PROXY_TOTAL"]) + 1
        audit_rows.append(
            {
                "sample_id": row["sample_id"],
                "split": split_name,
                "image_path": row["image_path"],
                "label_path": row["label_path"],
                "clean_proxy_path": str(proxy_path),
                "mask_path": str(mask_path),
                "image_sha256": row["image_sha256"],
                "label_sha256": row["label_sha256"],
                "clean_proxy_sha256": sha256_file(proxy_path),
                "method": audit.method,
                "mask_pixels": str(audit.mask_pixels),
                "outside_mask_unchanged": str(audit.outside_mask_unchanged),
                "is_negative": row["is_negative"],
                "anchor_class": row["anchor_class"],
                "class_ids": row["class_ids"],
            }
        )
        preview_key = "NEG" if row["is_negative"] == "true" else row["anchor_class"]
        if preview_counts.get(preview_key, 0) < preview_per_class:
            _write_preview(preview_dir / f"{split_name}_{row['sample_id']}_{preview_key}.jpg", original, mask, proxy)
            preview_counts[preview_key] = preview_counts.get(preview_key, 0) + 1
    qa["CLEAN_PROXY_READY"] = "PASS" if clean_proxy_qa_passed(qa) else "FAIL"
    _write_csv(manifest_path, audit_rows)
    (out_dir / "clean_proxy_audit.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return qa


def clean_proxy_qa_passed(qa: dict[str, object], expected_total: int = 576) -> bool:
    integer_gates = {
        "PROXY_TOTAL": expected_total,
        "PROXY_MISSING": 0,
        "PROXY_CORRUPT": 0,
        "PROXY_SHAPE_MISMATCH": 0,
    }
    for key, expected in integer_gates.items():
        if int(qa.get(key, -1)) != expected:
            return False
    return (
        qa.get("NEGATIVE_PROXY_EXACT_MATCH") == "PASS"
        and qa.get("POSITIVE_MASK_CHANGED") == "PASS"
        and qa.get("OUTSIDE_MASK_UNCHANGED") == "PASS"
        and qa.get("GRAY_RECTANGLE_METHOD_USED") == "NO"
        and qa.get("SAM_USED") == "NO"
    )


def _write_preview(path: Path, original: Image.Image, mask: Image.Image, proxy: Image.Image) -> None:
    mask_rgb = mask.convert("RGB")
    width, height = original.size
    canvas = Image.new("RGB", (width * 3, height), "white")
    canvas.paste(original, (0, 0))
    canvas.paste(mask_rgb, (width, 0))
    canvas.paste(proxy, (width * 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "original", fill=(255, 0, 0))
    draw.text((width + 8, 8), "mask", fill=(255, 0, 0))
    draw.text((width * 2 + 8, 8), "pseudo-clean", fill=(255, 0, 0))
    canvas.save(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

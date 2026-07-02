from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import as_path, cfg_path
from .io_utils import image_files, write_csv


@dataclass
class DatasetIndex:
    real_rows: list[dict]
    generated_rows: list[dict]


def label_path_for(image_path: Path, image_root: Path | None, label_root: Path | None) -> Path | None:
    if not image_root or not label_root:
        return None
    try:
        rel = image_path.relative_to(image_root)
    except ValueError:
        rel = Path(image_path.name)
    return (label_root / rel).with_suffix(".txt")


def parse_yolo_label(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        boxes.append({"class_id": cls, "area": max(w, 0.0) * max(h, 0.0)})
    return boxes


def main_class(boxes: list[dict]) -> tuple[str, str]:
    if not boxes:
        return "unknown", "no_label"
    classes = {b["class_id"] for b in boxes}
    if len(classes) == 1:
        return str(next(iter(classes))), "single_class"
    best = max(boxes, key=lambda item: item["area"])
    return str(best["class_id"]), "largest_bbox_area"


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return 0, 0


def build_rows(
    images_dir: Path | None,
    labels_dir: Path | None,
    split_type: str,
    class_names: dict,
    limit_per_class: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    per_class: dict[str, int] = {}
    for img in image_files(images_dir):
        label = label_path_for(img, images_dir, labels_dir)
        boxes = parse_yolo_label(label)
        cid, method = main_class(boxes)
        if limit_per_class is not None:
            if per_class.get(cid, 0) >= limit_per_class:
                continue
            per_class[cid] = per_class.get(cid, 0) + 1
        width, height = image_size(img)
        rows.append(
            {
                "image_path": str(img),
                "label_path": str(label) if label else "",
                "split_type": split_type,
                "class_id": cid,
                "class_name": class_names.get(cid, class_names.get(int(cid), "unknown")) if cid != "unknown" else "unknown",
                "num_boxes": len(boxes),
                "main_class_method": method,
                "width": width,
                "height": height,
            }
        )
    return rows


def build_index(cfg: dict, output_dir: Path, dry_run: bool = False) -> DatasetIndex:
    ds = cfg.get("dataset", {})
    classes = cfg_path(cfg, "classes.names", {}) or {}
    class_names = {str(k): v for k, v in classes.items()}
    limit = int(cfg_path(cfg, "runtime.dry_run_per_class", 5) or 5) if dry_run else None
    real_rows = build_rows(as_path(ds.get("real_images_dir")), as_path(ds.get("real_labels_dir")), "real", class_names, limit)
    gen_rows = build_rows(
        as_path(ds.get("generated_images_dir")),
        as_path(ds.get("generated_labels_dir")),
        "generated",
        class_names,
        limit,
    )
    write_csv(output_dir / "cache" / "real_index.csv", real_rows, INDEX_FIELDS)
    write_csv(output_dir / "cache" / "generated_index.csv", gen_rows, INDEX_FIELDS)
    return DatasetIndex(real_rows, gen_rows)


INDEX_FIELDS = [
    "image_path",
    "label_path",
    "split_type",
    "class_id",
    "class_name",
    "num_boxes",
    "main_class_method",
    "width",
    "height",
]

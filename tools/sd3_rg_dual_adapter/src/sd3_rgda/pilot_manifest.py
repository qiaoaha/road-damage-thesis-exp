"""Pilot1000 train/eval manifest selection for Czech RDD2022 data."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from sd3_rgda.cache import read_yolo_boxes

CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
PILOT_FIELDS = [
    "sample_id",
    "image_path",
    "label_path",
    "split",
    "is_negative",
    "anchor_class",
    "class_ids",
    "box_count",
    "image_sha256",
    "label_sha256",
    "selection_seed",
]


@dataclass(frozen=True)
class PilotManifestSummary:
    train_rows: int
    train_unique_images: int
    train_positive: int
    train_negative: int
    eval_rows: int
    eval_unique_images: int
    eval_positive: int
    eval_negative: int
    train_eval_overlap: int
    val_test_leakage: int
    class_minimum_gate: str
    train_anchor_counts: dict[str, int]
    eval_anchor_counts: dict[str, int]


@dataclass(frozen=True)
class DatasetCandidate:
    image_path: Path
    label_path: Path
    split: str
    class_ids: tuple[int, ...]
    box_count: int

    @property
    def is_negative(self) -> bool:
        return self.box_count == 0


def find_split_dirs(dataset_root: Path) -> tuple[Path, Path]:
    train_images = _first_existing(
        dataset_root / "images" / "train",
        dataset_root / "train" / "images",
        dataset_root / "images" / "Train",
    )
    train_labels = _first_existing(
        dataset_root / "labels" / "train",
        dataset_root / "train" / "labels",
        dataset_root / "labels" / "Train",
    )
    return train_images, train_labels


def collect_train_candidates(dataset_root: str | Path) -> list[DatasetCandidate]:
    root = Path(dataset_root)
    image_dir, label_dir = find_split_dirs(root)
    candidates: list[DatasetCandidate] = []
    for image_path in sorted(_image_files(image_dir)):
        label_path = label_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            boxes = read_yolo_boxes(label_path, image.size)
        class_ids = tuple(sorted({box.class_id for box in boxes}))
        candidates.append(
            DatasetCandidate(
                image_path=image_path.resolve(),
                label_path=label_path.resolve(),
                split="train",
                class_ids=class_ids,
                box_count=len(boxes),
            )
        )
    return candidates


def select_pilot_manifests(
    candidates: list[DatasetCandidate],
    *,
    seed: int = 2026,
    train_positive: int = 256,
    train_negative: int = 256,
    eval_positive: int = 32,
    eval_negative: int = 32,
) -> tuple[list[dict[str, str]], list[dict[str, str]], PilotManifestSummary]:
    rng = random.Random(seed)
    positives = [item for item in candidates if not item.is_negative]
    negatives = [item for item in candidates if item.is_negative]
    selected_train_pos = _select_positive_quota(positives, {0: 64, 1: 64, 2: 64, 3: 64}, rng)
    if len(selected_train_pos) < train_positive:
        selected_train_pos.extend(_fill_remaining(positives, selected_train_pos, train_positive, rng))
    selected_train_neg = _sample_unique(negatives, train_negative, rng)
    used = {item.image_path for item in [*selected_train_pos, *selected_train_neg]}
    eval_pool_pos = [item for item in positives if item.image_path not in used]
    eval_pool_neg = [item for item in negatives if item.image_path not in used]
    selected_eval_pos = _select_positive_quota(eval_pool_pos, {0: 8, 1: 8, 2: 8, 3: 8}, rng)
    if len(selected_eval_pos) < eval_positive:
        selected_eval_pos.extend(_fill_remaining(eval_pool_pos, selected_eval_pos, eval_positive, rng))
    selected_eval_neg = _sample_unique(eval_pool_neg, eval_negative, rng)
    train = [*selected_train_pos[:train_positive], *selected_train_neg]
    eval_rows = [*selected_eval_pos[:eval_positive], *selected_eval_neg]
    rng.shuffle(train)
    rng.shuffle(eval_rows)
    train_dicts = [_row_dict(index, item, seed, "pilot_train") for index, item in enumerate(train)]
    eval_dicts = [_row_dict(index, item, seed, "pilot_eval") for index, item in enumerate(eval_rows)]
    summary = summarize_manifest_rows(train_dicts, eval_dicts)
    if summary.class_minimum_gate != "PASS":
        raise ValueError("CLASS_MINIMUM_GATE=FAIL")
    return train_dicts, eval_dicts, summary


def summarize_manifest_rows(train: list[dict[str, str]], eval_rows: list[dict[str, str]]) -> PilotManifestSummary:
    train_images = {row["image_path"] for row in train}
    eval_images = {row["image_path"] for row in eval_rows}
    train_anchor_counts = _anchor_counts(train)
    eval_anchor_counts = _anchor_counts(eval_rows)
    return PilotManifestSummary(
        train_rows=len(train),
        train_unique_images=len(train_images),
        train_positive=sum(row["is_negative"] == "false" for row in train),
        train_negative=sum(row["is_negative"] == "true" for row in train),
        eval_rows=len(eval_rows),
        eval_unique_images=len(eval_images),
        eval_positive=sum(row["is_negative"] == "false" for row in eval_rows),
        eval_negative=sum(row["is_negative"] == "true" for row in eval_rows),
        train_eval_overlap=len(train_images & eval_images),
        val_test_leakage=sum(row["split"] != "train" for row in [*train, *eval_rows]),
        class_minimum_gate="PASS" if all(train_anchor_counts.get(CLASS_NAMES[i], 0) >= 32 for i in range(4)) else "FAIL",
        train_anchor_counts=train_anchor_counts,
        eval_anchor_counts=eval_anchor_counts,
    )


def write_pilot_manifest_outputs(
    dataset_root: str | Path,
    out_dir: str | Path,
    seed: int = 2026,
) -> PilotManifestSummary:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, eval_rows, summary = select_pilot_manifests(collect_train_candidates(dataset_root), seed=seed)
    _write_csv(out / "train512.csv", train)
    _write_csv(out / "eval64.csv", eval_rows)
    (out / "manifest_summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    lines = ["# Pilot1000 Manifest Audit", ""]
    for key, value in asdict(summary).items():
        lines.append(f"{key.upper()}={value}")
    (out / "manifest_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _select_positive_quota(
    positives: list[DatasetCandidate], quota: dict[int, int], rng: random.Random
) -> list[DatasetCandidate]:
    selected: list[DatasetCandidate] = []
    used: set[Path] = set()
    for class_id, target in quota.items():
        single = [item for item in positives if item.class_ids == (class_id,) and item.image_path not in used]
        multi = [item for item in positives if class_id in item.class_ids and item.image_path not in used and item not in single]
        rng.shuffle(single)
        rng.shuffle(multi)
        choices = [*single, *multi]
        if len(choices) < min(target, 32):
            raise ValueError(f"{CLASS_NAMES[class_id]} has fewer than 32 unique train images")
        for item in choices[:target]:
            if item.image_path not in used:
                selected.append(item)
                used.add(item.image_path)
    return selected


def _fill_remaining(
    pool: list[DatasetCandidate], selected: list[DatasetCandidate], target: int, rng: random.Random
) -> list[DatasetCandidate]:
    used = {item.image_path for item in selected}
    remaining = [item for item in pool if item.image_path not in used]
    rng.shuffle(remaining)
    needed = target - len(selected)
    if needed < 0:
        return []
    if len(remaining) < needed:
        raise ValueError("Not enough unique positive samples for Pilot1000 manifest")
    return remaining[:needed]


def _sample_unique(pool: list[DatasetCandidate], count: int, rng: random.Random) -> list[DatasetCandidate]:
    items = list(pool)
    rng.shuffle(items)
    if len(items) < count:
        raise ValueError(f"Need {count} unique samples, found {len(items)}")
    return items[:count]


def _row_dict(index: int, item: DatasetCandidate, seed: int, split_name: str) -> dict[str, str]:
    anchor = "" if item.is_negative else CLASS_NAMES[item.class_ids[0]]
    return {
        "sample_id": f"{split_name}_{index:04d}",
        "image_path": str(item.image_path),
        "label_path": str(item.label_path),
        "split": item.split,
        "is_negative": str(item.is_negative).lower(),
        "anchor_class": anchor,
        "class_ids": " ".join(str(class_id) for class_id in item.class_ids),
        "box_count": str(item.box_count),
        "image_sha256": _sha256_or_empty(item.image_path),
        "label_sha256": _sha256_or_empty(item.label_path),
        "selection_seed": str(seed),
    }


def _anchor_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counter = Counter(row["anchor_class"] for row in rows if row["anchor_class"])
    return {name: counter.get(name, 0) for name in CLASS_NAMES.values()}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_dir():
            return path
    raise FileNotFoundError("None of the expected Czech train directories exist")


def _image_files(path: Path) -> list[Path]:
    return [item for item in path.rglob("*") if item.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def _sha256_or_empty(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

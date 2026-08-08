"""Formal5000 full Czech train/val manifest and schedule construction."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from sd3_rgda.cache import read_yolo_boxes
from sd3_rgda.pilot_manifest import CLASS_NAMES, DatasetCandidate

FORMAL_MANIFEST_FIELDS = [
    "sample_id",
    "image_path",
    "label_path",
    "source_split",
    "is_negative",
    "anchor_class",
    "class_ids",
    "box_count",
    "image_sha256",
    "label_sha256",
    "selection_seed",
]

SCHEDULE_FIELDS = [
    "step",
    "pool_index",
    "source_sample_id",
    "is_negative",
    "anchor_class",
    "stratum",
    "cycle_index",
    "schedule_seed",
]


@dataclass(frozen=True)
class FormalManifestSummary:
    train_pool_rows: int
    train_pool_unique_images: int
    train_positive_pool: int
    train_negative_pool: int
    val_full_rows: int
    val_full_unique_images: int
    val_positive: int
    val_negative: int
    eval128_rows: int
    eval128_positive: int
    eval128_negative: int
    train_val_overlap: int
    train_test_overlap: int
    val_test_overlap: int
    eval_test_overlap: int
    test_leakage: int
    duplicate_image_sha256: int
    train_source_split_only: str
    class_minimum_gate: str
    train_anchor_pool_counts: dict[str, int]
    eval128_anchor_counts: dict[str, int]


@dataclass(frozen=True)
class FormalScheduleAudit:
    schedule_rows: int
    positive_steps: int
    negative_steps: int
    all_1980_train_images_used: str
    positive_stratum_usage_spread_le_1: str
    negative_usage_spread_le_1: str
    schedule_duplicate_step: int
    schedule_missing_step: int
    schedule_out_of_range: int
    schedule_audit: str


def collect_split_candidates(dataset_root: str | Path, split: str) -> list[DatasetCandidate]:
    root = Path(dataset_root)
    image_dir = _first_existing(root / "images" / split, root / split / "images", root / "images" / split.capitalize())
    label_dir = _first_existing(root / "labels" / split, root / split / "labels", root / "labels" / split.capitalize())
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
                split=split,
                class_ids=class_ids,
                box_count=len(boxes),
            )
        )
    return candidates


def build_formal_manifests(
    dataset_root: str | Path,
    seed: int = 2026,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], FormalManifestSummary]:
    train = collect_split_candidates(dataset_root, "train")
    val = collect_split_candidates(dataset_root, "val")
    test = collect_split_candidates(dataset_root, "test")
    train_rows = [_row_dict(index, item, seed, "formal_train", _assign_train_anchors(train)[index]) for index, item in enumerate(train)]
    val_anchors = _assign_train_anchors(val)
    val_rows = [_row_dict(index, item, seed, "formal_val", val_anchors[index]) for index, item in enumerate(val)]
    eval128 = select_eval128(val_rows, seed=seed)
    summary = summarize_formal(dataset_root, train_rows, val_rows, eval128, test)
    return train_rows, val_rows, eval128, summary


def select_eval128(val_rows: list[dict[str, str]], seed: int = 2026) -> list[dict[str, str]]:
    rng = random.Random(seed)
    positives = [row for row in val_rows if row["is_negative"] == "false"]
    negatives = [row for row in val_rows if row["is_negative"] == "true"]
    selected: list[dict[str, str]] = []
    used: set[str] = set()
    for cls in CLASS_NAMES.values():
        single = [row for row in positives if row["anchor_class"] == cls and len(row["class_ids"].split()) == 1]
        multi = [row for row in positives if row["anchor_class"] == cls and row not in single]
        rng.shuffle(single)
        rng.shuffle(multi)
        choices = [*single, *multi]
        if len(choices) < 16:
            raise ValueError(f"EVAL128_{cls}_AVAILABLE={len(choices)}")
        for row in choices:
            if row["image_path"] not in used:
                selected.append(row)
                used.add(row["image_path"])
            if sum(item["anchor_class"] == cls for item in selected) == 16:
                break
    rng.shuffle(negatives)
    for row in negatives:
        if row["image_path"] not in used:
            selected.append(row)
            used.add(row["image_path"])
        if sum(item["is_negative"] == "true" for item in selected) == 64:
            break
    if len(selected) != 128:
        raise ValueError(f"EVAL128_TOTAL={len(selected)}")
    rng.shuffle(selected)
    return [{**row, "sample_id": f"formal_eval128_{index:04d}"} for index, row in enumerate(selected)]


def build_schedule5000(train_rows: list[dict[str, str]], seed: int = 2026, steps: int = 5000) -> list[dict[str, str]]:
    positives = [(index, row) for index, row in enumerate(train_rows) if row["is_negative"] == "false"]
    negatives = [(index, row) for index, row in enumerate(train_rows) if row["is_negative"] == "true"]
    pos_targets = _positive_targets(positives, steps // 2)
    pos_sequence = _expand_by_anchor(positives, pos_targets, seed)
    neg_sequence = _cycle_items(negatives, steps // 2, seed + 17)
    start_positive = random.Random(seed).choice([True, False])
    pos_i = neg_i = 0
    rows: list[dict[str, str]] = []
    for step in range(1, steps + 1):
        use_positive = (step % 2 == 1) == start_positive
        pool_index, row = pos_sequence[pos_i] if use_positive else neg_sequence[neg_i]
        cycle_index = pos_i if use_positive else neg_i
        if use_positive:
            pos_i += 1
        else:
            neg_i += 1
        rows.append(
            {
                "step": str(step),
                "pool_index": str(pool_index),
                "source_sample_id": row["sample_id"],
                "is_negative": row["is_negative"],
                "anchor_class": row["anchor_class"],
                "stratum": row["anchor_class"] if use_positive else "NEG",
                "cycle_index": str(cycle_index),
                "schedule_seed": str(seed),
            }
        )
    return rows


def audit_schedule(train_rows: list[dict[str, str]], schedule: list[dict[str, str]], steps: int = 5000) -> FormalScheduleAudit:
    step_values = [int(row["step"]) for row in schedule]
    usage = Counter(int(row["pool_index"]) for row in schedule)
    positive_steps = sum(row["is_negative"] == "false" for row in schedule)
    negative_steps = sum(row["is_negative"] == "true" for row in schedule)
    pos_spreads: list[int] = []
    for cls in CLASS_NAMES.values():
        indices = [index for index, row in enumerate(train_rows) if row["is_negative"] == "false" and row["anchor_class"] == cls]
        values = [usage[index] for index in indices]
        if values:
            pos_spreads.append(max(values) - min(values))
    neg_indices = [index for index, row in enumerate(train_rows) if row["is_negative"] == "true"]
    neg_values = [usage[index] for index in neg_indices]
    duplicate = len(step_values) - len(set(step_values))
    missing = len(set(range(1, steps + 1)) - set(step_values))
    out_of_range = sum(step < 1 or step > steps for step in step_values)
    all_used = all(usage[index] > 0 for index in range(len(train_rows)))
    ok = (
        len(schedule) == steps
        and positive_steps == steps // 2
        and negative_steps == steps // 2
        and all_used
        and all(spread <= 1 for spread in pos_spreads)
        and (not neg_values or max(neg_values) - min(neg_values) <= 1)
        and duplicate == missing == out_of_range == 0
    )
    return FormalScheduleAudit(
        schedule_rows=len(schedule),
        positive_steps=positive_steps,
        negative_steps=negative_steps,
        all_1980_train_images_used="PASS" if all_used else "FAIL",
        positive_stratum_usage_spread_le_1="PASS" if all(spread <= 1 for spread in pos_spreads) else "FAIL",
        negative_usage_spread_le_1="PASS" if not neg_values or max(neg_values) - min(neg_values) <= 1 else "FAIL",
        schedule_duplicate_step=duplicate,
        schedule_missing_step=missing,
        schedule_out_of_range=out_of_range,
        schedule_audit="PASS" if ok else "FAIL",
    )


def write_formal_manifest_outputs(dataset_root: str | Path, out_dir: str | Path, seed: int = 2026) -> tuple[FormalManifestSummary, FormalScheduleAudit]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, eval_rows, summary = build_formal_manifests(dataset_root, seed)
    schedule = build_schedule5000(train_rows, seed)
    schedule_audit = audit_schedule(train_rows, schedule)
    _write_csv(out / "train_pool1980.csv", train_rows, FORMAL_MANIFEST_FIELDS)
    _write_csv(out / "val_full424.csv", val_rows, FORMAL_MANIFEST_FIELDS)
    _write_csv(out / "eval128.csv", eval_rows, FORMAL_MANIFEST_FIELDS)
    _write_csv(out / "schedule5000.csv", schedule, SCHEDULE_FIELDS)
    (out / "manifest_summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    (out / "schedule_audit.json").write_text(json.dumps(asdict(schedule_audit), indent=2) + "\n", encoding="utf-8")
    lines = ["# Formal5000 Manifest Audit", ""]
    for key, value in asdict(summary).items():
        lines.append(f"{key.upper()}={value}")
    for key, value in asdict(schedule_audit).items():
        lines.append(f"{key.upper()}={value}")
    (out / "manifest_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, schedule_audit


def summarize_formal(
    dataset_root: str | Path,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    test_candidates: list[DatasetCandidate],
) -> FormalManifestSummary:
    train_images = {row["image_path"] for row in train_rows}
    val_images = {row["image_path"] for row in val_rows}
    eval_images = {row["image_path"] for row in eval_rows}
    test_images = {str(item.image_path.resolve()) for item in test_candidates}
    image_shas = [row["image_sha256"] for row in train_rows]
    train_test_overlap = len(train_images & test_images)
    val_test_overlap = len(val_images & test_images)
    eval_test_overlap = len(eval_images & test_images)
    return FormalManifestSummary(
        train_pool_rows=len(train_rows),
        train_pool_unique_images=len(train_images),
        train_positive_pool=sum(row["is_negative"] == "false" for row in train_rows),
        train_negative_pool=sum(row["is_negative"] == "true" for row in train_rows),
        val_full_rows=len(val_rows),
        val_full_unique_images=len(val_images),
        val_positive=sum(row["is_negative"] == "false" for row in val_rows),
        val_negative=sum(row["is_negative"] == "true" for row in val_rows),
        eval128_rows=len(eval_rows),
        eval128_positive=sum(row["is_negative"] == "false" for row in eval_rows),
        eval128_negative=sum(row["is_negative"] == "true" for row in eval_rows),
        train_val_overlap=len(train_images & val_images),
        train_test_overlap=train_test_overlap,
        val_test_overlap=val_test_overlap,
        eval_test_overlap=eval_test_overlap,
        test_leakage=train_test_overlap + val_test_overlap + eval_test_overlap,
        duplicate_image_sha256=len(image_shas) - len(set(image_shas)),
        train_source_split_only="PASS" if all(row["source_split"] == "train" for row in train_rows) else "FAIL",
        class_minimum_gate="PASS" if len(eval_rows) == 128 and all(_anchor_counts(eval_rows).get(cls, 0) == 16 for cls in CLASS_NAMES.values()) else "FAIL",
        train_anchor_pool_counts=_anchor_counts(train_rows),
        eval128_anchor_counts=_anchor_counts(eval_rows),
    )


def _assign_train_anchors(candidates: list[DatasetCandidate]) -> dict[int, int | None]:
    counts: Counter[int] = Counter()
    anchors: dict[int, int | None] = {}
    for index, item in enumerate(candidates):
        if item.is_negative:
            anchors[index] = None
            continue
        if len(item.class_ids) == 1:
            anchor = item.class_ids[0]
        else:
            anchor = min(item.class_ids, key=lambda class_id: (counts[class_id], class_id))
        anchors[index] = anchor
        counts[anchor] += 1
    return anchors


def _row_dict(index: int, item: DatasetCandidate, seed: int, prefix: str, anchor_class: int | None) -> dict[str, str]:
    return {
        "sample_id": f"{prefix}_{index:04d}",
        "image_path": str(item.image_path),
        "label_path": str(item.label_path),
        "source_split": item.split,
        "is_negative": str(item.is_negative).lower(),
        "anchor_class": "" if anchor_class is None else CLASS_NAMES[anchor_class],
        "class_ids": " ".join(str(class_id) for class_id in item.class_ids),
        "box_count": str(item.box_count),
        "image_sha256": _sha256_or_empty(item.image_path),
        "label_sha256": _sha256_or_empty(item.label_path),
        "selection_seed": str(seed),
    }


def _positive_targets(positives: list[tuple[int, dict[str, str]]], target_steps: int) -> dict[str, int]:
    counts = Counter(row["anchor_class"] for _index, row in positives)
    targets = {cls: counts.get(cls, 0) for cls in CLASS_NAMES.values()}
    remaining = target_steps - sum(targets.values())
    order = list(CLASS_NAMES.values())
    while remaining > 0:
        cls = min(order, key=lambda name: (targets[name], order.index(name)))
        targets[cls] += 1
        remaining -= 1
    return targets


def _expand_by_anchor(
    positives: list[tuple[int, dict[str, str]]], targets: dict[str, int], seed: int
) -> list[tuple[int, dict[str, str]]]:
    by_anchor: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for item in positives:
        by_anchor[item[1]["anchor_class"]].append(item)
    expanded: list[tuple[int, dict[str, str]]] = []
    for cls in CLASS_NAMES.values():
        expanded.extend(_cycle_items(by_anchor[cls], targets[cls], seed + list(CLASS_NAMES.values()).index(cls)))
    random.Random(seed + 101).shuffle(expanded)
    return expanded


def _cycle_items(items: list[tuple[int, dict[str, str]]], count: int, seed: int) -> list[tuple[int, dict[str, str]]]:
    if not items and count:
        raise ValueError("Cannot schedule an empty stratum")
    rng = random.Random(seed)
    output: list[tuple[int, dict[str, str]]] = []
    cycle = 0
    while len(output) < count:
        current = list(items)
        rng.seed(seed + cycle)
        rng.shuffle(current)
        output.extend(current)
        cycle += 1
    return output[:count]


def _anchor_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counter = Counter(row["anchor_class"] for row in rows if row["anchor_class"])
    return {name: counter.get(name, 0) for name in CLASS_NAMES.values()}


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_dir():
            return path
    raise FileNotFoundError("None of the expected Czech split directories exist")


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

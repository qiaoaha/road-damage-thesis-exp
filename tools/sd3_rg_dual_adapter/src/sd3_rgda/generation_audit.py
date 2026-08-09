"""Generation1000 QA and YOLO ablation dataset construction."""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from sd3_rgda.generation_manifest import read_generation_manifest


@dataclass(frozen=True)
class GenerationAudit:
    generated_rows: int
    generated_success: int
    generated_failure: int
    unique_output_files: int
    unique_output_sha: int
    corrupt_images: int
    shape_mismatch: int
    rgb_image_gate: str
    source_label_sha_match: str
    negative_label_gate: str
    val_leakage: int
    test_leakage: int
    checkpoint_sha_match: str
    seed_pair_match: str
    base_rgda_source_pair_match: str
    base_rgda_seed_pair_match: str
    base_rgda_prompt_pair_match: str
    audit_gate: str


@dataclass(frozen=True)
class YoloDatasetAudit:
    group: str
    train_images: int
    val_images: int
    test_images: int
    synthetic_train_images: int
    synthetic_val_test_leakage: int
    real_synthetic_filename_collision: int
    dataset_gate: str


def audit_generation_results(
    generation_manifest: str | Path,
    results_csv: str | Path,
    *,
    expected_checkpoint_sha256: str,
    val_source_shas: set[str] | None = None,
    test_source_shas: set[str] | None = None,
) -> GenerationAudit:
    rows = read_generation_manifest(generation_manifest)
    results = _read_results(results_csv)
    corrupt = 0
    shape = 0
    label_matches = 0
    negative_ok = 0
    output_files: list[str] = []
    output_shas: list[str] = []
    source_by_mode: dict[tuple[str, str], dict[str, str]] = {}
    for result in results:
        output_files.append(result["output_image_path"])
        output_shas.append(result["output_image_sha256"])
        try:
            with Image.open(result["output_image_path"]) as image:
                if image.mode != "RGB":
                    corrupt += 1
                if image.size != (int(result["width"]), int(result["height"])):
                    shape += 1
        except OSError:
            corrupt += 1
        if result["source_label_sha256"] == result["output_label_sha256"]:
            label_matches += 1
        manifest_row = rows[int(result["generation_index"])]
        if manifest_row["is_negative"] == "true" and result["source_label_sha256"] == result["output_label_sha256"]:
            negative_ok += 1
        source_by_mode[(result["mode"], result["generation_index"])] = result
    base_pairs, seed_pairs, prompt_pairs = _pair_counts(source_by_mode)
    val = val_source_shas or set()
    test = test_source_shas or set()
    val_leakage = sum(row["source_image_sha256"] in val for row in rows)
    test_leakage = sum(row["source_image_sha256"] in test for row in rows)
    success = sum(result["status"] == "PASS" for result in results)
    failures = len(results) - success
    audit = GenerationAudit(
        generated_rows=len(results),
        generated_success=success,
        generated_failure=failures,
        unique_output_files=len(set(output_files)),
        unique_output_sha=len(set(output_shas)),
        corrupt_images=corrupt,
        shape_mismatch=shape,
        rgb_image_gate="PASS" if corrupt == 0 else "FAIL",
        source_label_sha_match=f"{label_matches}/{len(results)}",
        negative_label_gate="PASS" if negative_ok == sum(row["is_negative"] == "true" for row in rows) else "FAIL",
        val_leakage=val_leakage,
        test_leakage=test_leakage,
        checkpoint_sha_match="PASS" if all(r["rgda_checkpoint_sha256"] == expected_checkpoint_sha256 for r in results) else "FAIL",
        seed_pair_match="PASS" if seed_pairs == 1000 else "FAIL",
        base_rgda_source_pair_match=f"{base_pairs}/1000",
        base_rgda_seed_pair_match=f"{seed_pairs}/1000",
        base_rgda_prompt_pair_match=f"{prompt_pairs}/1000",
        audit_gate="PASS"
        if len(results) == 1000
        and success == 1000
        and failures == 0
        and corrupt == 0
        and shape == 0
        and val_leakage == 0
        and test_leakage == 0
        else "FAIL",
    )
    return audit


def write_generation_audit(path: str | Path, audit: GenerationAudit) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Generation1000 QA", ""]
    for key, value in asdict(audit).items():
        lines.append(f"{key.upper()}={value}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (p.with_suffix(".json")).write_text(json.dumps(asdict(audit), indent=2) + "\n", encoding="utf-8")


def build_yolo_ablation_dataset(
    real_dataset_root: str | Path,
    output_root: str | Path,
    *,
    group: str,
    synthetic_images: str | Path | None = None,
    synthetic_labels: str | Path | None = None,
    link_mode: str = "hardlink",
) -> YoloDatasetAudit:
    if group not in {"real", "sd3", "rgda"}:
        raise ValueError("group must be real, sd3, or rgda")
    real = Path(real_dataset_root)
    out = Path(output_root) / group
    _prepare_yolo_dirs(out)
    real_train = _link_split(real, out, "train", prefix="", link_mode=link_mode)
    val_count = _link_split(real, out, "val", prefix="", link_mode=link_mode)
    test_count = _link_split(real, out, "test", prefix="", link_mode=link_mode)
    synthetic_count = 0
    collisions = 0
    if group != "real":
        if synthetic_images is None or synthetic_labels is None:
            raise ValueError("synthetic groups require synthetic image and label dirs")
        prefix = "sd3base_" if group == "sd3" else "sd3rgda_"
        synthetic_count, collisions = _link_synthetic(
            Path(synthetic_images), Path(synthetic_labels), out, prefix=prefix, link_mode=link_mode
        )
    _write_dataset_yaml(out)
    leakage = _count_synthetic(out / "images" / "val") + _count_synthetic(out / "images" / "test")
    audit = YoloDatasetAudit(
        group=group,
        train_images=real_train + synthetic_count,
        val_images=val_count,
        test_images=test_count,
        synthetic_train_images=synthetic_count,
        synthetic_val_test_leakage=leakage,
        real_synthetic_filename_collision=collisions,
        dataset_gate="PASS" if leakage == 0 and collisions == 0 else "FAIL",
    )
    (out / "dataset_audit.json").write_text(json.dumps(asdict(audit), indent=2) + "\n", encoding="utf-8")
    return audit


def _read_results(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pair_counts(results: dict[tuple[str, str], dict[str, str]]) -> tuple[int, int, int]:
    source = seed = prompt = 0
    for index in {key[1] for key in results}:
        base = results.get(("base", index))
        rgda = results.get(("rgda", index))
        if not base or not rgda:
            continue
        source += base["source_sample_id"] == rgda["source_sample_id"]
        seed += base["seed"] == rgda["seed"]
        prompt += base["prompt"] == rgda["prompt"]
    return source, seed, prompt


def _prepare_yolo_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _link_split(real: Path, out: Path, split: str, *, prefix: str, link_mode: str) -> int:
    count = 0
    for image in sorted((real / "images" / split).glob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = real / "labels" / split / f"{image.stem}.txt"
        _link_or_copy(image, out / "images" / split / f"{prefix}{image.name}", link_mode)
        _link_or_copy(label, out / "labels" / split / f"{prefix}{label.name}", link_mode)
        count += 1
    return count


def _link_synthetic(images: Path, labels: Path, out: Path, *, prefix: str, link_mode: str) -> tuple[int, int]:
    count = 0
    collisions = 0
    existing = {p.name for p in (out / "images" / "train").glob("*")}
    for image in sorted(images.glob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = labels / f"{image.stem}.txt"
        image_name = f"{prefix}{image.name}"
        label_name = f"{prefix}{label.name}"
        collisions += image_name in existing
        _link_or_copy(image, out / "images" / "train" / image_name, link_mode)
        _link_or_copy(label, out / "labels" / "train" / label_name, link_mode)
        count += 1
    return count, collisions


def _link_or_copy(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if link_mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _count_synthetic(path: Path) -> int:
    return sum(p.name.startswith(("sd3base_", "sd3rgda_")) for p in path.glob("*"))


def _write_dataset_yaml(out: Path) -> None:
    text = "\n".join(
        [
            f"path: {out.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: D00",
            "  1: D10",
            "  2: D20",
            "  3: D40",
            "",
        ]
    )
    (out / "data.yaml").write_text(text, encoding="utf-8")

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
    result_rows: int
    base_rows: int
    rgda_rows: int
    base_success: int
    rgda_success: int
    failure_rows: int
    unique_output_files: int
    unique_output_sha: int
    corrupt_images: int
    shape_mismatch: int
    rgb_image_gate: str
    source_label_sha_match: str
    negative_label_gate: str
    val_leakage: int
    test_leakage: int
    rgda_checkpoint_sha_match: str
    base_checkpoint_field_none: str
    base_rgda_source_pair_match: str
    base_rgda_seed_pair_match: str
    base_rgda_prompt_pair_match: str
    base_rgda_generation_param_match: str
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
    image_count_label_count_match: str
    missing_labels: int
    orphan_labels: int
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
    result_by_key = {(result["mode"], result["generation_index"]): result for result in results}
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
    source_pairs, seed_pairs, prompt_pairs, param_pairs = _pair_counts(source_by_mode)
    val = val_source_shas or set()
    test = test_source_shas or set()
    val_leakage = sum(row["source_image_sha256"] in val for row in rows)
    test_leakage = sum(row["source_image_sha256"] in test for row in rows)
    success = sum(result["status"] == "PASS" for result in results)
    failures = len(results) - success
    base_rows = sum(result["mode"] == "base" for result in results)
    rgda_rows = sum(result["mode"] == "rgda" for result in results)
    base_success = sum(result["mode"] == "base" and result["status"] == "PASS" for result in results)
    rgda_success = sum(result["mode"] == "rgda" and result["status"] == "PASS" for result in results)
    base_none = sum(result["mode"] == "base" and result["rgda_checkpoint_sha256"] == "NONE" for result in results)
    rgda_sha = sum(
        result["mode"] == "rgda" and result["rgda_checkpoint_sha256"] == expected_checkpoint_sha256 for result in results
    )
    complete_pairs = all(("base", row["generation_index"]) in result_by_key and ("rgda", row["generation_index"]) in result_by_key for row in rows)
    negative_expected = sum(row["is_negative"] == "true" for row in rows) * 2
    audit = GenerationAudit(
        result_rows=len(results),
        base_rows=base_rows,
        rgda_rows=rgda_rows,
        base_success=base_success,
        rgda_success=rgda_success,
        failure_rows=failures,
        unique_output_files=len(set(output_files)),
        unique_output_sha=len(set(output_shas)),
        corrupt_images=corrupt,
        shape_mismatch=shape,
        rgb_image_gate="PASS" if corrupt == 0 else "FAIL",
        source_label_sha_match=f"{label_matches}/{len(results)}",
        negative_label_gate="PASS" if negative_ok == negative_expected else "FAIL",
        val_leakage=val_leakage,
        test_leakage=test_leakage,
        rgda_checkpoint_sha_match=f"{rgda_sha}/1000",
        base_checkpoint_field_none=f"{base_none}/1000",
        base_rgda_source_pair_match=f"{source_pairs}/1000",
        base_rgda_seed_pair_match=f"{seed_pairs}/1000",
        base_rgda_prompt_pair_match=f"{prompt_pairs}/1000",
        base_rgda_generation_param_match=f"{param_pairs}/1000",
        audit_gate="PASS"
        if len(rows) == 1000
        and len(results) == 2000
        and base_rows == 1000
        and rgda_rows == 1000
        and base_success == 1000
        and rgda_success == 1000
        and failures == 0
        and corrupt == 0
        and shape == 0
        and val_leakage == 0
        and test_leakage == 0
        and source_pairs == 1000
        and seed_pairs == 1000
        and prompt_pairs == 1000
        and param_pairs == 1000
        and rgda_sha == 1000
        and base_none == 1000
        and complete_pairs
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
    expected_real_train: int = 1980,
    expected_val: int = 424,
    expected_test: int = 425,
    expected_synthetic: int = 1000,
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
    missing_labels, orphan_labels = _label_problems(out)
    count_gate = (
        real_train == expected_real_train
        and val_count == expected_val
        and test_count == expected_test
        and (synthetic_count == 0 if group == "real" else synthetic_count == expected_synthetic)
    )
    audit = YoloDatasetAudit(
        group=group,
        train_images=real_train + synthetic_count,
        val_images=val_count,
        test_images=test_count,
        synthetic_train_images=synthetic_count,
        synthetic_val_test_leakage=leakage,
        real_synthetic_filename_collision=collisions,
        image_count_label_count_match="PASS" if missing_labels == 0 and orphan_labels == 0 else "FAIL",
        missing_labels=missing_labels,
        orphan_labels=orphan_labels,
        dataset_gate="PASS" if leakage == 0 and collisions == 0 and count_gate and missing_labels == 0 and orphan_labels == 0 else "FAIL",
    )
    (out / "dataset_audit.json").write_text(json.dumps(asdict(audit), indent=2) + "\n", encoding="utf-8")
    return audit


def _read_results(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pair_counts(results: dict[tuple[str, str], dict[str, str]]) -> tuple[int, int, int, int]:
    source = seed = prompt = params = 0
    for index in {key[1] for key in results}:
        base = results.get(("base", index))
        rgda = results.get(("rgda", index))
        if not base or not rgda:
            continue
        source += base["source_sample_id"] == rgda["source_sample_id"]
        seed += base["seed"] == rgda["seed"]
        prompt += base["prompt"] == rgda["prompt"]
        params += all(
            base[key] == rgda[key]
            for key in ["width", "height", "inference_steps", "guidance_scale", "scheduler", "source_image_sha256"]
        )
    return source, seed, prompt, params


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
        image_name = image.name if image.name.startswith(prefix) else f"{prefix}{image.name}"
        label_name = label.name if label.name.startswith(prefix) else f"{prefix}{label.name}"
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


def _label_problems(root: Path) -> tuple[int, int]:
    missing = orphan = 0
    for split in ("train", "val", "test"):
        images = {p.stem for p in (root / "images" / split).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
        missing += len(images - labels)
        orphan += len(labels - images)
    return missing, orphan


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

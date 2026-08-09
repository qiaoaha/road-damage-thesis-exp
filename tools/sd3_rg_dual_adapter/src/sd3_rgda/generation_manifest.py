"""Generation1000 source selection, deterministic prompts, and paired seeds."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from sd3_rgda.formal_manifest import collect_split_candidates
from sd3_rgda.pilot_manifest import CLASS_NAMES

GENERATION_FIELDS = [
    "generation_index",
    "source_sample_id",
    "source_image_path",
    "source_label_path",
    "source_image_sha256",
    "source_label_sha256",
    "source_split",
    "is_negative",
    "anchor_class",
    "class_ids",
    "prompt",
    "seed",
    "width",
    "height",
    "num_inference_steps",
    "guidance_scale",
    "scheduler",
    "base_output_filename",
    "rgda_output_filename",
    "rgda_checkpoint_sha256",
]

PROMPTS = {
    "NEG": "a clean Czech road surface without visible road damage",
    "D00": "a Czech road surface with longitudinal crack damage",
    "D10": "a Czech road surface with transverse crack damage",
    "D20": "a Czech road surface with alligator crack damage",
    "D40": "a Czech road surface with pothole damage",
}

CLASS_PHRASES = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack",
    "D40": "pothole",
}


@dataclass(frozen=True)
class Generation1000Summary:
    total: int
    unique_source_images: int
    positive: int
    negative: int
    val_leakage: int
    test_leakage: int
    source_image_sha_duplicate: int
    old_baseline_source_match: str
    seed_min: int
    seed_max: int
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    scheduler: str
    checkpoint_sha256: str
    manifest_gate: str


def derive_generation_seed(global_seed: int, generation_index: int) -> int:
    if generation_index < 0:
        raise ValueError("generation_index must be non-negative")
    return global_seed * 100000 + generation_index


def prompt_for_sample(is_negative: bool, class_ids: str, anchor_class: str) -> str:
    if is_negative:
        return PROMPTS["NEG"]
    classes = [CLASS_NAMES[int(item)] for item in class_ids.split() if item]
    if not classes and anchor_class:
        classes = [anchor_class]
    if len(classes) == 1:
        return PROMPTS[classes[0]]
    phrases = [CLASS_PHRASES[name] for name in sorted(set(classes))]
    return f"a Czech road surface with {' and '.join(phrases)} damage"


def build_generation1000_rows(
    dataset_root: str | Path,
    checkpoint_sha256: str,
    *,
    global_seed: int = 2026,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    scheduler: str = "FlowMatchEulerDiscreteScheduler",
    old_baseline_manifest: str | Path | None = None,
) -> tuple[list[dict[str, str]], Generation1000Summary]:
    train_rows = _train_rows(dataset_root, global_seed)
    positives = [row for row in train_rows if row["is_negative"] == "false"]
    negatives = [row for row in train_rows if row["is_negative"] == "true"]
    if len(positives) != 750 or len(negatives) != 1230:
        raise ValueError(f"Expected Czech train positive=750 negative=1230, got {len(positives)} {len(negatives)}")
    chosen = [*positives, *_sample_negatives(negatives, seed=global_seed, count=250)]
    rows: list[dict[str, str]] = []
    old_match = "NOT_PROVIDED"
    old_sources = _old_source_shas(old_baseline_manifest)
    if old_sources is not None:
        match_count = sum(row["image_sha256"] in old_sources for row in chosen)
        old_match = f"{match_count}/1000"
    for index, row in enumerate(chosen):
        seed = derive_generation_seed(global_seed, index)
        base_name = f"sd3base_{index:04d}_{row['sample_id']}.png"
        rgda_name = f"sd3rgda_{index:04d}_{row['sample_id']}.png"
        rows.append(
            {
                "generation_index": str(index),
                "source_sample_id": row["sample_id"],
                "source_image_path": row["image_path"],
                "source_label_path": row["label_path"],
                "source_image_sha256": row["image_sha256"],
                "source_label_sha256": row["label_sha256"],
                "source_split": row["source_split"],
                "is_negative": row["is_negative"],
                "anchor_class": row["anchor_class"],
                "class_ids": row["class_ids"],
                "prompt": prompt_for_sample(row["is_negative"] == "true", row["class_ids"], row["anchor_class"]),
                "seed": str(seed),
                "width": str(width),
                "height": str(height),
                "num_inference_steps": str(num_inference_steps),
                "guidance_scale": str(guidance_scale),
                "scheduler": scheduler,
                "base_output_filename": base_name,
                "rgda_output_filename": rgda_name,
                "rgda_checkpoint_sha256": checkpoint_sha256,
            }
        )
    val_shas = {_sha_candidate(item.image_path) for item in collect_split_candidates(dataset_root, "val")}
    test_shas = {_sha_candidate(item.image_path) for item in collect_split_candidates(dataset_root, "test")}
    shas = [row["source_image_sha256"] for row in rows]
    summary = Generation1000Summary(
        total=len(rows),
        unique_source_images=len(set(shas)),
        positive=sum(row["is_negative"] == "false" for row in rows),
        negative=sum(row["is_negative"] == "true" for row in rows),
        val_leakage=sum(sha in val_shas for sha in shas),
        test_leakage=sum(sha in test_shas for sha in shas),
        source_image_sha_duplicate=len(shas) - len(set(shas)),
        old_baseline_source_match=old_match,
        seed_min=min(int(row["seed"]) for row in rows),
        seed_max=max(int(row["seed"]) for row in rows),
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        scheduler=scheduler,
        checkpoint_sha256=checkpoint_sha256,
        manifest_gate="PASS" if _manifest_ok(rows, val_shas, test_shas) else "FAIL",
    )
    return rows, summary


def write_generation1000_outputs(
    dataset_root: str | Path,
    output_dir: str | Path,
    checkpoint_sha256: str,
    global_seed: int = 2026,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    scheduler: str = "FlowMatchEulerDiscreteScheduler",
    old_baseline_manifest: str | Path | None = None,
) -> Generation1000Summary:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = build_generation1000_rows(
        dataset_root,
        checkpoint_sha256,
        global_seed=global_seed,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        scheduler=scheduler,
        old_baseline_manifest=old_baseline_manifest,
    )
    write_generation_manifest(out / "generation1000.csv", rows)
    (out / "generation1000_summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    lines = ["# Generation1000 Manifest Audit", ""]
    for key, value in asdict(summary).items():
        lines.append(f"{key.upper()}={value}")
    (out / "generation1000_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_generation_manifest(path: str | Path, rows: list[dict[str, str]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_generation_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _train_rows(dataset_root: str | Path, seed: int) -> list[dict[str, str]]:
    from sd3_rgda.formal_manifest import _assign_train_anchors, _row_dict

    train = collect_split_candidates(dataset_root, "train")
    anchors = _assign_train_anchors(train)
    return [_row_dict(index, item, seed, "formal_train", anchors[index]) for index, item in enumerate(train)]


def _sample_negatives(rows: list[dict[str, str]], *, seed: int, count: int) -> list[dict[str, str]]:
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:count]


def _old_source_shas(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row.get("source_image_sha256", row.get("image_sha256", "")) for row in reader}


def _manifest_ok(rows: list[dict[str, str]], val_shas: set[str], test_shas: set[str]) -> bool:
    shas = [row["source_image_sha256"] for row in rows]
    seeds = [int(row["seed"]) for row in rows]
    return (
        len(rows) == 1000
        and len(set(shas)) == 1000
        and sum(row["is_negative"] == "false" for row in rows) == 750
        and sum(row["is_negative"] == "true" for row in rows) == 250
        and not set(shas) & val_shas
        and not set(shas) & test_shas
        and len(seeds) == len(set(seeds))
    )


def _sha_candidate(path: Path) -> str:
    return sha256_file(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

#!/usr/bin/env python3
"""Static no-GPU integration over fake paired outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from sd3_rgda.generation_audit import build_yolo_ablation_dataset
from sd3_rgda.generation_manifest import sha256_file, write_generation_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("tmp/generation_dry_integration"))
    args = parser.parse_args()
    root = args.output_root
    real = root / "real"
    manifest = root / "generation1000.csv"
    results = root / "generation_results.csv"
    _make_real(real)
    rows = _make_manifest(real)
    write_generation_manifest(manifest, rows)
    _make_results(root, rows, results)
    build_yolo_ablation_dataset(real, root / "yolo", group="real", link_mode="copy")
    build_yolo_ablation_dataset(
        real,
        root / "yolo",
        group="sd3",
        synthetic_images=root / "base" / "images",
        synthetic_labels=root / "base" / "labels",
        link_mode="copy",
    )
    build_yolo_ablation_dataset(
        real,
        root / "yolo",
        group="rgda",
        synthetic_images=root / "rgda" / "images",
        synthetic_labels=root / "rgda" / "labels",
        link_mode="copy",
    )
    print("GENERATION_DRY_INTEGRATION=PASS")
    print("REAL_SD3_USED=NO")
    print("GPU_USED=NO")
    return 0


def _make_real(root: Path) -> None:
    for split, count in [("train", 8), ("val", 4), ("test", 4)]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = root / "images" / split / f"{split}_{index}.png"
            label = root / "labels" / split / f"{split}_{index}.txt"
            Image.new("RGB", (16, 16), (index, 1, 2)).save(image)
            label.write_text("" if index >= 4 else f"{index % 4} 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def _make_manifest(real: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(8):
        image = real / "images" / "train" / f"train_{index}.png"
        label = real / "labels" / "train" / f"train_{index}.txt"
        rows.append(
            {
                "generation_index": str(index),
                "source_sample_id": f"dry_{index}",
                "source_image_path": str(image),
                "source_label_path": str(label),
                "source_image_sha256": sha256_file(image),
                "source_label_sha256": sha256_file(label),
                "source_split": "train",
                "is_negative": str(index >= 4).lower(),
                "anchor_class": "" if index >= 4 else ["D00", "D10", "D20", "D40"][index],
                "class_ids": "" if index >= 4 else str(index),
                "prompt": "dry prompt",
                "seed": str(202600000 + index),
                "width": "16",
                "height": "16",
                "num_inference_steps": "1",
                "guidance_scale": "1.0",
                "scheduler": "FakeScheduler",
                "base_output_filename": f"sd3base_{index}.png",
                "rgda_output_filename": f"sd3rgda_{index}.png",
                "rgda_checkpoint_sha256": "dry",
            }
        )
    return rows


def _make_results(root: Path, rows: list[dict[str, str]], results: Path) -> None:
    fields = [
        "generation_index",
        "mode",
        "source_sample_id",
        "seed",
        "prompt",
        "source_image_sha256",
        "source_label_sha256",
        "rgda_checkpoint_sha256",
        "output_image_path",
        "output_image_sha256",
        "output_label_path",
        "output_label_sha256",
        "width",
        "height",
        "inference_steps",
        "guidance_scale",
        "scheduler",
        "generation_seconds",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "status",
        "error_type",
        "error_message",
    ]
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for mode, directory, name_key in [
                ("base", "base", "base_output_filename"),
                ("rgda", "rgda", "rgda_output_filename"),
            ]:
                image = root / directory / "images" / row[name_key]
                label = root / directory / "labels" / f"{Path(row[name_key]).stem}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (16, 16), (int(row["generation_index"]), 2, 3)).save(image)
                Path(row["source_label_path"]).replace(label) if False else label.write_bytes(Path(row["source_label_path"]).read_bytes())
                writer.writerow(
                    {
                        "generation_index": row["generation_index"],
                        "mode": mode,
                        "source_sample_id": row["source_sample_id"],
                        "seed": row["seed"],
                        "prompt": row["prompt"],
                        "source_image_sha256": row["source_image_sha256"],
                        "source_label_sha256": row["source_label_sha256"],
                        "rgda_checkpoint_sha256": row["rgda_checkpoint_sha256"],
                        "output_image_path": image,
                        "output_image_sha256": sha256_file(image),
                        "output_label_path": label,
                        "output_label_sha256": sha256_file(label),
                        "width": row["width"],
                        "height": row["height"],
                        "inference_steps": row["num_inference_steps"],
                        "guidance_scale": row["guidance_scale"],
                        "scheduler": row["scheduler"],
                        "generation_seconds": "0",
                        "peak_allocated_mib": "0",
                        "peak_reserved_mib": "0",
                        "status": "PASS",
                        "error_type": "",
                        "error_message": "",
                    }
                )


if __name__ == "__main__":
    raise SystemExit(main())

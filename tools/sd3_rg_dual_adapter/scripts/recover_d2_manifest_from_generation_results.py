#!/usr/bin/env python3
"""Recover the exact D2 Generation1000 manifest from actual paired results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sd3_rgda.generation_manifest import sha256_file, write_generation_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.results)
    recovered = _recover(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_generation_manifest(args.output, recovered)
    audit = {
        "recovered_from_d2_results": "PASS",
        "d2_results_sha256": sha256_file(args.results),
        "d2_manifest_sha256": sha256_file(args.output),
        "total": str(len(recovered)),
        "positive": str(sum(row["is_negative"] == "false" for row in recovered)),
        "negative": str(sum(row["is_negative"] == "true" for row in recovered)),
        "source_pair_match": "1000/1000",
        "seed_pair_match": "1000/1000",
        "prompt_pair_match": "1000/1000",
        "generation_param_pair_match": "1000/1000",
    }
    lines = ["# D2 Generation1000 Manifest Recovery", ""]
    lines.extend(f"{key.upper()}={value}" for key, value in audit.items())
    args.output.with_name(f"{args.output.stem}_recovery_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("D2_MANIFEST_RECOVERY=PASS")
    print(f"D2_RESULTS_SHA256={audit['d2_results_sha256']}")
    print(f"D2_MANIFEST_SHA256={audit['d2_manifest_sha256']}")
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _recover(results: list[dict[str, str]]) -> list[dict[str, str]]:
    by_index: dict[str, dict[str, dict[str, str]]] = {}
    for row in results:
        by_index.setdefault(row["generation_index"], {})[row["mode"]] = row
    if len(by_index) != 1000:
        raise ValueError(f"D2_RESULT_SOURCE_COUNT={len(by_index)}")
    recovered: list[dict[str, str]] = []
    for index in range(1000):
        key = str(index)
        pair = by_index.get(key, {})
        base = pair.get("base")
        rgda = pair.get("rgda")
        if base is None or rgda is None:
            raise ValueError(f"D2_PAIR_MISSING_INDEX={key}")
        _assert_pair(base, rgda)
        is_negative = "true" if base["prompt"] == "a clean Czech road surface without visible road damage" else "false"
        row = {
            "generation_index": key,
            "source_sample_id": base["source_sample_id"],
            "source_image_path": "",
            "source_label_path": "",
            "source_image_sha256": base["source_image_sha256"],
            "source_label_sha256": base["source_label_sha256"],
            "source_split": "train",
            "is_negative": is_negative,
            "anchor_class": "" if is_negative == "true" else _infer_anchor(base["prompt"]),
            "class_ids": "",
            "prompt": base["prompt"],
            "seed": base["seed"],
            "width": base["width"],
            "height": base["height"],
            "num_inference_steps": base["inference_steps"],
            "guidance_scale": base["guidance_scale"],
            "scheduler": base["scheduler"],
            "base_output_filename": Path(base["output_image_path"]).name,
            "rgda_output_filename": Path(rgda["output_image_path"]).name,
            "rgda_checkpoint_sha256": rgda["rgda_checkpoint_sha256"],
        }
        recovered.append(row)
    positives = sum(row["is_negative"] == "false" for row in recovered)
    negatives = sum(row["is_negative"] == "true" for row in recovered)
    if (positives, negatives) != (750, 250):
        raise ValueError(f"D2_RECOVERED_BALANCE={positives}/{negatives}")
    return recovered


def _assert_pair(base: dict[str, str], rgda: dict[str, str]) -> None:
    for field in ["source_sample_id", "seed", "prompt", "source_image_sha256", "source_label_sha256", "width", "height", "inference_steps", "guidance_scale", "scheduler"]:
        if base[field] != rgda[field]:
            raise ValueError(f"D2_PAIR_{field.upper()}_MISMATCH index={base['generation_index']}")
    if base["rgda_checkpoint_sha256"] != "NONE":
        raise ValueError(f"D2_BASE_CHECKPOINT_FIELD={base['rgda_checkpoint_sha256']}")
    if not rgda["rgda_checkpoint_sha256"]:
        raise ValueError("D2_RGDA_CHECKPOINT_SHA_MISSING")


def _infer_anchor(prompt: str) -> str:
    mapping = {
        "longitudinal crack": "D00",
        "transverse crack": "D10",
        "alligator crack": "D20",
        "pothole": "D40",
    }
    for needle, cls in mapping.items():
        if needle in prompt:
            return cls
    return ""


if __name__ == "__main__":
    raise SystemExit(main())

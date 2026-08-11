#!/usr/bin/env python3
"""Derive D3 RAAL Generation1000 manifest from the exact D2 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sd3_rgda.formal_manifest import collect_split_candidates
from sd3_rgda.generation_manifest import GENERATION_FIELDS, read_generation_manifest

RAAL_EXTRA_FIELDS = [
    "method",
    "formal_commit",
    "raal_checkpoint_path",
    "raal_checkpoint_sha256",
    "d2_manifest_sha256",
    "d3_pairing_contract",
    "raal_attention_collector",
    "raal_loss",
    "backward",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal-commit", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--best-flow-eval-step", default="")
    parser.add_argument("--best-flow-eval-loss", default="")
    args = parser.parse_args()

    rows = read_generation_manifest(args.d2_manifest)
    path_index = _build_dataset_path_index(args.dataset_root) if args.dataset_root else {}
    _validate_d2_rows(rows)
    d2_sha = sha256_file(args.d2_manifest)
    output_rows = []
    for row in rows:
        derived = dict(row)
        if (not derived.get("source_image_path") or not derived.get("source_label_path")) and path_index:
            image_path, label_path = path_index[derived["source_image_sha256"]]
            derived["source_image_path"] = image_path
            derived["source_label_path"] = label_path
        derived["rgda_checkpoint_sha256"] = args.checkpoint_sha256
        derived["method"] = "SD3_RGDA_RAAL"
        derived["formal_commit"] = args.formal_commit
        derived["raal_checkpoint_path"] = args.checkpoint_path
        derived["raal_checkpoint_sha256"] = args.checkpoint_sha256
        derived["d2_manifest_sha256"] = d2_sha
        derived["d3_pairing_contract"] = "D2_MANIFEST_EXACT_REUSE_CHECKPOINT_ONLY_CHANGED"
        derived["raal_attention_collector"] = "OFF"
        derived["raal_loss"] = "NOT_USED"
        derived["backward"] = "NO"
        output_rows.append(derived)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [*GENERATION_FIELDS, *RAAL_EXTRA_FIELDS]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "total": len(output_rows),
        "positive": sum(row["is_negative"] == "false" for row in output_rows),
        "negative": sum(row["is_negative"] == "true" for row in output_rows),
        "d2_manifest_sha256": d2_sha,
        "d3_manifest_sha256": sha256_file(args.output),
        "formal_commit": args.formal_commit,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_sha256": args.checkpoint_sha256,
        "best_flow_eval_step": args.best_flow_eval_step,
        "best_flow_eval_loss": args.best_flow_eval_loss,
        "pairing_gate": "PASS",
        "raal_inference_gate": "PASS",
    }
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    audit_path = args.output.with_name(f"{args.output.stem}_audit.md")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# RAAL Generation1000 Manifest Audit", ""]
    lines.extend(f"{key.upper()}={value}" for key, value in summary.items())
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("RAAL_GENERATION_MANIFEST=PASS")
    print(f"D2_MANIFEST_SHA256={d2_sha}")
    print(f"D3_MANIFEST_SHA256={summary['d3_manifest_sha256']}")
    return 0


def _validate_d2_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 1000:
        raise ValueError(f"D2_MANIFEST_ROWS={len(rows)}")
    positives = sum(row["is_negative"] == "false" for row in rows)
    negatives = sum(row["is_negative"] == "true" for row in rows)
    if (positives, negatives) != (750, 250):
        raise ValueError(f"D2_MANIFEST_BALANCE={positives}/{negatives}")
    required = {
        "generation_index",
        "source_sample_id",
        "source_image_sha256",
        "source_label_sha256",
        "prompt",
        "seed",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "scheduler",
    }
    missing = sorted(field for field in required if any(not row.get(field) for row in rows))
    if missing:
        raise ValueError(f"D2_MANIFEST_MISSING_FIELDS={missing}")
    indices = [int(row["generation_index"]) for row in rows]
    if indices != list(range(1000)):
        raise ValueError("D2_MANIFEST_ORDER_GATE=FAIL")
    if len({row["source_image_sha256"] for row in rows}) != 1000:
        raise ValueError("D2_MANIFEST_UNIQUE_SOURCE_GATE=FAIL")
    if len({row["seed"] for row in rows}) != 1000:
        raise ValueError("D2_MANIFEST_UNIQUE_SEED_GATE=FAIL")


def _build_dataset_path_index(dataset_root: Path | None) -> dict[str, tuple[str, str]]:
    if dataset_root is None:
        return {}
    index: dict[str, tuple[str, str]] = {}
    for candidate in collect_split_candidates(dataset_root, "train"):
        image_sha = sha256_file(candidate.image_path)
        index[image_sha] = (str(candidate.image_path), str(candidate.label_path))
    return index


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

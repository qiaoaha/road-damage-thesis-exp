#!/usr/bin/env python3
"""No-GPU contract gate for D3 RAAL downstream generation and YOLO."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from sd3_rgda.generation_manifest import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal-commit", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    args = parser.parse_args()

    archive_sha = sha256_file(args.formal_archive)
    if archive_sha != args.expected_archive_sha256:
        raise ValueError(f"FORMAL_ARCHIVE_SHA256_GATE=FAIL actual={archive_sha}")

    report_member = ""
    best_member = ""
    with tarfile.open(args.formal_archive, "r:gz") as archive:
        names = archive.getnames()
        report_member = next((name for name in names if name.endswith("/35_RAAL_FORMAL5000_GPU_FINAL.md")), "")
        best_member = next((name for name in names if name.endswith("/checkpoints/raal_formal5000/best_eval.pt")), "")
        if not report_member:
            raise ValueError("FORMAL35_REPORT_GATE=FAIL")
        if not best_member:
            raise ValueError("RAAL_BEST_CHECKPOINT_GATE=FAIL")
        report = archive.extractfile(report_member)
        if report is None:
            raise ValueError("FORMAL35_REPORT_READ_GATE=FAIL")
        report_text = report.read().decode("utf-8")

    fields = _parse_report(report_text)
    if fields.get("FINAL_COMMIT") != args.formal_commit:
        raise ValueError("FINAL_COMMIT_GATE=FAIL")
    if fields.get("FINAL_VERDICT") != "PASS":
        raise ValueError("FORMAL5000_VERDICT_GATE=FAIL")
    if fields.get("READY_FOR_RAAL_GENERATION1000") != "PASS":
        raise ValueError("READY_FOR_RAAL_GENERATION1000_GATE=FAIL")

    summary = {
        "formal_archive_sha256": archive_sha,
        "formal_report_member": report_member,
        "formal_commit": fields.get("FINAL_COMMIT", ""),
        "formal_verdict": fields.get("FINAL_VERDICT", ""),
        "ready_for_raal_generation1000": fields.get("READY_FOR_RAAL_GENERATION1000", ""),
        "best_checkpoint_member": best_member,
        "best_checkpoint_path": fields.get("BEST_CHECKPOINT_PATH", ""),
        "best_checkpoint_sha256": fields.get("BEST_CHECKPOINT_SHA256", ""),
        "best_flow_eval_step": fields.get("BEST_FLOW_EVAL_STEP", ""),
        "best_flow_eval_loss": fields.get("BEST_FLOW_EVAL_LOSS", ""),
        "d3_contract": "REUSE_D2_MANIFEST_CHECKPOINT_ONLY_CHANGE",
        "gpu_used": "NO",
        "real_sd3_used": "NO",
        "ready_for_d3_gpu_smoke16": "PENDING_D2_MANIFEST",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = args.output.with_suffix(".md")
    lines = ["# RAAL Downstream No-GPU Contract", ""]
    lines.extend(f"{key.upper()}={value}" for key, value in summary.items())
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("RAAL_FORMAL_ARCHIVE_GATE=PASS")
    print("FORMAL35_REPORT_GATE=PASS")
    print("READY_FOR_D3_GPU_SMOKE16=PENDING_D2_MANIFEST")
    return 0


def _parse_report(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


if __name__ == "__main__":
    raise SystemExit(main())

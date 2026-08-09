#!/usr/bin/env python3
"""Trace whether the old Czech SD3-1000 baseline is exactly reusable."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("reports/18_SD3_BASELINE_TRACE.md"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = _search_evidence(args.repo_root)
    reuse = "PASS" if evidence["exact_source_manifest"] and evidence["prompt_seed_params"] and evidence["labels"] else "FAIL_REGENERATE_PAIRED_BASELINE"
    lines = [
        "# 18 SD3 Baseline Trace",
        "",
        f"OLD_SD3_SOURCE_MANIFEST_FOUND={_yn(evidence['exact_source_manifest'])}",
        f"OLD_SD3_OUTPUTS_FOUND={_yn(evidence['outputs'])}",
        f"OLD_SD3_SOURCE_COUNT={evidence['source_count']}",
        "OLD_SD3_POSITIVE_COUNT=UNVERIFIED",
        "OLD_SD3_NEGATIVE_COUNT=UNVERIFIED",
        f"OLD_SD3_PROMPTS_RECOVERED={_yn(evidence['prompt_seed_params'])}",
        f"OLD_SD3_SEEDS_RECOVERED={_yn(evidence['prompt_seed_params'])}",
        "OLD_SD3_INFERENCE_STEPS=UNVERIFIED",
        "OLD_SD3_GUIDANCE_SCALE=UNVERIFIED",
        "OLD_SD3_SCHEDULER=UNVERIFIED",
        "OLD_SD3_RESOLUTION=UNVERIFIED",
        f"OLD_SD3_LABEL_POLICY={_yn(evidence['labels'])}",
        f"OLD_SD3_YOLO_CONFIG_FOUND={_yn(evidence['yolo_config'])}",
        "OLD_REAL_MAP50=0.2230_UNVERIFIED_FROM_PLAN_ONLY",
        "OLD_REAL_MAP5095=0.0804_UNVERIFIED_FROM_PLAN_ONLY",
        "OLD_SD3_MAP50=0.2580_UNVERIFIED_FROM_PLAN_ONLY",
        "OLD_SD3_MAP5095=0.0977_UNVERIFIED_FROM_PLAN_ONLY",
        f"BASELINE_REUSE_VERDICT={reuse}",
        "",
        "Decision: exact old SD3-1000 sources, prompts, seeds, inference parameters, and labels were not all recovered from local evidence, so the next GPU stage must regenerate paired G0/G1 unless stronger historical artifacts are provided.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"BASELINE_REUSE_VERDICT={reuse}")
    print(f"TRACE={args.output}")
    return 0


def _search_evidence(root: Path) -> dict[str, bool | int]:
    text_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".md", ".csv", ".json", ".yaml", ".yml", ".txt"}
        and ".git" not in path.parts
        and "cache" not in path.parts
        and "generated" not in path.parts
    ]
    exact_manifest = False
    prompt_seed_params = False
    labels = False
    outputs = False
    yolo_config = False
    source_count = "UNVERIFIED"
    for path in text_files[:10000]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lowered = text.lower()
        if "source_image_sha256" in lowered and "prompt" in lowered and "seed" in lowered and "1000" in lowered:
            exact_manifest = True
            source_count = "1000_CANDIDATE"
        prompt_seed_params = prompt_seed_params or all(token in lowered for token in ["guidance", "scheduler", "seed"])
        labels = labels or ("copy" in lowered and "label" in lowered and "sha" in lowered)
        outputs = outputs or ("real+sd3" in lowered or "sd3-1000" in lowered or "sd3 background" in lowered)
        yolo_config = yolo_config or ("yolov11" in lowered and "args.yaml" in lowered)
    return {
        "exact_source_manifest": exact_manifest,
        "prompt_seed_params": prompt_seed_params,
        "labels": labels,
        "outputs": outputs,
        "yolo_config": yolo_config,
        "source_count": source_count,
    }


def _yn(value: object) -> str:
    return "PASS" if bool(value) else "FAIL"


if __name__ == "__main__":
    raise SystemExit(main())

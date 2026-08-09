#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.generation_manifest import write_generation1000_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("manifests/generation1000"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--old-baseline-manifest", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--scheduler", default="FlowMatchEulerDiscreteScheduler")
    args = parser.parse_args()
    summary = write_generation1000_outputs(
        args.dataset_root,
        args.output_dir,
        args.checkpoint_sha256,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        scheduler=args.scheduler,
        old_baseline_manifest=args.old_baseline_manifest,
    )
    print(f"GEN1000_MANIFEST={summary.manifest_gate}")
    print(f"TOTAL={summary.total}")
    print(f"POSITIVE={summary.positive}")
    print(f"NEGATIVE={summary.negative}")
    return 0 if summary.manifest_gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

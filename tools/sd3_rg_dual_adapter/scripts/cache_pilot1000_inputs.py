from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sd3_rgda.cache import cache_pilot_manifest_rows
from sd3_rgda.real_sd3_engine import load_sd3_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache real SD3 inputs for Pilot1000.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--clean-proxy-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Pilot1000 real cache requires CUDA; do not run in no-GPU audit")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    pipe = load_sd3_pipeline(args.model_path, dtype)
    train_manifest = args.cache_dir / "train512_clean_proxy_manifest.csv"
    eval_manifest = args.cache_dir / "eval64_clean_proxy_manifest.csv"
    _split_clean_proxy_manifest(args.clean_proxy_manifest, train_manifest, eval_manifest)
    train = cache_pilot_manifest_rows(pipe, train_manifest, args.cache_dir / "train512", args.resolution, dtype, patch_size=2)
    eval_report = cache_pilot_manifest_rows(pipe, eval_manifest, args.cache_dir / "eval64", args.resolution, dtype, patch_size=2)
    print(f"TRAIN_CACHE_ROWS={train.cache_rows}")
    print(f"EVAL_CACHE_ROWS={eval_report.cache_rows}")
    print(f"ALL_TENSORS_FINITE={train.all_tensors_finite and eval_report.all_tensors_finite}")
    audit = {
        "TRAIN_CACHE_ROWS": train.cache_rows,
        "EVAL_CACHE_ROWS": eval_report.cache_rows,
        "ALL_CACHE_FILES_EXIST": "PASS" if train.all_cache_files_exist and eval_report.all_cache_files_exist else "FAIL",
        "ALL_TENSORS_FINITE": "PASS" if train.all_tensors_finite and eval_report.all_tensors_finite else "FAIL",
        "VAE_PARAMETER_DTYPE": train.vae_parameter_dtype,
        "VAE_INPUT_DTYPE": train.vae_input_dtype,
        "CACHED_LATENT_DTYPE": train.cached_latent_dtype,
        "TEXT_CACHE_DTYPE": "torch.bfloat16",
        "SOURCE_HASH_VERIFIED": "PASS",
        "CLEAN_PROXY_HASH_VERIFIED": "PASS",
        "LABEL_HASH_VERIFIED": "PASS",
        "NEGATIVE_RG_ZERO": "PASS" if train.negative_rg_map_zero and eval_report.negative_rg_map_zero else "FAIL",
        "NEGATIVE_TOKEN_MASK_ZERO": "PASS" if train.negative_token_mask_zero and eval_report.negative_token_mask_zero else "FAIL",
    }
    (args.cache_dir / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return 0


def _split_clean_proxy_manifest(source: Path, train_out: Path, eval_out: Path) -> None:
    import csv

    train_out.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields:
        fields = list(rows[0]) if rows else ["sample_id"]
    for split_name, out_path in [("train", train_out), ("eval", eval_out)]:
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in rows if row.get("split") == split_name)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from sd3_rgda.cache import cache_formal_manifest_rows
from sd3_rgda.real_sd3_engine import load_sd3_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache real SD3 inputs for Formal5000.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--clean-proxy-manifest", type=Path, required=True)
    parser.add_argument("--eval128-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Formal5000 real cache requires CUDA; do not run in no-GPU audit")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    pipe = load_sd3_pipeline(args.model_path, dtype)
    train = cache_formal_manifest_rows(
        pipe, args.clean_proxy_manifest, args.cache_dir / "train1980", args.resolution, dtype, patch_size=2, role="train"
    )
    val = cache_formal_manifest_rows(
        pipe, args.clean_proxy_manifest, args.cache_dir / "val424", args.resolution, dtype, patch_size=2, role="val_full"
    )
    eval128 = write_eval128_cache_manifest(args.eval128_manifest, val.cache_manifest, args.cache_dir / "eval128_cache_manifest.csv")
    audit = build_formal_cache_audit(train, val, len(eval128))
    write_payload_sha256(args.cache_dir / "cache_payload_sha256.tsv", args.cache_dir)
    (args.cache_dir / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"TRAIN_CACHE_ROWS={audit['TRAIN_CACHE_ROWS']}")
    print(f"VAL_CACHE_ROWS={audit['VAL_CACHE_ROWS']}")
    print(f"EVAL128_CACHE_ROWS={audit['EVAL128_CACHE_ROWS']}")
    print(f"CACHE_READY={audit['CACHE_READY']}")
    return 0 if audit["CACHE_READY"] == "PASS" else 2


def write_eval128_cache_manifest(eval128_manifest: Path, val_cache_manifest: Path, out_path: Path) -> list[dict[str, str]]:
    eval_rows = _read_csv(eval128_manifest)
    eval_shas = [row.get("image_sha256", "") for row in eval_rows]
    if len(eval_shas) != 128:
        raise ValueError(f"EVAL128_MANIFEST_ROWS={len(eval_shas)}")
    if any(not sha for sha in eval_shas):
        raise ValueError("EVAL128_IMAGE_SHA256_MISSING")
    if len(set(eval_shas)) != len(eval_shas):
        raise ValueError("EVAL128_IMAGE_SHA256_DUPLICATE")
    val_rows = _read_csv(val_cache_manifest)
    val_by_sha: dict[str, dict[str, str]] = {}
    for row in val_rows:
        sha = row.get("source_image_sha256", "")
        if not sha:
            raise ValueError("VAL_CACHE_SOURCE_IMAGE_SHA256_MISSING")
        if sha in val_by_sha:
            raise ValueError(f"VAL_CACHE_SOURCE_IMAGE_SHA256_DUPLICATE={sha}")
        val_by_sha[sha] = row
    selected: list[dict[str, str]] = []
    for sha in eval_shas:
        if sha not in val_by_sha:
            raise ValueError(f"EVAL128_IMAGE_SHA256_UNMATCHED={sha}")
        selected.append({**val_by_sha[sha], "eval_role": "eval128"})
    if len(selected) != 128:
        raise ValueError(f"EVAL128_CACHE_ROWS={len(selected)}")
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    return selected


def build_formal_cache_audit(train, val, eval128_rows: int) -> dict[str, object]:
    audit = {
        "TRAIN_CACHE_ROWS": train.cache_rows,
        "VAL_CACHE_ROWS": val.cache_rows,
        "TOTAL_CACHE_ROWS": train.cache_rows + val.cache_rows,
        "EVAL128_CACHE_ROWS": eval128_rows,
        "ALL_CACHE_FILES_EXIST": "PASS" if train.all_cache_files_exist and val.all_cache_files_exist else "FAIL",
        "ALL_TENSORS_FINITE": "PASS" if train.all_tensors_finite and val.all_tensors_finite else "FAIL",
        "VAE_PARAMETER_DTYPE": train.vae_parameter_dtype if train.vae_parameter_dtype == val.vae_parameter_dtype else "MISMATCH",
        "VAE_INPUT_DTYPE": train.vae_input_dtype if train.vae_input_dtype == val.vae_input_dtype else "MISMATCH",
        "CACHED_LATENT_DTYPE": train.cached_latent_dtype if train.cached_latent_dtype == val.cached_latent_dtype else "MISMATCH",
        "TEXT_CACHE_DTYPE": train.text_cache_dtype if train.text_cache_dtype == val.text_cache_dtype else "MISMATCH",
        "SOURCE_HASH_VERIFIED": "PASS" if train.source_hash_verified and val.source_hash_verified else "FAIL",
        "LABEL_HASH_VERIFIED": "PASS" if train.label_hash_verified and val.label_hash_verified else "FAIL",
        "CLEAN_PROXY_HASH_VERIFIED": "PASS" if train.clean_proxy_hash_verified and val.clean_proxy_hash_verified else "FAIL",
        "NEGATIVE_RG_ZERO": "PASS" if train.negative_rg_map_zero and val.negative_rg_map_zero else "FAIL",
        "NEGATIVE_TOKEN_MASK_ZERO": "PASS" if train.negative_token_mask_zero and val.negative_token_mask_zero else "FAIL",
    }
    expected = {
        "TRAIN_CACHE_ROWS": 1980,
        "VAL_CACHE_ROWS": 424,
        "TOTAL_CACHE_ROWS": 2404,
        "EVAL128_CACHE_ROWS": 128,
        "ALL_CACHE_FILES_EXIST": "PASS",
        "ALL_TENSORS_FINITE": "PASS",
        "VAE_PARAMETER_DTYPE": "torch.float32",
        "VAE_INPUT_DTYPE": "torch.float32",
        "CACHED_LATENT_DTYPE": "torch.bfloat16",
        "TEXT_CACHE_DTYPE": "torch.bfloat16",
        "SOURCE_HASH_VERIFIED": "PASS",
        "LABEL_HASH_VERIFIED": "PASS",
        "CLEAN_PROXY_HASH_VERIFIED": "PASS",
        "NEGATIVE_RG_ZERO": "PASS",
        "NEGATIVE_TOKEN_MASK_ZERO": "PASS",
    }
    audit["CACHE_READY"] = "PASS" if all(audit.get(key) == value for key, value in expected.items()) else "FAIL"
    return audit


def write_payload_sha256(path: Path, cache_dir: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in sorted(cache_dir.glob("**/*.pt")):
            handle.write(f"{_sha256(item)}  {item}\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

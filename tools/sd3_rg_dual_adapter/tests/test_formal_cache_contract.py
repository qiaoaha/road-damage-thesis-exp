from __future__ import annotations

import csv

import pytest

from scripts.cache_formal5000_inputs import build_formal_cache_audit, write_eval128_cache_manifest


class _Report:
    cache_rows = 1980
    all_cache_files_exist = True
    all_tensors_finite = True
    vae_parameter_dtype = "torch.float32"
    vae_input_dtype = "torch.float32"
    cached_latent_dtype = "torch.bfloat16"
    text_cache_dtype = "torch.bfloat16"
    source_hash_verified = True
    label_hash_verified = True
    clean_proxy_hash_verified = True
    negative_rg_map_zero = True
    negative_token_mask_zero = True


def test_formal_cache_audit_contract() -> None:
    train = _Report()
    val = _Report()
    val.cache_rows = 424
    audit = build_formal_cache_audit(train, val, 128)
    assert audit["TOTAL_CACHE_ROWS"] == 2404
    assert audit["CACHE_READY"] == "PASS"
    val.text_cache_dtype = "torch.float32"
    assert build_formal_cache_audit(train, val, 128)["CACHE_READY"] == "FAIL"


def _write_csv(path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_eval128_cache_manifest_matches_by_image_sha256(tmp_path) -> None:
    eval_rows = [{"sample_id": f"formal_eval128_{i:04d}", "image_sha256": f"sha_{i:03d}"} for i in range(128)]
    eval_rows[0]["image_sha256"] = "sha_b"
    eval_rows[1]["image_sha256"] = "sha_a"
    val_rows = [
        {"sample_id": "0", "source_sample_id": "formal_val_0003", "source_image_sha256": "sha_a", "anchor_class": "D00", "is_negative": "false", "source_split": "val", "formal_role": "val_full"},
        {"sample_id": "1", "source_sample_id": "formal_val_0010", "source_image_sha256": "sha_b", "anchor_class": "D10", "is_negative": "false", "source_split": "val", "formal_role": "val_full"},
    ] + [
        {"sample_id": str(i), "source_sample_id": f"formal_val_{i:04d}", "source_image_sha256": f"sha_{i:03d}", "anchor_class": "", "is_negative": "true", "source_split": "val", "formal_role": "val_full"}
        for i in range(2, 128)
    ]
    eval_path = tmp_path / "eval128.csv"
    val_path = tmp_path / "val_cache.csv"
    _write_csv(eval_path, eval_rows)
    _write_csv(val_path, val_rows)
    selected = write_eval128_cache_manifest(eval_path, val_path, tmp_path / "eval_cache.csv")
    assert [row["source_sample_id"] for row in selected[:2]] == ["formal_val_0010", "formal_val_0003"]
    assert all(row["eval_role"] == "eval128" for row in selected)
    assert selected[0]["formal_role"] == "val_full"


def test_eval128_cache_rejects_missing_sha(tmp_path) -> None:
    eval_path = tmp_path / "eval128.csv"
    val_path = tmp_path / "val_cache.csv"
    _write_csv(eval_path, [{"sample_id": f"formal_eval128_{i:04d}", "image_sha256": "" if i == 0 else f"sha_{i:03d}"} for i in range(128)])
    _write_csv(val_path, [{"source_sample_id": f"formal_val_{i:04d}", "source_image_sha256": f"sha_{i:03d}"} for i in range(128)])
    with pytest.raises(ValueError, match="EVAL128_IMAGE_SHA256_MISSING"):
        write_eval128_cache_manifest(eval_path, val_path, tmp_path / "out.csv")


def test_eval128_cache_rejects_duplicate_sha(tmp_path) -> None:
    eval_path = tmp_path / "eval128.csv"
    val_path = tmp_path / "val_cache.csv"
    _write_csv(eval_path, [{"sample_id": f"formal_eval128_{i:04d}", "image_sha256": "dup" if i < 2 else f"sha_{i:03d}"} for i in range(128)])
    _write_csv(val_path, [{"source_sample_id": f"formal_val_{i:04d}", "source_image_sha256": f"sha_{i:03d}"} for i in range(128)])
    with pytest.raises(ValueError, match="EVAL128_IMAGE_SHA256_DUPLICATE"):
        write_eval128_cache_manifest(eval_path, val_path, tmp_path / "out.csv")


def test_eval128_cache_rejects_unmatched_sha(tmp_path) -> None:
    eval_path = tmp_path / "eval128.csv"
    val_path = tmp_path / "val_cache.csv"
    _write_csv(eval_path, [{"sample_id": f"formal_eval128_{i:04d}", "image_sha256": f"sha_{i:03d}"} for i in range(128)])
    _write_csv(val_path, [{"source_sample_id": f"formal_val_{i:04d}", "source_image_sha256": f"other_{i:03d}"} for i in range(128)])
    with pytest.raises(ValueError, match="EVAL128_IMAGE_SHA256_UNMATCHED"):
        write_eval128_cache_manifest(eval_path, val_path, tmp_path / "out.csv")

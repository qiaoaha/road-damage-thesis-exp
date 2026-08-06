from __future__ import annotations

from scripts.cache_formal5000_inputs import build_formal_cache_audit


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

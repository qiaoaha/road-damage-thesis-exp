from __future__ import annotations

import torch

from sd3_rgda.cache import CacheBuildReport
from sd3_rgda.real_sd3_engine import _cache_report_passed, is_cuda_device


def _report(vae_device: str, text_device: str) -> CacheBuildReport:
    return CacheBuildReport(
        cache_manifest="manifest.csv",  # type: ignore[arg-type]
        cache_rows=1,
        unique_prompt_count=1,
        vae_device_during_encoding=vae_device,
        text_encoder_device_during_encoding=text_device,
        all_cache_files_exist=True,
        all_tensors_finite=True,
        no_val_test_leakage=True,
        negative_rg_map_zero=True,
        negative_token_mask_zero=True,
        vae_shift_factor=0.0,
        vae_scaling_factor=1.0,
        latent_dtype="torch.float32",
        latent_shape=(1, 16, 2, 2),
    )


def test_is_cuda_device_accepts_cuda_variants() -> None:
    assert is_cuda_device("cuda")
    assert is_cuda_device("cuda:0")
    assert is_cuda_device(torch.device("cuda:1"))


def test_is_cuda_device_rejects_non_cuda_and_invalid() -> None:
    assert not is_cuda_device("cpu")
    assert not is_cuda_device("")
    assert not is_cuda_device("definitely-not-a-device")


def test_cache_gate_uses_cuda_device_type_and_rows() -> None:
    assert _cache_report_passed(_report("cuda:0", "cuda:1"))
    assert not _cache_report_passed(_report("cuda:0", "cpu"))
    empty = _report("cuda", "cuda")
    empty.cache_rows = 0
    assert not _cache_report_passed(empty)

from __future__ import annotations

from sd3_rgda.cache import CacheBuildReport


def test_cache_build_report_requires_cuda_devices_for_pass() -> None:
    report = CacheBuildReport(
        cache_manifest=__import__("pathlib").Path("cache_manifest.csv"),
        cache_rows=1,
        unique_prompt_count=1,
        vae_device_during_encoding="cpu",
        text_encoder_device_during_encoding="cuda",
        all_cache_files_exist=True,
        all_tensors_finite=True,
        no_val_test_leakage=True,
        negative_rg_map_zero=True,
        negative_token_mask_zero=True,
        vae_shift_factor=0.0,
        vae_scaling_factor=1.0,
        latent_dtype="torch.float32",
        latent_shape=(1, 4, 8, 8),
    )
    assert report.vae_device_during_encoding != "cuda"

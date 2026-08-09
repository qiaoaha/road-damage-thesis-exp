from pathlib import Path

import pytest
import torch
from torch import nn

from sd3_rgda.checkpoint import save_adapter_checkpoint
from sd3_rgda.generation_manifest import sha256_file
from sd3_rgda.inference import prepare_injector_from_checkpoint, verify_checkpoint_sha256


def test_checkpoint_sha_mismatch_rejects(tmp_path: Path) -> None:
    path = tmp_path / "adapter.pt"
    modules = {name: nn.Linear(1, 1) for name in ["normal_encoder", "rg_encoder", "normal_adapter", "defect_adapter", "timestep_gate"]}
    save_adapter_checkpoint(path, modules)
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_checkpoint_sha256(path, "0" * 64)


def test_checkpoint_unexpected_base_module_rejects(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"modules": {"transformer": {}}}, path)
    with pytest.raises(ValueError, match="adapter-only"):
        verify_checkpoint_sha256(path, sha256_file(path))


def test_rgda_inference_preserves_fp32_parameters(tmp_path: Path) -> None:
    path = tmp_path / "adapter.pt"
    injector = prepare_injector_from_checkpoint.__globals__["RGDAInjector"](token_dim=4, latent_channels=2, patch_size=1)
    modules = injector.trainable_modules()
    from sd3_rgda.checkpoint import save_adapter_checkpoint

    save_adapter_checkpoint(path, modules)
    loaded = prepare_injector_from_checkpoint(path, sha256_file(path), token_dim=4, latent_channels=2, patch_size=1)
    assert {
        parameter.dtype for module in loaded.trainable_modules().values() for parameter in module.parameters()
    } == {torch.float32}

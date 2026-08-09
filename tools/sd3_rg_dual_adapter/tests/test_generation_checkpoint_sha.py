from pathlib import Path

import pytest
import torch
from torch import nn

from sd3_rgda.checkpoint import save_adapter_checkpoint
from sd3_rgda.generation_manifest import sha256_file
from sd3_rgda.inference import verify_checkpoint_sha256


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

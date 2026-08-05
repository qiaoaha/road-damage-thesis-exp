from __future__ import annotations

import torch
from torch import nn

from sd3_rgda.checkpoint import inspect_adapter_checkpoint, save_adapter_checkpoint


def _modules() -> dict[str, nn.Module]:
    return {
        "normal_encoder": nn.Linear(1, 1),
        "rg_encoder": nn.Linear(1, 1),
        "normal_adapter": nn.Linear(1, 1),
        "defect_adapter": nn.Linear(1, 1),
        "timestep_gate": nn.Linear(1, 1),
    }


def test_adapter_checkpoint_scope_is_valid(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(path, _modules(), {"engine": "unit"})
    report = inspect_adapter_checkpoint(path)
    assert report["valid_adapter_only"] is True
    assert report["contains_base_sd3"] is False
    assert report["missing_modules"] == []


def test_adapter_checkpoint_rejects_base_transformer(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "bad.pt"
    modules = _modules()
    payload = {
        "metadata": {},
        "modules": {name: module.state_dict() for name, module in modules.items()} | {"transformer": {"w": torch.ones(1)}},
    }
    torch.save(payload, path)
    report = inspect_adapter_checkpoint(path)
    assert report["valid_adapter_only"] is False
    assert report["contains_base_sd3"] is True
    assert report["forbidden_modules"] == ["transformer"]


def test_adapter_checkpoint_rejects_missing_timestep_gate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "missing.pt"
    modules = _modules()
    modules.pop("timestep_gate")
    save_adapter_checkpoint(path, modules, None)
    report = inspect_adapter_checkpoint(path)
    assert report["valid_adapter_only"] is False
    assert report["missing_modules"] == ["timestep_gate"]

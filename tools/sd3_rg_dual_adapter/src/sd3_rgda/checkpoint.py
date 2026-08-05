"""Checkpoint helpers that exclude base SD3 weights."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

EXPECTED_ADAPTER_MODULES = {
    "normal_encoder",
    "rg_encoder",
    "normal_adapter",
    "defect_adapter",
    "timestep_gate",
}
FORBIDDEN_BASE_PREFIXES = ("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3")


def save_adapter_checkpoint(path: str | Path, modules: dict[str, nn.Module], metadata: dict[str, str] | None = None) -> None:
    payload = {
        "metadata": metadata or {},
        "modules": {name: module.state_dict() for name, module in modules.items()},
    }
    torch.save(payload, path)


def inspect_adapter_checkpoint(path: str | Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    modules = payload.get("modules") if isinstance(payload, dict) else None
    if not isinstance(modules, dict):
        module_names: list[str] = []
    else:
        module_names = sorted(str(name) for name in modules)
    missing = sorted(EXPECTED_ADAPTER_MODULES - set(module_names))
    unexpected = sorted(set(module_names) - EXPECTED_ADAPTER_MODULES)
    forbidden = sorted(
        name for name in module_names if name == "transformer" or name.startswith(FORBIDDEN_BASE_PREFIXES)
    )
    contains_base_sd3 = bool(forbidden)
    valid_adapter_only = bool(module_names) and not missing and not unexpected and not contains_base_sd3
    return {
        "module_names": module_names,
        "missing_modules": missing,
        "unexpected_modules": unexpected,
        "forbidden_modules": forbidden,
        "contains_base_sd3": contains_base_sd3,
        "valid_adapter_only": valid_adapter_only,
    }


def load_adapter_checkpoint(path: str | Path, modules: dict[str, nn.Module]) -> dict[str, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "modules" not in payload:
        raise ValueError(f"Invalid adapter checkpoint: {path}")
    state = payload["modules"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint modules entry must be a mapping")
    for name, module in modules.items():
        if name not in state:
            raise KeyError(f"Checkpoint missing module {name}")
        module.load_state_dict(state[name])
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}

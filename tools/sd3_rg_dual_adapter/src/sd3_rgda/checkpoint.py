"""Checkpoint helpers that exclude base SD3 weights."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_adapter_checkpoint(path: str | Path, modules: dict[str, nn.Module], metadata: dict[str, str] | None = None) -> None:
    payload = {
        "metadata": metadata or {},
        "modules": {name: module.state_dict() for name, module in modules.items()},
    }
    torch.save(payload, path)


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

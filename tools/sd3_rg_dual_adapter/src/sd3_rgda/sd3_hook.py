"""Controlled hooks for SD3 transformer compatibility experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


def assert_diffusers_compatibility(transformer: object) -> None:
    config = getattr(transformer, "config", None)
    required = ["patch_size", "num_attention_heads", "attention_head_dim"]
    missing = [name for name in required if config is None or not hasattr(config, name)]
    if missing:
        raise RuntimeError(f"Incompatible SD3 transformer config; missing {missing}")
    candidates = ["pos_embed", "x_embedder", "patch_embed"]
    if not any(hasattr(transformer, name) for name in candidates):
        raise RuntimeError(f"Could not locate SD3 patch embedding module; checked {candidates}")


@contextmanager
def temporary_forward_hook(module: nn.Module, hook: Callable[..., Any]) -> Iterator[None]:
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class AddOnceHook:
    def __init__(self, residual: torch.Tensor) -> None:
        self.residual = residual
        self.calls = 0

    def __call__(self, _module: nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return output + self.residual

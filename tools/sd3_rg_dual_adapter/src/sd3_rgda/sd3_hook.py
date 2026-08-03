"""Controlled hooks for SD3 transformer compatibility experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

PATCH_EMBED_CANDIDATES = ("pos_embed", "x_embedder", "patch_embed")


def assert_diffusers_compatibility(transformer: object) -> None:
    config = getattr(transformer, "config", None)
    required = ["patch_size", "num_attention_heads", "attention_head_dim"]
    missing = [name for name in required if config is None or not hasattr(config, name)]
    if missing:
        raise RuntimeError(f"Incompatible SD3 transformer config; missing {missing}")
    resolve_patch_embedding(transformer)


def resolve_patch_embedding(transformer: object) -> tuple[str, nn.Module]:
    """Find the SD3 image patch embedding module without assuming its exact name."""

    for name in PATCH_EMBED_CANDIDATES:
        module = getattr(transformer, name, None)
        if isinstance(module, nn.Module):
            return name, module
    named_modules = getattr(transformer, "named_modules", None)
    if callable(named_modules):
        for name, module in named_modules():
            lowered = str(name).lower()
            if isinstance(module, nn.Module) and "patch" in lowered and "embed" in lowered:
                return str(name), module
    raise RuntimeError(f"Could not locate SD3 patch embedding module; checked {PATCH_EMBED_CANDIDATES}")


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

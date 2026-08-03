"""Validation helpers for local CPU gates."""

from __future__ import annotations

from pathlib import Path

FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".zip"}


def assert_no_large_or_forbidden_files(paths: list[Path], max_bytes: int = 20 * 1024 * 1024) -> None:
    for path in paths:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Forbidden tracked artifact suffix: {path}")
        if path.is_file() and path.stat().st_size > max_bytes:
            raise ValueError(f"Tracked file exceeds {max_bytes} bytes: {path}")

from __future__ import annotations

from pathlib import Path


def require_gpu_step(step_name: str, cfg: dict) -> None:
    cpu_only = bool(cfg.get("runtime", {}).get("cpu_only_check", False))
    if cpu_only:
        raise RuntimeError(f"{step_name} requires model inference and is disabled by cpu_only_check=true")


def cache_exists(path: Path | None) -> bool:
    return bool(path and path.exists())

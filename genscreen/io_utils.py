from __future__ import annotations

import csv
import logging
import random
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def setup_logging(log_dir: Path | None = None, name: str = "genscreen") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "genscreen.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_files(root: Path | None) -> list[Path]:
    if not root or not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_copy(src: Path, dst_dir: Path, preserve_relative_to: Path | None = None) -> Path:
    ensure_dir(dst_dir)
    if preserve_relative_to:
        try:
            rel = src.relative_to(preserve_relative_to)
            target = dst_dir / rel
            ensure_dir(target.parent)
        except ValueError:
            target = dst_dir / src.name
    else:
        target = dst_dir / src.name
    if not target.exists():
        shutil.copy2(src, target)
        return target
    stem, suffix = target.stem, target.suffix
    for idx in range(1, 100000):
        candidate = target.with_name(f"{stem}__{idx:04d}{suffix}")
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
    raise RuntimeError(f"Could not find non-overwriting copy target for {src}")

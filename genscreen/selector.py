from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .config import cfg_path
from .io_utils import read_csv, safe_copy, write_csv
from .mmr import mmr_select


def quota_map(cfg: dict, classes: list[str]) -> dict[str, int]:
    configured = cfg_path(cfg, "selection.class_quota", {}) or {}
    if configured:
        return {str(k): int(v) for k, v in configured.items()}
    total = int(cfg_path(cfg, "selection.total_num", 0) or 0)
    if not classes or total <= 0:
        return {}
    base = total // len(classes)
    quotas = {c: base for c in classes}
    for c in classes[: total - base * len(classes)]:
        quotas[c] += 1
    return quotas


def select_samples(cfg: dict, quality_csv: Path, output_dir: Path, copy_files: bool = True) -> list[dict]:
    rows = [r for r in read_csv(quality_csv) if r.get("is_valid") == "true"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["class_id"])].append(row)
    quotas = quota_map(cfg, sorted(groups.keys()))
    allow_redistribution = bool(cfg_path(cfg, "selection.allow_quota_redistribution", True))
    selected: list[dict] = []
    shortfall = 0
    lamb_q = float(cfg_path(cfg, "mmr.lambda_quality", 0.7))
    lamb_d = float(cfg_path(cfg, "mmr.lambda_diversity", 0.3))
    mmr_enabled = bool(cfg_path(cfg, "mmr.enabled", True))
    for cls, quota in quotas.items():
        available = groups.get(cls, [])
        take = min(quota, len(available))
        if take < quota:
            shortfall += quota - take
            if not allow_redistribution:
                raise RuntimeError(f"class {cls} has {len(available)} candidates but quota is {quota}")
        selected.extend(mmr_select(available, take, lamb_q, lamb_d, mmr_enabled))
    if shortfall and allow_redistribution:
        already = {r["image_path"] for r in selected}
        leftovers = [r for r in rows if r["image_path"] not in already]
        selected.extend(mmr_select(leftovers, shortfall, lamb_q, lamb_d, mmr_enabled))
    selected = sorted(selected, key=lambda r: float(r.get("Q", 0) or 0), reverse=True)
    for rank, row in enumerate(selected, 1):
        row["rank_global"] = rank
    fields = [
        "rank_global",
        "rank_class",
        "image_path",
        "label_path",
        "class_id",
        "class_name",
        "Q",
        "mmr_score",
        "diversity_penalty",
        "S_DINO",
        "S_PCS",
        "S_conf",
        "S_margin",
    ]
    write_csv(output_dir / "scores" / "selected_gen200.csv", selected, fields)
    if copy_files:
        image_out = output_dir / "selected" / "images"
        label_out = output_dir / "selected" / "labels"
        for row in selected:
            image = Path(row["image_path"])
            label = Path(row["label_path"]) if row.get("label_path") else None
            if image.exists():
                safe_copy(image, image_out)
            if label and label.exists():
                safe_copy(label, label_out)
    return selected

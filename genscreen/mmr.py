from __future__ import annotations

from pathlib import Path

import numpy as np

from .dino_score import deterministic_score
from .features import read_feature_cache


def pair_similarity(a: dict, b: dict) -> float:
    key = "|".join(sorted([a["image_path"], b["image_path"]]))
    return deterministic_score(key, "sim")


def load_similarity_features(output_dir: Path) -> dict[str, np.ndarray]:
    cache_path = output_dir / "cache" / "generated_dino_features.npz"
    if not cache_path.exists():
        return {}
    cache = read_feature_cache(cache_path)
    return {str(path): cache["feature"][idx] for idx, path in enumerate(cache["image_path"])}


def real_similarity(a: dict, b: dict, features: dict[str, np.ndarray]) -> float:
    fa = features.get(a["image_path"])
    fb = features.get(b["image_path"])
    if fa is None or fb is None:
        raise KeyError("missing generated DINO feature")
    return float((fa @ fb + 1.0) / 2.0)


def mmr_select(
    rows: list[dict],
    quota: int,
    lambda_quality: float,
    lambda_diversity: float,
    enabled: bool = True,
    features: dict[str, np.ndarray] | None = None,
    allow_dummy_similarity: bool = False,
) -> list[dict]:
    candidates = sorted(rows, key=lambda r: float(r.get("Q", 0) or 0), reverse=True)
    if not enabled:
        selected = candidates[:quota]
        for rank, row in enumerate(selected, 1):
            row["rank_class"] = rank
            row["diversity_penalty"] = 0.0
            row["max_similarity_to_selected"] = 0.0
            row["mmr_score"] = row.get("Q", 0)
        return selected
    features = features or {}
    selected: list[dict] = []
    while candidates and len(selected) < quota:
        if not selected:
            best = candidates.pop(0)
            best["diversity_penalty"] = 0.0
            best["max_similarity_to_selected"] = 0.0
            best["mmr_score"] = best.get("Q", 0)
        else:
            scored = []
            for row in candidates:
                if features:
                    penalty = max(real_similarity(row, chosen, features) for chosen in selected)
                elif allow_dummy_similarity:
                    penalty = max(pair_similarity(row, chosen) for chosen in selected)
                else:
                    raise RuntimeError("MMR requires generated_dino_features.npz. Use --dry-run to allow dummy similarity.")
                score = lambda_quality * float(row.get("Q", 0) or 0) - lambda_diversity * penalty
                scored.append((score, penalty, row))
            score, penalty, best = max(scored, key=lambda item: item[0])
            candidates.remove(best)
            best["diversity_penalty"] = round(penalty, 6)
            best["max_similarity_to_selected"] = round(penalty, 6)
            best["mmr_score"] = round(score, 6)
        best["rank_class"] = len(selected) + 1
        selected.append(best)
    return selected

from __future__ import annotations

from .dino_score import deterministic_score


def pair_similarity(a: dict, b: dict) -> float:
    key = "|".join(sorted([a["image_path"], b["image_path"]]))
    return deterministic_score(key, "sim")


def mmr_select(rows: list[dict], quota: int, lambda_quality: float, lambda_diversity: float, enabled: bool = True) -> list[dict]:
    candidates = sorted(rows, key=lambda r: float(r.get("Q", 0) or 0), reverse=True)
    if not enabled:
        selected = candidates[:quota]
        for rank, row in enumerate(selected, 1):
            row["rank_class"] = rank
            row["diversity_penalty"] = 0.0
            row["mmr_score"] = row.get("Q", 0)
        return selected
    selected: list[dict] = []
    while candidates and len(selected) < quota:
        if not selected:
            best = candidates.pop(0)
            best["diversity_penalty"] = 0.0
            best["mmr_score"] = best.get("Q", 0)
        else:
            scored = []
            for row in candidates:
                penalty = max(pair_similarity(row, chosen) for chosen in selected)
                score = lambda_quality * float(row.get("Q", 0) or 0) - lambda_diversity * penalty
                scored.append((score, penalty, row))
            score, penalty, best = max(scored, key=lambda item: item[0])
            candidates.remove(best)
            best["diversity_penalty"] = round(penalty, 6)
            best["mmr_score"] = round(score, 6)
        best["rank_class"] = len(selected) + 1
        selected.append(best)
    return selected

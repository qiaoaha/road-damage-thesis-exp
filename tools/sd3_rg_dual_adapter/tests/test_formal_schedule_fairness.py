from __future__ import annotations

from collections import Counter

from sd3_rgda.formal_manifest import build_schedule5000


def test_formal_schedule_usage_spread_is_at_most_one() -> None:
    rows = [
        {"sample_id": f"p{i}", "source_sample_id": f"p{i}", "is_negative": "false", "anchor_class": ["D00", "D10", "D20", "D40"][i % 4]}
        for i in range(40)
    ] + [{"sample_id": f"n{i}", "source_sample_id": f"n{i}", "is_negative": "true", "anchor_class": ""} for i in range(30)]
    schedule = build_schedule5000(rows)
    counts = Counter(int(row["pool_index"]) for row in schedule)
    for indices in ([i for i in range(40) if i % 4 == cls] for cls in range(4)):
        values = [counts[index] for index in indices]
        assert max(values) - min(values) <= 1
    neg_values = [counts[index] for index in range(40, 70)]
    assert max(neg_values) - min(neg_values) <= 1

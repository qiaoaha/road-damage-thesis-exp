from __future__ import annotations

from sd3_rgda.pilot_engine import audit_sample_schedule, deterministic_sample_order


def test_pilot_sampling_covers_all_512_with_expected_use_bounds() -> None:
    rows = [
        {"sample_id": str(i), "is_negative": str(i % 2 == 0).lower(), "anchor_class": "D00"}
        for i in range(512)
    ]
    order = deterministic_sample_order(rows, 1000, seed=2026)
    audit = audit_sample_schedule(rows, order)
    assert audit.unique_train_samples_used == 512
    assert audit.train_sample_use_min >= 1
    assert audit.train_sample_use_max <= 2
    assert audit.train_positive_steps + audit.train_negative_steps == 1000

from __future__ import annotations

from sd3_rgda.pilot_engine import deterministic_sample_order


def test_resume_schedule_does_not_repeat_or_skip_steps() -> None:
    rows = [{"sample_id": str(i)} for i in range(12)]
    continuous = deterministic_sample_order(rows, 10, seed=2026)
    resumed = deterministic_sample_order(rows, 5, seed=2026) + deterministic_sample_order(rows, 10, seed=2026)[5:10]
    assert resumed == continuous

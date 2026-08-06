from __future__ import annotations

from sd3_rgda.formal_manifest import audit_schedule, build_schedule5000


def test_formal_schedule_has_5000_balanced_steps() -> None:
    rows = [
        {"sample_id": f"p{i}", "source_sample_id": f"p{i}", "is_negative": "false", "anchor_class": ["D00", "D10", "D20", "D40"][i % 4]}
        for i in range(12)
    ] + [{"sample_id": f"n{i}", "source_sample_id": f"n{i}", "is_negative": "true", "anchor_class": ""} for i in range(8)]
    schedule = build_schedule5000(rows)
    audit = audit_schedule(rows, schedule)
    assert audit.schedule_rows == 5000
    assert audit.positive_steps == 2500
    assert audit.negative_steps == 2500
    assert audit.schedule_audit == "PASS"

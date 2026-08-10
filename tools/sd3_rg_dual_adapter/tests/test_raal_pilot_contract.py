from __future__ import annotations

from sd3_rgda.raal_engine import build_pilot_arm_configs


def test_r0_r1_schedule_identity() -> None:
    schedule = [{"step": step, "source_id": f"src-{step:04d}"} for step in range(1, 1001)]
    r0, r1 = build_pilot_arm_configs(schedule)
    assert r0["source_schedule_sha256"] == r1["source_schedule_sha256"]
    assert r0["raal_enabled"] is False
    assert r0["raal_weight"] == 0.0
    assert r1["raal_enabled"] is True
    assert r1["raal_weight"] == 0.02

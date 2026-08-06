from __future__ import annotations

from sd3_rgda.pilot_engine import build_fixed_eval_plan


def test_fixed_eval_plan_is_deterministic_and_complete() -> None:
    rows = [{"sample_id": f"eval_{i:04d}"} for i in range(64)]
    first = build_fixed_eval_plan(rows, seed=2026)
    second = build_fixed_eval_plan(rows, seed=2026)
    assert first == second
    assert len(first) == 64
    assert {"noise_seed", "timestep_seed", "sigma_seed"} <= set(first[0])

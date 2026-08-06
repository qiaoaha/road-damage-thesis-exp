from __future__ import annotations

from sd3_rgda.pilot_engine import PilotGateInputs, evaluate_pilot_gates


def test_pilot_gate_logic_enforces_train_and_eval_improvement() -> None:
    passing = PilotGateInputs(
        train_steps_completed=1000,
        oom_count=0,
        nan_inf_count=0,
        base_hash_before="a",
        base_hash_after="a",
        checkpoint_reload_max_abs_diff=1e-4,
        adapter_parameter_delta=1.0,
        median_first100=1.0,
        median_last100=0.8,
        eval_loss_step0=1.0,
        eval_loss_step1000=0.94,
    )
    assert evaluate_pilot_gates(passing)["FINAL_VERDICT"] == "PASS"
    failing = passing.__class__(**{**passing.__dict__, "eval_loss_step1000": 0.99})
    assert evaluate_pilot_gates(failing)["FINAL_VERDICT"] == "FAIL"

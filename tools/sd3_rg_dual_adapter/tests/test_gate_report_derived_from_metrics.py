from __future__ import annotations

from sd3_rgda.pilot_engine import PilotGateInputs, evaluate_pilot_gates


def test_gate_report_derived_from_metrics() -> None:
    fields = evaluate_pilot_gates(
        PilotGateInputs(
            train_steps_completed=999,
            oom_count=0,
            nan_inf_count=0,
            base_hash_before="a",
            base_hash_after="a",
            checkpoint_reload_max_abs_diff=0.0,
            adapter_parameter_delta=1.0,
            median_first100=1.0,
            median_last100=0.7,
            eval_loss_step0=1.0,
            eval_loss_step1000=0.8,
        )
    )
    assert fields["TRAIN_1000_STEPS"] == "FAIL"
    assert "FINAL_VERDICT" not in fields

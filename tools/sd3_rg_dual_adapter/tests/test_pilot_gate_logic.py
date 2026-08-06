from __future__ import annotations

from sd3_rgda.pilot_engine import PilotGateInputs, evaluate_pilot_gates, finalize_gate_report


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
    assert "FINAL_VERDICT" not in evaluate_pilot_gates(passing)
    failing = passing.__class__(**{**passing.__dict__, "eval_loss_step1000": 0.99})
    assert evaluate_pilot_gates(failing)["EVAL_LOSS_IMPROVED"] == "FAIL"


def test_final_verdict_includes_late_gates() -> None:
    gates = {
        "SD3_FULL_LOAD": "PASS",
        "PILOT_MANIFEST": "PASS",
        "CLEAN_PROXY": "PASS",
        "REAL_PILOT_CACHE": "PASS",
        "ZERO_INIT_EQUIVALENCE": "PASS",
        "BASE_SD3_FROZEN": "PASS",
        "TRAIN_1000_STEPS": "PASS",
        "ALL_512_SAMPLES_USED": "PASS",
        "SAMPLE_USE_RANGE": "PASS",
        "POSITIVE_NEGATIVE_MIX": "PASS",
        "DEFECT_BRANCH_ACTIVE_POSITIVE": "PASS",
        "DEFECT_BRANCH_BLOCKED_NEGATIVE": "PASS",
        "NORMAL_BRANCH_ACTIVE": "PASS",
        "EVAL_FIXED_BATCH": "PASS",
        "TRAIN_LOSS_IMPROVED": "PASS",
        "EVAL_LOSS_IMPROVED": "PASS",
        "CHECKPOINT_SAVE": "PASS",
        "CHECKPOINT_RELOAD": "PASS",
        "BASE_HASH_UNCHANGED": "PASS",
        "OOM_GATE": "PASS",
        "NAN_INF_GATE": "PASS",
    }
    assert finalize_gate_report(gates)["FINAL_VERDICT"] == "PASS"
    gates["REAL_PILOT_CACHE"] = "FAIL"
    assert finalize_gate_report(gates)["FINAL_VERDICT"] == "FAIL"

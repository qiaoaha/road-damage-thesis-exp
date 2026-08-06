from __future__ import annotations

from pathlib import Path

from sd3_rgda.formal_engine import finalize_formal_gates
from sd3_rgda.pilot_engine import BranchGradientCounts


def test_formal_gate_logic_requires_late_gates(tmp_path: Path) -> None:
    counts = BranchGradientCounts()
    counts.positive_defect_adapter_grad_nonzero_steps = 1
    counts.positive_rg_encoder_grad_nonzero_steps = 1
    counts.positive_normal_adapter_grad_nonzero_steps = 1
    counts.negative_normal_adapter_grad_nonzero_steps = 1
    checkpoint = tmp_path / "best_eval.pt"
    checkpoint.write_bytes(b"adapter")
    gates = finalize_formal_gates(
        state=type("S", (), {"step": 5000, "best_eval_loss": 0.5, "best_eval_step": 250})(),
        losses=[1.0] * 250 + [0.8] * 4500 + [0.5] * 250,
        eval_rows=[{"eval_loss_all": "1.0"}] + [{"eval_loss_all": "0.9"}] * 20,
        full_val_rows=[{"model_tag": "zero_init", "eval_loss_all": "1.0"}, {"model_tag": "best_eval", "eval_loss_all": "0.9"}, {"model_tag": "last", "eval_loss_all": "0.95"}],
        usage_audit={"ALL_1980_TRAIN_IMAGES_USED": "PASS", "POSITIVE_STEPS": 2500, "NEGATIVE_STEPS": 2500, "STRATUM_USAGE_FAIR": "PASS"},
        branch_counts=counts,
        base_before="a",
        base_after="a",
        zero_evidence={
            "FINAL_OUTPUT_MAX_ABS_DIFF": 0.0,
            "PATCH_TOKEN_MAX_ABS_DIFF": 0.0,
            "RGDA_RESIDUAL_MAX_ABS": 0.0,
            "FORMAL_STARTED_FROM_ZERO_INIT": "PASS",
        },
        checkpoint_paths=[checkpoint],
        best_diff=0.0,
        last_diff=0.0,
        oom_count=0,
        nan_inf_count=0,
        preflight={"FORMAL_MANIFEST": "PASS", "FORMAL_SCHEDULE": "PASS", "FORMAL_CLEAN_PROXY": "PASS", "FORMAL_CACHE": "PASS"},
    )
    assert gates["FINAL_VERDICT"] == "PASS"

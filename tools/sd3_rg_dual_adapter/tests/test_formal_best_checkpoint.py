from __future__ import annotations

from pathlib import Path

from sd3_rgda.formal_engine import finalize_formal_gates, run_checkpoint_reload_evaluations
from sd3_rgda.pilot_engine import BranchGradientCounts


def test_formal_best_checkpoint_controls_final_verdict(tmp_path: Path) -> None:
    counts = BranchGradientCounts()
    counts.positive_defect_adapter_grad_nonzero_steps = 1
    counts.positive_rg_encoder_grad_nonzero_steps = 1
    counts.positive_normal_adapter_grad_nonzero_steps = 1
    counts.negative_normal_adapter_grad_nonzero_steps = 1
    checkpoint = tmp_path / "best_eval.pt"
    checkpoint.write_bytes(b"adapter")
    gates = finalize_formal_gates(
        state=type("S", (), {"step": 5000, "best_eval_loss": 1.1, "best_eval_step": 250})(),
        losses=[1.0] * 250 + [0.8] * 4500 + [0.5] * 250,
        eval_rows=[{"eval_loss_all": "1.0"}] + [{"eval_loss_all": "0.9"}] * 20,
        full_val_rows=[{"model_tag": "zero_init", "eval_loss_all": "1.0"}, {"model_tag": "best_eval", "eval_loss_all": "1.1"}, {"model_tag": "last", "eval_loss_all": "0.95"}],
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
    assert gates["FULL_VAL_BEST_IMPROVED"] == "FAIL"
    assert gates["FINAL_VERDICT"] == "FAIL"


def test_best_eval128_gate_controls_final_verdict(tmp_path: Path) -> None:
    counts = BranchGradientCounts()
    counts.positive_defect_adapter_grad_nonzero_steps = 1
    counts.positive_rg_encoder_grad_nonzero_steps = 1
    counts.positive_normal_adapter_grad_nonzero_steps = 1
    counts.negative_normal_adapter_grad_nonzero_steps = 1
    checkpoint = tmp_path / "best_eval.pt"
    checkpoint.write_bytes(b"adapter")
    gates = finalize_formal_gates(
        state=type("S", (), {"step": 5000, "best_eval_loss": 0.91, "best_eval_step": 250})(),
        losses=[1.0] * 250 + [0.8] * 4500 + [0.5] * 250,
        eval_rows=[{"eval_loss_all": "1.0"}] + [{"eval_loss_all": "0.94"}] * 20,
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
    assert gates["EVAL128_LOSS_IMPROVED"] == "PASS"
    assert gates["BEST_EVAL128_IMPROVED"] == "FAIL"
    assert gates["FINAL_VERDICT"] == "FAIL"


def test_best_checkpoint_reload_is_self_consistency(tmp_path: Path) -> None:
    best = tmp_path / "best_eval.pt"
    last = tmp_path / "last.pt"
    best.write_text("1", encoding="utf-8")
    last.write_text("2", encoding="utf-8")

    class _Injector:
        def __init__(self) -> None:
            self.value = "2"

        def trainable_modules(self):
            return {"adapter": self}

    class _Trainer:
        def __init__(self) -> None:
            self.injector = _Injector()

        def compare_outputs(self, checkpoint: Path, _batch) -> float:
            return 0.0 if self.injector.value == checkpoint.read_text(encoding="utf-8") else 1.0

    def loader(path: Path, modules) -> None:
        modules["adapter"].value = path.read_text(encoding="utf-8")

    full_val_calls = []
    best_diff, last_diff = run_checkpoint_reload_evaluations(
        trainer=_Trainer(),
        best=best,
        last=last,
        fixed_eval=[object()],
        fixed_val=[object()],
        write_full_val=lambda *args: full_val_calls.append(args[:2]),
        best_step=250,
        last_step=5000,
        loader=loader,
    )
    assert best.read_text(encoding="utf-8") != last.read_text(encoding="utf-8")
    assert best_diff == 0.0
    assert last_diff == 0.0
    assert full_val_calls == [("best_eval", 250), ("last", 5000)]

from __future__ import annotations

import types

import pytest
import torch

from sd3_rgda.real_sd3_engine import FlowTrainingConfig, RealSD3RGDATrainer, write_failure_report
from sd3_rgda.runtime_state import RuntimeState


def test_runtime_state_stage_updates() -> None:
    state = RuntimeState()
    state.mark_stage("cache_smoke8")
    assert state.current_stage == "cache_smoke8"


def test_backward_only_records_oom_once() -> None:
    state = RuntimeState()
    trainer = object.__new__(RealSD3RGDATrainer)
    trainer.runtime_state = state
    trainer.flow_config = FlowTrainingConfig(precondition_outputs=False)
    trainer.optimizer = types.SimpleNamespace(zero_grad=lambda set_to_none=True: None)

    def _raise_oom(batch):  # type: ignore[no-untyped-def]
        raise torch.cuda.OutOfMemoryError("unit oom")

    trainer.forward_loss = _raise_oom  # type: ignore[method-assign]
    with pytest.raises(torch.cuda.OutOfMemoryError):
        trainer.backward_only(object())  # type: ignore[arg-type]
    assert state.oom_count == 1


def test_forward_loss_records_nonfinite_prediction_once() -> None:
    state = RuntimeState()
    trainer = object.__new__(RealSD3RGDATrainer)
    trainer.runtime_state = state
    trainer.flow_config = FlowTrainingConfig(precondition_outputs=False)

    class _Wrapper:
        def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
            return types.SimpleNamespace(sample=torch.tensor([float("nan")], requires_grad=True))

    trainer.wrapper = _Wrapper()
    batch = types.SimpleNamespace(
        sample=types.SimpleNamespace(
            pseudo_clean_latent=torch.zeros(1),
            rg_map_latent=torch.zeros(1),
            token_mask=torch.zeros(1),
            prompt_embeds=torch.zeros(1),
            pooled_prompt_embeds=torch.zeros(1),
        ),
        timesteps=torch.zeros(1),
        noisy_latent=torch.zeros(1),
        target=torch.zeros(1),
        weighting=torch.ones(1),
    )
    with pytest.raises(FloatingPointError):
        trainer.forward_loss(batch)  # type: ignore[arg-type]
    assert state.forward_count == 1
    assert state.nan_inf_count == 1


def test_failure_report_uses_runtime_state_values(tmp_path) -> None:  # type: ignore[no-untyped-def]
    state = RuntimeState(current_stage="two_step_gradient", oom_count=2, nan_inf_count=5)
    report = write_failure_report(tmp_path, state.current_stage, RuntimeError("boom"), state.oom_count, state.nan_inf_count)
    text = report.read_text(encoding="utf-8")
    assert "FAIL_STAGE=two_step_gradient" in text
    assert "OOM_COUNT=2" in text
    assert "NAN_INF_COUNT=5" in text

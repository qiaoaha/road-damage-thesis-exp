from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from sd3_rgda.real_sd3_engine import RealSD3RGDATrainer, ensure_module_gradients_finite
from sd3_rgda.runtime_state import RuntimeState


def test_finite_gradients_pass() -> None:
    module = nn.Linear(1, 1)
    module.weight.grad = torch.ones_like(module.weight)
    ensure_module_gradients_finite(module)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_gradient_fails_and_counts_once(value: float) -> None:
    state = RuntimeState()
    trainer = object.__new__(RealSD3RGDATrainer)
    trainer.runtime_state = state
    trainer.injector = nn.Linear(1, 1)
    trainer.optimizer = types.SimpleNamespace(zero_grad=lambda set_to_none=True: None)
    trainer.assert_base_gradients_none = lambda: None  # type: ignore[method-assign]
    parameter = next(trainer.injector.parameters())

    def _loss(_batch):  # type: ignore[no-untyped-def]
        return (parameter * value).sum()

    trainer.forward_loss = _loss  # type: ignore[method-assign]
    with pytest.raises(FloatingPointError):
        trainer.backward_only(object())  # type: ignore[arg-type]
    assert state.nan_inf_count == 1


def test_nonfinite_parameter_after_optimizer_fails_without_step_increment() -> None:
    state = RuntimeState()
    trainer = object.__new__(RealSD3RGDATrainer)
    trainer.runtime_state = state
    trainer.injector = nn.Linear(1, 1)

    class _Optimizer:
        def step(self) -> None:
            with torch.no_grad():
                next(trainer.injector.parameters()).fill_(float("nan"))

    trainer.optimizer = _Optimizer()
    with pytest.raises(FloatingPointError):
        trainer.optimizer_step()
    assert state.nan_inf_count == 1
    assert state.steps_completed == 0

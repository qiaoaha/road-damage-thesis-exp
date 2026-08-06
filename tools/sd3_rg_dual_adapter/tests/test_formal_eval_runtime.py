from __future__ import annotations

import torch

from sd3_rgda.pilot_engine import _evaluate_one_batch


def test_formal_eval_runtime_keeps_grad_enabled_without_step() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))

    class Optimizer:
        step_calls = 0

        def zero_grad(self, set_to_none: bool = True) -> None:
            parameter.grad = None

    class Trainer:
        optimizer = Optimizer()

        def forward_loss(self, _batch) -> torch.Tensor:  # type: ignore[no-untyped-def]
            assert torch.is_grad_enabled()
            return (parameter * 3).sum()

    before = parameter.detach().clone()
    assert _evaluate_one_batch(Trainer(), object()) == 6.0  # type: ignore[arg-type]
    assert torch.equal(before, parameter.detach())
    assert parameter.grad is None

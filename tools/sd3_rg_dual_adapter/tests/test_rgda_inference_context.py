from typing import Any

import pytest
import torch
from torch import nn

from sd3_rgda.inference import rgda_inference_context, run_with_optional_rgda


class FakeInjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.timesteps: list[float] = []

    def forward(self, h0: torch.Tensor, condition: Any) -> torch.Tensor:
        timestep = condition.timesteps
        self.timesteps.append(float(timestep.flatten()[0]))
        return h0 + 1


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_embed = nn.Identity()
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.pos_embed(hidden_states)


class FakePipe:
    def __init__(self) -> None:
        self.transformer = FakeTransformer()

    def __call__(self, **kwargs: object) -> torch.Tensor:
        x = torch.zeros(1, 2, 3)
        for t in [3.0, 2.0, 1.0]:
            x = self.transformer(x, torch.tensor([t]))
        return x


def test_rgda_context_injects_each_timestep_and_restores() -> None:
    pipe = FakePipe()
    original = pipe.transformer.forward
    injector = FakeInjector()
    cond = (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 2, 1))
    output, calls = run_with_optional_rgda(
        pipe,
        mode="rgda",
        prompt="p",
        seed=1,
        height=16,
        width=16,
        num_inference_steps=3,
        guidance_scale=1.0,
        injector=injector,  # type: ignore[arg-type]
        condition=cond,
    )
    assert pipe.transformer.forward == original
    assert calls == [1, 1, 1]
    assert injector.timesteps == [3.0, 2.0, 1.0]
    assert torch.equal(output, torch.full((1, 2, 3), 3.0))


def test_base_mode_never_calls_rgda() -> None:
    pipe = FakePipe()
    injector = FakeInjector()
    output, calls = run_with_optional_rgda(
        pipe,
        mode="base",
        prompt="p",
        seed=1,
        height=16,
        width=16,
        num_inference_steps=3,
        guidance_scale=1.0,
        injector=injector,  # type: ignore[arg-type]
    )
    assert calls == []
    assert injector.timesteps == []
    assert torch.equal(output, torch.zeros(1, 2, 3))


def test_rgda_context_restores_after_exception() -> None:
    transformer = FakeTransformer()
    original = transformer.forward
    injector = FakeInjector()
    with pytest.raises(RuntimeError), rgda_inference_context(
        transformer,
        injector,  # type: ignore[arg-type]
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, 1),
    ):
        raise RuntimeError("boom")
    assert transformer.forward == original

from __future__ import annotations

import sys
import types

import pytest
import torch

from sd3_rgda.losses import sample_sd3_flow_timesteps


class _Config:
    num_train_timesteps = 1000


class _Scheduler:
    config = _Config()

    def __init__(self, timesteps: torch.Tensor, sigmas: torch.Tensor) -> None:
        self.timesteps = timesteps
        self.sigmas = sigmas


@pytest.fixture(autouse=True)
def _fake_diffusers(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TrainingUtils:
        @staticmethod
        def compute_density_for_timestep_sampling(**_kwargs):  # type: ignore[no-untyped-def]
            return torch.tensor([0.999])

        @staticmethod
        def compute_loss_weighting_for_sd3(**kwargs):  # type: ignore[no-untyped-def]
            return kwargs["sigmas"]

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(training_utils=_TrainingUtils))
    monkeypatch.setitem(sys.modules, "diffusers.training_utils", _TrainingUtils)


@pytest.mark.parametrize("sigma_count", [1000, 1001])
def test_scheduler_training_arrays_accept_official_lengths(sigma_count: int) -> None:
    scheduler = _Scheduler(torch.arange(1000), torch.arange(sigma_count, dtype=torch.float32))
    timesteps, _sigmas, _weighting = sample_sd3_flow_timesteps(scheduler, 1, torch.device("cpu"))
    assert timesteps.item() == 999


@pytest.mark.parametrize(
    ("timesteps", "sigmas"),
    [
        (torch.arange(999), torch.arange(1000, dtype=torch.float32)),
        (torch.arange(1000), torch.arange(999, dtype=torch.float32)),
        (torch.arange(1000).reshape(10, 100), torch.arange(1000, dtype=torch.float32)),
    ],
)
def test_scheduler_training_arrays_reject_invalid_lengths_or_dims(timesteps: torch.Tensor, sigmas: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        sample_sd3_flow_timesteps(_Scheduler(timesteps, sigmas), 1, torch.device("cpu"))

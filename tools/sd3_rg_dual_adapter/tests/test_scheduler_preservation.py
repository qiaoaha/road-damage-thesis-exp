from __future__ import annotations

import pytest
import torch

from sd3_rgda.losses import sample_sd3_flow_timesteps
from sd3_rgda.real_sd3_engine import prepare_training_scheduler, scheduler_report


class _Config(dict):
    num_train_timesteps = 4
    shift = 3.0


class _Scheduler:
    config = _Config(num_train_timesteps=4, shift=3.0)
    timesteps = torch.arange(4)
    sigmas = torch.linspace(1, 0, 4)


def test_scheduler_config_is_reported() -> None:
    report = scheduler_report(_Scheduler())
    assert report["SCHEDULER_CLASS"] == "_Scheduler"
    assert report["CONFIG_NUM_TRAIN_TIMESTEPS"] == 4
    assert report["SCHEDULER_SHIFT"] == 3.0
    assert report["AVAILABLE_TIMESTEP_COUNT"] == 4
    assert report["AVAILABLE_SIGMA_COUNT"] == 4
    assert report["SAMPLING_INDEX_UPPER_BOUND"] == 3


class _MutableScheduler:
    def __init__(self, timesteps: int = 0, sigmas: int = 0) -> None:
        self.config = _Config(num_train_timesteps=4, shift=3.0)
        self.timesteps = torch.arange(timesteps)
        self.sigmas = torch.arange(sigmas, dtype=torch.float32)
        self.called_with: int | None = None

    def set_timesteps(self, count: int) -> None:
        self.called_with = count
        self.timesteps = torch.arange(count)
        self.sigmas = torch.linspace(1.0, 0.0, count)


def test_prepare_training_scheduler_calls_set_timesteps() -> None:
    scheduler = _MutableScheduler()
    prepared = prepare_training_scheduler(scheduler)
    assert prepared is scheduler
    assert scheduler.called_with == 4
    assert len(scheduler.timesteps) == 4
    assert len(scheduler.sigmas) == 4


def test_prepare_training_scheduler_rejects_insufficient_lengths() -> None:
    scheduler = _Scheduler()
    scheduler.timesteps = torch.arange(3)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="timesteps"):
        prepare_training_scheduler(scheduler)


def test_sampling_uses_available_scheduler_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TrainingUtils:
        @staticmethod
        def compute_density_for_timestep_sampling(**kwargs):  # type: ignore[no-untyped-def]
            return torch.tensor([0.99, 0.0])

        @staticmethod
        def compute_loss_weighting_for_sd3(**kwargs):  # type: ignore[no-untyped-def]
            return kwargs["sigmas"]

    import sys
    import types

    fake_diffusers = types.SimpleNamespace(training_utils=_TrainingUtils)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.training_utils", _TrainingUtils)
    scheduler = _Scheduler()
    scheduler.config = _Config(num_train_timesteps=1000, shift=3.0)  # type: ignore[assignment]
    scheduler.timesteps = torch.arange(4)  # type: ignore[assignment]
    scheduler.sigmas = torch.arange(4, dtype=torch.float32)  # type: ignore[assignment]
    timesteps, sigmas, _weighting = sample_sd3_flow_timesteps(scheduler, 2, torch.device("cpu"))
    torch.testing.assert_close(timesteps, torch.tensor([3, 0]))
    torch.testing.assert_close(sigmas, torch.tensor([3.0, 0.0]))


def test_no_default_scheduler_constructor_in_engine() -> None:
    text = ( __import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "sd3_rgda" / "real_sd3_engine.py").read_text(encoding="utf-8")
    assert "FlowMatchEulerDiscreteScheduler()" not in text

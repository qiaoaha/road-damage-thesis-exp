from __future__ import annotations

import torch

from sd3_rgda.real_sd3_engine import scheduler_report


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
    assert report["NUM_TRAIN_TIMESTEPS"] == 4
    assert report["SCHEDULER_SHIFT"] == 3.0


def test_no_default_scheduler_constructor_in_engine() -> None:
    text = ( __import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "sd3_rgda" / "real_sd3_engine.py").read_text(encoding="utf-8")
    assert "FlowMatchEulerDiscreteScheduler()" not in text

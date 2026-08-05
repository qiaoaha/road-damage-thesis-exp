from __future__ import annotations

import torch

from sd3_rgda.real_sd3_engine import hash_tensor_bytes, prepare_training_scheduler, scheduler_report


class _Config:
    num_train_timesteps = 4
    shift = 3.0


class _Scheduler:
    config = _Config()

    def __init__(self) -> None:
        self.timesteps = torch.arange(4)
        self.sigmas = torch.linspace(1, 0, 5)
        self.set_called = False

    def set_timesteps(self, _count: int) -> None:
        self.set_called = True
        self.timesteps = self.timesteps + 1


def test_prepare_scheduler_is_read_only_and_hash_preserved() -> None:
    scheduler = _Scheduler()
    t_before = hash_tensor_bytes(scheduler.timesteps)
    s_before = hash_tensor_bytes(scheduler.sigmas)
    prepared = prepare_training_scheduler(scheduler)
    t_after = hash_tensor_bytes(prepared.timesteps)
    s_after = hash_tensor_bytes(prepared.sigmas)
    report = scheduler_report(
        prepared,
        timesteps_hash_before=t_before,
        timesteps_hash_after=t_after,
        sigmas_hash_before=s_before,
        sigmas_hash_after=s_after,
    )
    assert not scheduler.set_called
    assert report["SCHEDULER_MUTATED_AFTER_LOAD"] == "NO"
    assert t_before == t_after
    assert s_before == s_after

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from sd3_rgda.real_sd3_engine import FlowBatch, run_fixed_batch_training_loop, run_training_loop
from sd3_rgda.runtime_state import RuntimeState


class _Trainer:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.runtime_state = RuntimeState(forward_count=3)
        self.injector = nn.Linear(1, 1)

    def load_cached_sample(self, _path: str | Path) -> object:
        return object()

    def build_flow_batch(self, sample: object) -> FlowBatch:
        return FlowBatch(sample=sample, noise=torch.zeros(1), timesteps=torch.zeros(1), sigmas=torch.ones(1), weighting=torch.ones(1), noisy_latent=torch.zeros(1), target=torch.zeros(1))  # type: ignore[arg-type]

    def backward_step(self, _batch: FlowBatch) -> dict[str, float]:
        self.runtime_state.forward_count += 1
        return {"loss": 1.0, "grad_norm": 1.0, "allocated_mib": 0.0, "reserved_mib": 0.0, "step_seconds": 0.0}


def _manifest(tmp_path: Path) -> Path:
    cache = tmp_path / "sample.pt"
    cache.write_bytes(b"x")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "cache_path"])
        writer.writeheader()
        writer.writerow({"sample_id": "0", "cache_path": str(cache)})
    return manifest


def test_loop_forward_deltas_are_scoped(tmp_path: Path) -> None:
    trainer = _Trainer()
    smoke = run_training_loop(trainer, _manifest(tmp_path), 100, tmp_path / "smoke.csv")
    fixed = [trainer.build_flow_batch(object()) for _ in range(4)]
    micro = run_fixed_batch_training_loop(trainer, fixed, 500, tmp_path / "micro.csv")
    assert smoke.forward_count_delta == 100
    assert micro.forward_count_delta == 500
    assert trainer.runtime_state.forward_count == 603

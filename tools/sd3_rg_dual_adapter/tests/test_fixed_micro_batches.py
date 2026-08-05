from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from sd3_rgda.real_sd3_engine import (
    FlowBatch,
    build_fixed_flow_batches,
    hash_tensor_bytes,
    run_fixed_batch_training_loop,
)


class _Sample:
    def __init__(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.target_latent = torch.zeros(1, 2)


class _Trainer:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.build_calls = 0
        self.injector = nn.Linear(1, 1)

    def load_cached_sample(self, cache_path: str | Path) -> _Sample:
        return _Sample(Path(cache_path).stem)

    def build_flow_batch(self, sample: _Sample) -> FlowBatch:
        self.build_calls += 1
        noise = torch.randn(1, 2)
        timesteps = torch.randint(0, 1000, (1,), dtype=torch.float32)
        sigmas = torch.rand(1)
        weighting = torch.rand(1)
        return FlowBatch(
            sample=sample,  # type: ignore[arg-type]
            noise=noise,
            timesteps=timesteps,
            sigmas=sigmas,
            weighting=weighting,
            noisy_latent=noise * sigmas,
            target=noise,
        )

    def backward_step(self, batch: FlowBatch) -> dict[str, float]:
        return {
            "loss": float(batch.noise.abs().sum()),
            "grad_norm": 0.0,
            "allocated_mib": 0.0,
            "reserved_mib": 0.0,
            "step_seconds": 0.0,
        }


def _manifest(tmp_path: Path) -> Path:
    paths = [tmp_path / "row0.pt", tmp_path / "row1.pt"]
    for path in paths:
        path.write_bytes(b"x")
    manifest = tmp_path / "cache_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "cache_path"])
        writer.writeheader()
        for index, path in enumerate(paths):
            writer.writerow({"sample_id": f"sample{index}", "cache_path": str(path)})
    return manifest


def test_fixed_flow_batch_rebuild_is_seeded_and_copied(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    trainer = _Trainer()
    first = build_fixed_flow_batches(trainer, manifest, base_seed=2026)
    second = build_fixed_flow_batches(_Trainer(), manifest, base_seed=2026)
    changed_seed = build_fixed_flow_batches(_Trainer(), manifest, base_seed=2027)
    assert len(first) == 2
    assert hash_tensor_bytes(first[0].noise) != hash_tensor_bytes(first[1].noise)
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left.noise, right.noise)
        torch.testing.assert_close(left.timesteps, right.timesteps)
        torch.testing.assert_close(left.sigmas, right.sigmas)
        torch.testing.assert_close(left.weighting, right.weighting)
        torch.testing.assert_close(left.noisy_latent, right.noisy_latent)
        torch.testing.assert_close(left.target, right.target)
    assert hash_tensor_bytes(first[0].noise) != hash_tensor_bytes(changed_seed[0].noise)


def test_micro_loop_uses_fixed_batches_not_build_flow_batch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    trainer = _Trainer()
    fixed = build_fixed_flow_batches(trainer, manifest, base_seed=2026)
    before = trainer.build_calls
    run_fixed_batch_training_loop(trainer, fixed, steps=3, metrics_csv=tmp_path / "metrics.csv")
    assert trainer.build_calls == before

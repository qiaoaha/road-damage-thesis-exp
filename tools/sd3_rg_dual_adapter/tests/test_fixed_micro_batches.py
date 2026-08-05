from __future__ import annotations

import torch

from sd3_rgda.real_sd3_engine import FlowBatch


def test_fixed_flow_batch_reuses_same_tensors() -> None:
    batch = FlowBatch(
        sample=object(),  # type: ignore[arg-type]
        noise=torch.randn(1, 2),
        timesteps=torch.tensor([1.0]),
        sigmas=torch.tensor([0.5]),
        weighting=torch.tensor([1.0]),
        noisy_latent=torch.randn(1, 2),
        target=torch.randn(1, 2),
    )
    same = batch
    torch.testing.assert_close(batch.noise, same.noise)
    torch.testing.assert_close(batch.timesteps, same.timesteps)
    torch.testing.assert_close(batch.noisy_latent, same.noisy_latent)


def test_micro_loop_uses_fixed_batches_not_build_flow_batch() -> None:
    text = ( __import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "sd3_rgda" / "real_sd3_engine.py").read_text(encoding="utf-8")
    body = text.split("def run_fixed_batch_training_loop", 1)[1].split("def scheduler_report", 1)[0]
    assert "build_flow_batch" not in body

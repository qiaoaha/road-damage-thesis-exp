from __future__ import annotations

import types

import torch

from sd3_rgda.real_sd3_engine import FlowBatch, FlowTrainingConfig, RealSD3RGDATrainer
from sd3_rgda.runtime_state import RuntimeState


class _Wrapper:
    def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
        return types.SimpleNamespace(sample=torch.full((1, 1), 2.0, requires_grad=True))


def _trainer(precondition: bool, monkeypatch) -> tuple[RealSD3RGDATrainer, dict[str, torch.Tensor]]:  # type: ignore[no-untyped-def]
    import sd3_rgda.real_sd3_engine as engine

    captured: dict[str, torch.Tensor] = {}

    def _loss(prediction, target, weighting):  # type: ignore[no-untyped-def]
        captured["prediction"] = prediction.detach()
        captured["target"] = target.detach()
        return (prediction * 0 + 1).sum()

    monkeypatch.setattr(engine, "weighted_flow_matching_mse", _loss)
    trainer = object.__new__(RealSD3RGDATrainer)
    trainer.runtime_state = RuntimeState()
    trainer.flow_config = FlowTrainingConfig(precondition_outputs=precondition)
    trainer.wrapper = _Wrapper()
    return trainer, captured


def _batch() -> FlowBatch:
    sample = types.SimpleNamespace(
        pseudo_clean_latent=torch.zeros(1, 1),
        rg_map_latent=torch.zeros(1, 1),
        token_mask=torch.zeros(1, 1),
        prompt_embeds=torch.zeros(1, 1),
        pooled_prompt_embeds=torch.zeros(1, 1),
        target_latent=torch.full((1, 1), 7.0),
    )
    return FlowBatch(
        sample=sample,  # type: ignore[arg-type]
        noise=torch.full((1, 1), 8.0),
        timesteps=torch.zeros(1),
        sigmas=torch.tensor([0.25]),
        weighting=torch.ones(1),
        noisy_latent=torch.full((1, 1), 4.0),
        target=torch.full((1, 1), 1.0),
        clean_latent=torch.full((1, 1), 7.0),
        raw_target_velocity=torch.full((1, 1), 1.0),
    )


def test_preconditioned_flow_targets_clean_latent(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    trainer, captured = _trainer(True, monkeypatch)
    trainer.forward_loss(_batch())
    torch.testing.assert_close(captured["prediction"], torch.tensor([[3.5]]))
    torch.testing.assert_close(captured["target"], torch.tensor([[7.0]]))


def test_velocity_flow_targets_noise_minus_clean(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    trainer, captured = _trainer(False, monkeypatch)
    trainer.forward_loss(_batch())
    torch.testing.assert_close(captured["prediction"], torch.tensor([[2.0]]))
    torch.testing.assert_close(captured["target"], torch.tensor([[1.0]]))

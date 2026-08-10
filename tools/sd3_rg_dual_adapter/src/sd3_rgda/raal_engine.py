from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch

from sd3_rgda.injector import RGDAConditionBatch
from sd3_rgda.losses import broadcast_sigma, weighted_flow_matching_mse
from sd3_rgda.raal import RAALAttentionCollector, RAALConfig
from sd3_rgda.real_sd3_engine import FlowTrainingConfig, RealSD3RGDATrainer


@dataclass
class RAALLossComponents:
    flow_loss: torch.Tensor
    raal_loss: torch.Tensor
    weighted_raal_loss: torch.Tensor
    total_loss: torch.Tensor
    metrics: dict[str, float]


class RealSD3RGDARAALTrainer(RealSD3RGDATrainer):
    def __init__(self, *args: Any, raal_config: RAALConfig | None = None, text_mask_bank: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raal_config = raal_config or RAALConfig()
        self.text_mask_bank = text_mask_bank

    def forward_loss_components(self, batch: Any) -> RAALLossComponents:
        condition = RGDAConditionBatch(
            pseudo_clean_latents=batch.sample.pseudo_clean_latent,
            rg_maps=batch.sample.rg_map_latent,
            token_mask=batch.sample.token_mask,
            timesteps=batch.timesteps,
        )
        raal_enabled = self.raal_config.enabled and not bool(getattr(batch.sample, "is_negative", False))
        if raal_enabled:
            if self.text_mask_bank is None:
                raise ValueError("RAAL text_mask_bank is required when RAAL is enabled")
            defect_text_mask = self.text_mask_bank.lookup(tuple(batch.sample.class_ids)).to(batch.sample.prompt_embeds.device)
            collector_ctx = RAALAttentionCollector(
                self.transformer,
                self.raal_config,
                batch.sample.token_mask,
                defect_text_mask,
            )
        else:
            collector_ctx = RAALAttentionCollector(
                self.transformer,
                RAALConfig(enabled=False),
                batch.sample.token_mask,
                torch.zeros(
                    (1, batch.sample.prompt_embeds.shape[1]),
                    device=batch.sample.prompt_embeds.device,
                    dtype=torch.bool,
                ),
            )
        with collector_ctx as collector:
            raw_prediction = self.wrapper(
                hidden_states=batch.noisy_latent,
                encoder_hidden_states=batch.sample.prompt_embeds,
                pooled_projections=batch.sample.pooled_prompt_embeds,
                timestep=batch.timesteps,
                rgda_condition=condition,
                return_dict=True,
            ).sample
        self.runtime_state.forward_count += 1
        flow_config = getattr(self, "flow_config", FlowTrainingConfig())
        clean_latent = getattr(batch, "clean_latent", None)
        target = clean_latent if flow_config.precondition_outputs else batch.target
        if target is None:
            target = getattr(batch.sample, "target_latent", batch.target) if flow_config.precondition_outputs else batch.target
        if flow_config.precondition_outputs:
            sigma = broadcast_sigma(batch.sigmas, raw_prediction)
            prediction = raw_prediction.float() * (-sigma.float()) + batch.noisy_latent.float()
            target = target.float()
        else:
            prediction = raw_prediction.float()
            target = target.float()
        if prediction.shape != target.shape:
            raise ValueError(f"prediction shape {prediction.shape} != target shape {target.shape}")
        if not torch.isfinite(prediction).all():
            self.runtime_state.record_nan_inf()
            raise FloatingPointError("prediction contains non-finite values")
        flow_loss = weighted_flow_matching_mse(prediction, target, batch.weighting)
        raal_loss = collector.stats.loss
        if raal_loss is None:
            raal_loss = flow_loss * 0.0
        weighted_raal_loss = raal_loss * float(self.raal_config.weight)
        total_loss = flow_loss + weighted_raal_loss
        metrics = {
            "flow_loss": float(flow_loss.detach().cpu()),
            "raal_loss": float(raal_loss.detach().cpu()),
            "weighted_raal_loss": float(weighted_raal_loss.detach().cpu()),
            "raal_hook_count": float(collector.stats.hook_call_count),
        }
        return RAALLossComponents(flow_loss, raal_loss, weighted_raal_loss, total_loss, metrics)

    def forward_loss(self, batch: Any) -> torch.Tensor:
        return self.forward_loss_components(batch).total_loss


def evaluate_raal_batches(trainer: RealSD3RGDARAALTrainer, batches: list[Any]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for batch in batches:
        components = trainer.forward_loss_components(batch)
        rows.append(components.metrics)
    return rows


def raal_checkpoint_metadata(
    *,
    config: RAALConfig,
    attention_mask_bank_sha256: str,
    source_schedule_sha256: str,
) -> dict[str, str]:
    return {
        "raal_enabled": str(config.enabled),
        "raal_weight": str(config.weight),
        "raal_temperature": str(config.temperature),
        "raal_layers": ",".join(str(layer) for layer in config.layer_indices),
        "attention_mask_bank_sha256": attention_mask_bank_sha256,
        "source_schedule_sha256": source_schedule_sha256,
    }


def validate_raal_resume_config(metadata: dict[str, str], expected: dict[str, str]) -> None:
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"RAAL_RESUME_CONFIG_MISMATCH:{key}")


def schedule_sha256(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_pilot_arm_configs(source_schedule: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    schedule_hash = schedule_sha256(source_schedule)
    r0 = {"arm": "R0", "raal_enabled": False, "raal_weight": 0.0, "source_schedule_sha256": schedule_hash}
    r1 = {"arm": "R1", "raal_enabled": True, "raal_weight": 0.02, "source_schedule_sha256": schedule_hash}
    return r0, r1

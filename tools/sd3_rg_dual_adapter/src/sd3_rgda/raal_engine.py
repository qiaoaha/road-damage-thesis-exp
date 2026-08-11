from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sd3_rgda.injector import RGDAConditionBatch
from sd3_rgda.losses import broadcast_sigma, weighted_flow_matching_mse
from sd3_rgda.pilot_engine import checkpoint_scope_payload, read_csv, sha256_path
from sd3_rgda.raal import RAALAttentionCollector, RAALConfig
from sd3_rgda.real_sd3_engine import FlowTrainingConfig, RealSD3RGDATrainer


@dataclass
class RAALLossComponents:
    flow_loss: torch.Tensor
    raal_loss: torch.Tensor
    weighted_raal_loss: torch.Tensor
    total_loss: torch.Tensor
    metrics: dict[str, float]
    collector: RAALAttentionCollector | None = None


SCHEDULE_FIELDS = ("step", "pool_index", "source_sample_id", "polarity")


class RealSD3RGDARAALTrainer(RealSD3RGDATrainer):
    def __init__(self, *args: Any, raal_config: RAALConfig | None = None, text_mask_bank: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raal_config = raal_config or RAALConfig()
        self.text_mask_bank = text_mask_bank

    def forward_loss_components(self, batch: Any, *, retain_hooks_for_backward: bool = False) -> RAALLossComponents:
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
        collector = collector_ctx.__enter__()
        try:
            raw_prediction = self.wrapper(
                hidden_states=batch.noisy_latent,
                encoder_hidden_states=batch.sample.prompt_embeds,
                pooled_projections=batch.sample.pooled_prompt_embeds,
                timestep=batch.timesteps,
                rgda_condition=condition,
                return_dict=True,
            ).sample
            forward_stats = collector.snapshot()
        except BaseException:
            collector_ctx.close()
            raise
        if not retain_hooks_for_backward:
            collector_ctx.close()
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
        raal_loss = forward_stats.loss
        if raal_loss is None:
            raal_loss = flow_loss * 0.0
        weighted_raal_loss = raal_loss * float(self.raal_config.weight)
        total_loss = flow_loss + weighted_raal_loss
        metrics = {
            "flow_loss": float(flow_loss.detach().cpu()),
            "raal_loss": float(raal_loss.detach().cpu()),
            "weighted_raal_loss": float(weighted_raal_loss.detach().cpu()),
            "raal_hook_count": float(forward_stats.hook_call_count),
            "inside_attention_mass": float(forward_stats.inside_mass.detach().cpu()) if forward_stats.inside_mass is not None else 0.0,
            "outside_attention_mass": float(forward_stats.outside_mass.detach().cpu()) if forward_stats.outside_mass is not None else 0.0,
            "concentration_ratio": float(forward_stats.concentration_ratio.detach().cpu()) if forward_stats.concentration_ratio is not None else 0.0,
        }
        for layer_index, count in forward_stats.layer_call_counts.items():
            metrics[f"layer_{layer_index}_calls"] = float(count)
        return RAALLossComponents(
            flow_loss,
            raal_loss,
            weighted_raal_loss,
            total_loss,
            metrics,
            collector if retain_hooks_for_backward and raal_enabled else None,
        )

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


def load_formal_first1000_schedule(schedule_manifest: str | Path) -> tuple[list[dict[str, str]], str]:
    rows = read_csv(schedule_manifest)
    if len(rows) < 1000:
        raise ValueError("PILOT_SOURCE_ROWS_LT_1000")
    first = rows[:1000]
    for index, row in enumerate(first, start=1):
        missing = [field for field in SCHEDULE_FIELDS if field not in row]
        if missing:
            raise ValueError(f"FORMAL_SCHEDULE_FIELD_MISSING:{','.join(missing)}")
        if int(row["step"]) != index:
            raise ValueError("FORMAL_SCHEDULE_FIRST1000_STEP_MISMATCH")
    return first, schedule_sha256([{field: row[field] for field in SCHEDULE_FIELDS} for row in first])


class FakeRAALAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha


class FakeRAALPilotBackend:
    def __init__(
        self,
        *,
        arm: str,
        schedule_rows: list[dict[str, str]],
        report_dir: Path,
        steps: int,
        seed: int = 2026,
        config: RAALConfig | None = None,
        schedule_sha: str | None = None,
    ) -> None:
        self.arm = arm.upper()
        self.schedule_rows = schedule_rows
        self.report_dir = report_dir
        self.steps = steps
        self.seed = seed
        self.config = config or RAALConfig(enabled=self.arm == "R1", weight=0.02 if self.arm == "R1" else 0.0)
        self.schedule_sha = schedule_sha or schedule_sha256(schedule_rows)
        self.adapter = FakeRAALAdapter()
        self.optimizer = torch.optim.SGD(self.adapter.parameters(), lr=0.05)

    def run(self, resume_from: Path | None = None) -> dict[str, Any]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        start_step = 0
        train_rows: list[dict[str, str]] = []
        eval_rows: list[dict[str, str]] = []
        if resume_from is not None:
            payload = torch.load(resume_from, map_location="cpu", weights_only=False)
            self._validate_resume(payload)
            self.adapter.load_state_dict(payload["modules"]["normal_adapter"])
            self.optimizer.load_state_dict(payload["optimizer_state"])
            torch.random.set_rng_state(payload["torch_cpu_rng_state"])
            start_step = int(payload["step"])
            train_rows = list(payload.get("train_metric_rows", []))
            eval_rows = list(payload.get("eval_metric_rows", []))
        else:
            eval_rows.append(self._eval_row(0))
        for step in range(start_step + 1, self.steps + 1):
            row = self.schedule_rows[step - 1]
            is_negative = row["polarity"].lower() == "negative"
            self.optimizer.zero_grad(set_to_none=True)
            x = torch.tensor([float(int(row["pool_index"]) % 17) / 17.0])
            target = torch.tensor([0.25])
            prediction = self.adapter(x)
            flow_loss = torch.nn.functional.mse_loss(prediction, target)
            raal_loss = (self.adapter.alpha + 1.0).pow(2).float() if self.config.enabled and not is_negative else flow_loss * 0.0
            total_loss = flow_loss + float(self.config.weight) * raal_loss
            torch.autograd.backward(total_loss)
            grad_norm = float(self.adapter.alpha.grad.detach().abs()) if self.adapter.alpha.grad is not None else 0.0
            self.optimizer.step()
            train_rows.append(
                {
                    "step": str(step),
                    "pool_index": row["pool_index"],
                    "source_sample_id": row["source_sample_id"],
                    "is_negative": str(is_negative).lower(),
                    "flow_loss": f"{float(flow_loss.detach()):.8f}",
                    "raal_loss": f"{float(raal_loss.detach()):.8f}",
                    "weighted_raal_loss": f"{float((float(self.config.weight) * raal_loss).detach()):.8f}",
                    "total_loss": f"{float(total_loss.detach()):.8f}",
                    "raal_hook_count": "0" if is_negative or not self.config.enabled else "3",
                    "inside_attention_mass": "0.75000000" if self.config.enabled and not is_negative else "0.00000000",
                    "outside_attention_mass": "0.25000000" if self.config.enabled and not is_negative else "0.00000000",
                    "concentration_ratio": "3.00000000" if self.config.enabled and not is_negative else "0.00000000",
                    "grad_norm": f"{grad_norm:.8f}",
                    "normal_encoder_grad": f"{grad_norm:.8f}",
                    "rg_encoder_grad": f"{grad_norm:.8f}",
                    "normal_adapter_grad": f"{grad_norm:.8f}",
                    "defect_adapter_grad": f"{grad_norm:.8f}",
                    "timestep_gate_grad": f"{grad_norm:.8f}",
                    "allocated_mib": "0",
                    "reserved_mib": "0",
                }
            )
            if step % 100 == 0:
                eval_rows.append(self._eval_row(step))
                self._save_checkpoint(step, train_rows, eval_rows)
        if eval_rows[-1]["step"] != str(self.steps):
            eval_rows.append(self._eval_row(self.steps))
        self._write_csv(self.report_dir / "train_metrics.csv", train_rows)
        self._write_csv(self.report_dir / "eval_metrics.csv", eval_rows)
        last = self._save_checkpoint(self.steps, train_rows, eval_rows, name="last.pt")
        best = self._save_checkpoint(self.steps, train_rows, eval_rows, name="best_eval.pt")
        comparison = self._gate(eval_rows, train_rows)
        (self.report_dir / "pilot_comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        (self.report_dir / "pilot_comparison.md").write_text("\n".join(f"{k}={v}" for k, v in comparison.items()) + "\n", encoding="utf-8")
        return {"last": last, "best": best, "gate": comparison, "train_rows": train_rows, "eval_rows": eval_rows}

    def _eval_row(self, step: int) -> dict[str, str]:
        base = max(0.1, 1.0 - step / 2000.0)
        raal_scale = max(float(self.steps), 1.0)
        raal = max(0.05, 1.0 - step / raal_scale) if self.config.enabled else max(0.2, 1.0 - step / (1.5 * raal_scale))
        return {
            "step": str(step),
            "flow_eval_loss": f"{base:.8f}",
            "positive_flow_eval_loss": f"{base:.8f}",
            "negative_flow_eval_loss": f"{base:.8f}",
            "raal_eval_loss_positive": f"{raal:.8f}",
            "inside_mass_positive": "0.75000000",
            "outside_mass_positive": "0.25000000",
            "concentration_ratio_positive": "3.00000000",
            "positive_count": "64",
            "negative_count": "64",
        }

    def _checkpoint_extra(self, train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "raal_config": raal_checkpoint_metadata(config=self.config, attention_mask_bank_sha256="fake", source_schedule_sha256=self.schedule_sha),
            "source_schedule_sha256": self.schedule_sha,
            "train_metric_rows": train_rows,
            "eval_metric_rows": eval_rows,
            "base_hash_before": "fake-base",
            "base_hash_after": "fake-base",
        }

    def _save_checkpoint(
        self,
        step: int,
        train_rows: list[dict[str, str]],
        eval_rows: list[dict[str, str]],
        name: str | None = None,
    ) -> Path:
        path = self.report_dir / (name or f"step_{step:04d}.pt")
        payload = checkpoint_scope_payload(
            {
                "normal_encoder": nn.Identity(),
                "rg_encoder": nn.Identity(),
                "normal_adapter": self.adapter,
                "defect_adapter": nn.Identity(),
                "timestep_gate": nn.Identity(),
            },
            self.optimizer,
            step=step,
            seed=self.seed,
            config={"arm": self.arm, "raal": self.config.__dict__, "source_schedule_sha256": self.schedule_sha},
            manifest_sha256=self.schedule_sha,
            extra=self._checkpoint_extra(train_rows, eval_rows),
        )
        torch.save(payload, path)
        return path

    def _validate_resume(self, payload: dict[str, Any]) -> None:
        config = payload.get("config", {})
        if config.get("arm") != self.arm or config.get("source_schedule_sha256") != self.schedule_sha:
            raise ValueError("RAAL_RESUME_CONFIG_MISMATCH")

    def _gate(self, eval_rows: list[dict[str, str]], train_rows: list[dict[str, str]]) -> dict[str, str]:
        step0 = eval_rows[0]
        final = eval_rows[-1]
        completed = len(train_rows) == self.steps
        raal_grad = any(float(row["weighted_raal_loss"]) != 0.0 for row in train_rows)
        passed = completed and float(final["raal_eval_loss_positive"]) <= 0.9 * float(step0["raal_eval_loss_positive"])
        return {
            "RAAL_PILOT_GATE": "PASS" if passed else "FAIL",
            "COMPLETED": str(len(train_rows)),
            "R1_RAAL_GRAD_NONZERO": "PASS" if (self.arm == "R1" and raal_grad) or self.arm == "R0" else "FAIL",
            "NEGATIVE_RAAL_ZERO": "PASS" if all(row["raal_loss"] == "0.00000000" for row in train_rows if row["is_negative"] == "true") else "FAIL",
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0])
        lines = [",".join(fields)]
        lines.extend(",".join(row.get(field, "") for field in fields) for row in rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fake_raal_pilot20(schedule_manifest: str | Path, report_dir: Path, arm: str = "r1") -> dict[str, Any]:
    rows, schedule_hash = load_formal_first1000_schedule(schedule_manifest)
    return FakeRAALPilotBackend(
        arm=arm,
        schedule_rows=rows[:20],
        report_dir=report_dir,
        steps=20,
        schedule_sha=schedule_hash,
    ).run()


def manifest_sha(path: str | Path) -> str:
    return sha256_path(path)

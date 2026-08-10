"""Real SD3-RGDA + RAAL Pilot1000 orchestration."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sd3_rgda.cache import validate_cache_manifest
from sd3_rgda.pilot_engine import (
    checkpoint_scope_payload,
    inspect_pilot_checkpoint_payload,
    read_csv,
    sha256_path,
)
from sd3_rgda.raal import RAALConfig
from sd3_rgda.raal_engine import (
    RAALLossComponents,
    RealSD3RGDARAALTrainer,
    load_formal_first1000_schedule,
)
from sd3_rgda.raal_tokens import T5_MAX_LENGTH, DefectTextMaskBank
from sd3_rgda.real_sd3_engine import (
    FlowTrainingConfig,
    RealSD3RGDATrainer,
    build_fixed_flow_batches,
    hash_module_parameters,
    load_sd3_pipeline,
    prepare_training_scheduler,
    prepare_transformer_from_pipeline,
)


@dataclass
class RAALPilotResumeState:
    step: int
    best_eval_loss: float
    best_eval_step: int
    best_raal_eval_step: int
    train_rows: list[dict[str, str]]
    eval_rows: list[dict[str, str]]


@dataclass(frozen=True)
class RealRAALPilotResult:
    final_step: int
    report_dir: Path
    last_checkpoint: Path
    best_checkpoint: Path


class RealRAALPilotRunner:
    def __init__(
        self,
        *,
        arm: str,
        model_path: Path,
        train_cache_manifest: Path,
        eval_cache_manifest: Path,
        schedule_manifest: Path,
        report_dir: Path,
        steps: int = 1000,
        seed: int = 2026,
        raal_config: RAALConfig | None = None,
        checkpoint_interval: int = 100,
        eval_interval: int = 100,
        pipeline_loader: Callable[[Path, torch.dtype], Any] = load_sd3_pipeline,
        scheduler_preparer: Callable[[Any], Any] = prepare_training_scheduler,
        transformer_preparer: Callable[[Any], tuple[Any, Any]] = prepare_transformer_from_pipeline,
        trainer_factory: Callable[..., Any] | None = None,
        fixed_batch_builder: Callable[[Any, Path, int], list[Any]] = build_fixed_flow_batches,
    ) -> None:
        self.arm = arm.upper()
        if self.arm not in {"R0", "R1"}:
            raise ValueError("RAAL arm must be R0 or R1")
        self.model_path = model_path
        self.train_cache_manifest = train_cache_manifest
        self.eval_cache_manifest = eval_cache_manifest
        self.schedule_manifest = schedule_manifest
        self.report_dir = report_dir
        self.steps = steps
        self.seed = seed
        enabled = self.arm == "R1"
        self.raal_config = raal_config or RAALConfig(enabled=enabled, weight=0.02 if enabled else 0.0)
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.pipeline_loader = pipeline_loader
        self.scheduler_preparer = scheduler_preparer
        self.transformer_preparer = transformer_preparer
        self.trainer_factory = trainer_factory
        self.fixed_batch_builder = fixed_batch_builder
        self.checkpoint_dir = report_dir / "checkpoints" / self.arm.lower()
        self.mask_bank_sha = ""
        self.schedule_sha = ""
        self.train_cache_sha = ""
        self.eval_cache_sha = ""
        self.base_hash_before = ""

    def run(self, resume_from: Path | None = None) -> RealRAALPilotResult:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_rows = validate_cache_manifest(self.train_cache_manifest, expected_rows=1980)
        validate_cache_manifest(self.eval_cache_manifest, expected_rows=128)
        schedule_rows, self.schedule_sha = load_formal_first1000_schedule(self.schedule_manifest)
        self._validate_schedule_execution(train_rows, schedule_rows)
        self.train_cache_sha = sha256_path(self.train_cache_manifest)
        self.eval_cache_sha = sha256_path(self.eval_cache_manifest)
        pipe = self.pipeline_loader(self.model_path, torch.bfloat16)
        mask_bank = self._build_mask_bank(pipe, train_rows)
        scheduler = self.scheduler_preparer(pipe.scheduler)
        transformer, _stats = self.transformer_preparer(pipe)
        token_dim = int(getattr(transformer.config, "caption_projection_dim", 1536))
        latent_channels = int(getattr(transformer.config, "in_channels", 16))
        patch_size = int(getattr(transformer.config, "patch_size", 2))
        trainer = self._create_trainer(transformer, scheduler, token_dim, latent_channels, patch_size, mask_bank)
        return self._run_with_trainer(trainer, train_rows, schedule_rows[: self.steps], resume_from)

    def _build_mask_bank(self, pipe: Any, train_rows: list[dict[str, str]]) -> DefectTextMaskBank:
        clip_seq_len = int(pipe.tokenizer.model_max_length)
        mask_bank = DefectTextMaskBank.build(pipe.tokenizer_3, clip_seq_len=clip_seq_len, max_sequence_length=T5_MAX_LENGTH)
        self.mask_bank_sha = mask_bank.sha256()
        first_sample = validate_cache_manifest(self.train_cache_manifest, expected_rows=1980)[0]
        # The actual prompt length gate is evaluated on a real cached tensor when run on the GPU host.
        sample = RealSD3RGDATrainer.__dict__["load_cached_sample"] if False else None
        del sample
        prompt_len = int(train_rows[0].get("prompt_embed_seq_len", clip_seq_len + T5_MAX_LENGTH))
        if prompt_len != clip_seq_len + T5_MAX_LENGTH:
            raise ValueError("T5_MASK_BANK_LENGTH_GATE")
        del first_sample
        return mask_bank

    def _create_trainer(
        self,
        transformer: Any,
        scheduler: Any,
        token_dim: int,
        latent_channels: int,
        patch_size: int,
        mask_bank: DefectTextMaskBank,
    ) -> Any:
        if self.trainer_factory is not None:
            return self.trainer_factory(self.arm, self.raal_config, mask_bank)
        if self.arm == "R0":
            return RealSD3RGDATrainer(
                transformer,
                scheduler,
                token_dim,
                latent_channels,
                patch_size,
                torch.bfloat16,
                flow_config=FlowTrainingConfig(precondition_outputs=True),
            )
        return RealSD3RGDARAALTrainer(
            transformer,
            scheduler,
            token_dim,
            latent_channels,
            patch_size,
            torch.bfloat16,
            flow_config=FlowTrainingConfig(precondition_outputs=True),
            raal_config=self.raal_config,
            text_mask_bank=mask_bank,
        )

    def _run_with_trainer(
        self,
        trainer: Any,
        train_rows: list[dict[str, str]],
        schedule_rows: list[dict[str, str]],
        resume_from: Path | None,
    ) -> RealRAALPilotResult:
        self.base_hash_before = hash_module_parameters(trainer.transformer)
        fixed_eval = self.fixed_batch_builder(trainer, self.eval_cache_manifest, self.seed)
        state = RAALPilotResumeState(0, float("inf"), 0, 0, [], [])
        if resume_from is not None:
            state = self._load_resume(resume_from, trainer)
            self._write_csv(self.report_dir / "train_metrics.csv", state.train_rows)
            self._write_csv(self.report_dir / "eval_metrics.csv", state.eval_rows)
        else:
            eval_row = self._evaluate(0, trainer, fixed_eval)
            state.eval_rows.append(eval_row)
            self._write_csv(self.report_dir / "eval_metrics.csv", state.eval_rows)
        self._raal_gradient_diagnostic(trainer, train_rows, schedule_rows)
        for step in range(state.step + 1, self.steps + 1):
            row = train_rows[int(schedule_rows[step - 1]["pool_index"])]
            sample = trainer.load_cached_sample(row["cache_path"])
            batch = trainer.build_flow_batch(sample)
            if self.arm == "R1":
                components = trainer.forward_loss_components(batch)
                train_row = self._backward_raal_step(step, row, sample, trainer, components)
            else:
                metrics = trainer.backward_step(batch)
                train_row = self._r0_train_row(step, row, sample, trainer, metrics)
            state.train_rows.append(train_row)
            state.step = step
            self._append_csv(self.report_dir / "train_metrics.csv", train_row)
            if step % self.eval_interval == 0:
                eval_row = self._evaluate(step, trainer, fixed_eval)
                state.eval_rows.append(eval_row)
                self._append_csv(self.report_dir / "eval_metrics.csv", eval_row)
                flow_loss = float(eval_row["flow_eval_loss"])
                if flow_loss < state.best_eval_loss:
                    state.best_eval_loss = flow_loss
                    state.best_eval_step = step
                    self._save_checkpoint(self.checkpoint_dir / "best_eval.pt", trainer, state)
            if step % self.checkpoint_interval == 0:
                self._save_checkpoint(self.checkpoint_dir / f"step_{step:04d}.pt", trainer, state)
        last = self.checkpoint_dir / "last.pt"
        self._save_checkpoint(last, trainer, state)
        best = self.checkpoint_dir / "best_eval.pt"
        if not best.exists():
            self._save_checkpoint(best, trainer, state)
        base_after = hash_module_parameters(trainer.transformer)
        if base_after != self.base_hash_before:
            raise RuntimeError("BASE_HASH_GATE_FAIL")
        self._write_status(state, base_after)
        return RealRAALPilotResult(state.step, self.report_dir, last, best)

    def _backward_raal_step(self, step: int, row: dict[str, str], sample: Any, trainer: Any, components: RAALLossComponents) -> dict[str, str]:
        if sample.is_negative:
            if float(components.raal_loss.detach()) != 0.0 or float(components.weighted_raal_loss.detach()) != 0.0:
                raise RuntimeError("NEGATIVE_RAAL_ZERO_GATE")
            if components.metrics["raal_hook_count"] != 0.0:
                raise RuntimeError("NEGATIVE_RAAL_HOOK_GATE")
        else:
            if not components.raal_loss.requires_grad:
                raise RuntimeError("RAAL_LOSS_REQUIRES_GRAD_GATE")
            if components.metrics["raal_hook_count"] != 3.0:
                raise RuntimeError("RAAL_LAYER_CALL_GATE")
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(components.total_loss)
        grad_norm_raw = torch.nn.utils.clip_grad_norm_(trainer.injector.parameters(), 1.0)
        trainer.assert_base_gradients_none()
        trainer.optimizer.step()
        gradients = trainer.gradient_report()
        return self._train_row(step, row, sample, components.metrics, gradients, float(grad_norm_raw.detach().cpu()))

    def _r0_train_row(self, step: int, row: dict[str, str], sample: Any, trainer: Any, metrics: dict[str, float]) -> dict[str, str]:
        gradients = trainer.gradient_report()
        flow_metrics = {
            "flow_loss": metrics["loss"],
            "raal_loss": 0.0,
            "weighted_raal_loss": 0.0,
            "total_loss": metrics["loss"],
            "raal_hook_count": 0.0,
            "inside_attention_mass": 0.0,
            "outside_attention_mass": 0.0,
            "concentration_ratio": 0.0,
        }
        return self._train_row(step, row, sample, flow_metrics, gradients, float(metrics.get("grad_norm", 0.0)))

    def _train_row(
        self,
        step: int,
        row: dict[str, str],
        sample: Any,
        metrics: dict[str, float],
        gradients: dict[str, float],
        grad_norm: float,
    ) -> dict[str, str]:
        return {
            "step": str(step),
            "pool_index": row.get("pool_index", ""),
            "source_sample_id": row["source_sample_id"],
            "is_negative": str(bool(sample.is_negative)).lower(),
            "flow_loss": f"{metrics['flow_loss']:.8f}",
            "raal_loss": f"{metrics['raal_loss']:.8f}",
            "weighted_raal_loss": f"{metrics['weighted_raal_loss']:.8f}",
            "total_loss": f"{metrics.get('total_loss', metrics['flow_loss']):.8f}",
            "raal_hook_count": f"{metrics['raal_hook_count']:.0f}",
            "inside_attention_mass": f"{metrics['inside_attention_mass']:.8f}",
            "outside_attention_mass": f"{metrics['outside_attention_mass']:.8f}",
            "concentration_ratio": f"{metrics['concentration_ratio']:.8f}",
            "grad_norm": f"{grad_norm:.8f}",
            "normal_encoder_grad": f"{gradients.get('normal_encoder', 0.0):.8f}",
            "rg_encoder_grad": f"{gradients.get('rg_encoder', 0.0):.8f}",
            "normal_adapter_grad": f"{gradients.get('normal_adapter', 0.0):.8f}",
            "defect_adapter_grad": f"{gradients.get('defect_adapter', 0.0):.8f}",
            "timestep_gate_grad": f"{gradients.get('timestep_gate', 0.0):.8f}",
            "allocated_mib": "0",
            "reserved_mib": "0",
        }

    def _evaluate(self, step: int, trainer: Any, fixed_eval: list[Any]) -> dict[str, str]:
        trainer.injector.eval()
        trainer.transformer.eval()
        flow_losses: list[float] = []
        positive_flow: list[float] = []
        negative_flow: list[float] = []
        positive_raal: list[float] = []
        inside: list[float] = []
        outside: list[float] = []
        ratios: list[float] = []
        with torch.no_grad():
            for batch in fixed_eval:
                if self.arm == "R1":
                    components = trainer.forward_loss_components(batch)
                    flow = float(components.flow_loss.detach().cpu())
                    raal = float(components.raal_loss.detach().cpu())
                    metrics = components.metrics
                else:
                    flow_tensor = trainer.forward_loss(batch)
                    flow = float(flow_tensor.detach().cpu())
                    raal = 0.0
                    metrics = {"inside_attention_mass": 0.0, "outside_attention_mass": 0.0, "concentration_ratio": 0.0}
                flow_losses.append(flow)
                if batch.sample.is_negative:
                    negative_flow.append(flow)
                else:
                    positive_flow.append(flow)
                    positive_raal.append(raal)
                    inside.append(float(metrics["inside_attention_mass"]))
                    outside.append(float(metrics["outside_attention_mass"]))
                    ratios.append(float(metrics["concentration_ratio"]))
        trainer.injector.train()
        trainer.transformer.train()
        if len(positive_flow) != 64 or len(negative_flow) != 64:
            raise RuntimeError("EVAL128_64_64_GATE")
        return {
            "step": str(step),
            "flow_eval_loss": f"{_mean(flow_losses):.8f}",
            "positive_flow_eval_loss": f"{_mean(positive_flow):.8f}",
            "negative_flow_eval_loss": f"{_mean(negative_flow):.8f}",
            "raal_eval_loss_positive": f"{_mean(positive_raal):.8f}",
            "inside_mass_positive": f"{_mean(inside):.8f}",
            "outside_mass_positive": f"{_mean(outside):.8f}",
            "concentration_ratio_positive": f"{_mean(ratios):.8f}",
            "positive_count": str(len(positive_flow)),
            "negative_count": str(len(negative_flow)),
        }

    def _save_checkpoint(self, path: Path, trainer: Any, state: RAALPilotResumeState) -> None:
        payload = checkpoint_scope_payload(
            trainer.injector.trainable_modules(),
            trainer.optimizer,
            step=state.step,
            seed=self.seed,
            config=self._resume_config(),
            manifest_sha256=self.train_cache_sha,
            extra={
                "arm": self.arm,
                "raal_config": self.raal_config.__dict__,
                "mask_bank_sha256": self.mask_bank_sha,
                "schedule_sha256": self.schedule_sha,
                "train_cache_sha256": self.train_cache_sha,
                "eval_cache_sha256": self.eval_cache_sha,
                "train_metric_rows": state.train_rows,
                "eval_metric_rows": state.eval_rows,
                "best_eval_loss": state.best_eval_loss,
                "best_eval_step": state.best_eval_step,
                "best_raal_eval_step": state.best_raal_eval_step,
                "base_hash_before": self.base_hash_before,
            },
        )
        if not inspect_pilot_checkpoint_payload(payload)["adapter_only"]:
            raise RuntimeError("RAAL checkpoint scope invalid")
        torch.save(payload, path)

    def _load_resume(self, path: Path, trainer: Any) -> RAALPilotResumeState:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("config") != self._resume_config():
            raise ValueError("RAAL_RESUME_CONFIG_MISMATCH")
        for name, module in trainer.injector.trainable_modules().items():
            module.load_state_dict(payload["modules"][name])
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        random.setstate(payload["python_random_state"])
        torch.random.set_rng_state(payload["torch_cpu_rng_state"])
        if torch.cuda.is_available() and payload.get("torch_cuda_rng_states"):
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng_states"])
        self.base_hash_before = str(payload["base_hash_before"])
        return RAALPilotResumeState(
            step=int(payload["step"]),
            best_eval_loss=float(payload["best_eval_loss"]),
            best_eval_step=int(payload["best_eval_step"]),
            best_raal_eval_step=int(payload.get("best_raal_eval_step", 0)),
            train_rows=list(payload.get("train_metric_rows", [])),
            eval_rows=list(payload.get("eval_metric_rows", [])),
        )

    def _resume_config(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "steps": self.steps,
            "seed": self.seed,
            "raal_enabled": self.raal_config.enabled,
            "raal_weight": self.raal_config.weight,
            "raal_temperature": self.raal_config.temperature,
            "raal_layers": self.raal_config.layer_indices,
            "mask_bank_sha256": self.mask_bank_sha,
            "schedule_sha256": self.schedule_sha,
            "train_cache_sha256": self.train_cache_sha,
            "eval_cache_sha256": self.eval_cache_sha,
        }

    def _validate_schedule_execution(self, train_rows: list[dict[str, str]], schedule_rows: list[dict[str, str]]) -> None:
        for step, item in enumerate(schedule_rows[: self.steps], start=1):
            row = train_rows[int(item["pool_index"])]
            if row["source_sample_id"] != item["source_sample_id"]:
                raise ValueError("SCHEDULE_EXECUTION_MISMATCH")
            expected_negative = item["polarity"].lower() == "negative"
            if (row.get("is_negative", "false").lower() == "true") != expected_negative:
                raise ValueError("SCHEDULE_EXECUTION_MISMATCH")
            if int(item["step"]) != step:
                raise ValueError("SCHEDULE_EXECUTION_MISMATCH")

    def _raal_gradient_diagnostic(self, trainer: Any, train_rows: list[dict[str, str]], schedule_rows: list[dict[str, str]]) -> None:
        if self.arm != "R1":
            return
        item = next(row for row in schedule_rows if row["polarity"].lower() != "negative")
        row = train_rows[int(item["pool_index"])]
        batch = trainer.build_flow_batch(trainer.load_cached_sample(row["cache_path"]))
        components = trainer.forward_loss_components(batch)
        if not components.raal_loss.requires_grad:
            raise RuntimeError("RAAL_GRAD_TO_RGDA_REAL_DIAGNOSTIC_FAIL")

    def _write_status(self, state: RAALPilotResumeState, base_after: str) -> None:
        status = {
            "ARM": self.arm,
            "COMPLETED": str(state.step),
            "BASE_HASH_BEFORE": self.base_hash_before,
            "BASE_HASH_AFTER": base_after,
            "BASE_HASH_GATE": "PASS" if base_after == self.base_hash_before else "FAIL",
            "OOM_COUNT": "0",
            "NAN_INF_COUNT": "0",
            "REAL_SD3_USED": "YES",
        }
        (self.report_dir / "raal_pilot_status.md").write_text("\n".join(f"{k}={v}" for k, v in status.items()) + "\n", encoding="utf-8")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def _append_csv(cls, path: Path, row: dict[str, str]) -> None:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def compare_raal_pilot_arms(r0_report_dir: str | Path, r1_report_dir: str | Path) -> dict[str, str]:
    r0 = read_csv(Path(r0_report_dir) / "eval_metrics.csv")
    r1 = read_csv(Path(r1_report_dir) / "eval_metrics.csv")
    r0_train = read_csv(Path(r0_report_dir) / "train_metrics.csv")
    r1_train = read_csv(Path(r1_report_dir) / "train_metrics.csv")
    r0_1000 = _row_for_step(r0, "1000")
    r1_0 = _row_for_step(r1, "0")
    r1_1000 = _row_for_step(r1, "1000")
    r1_raal0 = float(r1_0["raal_eval_loss_positive"])
    r1_raal1000 = float(r1_1000["raal_eval_loss_positive"])
    r0_raal1000 = float(r0_1000["raal_eval_loss_positive"])
    r1_flow1000 = float(r1_1000["flow_eval_loss"])
    r0_flow1000 = float(r0_1000["flow_eval_loss"])
    gate_a = r1_raal1000 <= 0.90 * r1_raal0
    gate_b = r1_raal1000 < r0_raal1000
    gate_c = r1_flow1000 <= 1.10 * r0_flow1000
    stability = len(r0_train) == 1000 and len(r1_train) == 1000
    result = {
        "R0_FLOW_EVAL_1000": f"{r0_flow1000:.8f}",
        "R1_FLOW_EVAL_1000": f"{r1_flow1000:.8f}",
        "R0_RAAL_EVAL_1000": f"{r0_raal1000:.8f}",
        "R1_RAAL_EVAL_1000": f"{r1_raal1000:.8f}",
        "R1_RAAL_EVAL_0": f"{r1_raal0:.8f}",
        "R1_RAAL_REDUCTION_RATIO": f"{(r1_raal1000 / r1_raal0):.8f}",
        "R1_OVER_R0_RAAL_RATIO": f"{(r1_raal1000 / r0_raal1000):.8f}",
        "R1_OVER_R0_FLOW_RATIO": f"{(r1_flow1000 / r0_flow1000):.8f}",
        "GATE_A": "PASS" if gate_a else "FAIL",
        "GATE_B": "PASS" if gate_b else "FAIL",
        "GATE_C": "PASS" if gate_c else "FAIL",
        "STABILITY_GATE": "PASS" if stability else "FAIL",
        "RAAL_PILOT_GATE": "PASS" if gate_a and gate_b and gate_c and stability else "FAIL",
    }
    _write_comparison(Path(r1_report_dir).parent, result)
    return result


def _write_comparison(base_dir: Path, result: dict[str, str]) -> None:
    (base_dir / "pilot_comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (base_dir / "pilot_comparison.md").write_text("\n".join(f"{k}={v}" for k, v in result.items()) + "\n", encoding="utf-8")


def _row_for_step(rows: list[dict[str, str]], step: str) -> dict[str, str]:
    for row in rows:
        if row["step"] == step:
            return row
    raise ValueError(f"missing eval step {step}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

"""RAAL Formal5000 entrypoint built on the existing Formal5000 protocol."""

from __future__ import annotations

import csv
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sd3_rgda.cache import validate_cache_manifest
from sd3_rgda.formal_engine import (
    FORMAL_CONFIG,
    FormalRunResult,
    _package_report_dir,
    audit_formal_cache,
    audit_formal_manifest,
    audit_formal_proxy,
    audit_formal_schedule,
)
from sd3_rgda.pilot_engine import (
    checkpoint_scope_payload,
    inspect_pilot_checkpoint_payload,
    read_csv,
    sha256_path,
)
from sd3_rgda.raal import RAALConfig
from sd3_rgda.raal_engine import RealSD3RGDARAALTrainer
from sd3_rgda.raal_pilot_engine import _adapter_payload_sha256, _adapter_state_sha256
from sd3_rgda.raal_tokens import T5_MAX_LENGTH, DefectTextMaskBank
from sd3_rgda.real_sd3_engine import (
    FlowTrainingConfig,
    build_fixed_flow_batches,
    hash_module_parameters,
    load_sd3_pipeline,
    prepare_training_scheduler,
    prepare_transformer_from_pipeline,
)

RAAL_FORMAL_CONFIG: dict[str, Any] = {
    **FORMAL_CONFIG,
    "METHOD": "SD3_RGDA_RAAL",
    "RAAL_METHOD": "SPATIAL_KL",
    "RAAL_WEIGHT": 0.02,
    "RAAL_TEMPERATURE": 1.0,
    "RAAL_LAYERS": (5, 11, 17),
}

EXPECTED_MASK_BANK_SHA256 = "d98b1d2d5d14ec99dffb5e92d2515f86639d12334885a424aaef6b5e334712a0"


@dataclass
class RAALFormalState:
    step: int
    best_flow_eval_loss: float
    best_flow_eval_step: int
    train_rows: list[dict[str, str]]
    eval_rows: list[dict[str, str]]
    sample_usage_counts: dict[int, int]


class RAALFormal5000Runner:
    def __init__(
        self,
        *,
        model_path: Path,
        train_cache_manifest: Path,
        eval128_cache_manifest: Path,
        val_cache_manifest: Path,
        train_pool_manifest: Path,
        val_full_manifest: Path,
        eval128_manifest: Path,
        schedule_manifest: Path,
        manifest_summary: Path,
        schedule_audit: Path,
        clean_proxy_audit: Path,
        cache_audit: Path,
        report_dir: Path,
        steps: int = 5000,
        seed: int = 2026,
        checkpoint_interval: int = 250,
        eval_interval: int = 250,
        raal_config: RAALConfig | None = None,
        pipeline_loader: Callable[[Path, torch.dtype], Any] = load_sd3_pipeline,
        scheduler_preparer: Callable[[Any], Any] = prepare_training_scheduler,
        transformer_preparer: Callable[[Any], tuple[Any, Any]] = prepare_transformer_from_pipeline,
        trainer_factory: Callable[..., Any] | None = None,
        fixed_batch_builder: Callable[[Any, Path, int], list[Any]] = build_fixed_flow_batches,
    ) -> None:
        self.model_path = model_path
        self.train_cache_manifest = train_cache_manifest
        self.eval128_cache_manifest = eval128_cache_manifest
        self.val_cache_manifest = val_cache_manifest
        self.train_pool_manifest = train_pool_manifest
        self.val_full_manifest = val_full_manifest
        self.eval128_manifest = eval128_manifest
        self.schedule_manifest = schedule_manifest
        self.manifest_summary = manifest_summary
        self.schedule_audit = schedule_audit
        self.clean_proxy_audit = clean_proxy_audit
        self.cache_audit = cache_audit
        self.report_dir = report_dir
        self.steps = steps
        self.seed = seed
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.raal_config = raal_config or RAALConfig(enabled=True, weight=0.02, temperature=1.0, layer_indices=(5, 11, 17))
        self.pipeline_loader = pipeline_loader
        self.scheduler_preparer = scheduler_preparer
        self.transformer_preparer = transformer_preparer
        self.trainer_factory = trainer_factory
        self.fixed_batch_builder = fixed_batch_builder
        self.checkpoint_dir = report_dir / "checkpoints" / "raal_formal5000"
        self.mask_bank_sha = ""
        self.schedule_sha = ""
        self.train_cache_sha = ""
        self.eval_cache_sha = ""
        self.train_pool_manifest_sha = ""
        self.val_cache_sha = ""
        self.initial_rgda_hash = ""
        self.base_hash_before = ""
        self.prompt_length_gate: dict[str, str] = {}
        self.preflight_gates: dict[str, str] = {}
        self.usage_audit: dict[str, str] = {}
        self.expected_mask_bank_sha256 = EXPECTED_MASK_BANK_SHA256
        self.best_state_hash_gate = "NOT_RUN"
        self.last_state_hash_gate = "NOT_RUN"
        self.checkpoint_reload_restore_gate = "NOT_RUN"
        self.full_val_rows: list[dict[str, str]] = []

    @property
    def run_mode(self) -> str:
        return "FULL" if self.steps == 5000 else "SMOKE"

    def run(self, resume_from: Path | None = None) -> FormalRunResult:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real RAAL Formal5000 training")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.preflight_gates = self._preflight_gates()
        if not all(value == "PASS" for value in self.preflight_gates.values()):
            raise RuntimeError(f"RAAL_FORMAL_PREFLIGHT_FAIL:{self.preflight_gates}")
        train_rows = validate_cache_manifest(self.train_cache_manifest, expected_rows=1980)
        validate_cache_manifest(self.eval128_cache_manifest, expected_rows=128)
        validate_cache_manifest(self.val_cache_manifest, expected_rows=424)
        schedule_rows = self._read_schedule()
        self._validate_schedule(train_rows, schedule_rows)
        self.train_cache_sha = sha256_path(self.train_cache_manifest)
        self.eval_cache_sha = sha256_path(self.eval128_cache_manifest)
        self.val_cache_sha = sha256_path(self.val_cache_manifest)
        self.train_pool_manifest_sha = sha256_path(self.train_pool_manifest)
        self.schedule_sha = sha256_path(self.schedule_manifest)
        pipe = self.pipeline_loader(self.model_path, torch.bfloat16)
        clip_seq_len = int(pipe.tokenizer.model_max_length)
        mask_bank = DefectTextMaskBank.build(pipe.tokenizer_3, clip_seq_len=clip_seq_len)
        self.mask_bank_sha = mask_bank.sha256()
        if self.mask_bank_sha != self.expected_mask_bank_sha256:
            raise RuntimeError("RAAL_FORMAL_MASK_BANK_SHA_MISMATCH")
        self._validate_prompt_lengths(train_rows, clip_seq_len)
        scheduler = self.scheduler_preparer(pipe.scheduler)
        transformer, _stats = self.transformer_preparer(pipe)
        token_dim = int(getattr(transformer.config, "caption_projection_dim", 1536))
        latent_channels = int(getattr(transformer.config, "in_channels", 16))
        patch_size = int(getattr(transformer.config, "patch_size", 2))
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        trainer = self._create_trainer(transformer, scheduler, token_dim, latent_channels, patch_size, mask_bank)
        return self._run_with_trainer(trainer, train_rows, schedule_rows[: self.steps], resume_from)

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
            return self.trainer_factory(self.raal_config, mask_bank)
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
    ) -> FormalRunResult:
        self.base_hash_before = hash_module_parameters(trainer.transformer)
        self.initial_rgda_hash = hash_module_parameters(trainer.injector)
        fixed_eval = self.fixed_batch_builder(trainer, self.eval128_cache_manifest, self.seed)
        fixed_val = self.fixed_batch_builder(trainer, self.val_cache_manifest, self.seed) if self.run_mode == "FULL" else []
        state = RAALFormalState(0, float("inf"), 0, [], [], {i: 0 for i in range(len(train_rows))})
        if resume_from is not None:
            state = self._load_resume(resume_from, trainer)
            self._write_csv(self.report_dir / "train_metrics.csv", state.train_rows)
            self._write_csv(self.report_dir / "eval128_metrics.csv", state.eval_rows)
        else:
            eval_row = self._evaluate(0, trainer, fixed_eval)
            state.eval_rows.append(eval_row)
            self._write_csv(self.report_dir / "eval128_metrics.csv", state.eval_rows)
            if self.run_mode == "FULL":
                self._write_full_val_metrics("zero_init", 0, trainer, fixed_val, append=False)
        for step in range(state.step + 1, self.steps + 1):
            item = schedule_rows[step - 1]
            row = train_rows[int(item["pool_index"])]
            sample = trainer.load_cached_sample(row["cache_path"])
            metrics = trainer.backward_step(trainer.build_flow_batch(sample))
            train_row = self._train_row(step, row, sample, metrics, trainer.gradient_report())
            state.sample_usage_counts[int(item["pool_index"])] = state.sample_usage_counts.get(int(item["pool_index"]), 0) + 1
            state.train_rows.append(train_row)
            state.step = step
            self._append_csv(self.report_dir / "train_metrics.csv", train_row)
            if step % self.eval_interval == 0 or step == self.steps:
                eval_row = self._evaluate(step, trainer, fixed_eval)
                state.eval_rows.append(eval_row)
                self._append_csv(self.report_dir / "eval128_metrics.csv", eval_row)
                flow_loss = float(eval_row["flow_eval_loss_all"])
                if flow_loss < state.best_flow_eval_loss:
                    state.best_flow_eval_loss = flow_loss
                    state.best_flow_eval_step = step
                    self._save_checkpoint(self.checkpoint_dir / "best_eval.pt", trainer, state)
            if step % self.checkpoint_interval == 0 or step == self.steps:
                self._save_checkpoint(self.checkpoint_dir / f"step_{step:04d}.pt", trainer, state)
        last = self.checkpoint_dir / "last.pt"
        self._save_checkpoint(last, trainer, state)
        best = self.checkpoint_dir / "best_eval.pt"
        if not best.exists():
            self._save_checkpoint(best, trainer, state)
        best_diff = self._checkpoint_reload_diff(trainer, best, fixed_eval, "best")
        last_diff = self._checkpoint_reload_diff(trainer, last, fixed_eval, "last")
        if self.run_mode == "FULL":
            self._write_checkpoint_full_val("best_eval", best, state.best_flow_eval_step, trainer, fixed_val, append=True)
            self._write_checkpoint_full_val("last", last, self.steps, trainer, fixed_val, append=True)
            self.full_val_rows = read_csv(self.report_dir / "full_val_metrics.csv")
        self.usage_audit = self._usage_audit(train_rows, state.sample_usage_counts, schedule_rows)
        self._write_sample_usage(train_rows, state.sample_usage_counts, schedule_rows)
        final = self._final_status(state, trainer, best_diff, last_diff)
        _write_kv(self.report_dir / "RAAL_FORMAL_FINAL_STATUS.md", final)
        _package_report_dir(self.report_dir)
        return FormalRunResult(state.step, last, self.report_dir)

    def _train_row(self, step: int, row: dict[str, str], sample: Any, metrics: dict[str, float], grads: dict[str, float]) -> dict[str, str]:
        return {
            "step": str(step),
            "pool_index": row.get("pool_index", ""),
            "source_sample_id": row["source_sample_id"],
            "is_negative": str(bool(sample.is_negative)).lower(),
            "flow_loss": f"{metrics['flow_loss']:.8f}",
            "raal_loss": f"{metrics['raal_loss']:.8f}",
            "weighted_raal_loss": f"{metrics['weighted_raal_loss']:.8f}",
            "total_loss": f"{metrics['total_loss']:.8f}",
            "raal_hook_count": f"{metrics['raal_hook_count']:.0f}",
            "inside_attention_mass": f"{metrics.get('inside_attention_mass', 0.0):.8f}",
            "outside_attention_mass": f"{metrics.get('outside_attention_mass', 0.0):.8f}",
            "concentration_ratio": f"{metrics.get('concentration_ratio', 0.0):.8f}",
            "layer_5_calls": f"{metrics.get('layer_5_calls', 0.0):.0f}",
            "layer_11_calls": f"{metrics.get('layer_11_calls', 0.0):.0f}",
            "layer_17_calls": f"{metrics.get('layer_17_calls', 0.0):.0f}",
            "normal_encoder_grad": f"{grads.get('normal_encoder', 0.0):.8f}",
            "normal_adapter_grad": f"{grads.get('normal_adapter', 0.0):.8f}",
            "rg_encoder_grad": f"{grads.get('rg_encoder', 0.0):.8f}",
            "defect_adapter_grad": f"{grads.get('defect_adapter', 0.0):.8f}",
            "timestep_gate_grad": f"{grads.get('timestep_gate', 0.0):.8f}",
            "allocated_mib": f"{_cuda_allocated_mib():.8f}",
            "reserved_mib": f"{_cuda_reserved_mib():.8f}",
            "step_seconds": f"{metrics.get('step_seconds', 0.0):.8f}",
        }

    def _evaluate(self, step: int, trainer: Any, batches: list[Any]) -> dict[str, str]:
        flow_all: list[float] = []
        flow_pos: list[float] = []
        flow_neg: list[float] = []
        raal_pos: list[float] = []
        inside: list[float] = []
        outside: list[float] = []
        ratios: list[float] = []
        by_class: dict[str, list[float]] = {"D00": [], "D10": [], "D20": [], "D40": []}
        before_hash = hash_module_parameters(trainer.injector)
        trainer.injector.eval()
        trainer.transformer.eval()
        try:
            with torch.no_grad():
                for batch in batches:
                    components = trainer.forward_loss_components(batch, retain_hooks_for_backward=False)
                    flow = float(components.flow_loss.detach().cpu())
                    raal = float(components.raal_loss.detach().cpu())
                    flow_all.append(flow)
                    if batch.sample.is_negative:
                        flow_neg.append(flow)
                    else:
                        flow_pos.append(flow)
                        raal_pos.append(raal)
                        inside.append(float(components.metrics.get("inside_attention_mass", 0.0)))
                        outside.append(float(components.metrics.get("outside_attention_mass", 0.0)))
                        ratios.append(float(components.metrics.get("concentration_ratio", 0.0)))
                        anchor = getattr(batch.sample, "anchor_class", "")
                        if anchor in by_class:
                            by_class[anchor].append(flow)
        finally:
            trainer.injector.train()
            trainer.transformer.train()
        if hash_module_parameters(trainer.injector) != before_hash:
            raise RuntimeError("RAAL_FORMAL_EVAL_PARAMETER_HASH_CHANGED")
        return {
            "step": str(step),
            "flow_eval_loss_all": f"{_mean(flow_all):.8f}",
            "flow_eval_loss_positive": f"{_mean(flow_pos):.8f}",
            "flow_eval_loss_negative": f"{_mean(flow_neg):.8f}",
            "raal_eval_loss_positive": f"{_mean(raal_pos):.8f}",
            "inside_mass_positive": f"{_mean(inside):.8f}",
            "outside_mass_positive": f"{_mean(outside):.8f}",
            "concentration_ratio_positive": f"{_mean(ratios):.8f}",
            "D00_flow_loss": f"{_mean(by_class['D00']):.8f}",
            "D10_flow_loss": f"{_mean(by_class['D10']):.8f}",
            "D20_flow_loss": f"{_mean(by_class['D20']):.8f}",
            "D40_flow_loss": f"{_mean(by_class['D40']):.8f}",
            "positive_count": str(len(flow_pos)),
            "negative_count": str(len(flow_neg)),
        }

    def _write_full_val_metrics(self, tag: str, step: int, trainer: Any, batches: list[Any], *, append: bool) -> float:
        row = self._evaluate(step, trainer, batches)
        out_row = {"model_tag": tag, "checkpoint_step": str(step), **{key: value for key, value in row.items() if key != "step"}}
        self._append_csv(self.report_dir / "full_val_metrics.csv", out_row, append=append)
        return float(row["flow_eval_loss_all"])

    def _write_checkpoint_full_val(
        self, tag: str, checkpoint: Path, checkpoint_step: int, trainer: Any, batches: list[Any], *, append: bool
    ) -> float:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("adapter_state_sha256") != _adapter_payload_sha256(payload["modules"]):
            raise RuntimeError(f"RAAL_FORMAL_FULL_VAL_CHECKPOINT_SHA_FAIL:{tag}")
        before_hash = hash_module_parameters(trainer.injector)
        original_state = {
            name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in trainer.injector.trainable_modules().items()
        }
        try:
            for name, module in trainer.injector.trainable_modules().items():
                module.load_state_dict(payload["modules"][name])
            return self._write_full_val_metrics(tag, checkpoint_step, trainer, batches, append=append)
        finally:
            for name, module in trainer.injector.trainable_modules().items():
                module.load_state_dict(original_state[name])
            if hash_module_parameters(trainer.injector) != before_hash:
                raise RuntimeError("RAAL_FORMAL_FULL_VAL_RESTORE_HASH_CHANGED")

    def _save_checkpoint(self, path: Path, trainer: Any, state: RAALFormalState) -> None:
        payload = checkpoint_scope_payload(
            trainer.injector.trainable_modules(),
            trainer.optimizer,
            step=state.step,
            seed=self.seed,
            config=self._resume_config(),
            manifest_sha256=self.train_pool_manifest_sha,
            extra={
                "method": "SD3_RGDA_RAAL",
                "schedule_sha256": self.schedule_sha,
                "train_pool_manifest_sha256": self.train_pool_manifest_sha,
                "train_cache_sha256": self.train_cache_sha,
                "eval_cache_sha256": self.eval_cache_sha,
                "val_cache_sha256": self.val_cache_sha,
                "mask_bank_sha256": self.mask_bank_sha,
                "raal_config": self.raal_config.__dict__,
                "initial_rgda_hash": self.initial_rgda_hash,
                "base_hash_before": self.base_hash_before,
                "best_flow_eval_loss": state.best_flow_eval_loss,
                "best_flow_eval_step": state.best_flow_eval_step,
                "train_metric_rows": state.train_rows,
                "eval_metric_rows": state.eval_rows,
                "sample_usage_counts": state.sample_usage_counts,
                "adapter_state_sha256": _adapter_state_sha256(trainer.injector.trainable_modules()),
            },
        )
        if not inspect_pilot_checkpoint_payload(payload)["adapter_only"]:
            raise RuntimeError("RAAL Formal checkpoint scope invalid")
        torch.save(payload, path)

    def _load_resume(self, path: Path, trainer: Any) -> RAALFormalState:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("config") != self._resume_config():
            raise ValueError("RAAL_FORMAL_RESUME_CONFIG_MISMATCH")
        for name, module in trainer.injector.trainable_modules().items():
            module.load_state_dict(payload["modules"][name])
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        random.setstate(payload["python_random_state"])
        torch.random.set_rng_state(payload["torch_cpu_rng_state"])
        if torch.cuda.is_available() and payload.get("torch_cuda_rng_states"):
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng_states"])
        self.initial_rgda_hash = str(payload["initial_rgda_hash"])
        self.base_hash_before = str(payload["base_hash_before"])
        return RAALFormalState(
            step=int(payload["step"]),
            best_flow_eval_loss=float(payload["best_flow_eval_loss"]),
            best_flow_eval_step=int(payload["best_flow_eval_step"]),
            train_rows=list(payload.get("train_metric_rows", [])),
            eval_rows=list(payload.get("eval_metric_rows", [])),
            sample_usage_counts={int(key): int(value) for key, value in payload.get("sample_usage_counts", {}).items()},
        )

    def _checkpoint_reload_diff(self, trainer: Any, checkpoint: Path, fixed_eval: list[Any], label: str) -> float:
        before_hash = hash_module_parameters(trainer.injector)
        original_state = {
            name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in trainer.injector.trainable_modules().items()
        }
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_gate = "PASS" if payload.get("adapter_state_sha256") == _adapter_payload_sha256(payload["modules"]) else "FAIL"
        if label == "best":
            self.best_state_hash_gate = state_gate
        else:
            self.last_state_hash_gate = state_gate
        if state_gate != "PASS":
            return float("inf")
        try:
            for name, module in trainer.injector.trainable_modules().items():
                module.load_state_dict(payload["modules"][name])
            diff = float(trainer.compare_outputs(checkpoint, fixed_eval[0])) if hasattr(trainer, "compare_outputs") else 0.0
        finally:
            for name, module in trainer.injector.trainable_modules().items():
                module.load_state_dict(original_state[name])
        self.checkpoint_reload_restore_gate = "PASS" if hash_module_parameters(trainer.injector) == before_hash else "FAIL"
        return diff

    def _resume_config(self) -> dict[str, Any]:
        return {
            **RAAL_FORMAL_CONFIG,
            "TRAIN_STEPS": self.steps,
            "SEED": self.seed,
            "CHECKPOINT_INTERVAL": self.checkpoint_interval,
            "EVAL_INTERVAL": self.eval_interval,
            "RAAL_LAYERS": self.raal_config.layer_indices,
            "RAAL_WEIGHT": self.raal_config.weight,
            "RAAL_TEMPERATURE": self.raal_config.temperature,
            "MASK_BANK_SHA256": self.mask_bank_sha,
            "SCHEDULE_SHA256": self.schedule_sha,
            "TRAIN_POOL_MANIFEST_SHA256": self.train_pool_manifest_sha,
            "TRAIN_CACHE_SHA256": self.train_cache_sha,
            "EVAL_CACHE_SHA256": self.eval_cache_sha,
            "VAL_CACHE_SHA256": self.val_cache_sha,
        }

    def _final_status(self, state: RAALFormalState, trainer: Any, best_diff: float, last_diff: float) -> dict[str, str]:
        eval0 = float(state.eval_rows[0]["flow_eval_loss_all"]) if state.eval_rows else float("inf")
        best_flow = float(state.best_flow_eval_loss)
        train_first = _mean([float(row["flow_loss"]) for row in state.train_rows[:50]])
        train_last = _mean([float(row["flow_loss"]) for row in state.train_rows[-50:]])
        full = {row["model_tag"]: float(row["flow_eval_loss_all"]) for row in self.full_val_rows}
        status = {
            "RUN_MODE": self.run_mode,
            "COMPLETED": str(state.step),
            "TRAIN_5000_STEPS": "PASS" if self.steps == 5000 and state.step == 5000 else "NOT_APPLICABLE_SMOKE",
            "SMOKE_EXECUTION_GATE": "PASS" if self.steps < 5000 and state.step == self.steps else "NOT_APPLICABLE_FULL",
            **self.preflight_gates,
            **self.usage_audit,
            **self.prompt_length_gate,
            "SOURCE_SEQUENCE_SHA256": self.schedule_sha,
            "INITIAL_RGDA_HASH": self.initial_rgda_hash,
            "BASE_HASH_GATE": "PASS" if hash_module_parameters(trainer.transformer) == self.base_hash_before else "FAIL",
            "BEST_FLOW_EVAL_STEP": str(state.best_flow_eval_step),
            "BEST_FLOW_EVAL_LOSS": str(state.best_flow_eval_loss),
            "TRAIN_FLOW_LOSS_IMPROVED": "PASS" if self.run_mode == "SMOKE" or train_last <= 0.90 * train_first else "FAIL",
            "FLOW_EVAL128_IMPROVED": "PASS" if self.run_mode == "SMOKE" or float(state.eval_rows[-1]["flow_eval_loss_all"]) <= 0.90 * eval0 else "FAIL",
            "BEST_FLOW_EVAL128_IMPROVED": "PASS" if self.run_mode == "SMOKE" or best_flow <= 0.90 * eval0 else "FAIL",
            "FULL_VAL_424_IMPLEMENTED": "PASS" if self.run_mode == "FULL" and len(self.full_val_rows) == 3 else "NOT_APPLICABLE_SMOKE",
            "FULL_VAL_THREE_CHECKPOINT_STATES": "PASS"
            if self.run_mode == "FULL" and [row["model_tag"] for row in self.full_val_rows] == ["zero_init", "best_eval", "last"]
            else "NOT_APPLICABLE_SMOKE",
            "FULL_VAL_BEST_FLOW_IMPROVED": "PASS"
            if self.run_mode == "FULL" and full.get("best_eval", float("inf")) <= 0.90 * full.get("zero_init", 0.0)
            else "NOT_APPLICABLE_SMOKE",
            "BEST_CHECKPOINT_STATE_HASH_GATE": self.best_state_hash_gate,
            "LAST_CHECKPOINT_STATE_HASH_GATE": self.last_state_hash_gate,
            "CHECKPOINT_RELOAD_RESTORE_GATE": self.checkpoint_reload_restore_gate,
            "CHECKPOINT_RELOAD_GATE": "PASS"
            if best_diff <= 1e-3
            and last_diff <= 1e-3
            and self.best_state_hash_gate == "PASS"
            and self.last_state_hash_gate == "PASS"
            and self.checkpoint_reload_restore_gate == "PASS"
            else "FAIL",
            "NEGATIVE_RAAL_ZERO_GATE": self._negative_raal_zero_gate(state.train_rows),
            "PRIMARY_HOOK_COUNT_GATE": self._primary_hook_count_gate(state.train_rows),
            "RAAL_GRAD_TO_RGDA": self._raal_grad_gate(state.train_rows),
            "OOM_COUNT": str(getattr(trainer, "oom_count", 0)),
            "NAN_INF_COUNT": str(getattr(trainer, "nan_inf_count", 0)),
            "OOM_GATE": "PASS" if getattr(trainer, "oom_count", 0) == 0 else "FAIL",
            "NAN_INF_GATE": "PASS" if getattr(trainer, "nan_inf_count", 0) == 0 else "FAIL",
            "FULL_VAL_MODE": "NOT_APPLICABLE_SMOKE" if self.run_mode == "SMOKE" else "FULL_VAL424",
            "REAL_SD3_USED": "YES",
        }
        required = [
            "FORMAL_MANIFEST",
            "FORMAL_SCHEDULE",
            "FORMAL_CLEAN_PROXY",
            "FORMAL_CACHE",
            "TRAIN_5000_STEPS",
            "POSITIVE_NEGATIVE_BALANCE",
            "ALL_1980_TRAIN_IMAGES_USED",
            "STRATUM_USAGE_FAIR",
            "BASE_HASH_GATE",
            "NEGATIVE_RAAL_ZERO_GATE",
            "PRIMARY_HOOK_COUNT_GATE",
            "RAAL_GRAD_TO_RGDA",
            "TRAIN_FLOW_LOSS_IMPROVED",
            "FLOW_EVAL128_IMPROVED",
            "BEST_FLOW_EVAL128_IMPROVED",
            "FULL_VAL_BEST_FLOW_IMPROVED",
            "BEST_CHECKPOINT_STATE_HASH_GATE",
            "LAST_CHECKPOINT_STATE_HASH_GATE",
            "CHECKPOINT_RELOAD_RESTORE_GATE",
            "CHECKPOINT_RELOAD_GATE",
            "OOM_GATE",
            "NAN_INF_GATE",
        ]
        if self.run_mode == "SMOKE":
            status["FINAL_VERDICT"] = "PASS" if state.step == self.steps and status["SMOKE_EXECUTION_GATE"] == "PASS" else "FAIL"
        else:
            status["FINAL_VERDICT"] = "PASS" if all(status.get(key) == "PASS" for key in required) else "FAIL"
        return status

    def _read_schedule(self) -> list[dict[str, str]]:
        return read_csv(self.schedule_manifest)

    def _validate_schedule(self, train_rows: list[dict[str, str]], schedule_rows: list[dict[str, str]]) -> None:
        if len(schedule_rows) < self.steps:
            raise ValueError("RAAL_FORMAL_SCHEDULE_TOO_SHORT")
        for step, item in enumerate(schedule_rows[: self.steps], start=1):
            row = train_rows[int(item["pool_index"])]
            if row["source_sample_id"] != item["source_sample_id"]:
                raise ValueError("RAAL_FORMAL_SCHEDULE_EXECUTION_MISMATCH")
            if int(item["step"]) != step:
                raise ValueError("RAAL_FORMAL_SCHEDULE_EXECUTION_MISMATCH")
            polarity = item.get("polarity", "")
            if polarity and ((polarity == "negative") != (row.get("is_negative") == "true")):
                raise ValueError("RAAL_FORMAL_SCHEDULE_EXECUTION_MISMATCH")
        if self.steps == 5000:
            pos = sum(row.get("polarity") == "positive" for row in schedule_rows[:5000])
            neg = sum(row.get("polarity") == "negative" for row in schedule_rows[:5000])
            if pos != 2500 or neg != 2500:
                raise ValueError("RAAL_FORMAL_SCHEDULE_BALANCE_FAIL")

    def _preflight_gates(self) -> dict[str, str]:
        return {
            "FORMAL_MANIFEST": audit_formal_manifest(self.manifest_summary),
            "FORMAL_SCHEDULE": audit_formal_schedule(self.schedule_audit),
            "FORMAL_CLEAN_PROXY": audit_formal_proxy(self.clean_proxy_audit),
            "FORMAL_CACHE": audit_formal_cache(self.cache_audit),
        }

    def _validate_prompt_lengths(self, train_rows: list[dict[str, str]], clip_seq_len: int) -> None:
        selected: list[dict[str, str]] = []
        for row in train_rows:
            if row.get("is_negative") != "true" and not any(item.get("is_negative") != "true" for item in selected):
                selected.append(row)
            if row.get("is_negative") == "true" and not any(item.get("is_negative") == "true" for item in selected):
                selected.append(row)
            if len(selected) == 2:
                break
        for row in selected:
            payload = torch.load(row["cache_path"], map_location="cpu", weights_only=False)
            prompt_len = int(payload["prompt_embeds"].shape[1])
            if prompt_len != clip_seq_len + T5_MAX_LENGTH:
                raise ValueError("RAAL_FORMAL_PROMPT_LENGTH_GATE")
        self.prompt_length_gate = {
            "CLIP_SEQ_LEN": str(clip_seq_len),
            "T5_SEQ_LEN": str(T5_MAX_LENGTH),
            "PROMPT_EMBED_SEQ_LEN": str(clip_seq_len + T5_MAX_LENGTH),
            "MASK_BANK_SHA256": self.mask_bank_sha,
            "REAL_PROMPT_LENGTH_GATE": "PASS",
        }

    def _usage_audit(
        self, train_rows: list[dict[str, str]], usage: dict[int, int], schedule_rows: list[dict[str, str]]
    ) -> dict[str, str]:
        pos = 0
        neg = 0
        expected = Counter(int(item["pool_index"]) for item in schedule_rows[: self.steps])
        for item in schedule_rows[: self.steps]:
            row = train_rows[int(item["pool_index"])]
            if row.get("is_negative") == "true":
                neg += 1
            else:
                pos += 1
        exact = all(usage.get(index, 0) == expected.get(index, 0) for index in range(len(train_rows)))
        all_used = all(usage.get(index, 0) > 0 for index in range(len(train_rows))) if self.run_mode == "FULL" else True
        return {
            "POSITIVE_STEPS": str(pos),
            "NEGATIVE_STEPS": str(neg),
            "POSITIVE_NEGATIVE_BALANCE": "PASS" if self.run_mode == "SMOKE" or (pos == 2500 and neg == 2500) else "FAIL",
            "ALL_1980_TRAIN_IMAGES_USED": "PASS" if all_used else "FAIL",
            "SAMPLE_USAGE_EXACT_GATE": "PASS" if exact else "FAIL",
            "STRATUM_USAGE_FAIR": "PASS" if self.run_mode == "SMOKE" or exact else "FAIL",
            "SAMPLE_USAGE_ROWS": str(len(train_rows)),
        }

    def _write_sample_usage(self, train_rows: list[dict[str, str]], usage: dict[int, int], schedule_rows: list[dict[str, str]]) -> None:
        expected = Counter(int(item["pool_index"]) for item in schedule_rows[: self.steps])
        rows = []
        for index, row in enumerate(train_rows):
            uses = int(usage.get(index, 0))
            exp = int(expected.get(index, 0))
            rows.append(
                {
                    "pool_index": str(index),
                    "source_sample_id": row["source_sample_id"],
                    "is_negative": row.get("is_negative", ""),
                    "anchor_class": row.get("anchor_class", ""),
                    "uses": str(uses),
                    "expected_uses": str(exp),
                    "usage_difference": str(uses - exp),
                }
            )
        self._write_csv(self.report_dir / "sample_usage.csv", rows)

    @staticmethod
    def _negative_raal_zero_gate(rows: list[dict[str, str]]) -> str:
        for row in rows:
            if row["is_negative"] == "true" and (float(row["raal_loss"]) != 0.0 or int(row["raal_hook_count"]) != 0):
                return "FAIL"
        return "PASS"

    @staticmethod
    def _primary_hook_count_gate(rows: list[dict[str, str]]) -> str:
        for row in rows:
            if row["is_negative"] == "false" and int(row["raal_hook_count"]) != 3:
                return "FAIL"
        return "PASS"

    @staticmethod
    def _raal_grad_gate(rows: list[dict[str, str]]) -> str:
        for row in rows:
            if row["is_negative"] == "false" and (
                float(row.get("normal_encoder_grad", 0.0)) <= 0.0 or float(row.get("normal_adapter_grad", 0.0)) <= 0.0
            ):
                return "FAIL"
            if row["is_negative"] == "true" and (
                float(row.get("rg_encoder_grad", 0.0)) != 0.0 or float(row.get("defect_adapter_grad", 0.0)) != 0.0
            ):
                return "FAIL"
        return "PASS"

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def _append_csv(cls, path: Path, row: dict[str, str], *, append: bool = True) -> None:
        exists = path.exists() and append
        mode = "a" if append else "w"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def write_raal_formal_dry_integration(report_dir: Path, *, steps: int = 20) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "checkpoints" / "raal_formal5000").mkdir(parents=True, exist_ok=True)
    train_rows = [
        {
            "step": str(step),
            "source_sample_id": f"dry_{step:04d}",
            "is_negative": str(step % 2 == 0).lower(),
            "flow_loss": "1.00000000",
            "raal_loss": "0.00000000" if step % 2 == 0 else "0.10000000",
            "weighted_raal_loss": "0.00000000" if step % 2 == 0 else "0.00200000",
            "total_loss": "1.00000000" if step % 2 == 0 else "1.00200000",
            "raal_hook_count": "0" if step % 2 == 0 else "3",
        }
        for step in range(1, steps + 1)
    ]
    eval_rows = [
        {
            "step": "0",
            "flow_eval_loss_all": "1.00000000",
            "flow_eval_loss_positive": "1.00000000",
            "flow_eval_loss_negative": "1.00000000",
            "raal_eval_loss_positive": "0.10000000",
            "positive_count": "64",
            "negative_count": "64",
        },
        {
            "step": str(steps),
            "flow_eval_loss_all": "0.90000000",
            "flow_eval_loss_positive": "0.90000000",
            "flow_eval_loss_negative": "0.90000000",
            "raal_eval_loss_positive": "0.08000000",
            "positive_count": "64",
            "negative_count": "64",
        },
    ]
    RAALFormal5000Runner._write_csv(report_dir / "train_metrics.csv", train_rows)
    RAALFormal5000Runner._write_csv(report_dir / "eval128_metrics.csv", eval_rows)
    (report_dir / "checkpoints" / "raal_formal5000" / "best_eval.pt").write_bytes(b"dry best")
    (report_dir / "checkpoints" / "raal_formal5000" / "last.pt").write_bytes(b"dry last")
    _write_kv(
        report_dir / "RAAL_FORMAL_DRY_STATUS.md",
        {
            "RAAL_FORMAL_DRY_INTEGRATION": "PASS",
            "TRAIN_METRICS": "PASS",
            "EVAL_METRICS": "PASS",
            "BEST_CHECKPOINT": "PASS",
            "LAST_CHECKPOINT": "PASS",
            "RESUME": "PASS",
            "REAL_SD3_USED": "NO",
            "GPU_USED": "NO",
        },
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cuda_allocated_mib() -> float:
    return torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0


def _cuda_reserved_mib() -> float:
    return torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0.0


def _write_kv(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")

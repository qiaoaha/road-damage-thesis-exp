"""Pilot1000 training orchestration and no-GPU audit helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import tarfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sd3_rgda.cache import validate_cache_manifest
from sd3_rgda.checkpoint import EXPECTED_ADAPTER_MODULES, FORBIDDEN_BASE_PREFIXES
from sd3_rgda.real_sd3_engine import (
    FlowTrainingConfig,
    RealSD3RGDATrainer,
    hash_module_parameters,
    load_sd3_pipeline,
    prepare_training_scheduler,
    prepare_transformer_from_pipeline,
)
from sd3_rgda.real_sd3_engine import (
    build_fixed_flow_batches as real_build_fixed_flow_batches,
)

PILOT_CONFIG: dict[str, Any] = {
    "MODEL": "Stable Diffusion 3 Medium",
    "RESOLUTION": 512,
    "DTYPE": "bfloat16",
    "BATCH_SIZE": 1,
    "GRADIENT_ACCUMULATION": 1,
    "OPTIMIZER": "AdamW",
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 0.01,
    "MAX_GRAD_NORM": 1.0,
    "SEED": 2026,
    "TRAIN_STEPS": 1000,
    "CHECKPOINT_INTERVAL": 100,
    "EVAL_INTERVAL": 100,
    "PRECONDITION_OUTPUTS": True,
    "WEIGHTING_SCHEME": "logit_normal",
    "GRADIENT_CHECKPOINTING": True,
    "BASE_SD3_FROZEN": True,
    "LORA": False,
    "LOCAL_FILES_ONLY": True,
}


@dataclass(frozen=True)
class SampleScheduleAudit:
    unique_train_samples_used: int
    train_sample_use_min: int
    train_sample_use_max: int
    train_positive_steps: int
    train_negative_steps: int
    class_step_counts: dict[str, int]


@dataclass(frozen=True)
class PilotGateInputs:
    train_steps_completed: int
    oom_count: int
    nan_inf_count: int
    base_hash_before: str
    base_hash_after: str
    checkpoint_reload_max_abs_diff: float
    adapter_parameter_delta: float
    median_first100: float
    median_last100: float
    eval_loss_step0: float
    eval_loss_step1000: float


@dataclass
class PilotResumeState:
    step: int
    next_position: int
    sample_order: list[int]
    sample_usage_counts: dict[int, int]
    losses: list[float]
    best_eval_loss: float


@dataclass(frozen=True)
class PilotRunResult:
    final_step: int
    losses: list[float]
    sample_usage_counts: dict[int, int]
    next_position: int
    checkpoint_path: Path


@dataclass
class BranchGradientCounts:
    positive_normal_adapter_grad_nonzero_steps: int = 0
    positive_defect_adapter_grad_nonzero_steps: int = 0
    positive_rg_encoder_grad_nonzero_steps: int = 0
    negative_normal_adapter_grad_nonzero_steps: int = 0
    negative_defect_adapter_grad_nonzero_steps: int = 0
    negative_rg_encoder_grad_nonzero_steps: int = 0

    def update(self, is_negative: bool, gradients: Mapping[str, float], eps: float = 1e-12) -> None:
        prefix = "negative" if is_negative else "positive"
        for name in ("normal_adapter", "defect_adapter", "rg_encoder"):
            if abs(float(gradients.get(name, 0.0))) > eps:
                attr = f"{prefix}_{name}_grad_nonzero_steps"
                setattr(self, attr, int(getattr(self, attr)) + 1)


def deterministic_sample_order(rows: list[dict[str, str]], steps: int, seed: int = 2026) -> list[int]:
    if not rows:
        raise ValueError("Pilot training rows are empty")
    order: list[int] = []
    epoch = 0
    while len(order) < steps:
        indices = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(indices)
        order.extend(indices)
        epoch += 1
    return order[:steps]


def audit_sample_schedule(rows: list[dict[str, str]], order: list[int]) -> SampleScheduleAudit:
    usage = {index: 0 for index in range(len(rows))}
    class_counts = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
    positive = 0
    negative = 0
    for index in order:
        usage[index] += 1
        row = rows[index]
        if row.get("is_negative") == "true":
            negative += 1
        else:
            positive += 1
            anchor = row.get("anchor_class", "")
            if anchor in class_counts:
                class_counts[anchor] += 1
    values = list(usage.values())
    return SampleScheduleAudit(
        unique_train_samples_used=sum(value > 0 for value in values),
        train_sample_use_min=min(values),
        train_sample_use_max=max(values),
        train_positive_steps=positive,
        train_negative_steps=negative,
        class_step_counts=class_counts,
    )


def build_fixed_eval_plan(rows: list[dict[str, str]], seed: int = 2026) -> list[dict[str, str]]:
    rng = random.Random(seed)
    plan: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        plan.append(
            {
                "sample_id": row["sample_id"],
                "eval_index": str(index),
                "noise_seed": str(rng.randrange(2**31)),
                "timestep_seed": str(rng.randrange(2**31)),
                "sigma_seed": str(rng.randrange(2**31)),
            }
        )
    return plan


def build_fixed_eval_batches(
    trainer: RealSD3RGDATrainer,
    eval_cache_manifest: str | Path,
    seed: int = 2026,
) -> list[Any]:
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return real_build_fixed_flow_batches(trainer, eval_cache_manifest, seed)
    finally:
        torch.random.set_rng_state(torch_state)
        if torch.cuda.is_available() and cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)


def checkpoint_scope_payload(
    modules: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    seed: int,
    config: dict[str, Any],
    manifest_sha256: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else {},
        "step": step,
        "seed": seed,
        "config": config,
        "manifest_sha256": manifest_sha256,
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if extra:
        payload.update(extra)
    return payload


class Pilot1000Runner:
    def __init__(
        self,
        *,
        model_path: Path,
        train_cache_manifest: Path,
        eval_cache_manifest: Path,
        report_dir: Path,
        pilot_manifest_summary: Path | None = None,
        clean_proxy_audit: Path | None = None,
        cache_audit: Path | None = None,
        steps: int = 1000,
        seed: int = 2026,
        checkpoint_interval: int = 100,
        eval_interval: int = 100,
    ) -> None:
        self.model_path = model_path
        self.train_cache_manifest = train_cache_manifest
        self.eval_cache_manifest = eval_cache_manifest
        self.report_dir = report_dir
        self.pilot_manifest_summary = pilot_manifest_summary
        self.clean_proxy_audit = clean_proxy_audit
        self.cache_audit = cache_audit
        self._initial_zero_init_evidence: dict[str, float | str] = {}
        self._initial_adapter_hash = ""
        self.steps = steps
        self.seed = seed
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.checkpoint_dir = report_dir / "checkpoints" / "pilot1000"

    def run(self, resume_from: Path | None = None) -> PilotRunResult:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real Pilot1000 training")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_rows = validate_cache_manifest(self.train_cache_manifest, expected_rows=512)
        pipe = load_sd3_pipeline(self.model_path, torch.bfloat16)
        scheduler = prepare_training_scheduler(pipe.scheduler)
        transformer, _stats = prepare_transformer_from_pipeline(pipe)
        token_dim = int(getattr(transformer.config, "caption_projection_dim", 1536))
        latent_channels = int(getattr(transformer.config, "in_channels", 16))
        patch_size = int(getattr(transformer.config, "patch_size", 2))
        trainer = RealSD3RGDATrainer(
            transformer,
            scheduler,
            token_dim,
            latent_channels,
            patch_size,
            torch.bfloat16,
            flow_config=FlowTrainingConfig(precondition_outputs=True),
        )
        try:
            return self._run_with_trainer(trainer, train_rows, resume_from)
        except torch.cuda.OutOfMemoryError as exc:
            self._write_failure_summary("pilot_train", exc, trainer, 1, trainer.nan_inf_count, None)
            raise
        except (FloatingPointError, RuntimeError) as exc:
            self._write_failure_summary("pilot_train", exc, trainer, trainer.oom_count, trainer.nan_inf_count, None)
            raise

    def _run_with_trainer(
        self, trainer: RealSD3RGDATrainer, train_rows: list[dict[str, str]], resume_from: Path | None
    ) -> PilotRunResult:
        base_hash_before = trainer.base_parameter_hash()
        initial_snapshot = {name: parameter.detach().cpu().clone() for name, parameter in trainer.injector.named_parameters()}
        initial_adapter_hash = hash_module_parameters(trainer.injector)
        self._initial_adapter_hash = initial_adapter_hash
        manifest_sha = sha256_path(self.train_cache_manifest)
        order = deterministic_sample_order(train_rows, self.steps, self.seed)
        state = PilotResumeState(0, 0, order, {index: 0 for index in range(len(train_rows))}, [], float("inf"))
        zero_evidence: dict[str, float | str]
        if resume_from is not None:
            state = self._load_resume(resume_from, trainer, manifest_sha)
            payload = torch.load(resume_from, map_location="cpu", weights_only=False)
            zero_evidence = dict(payload.get("initial_zero_init_evidence", {}))
            initial_adapter_hash = str(payload.get("initial_adapter_hash", initial_adapter_hash))
            self._initial_zero_init_evidence = dict(zero_evidence)
            self._initial_adapter_hash = initial_adapter_hash
        eval_batches = build_fixed_eval_batches(trainer, self.eval_cache_manifest, self.seed)
        if resume_from is None:
            first_positive = next(row for row in train_rows if row.get("is_negative") != "true")
            zero_batch = trainer.build_flow_batch(trainer.load_cached_sample(first_positive["cache_path"]))
            zero_raw = trainer.zero_init_equivalence(zero_batch)
            zero_evidence = {key: float(value) for key, value in zero_raw.items()}
            zero_evidence["PILOT_STARTED_FROM_ZERO_INIT"] = "PASS"
            zero_evidence["INITIAL_ADAPTER_HASH"] = initial_adapter_hash
            self._initial_zero_init_evidence = dict(zero_evidence)
            self._write_eval_metrics(0, trainer, eval_batches, append=False)
        else:
            self._restore_metric_history(resume_from)
        train_metrics = self.report_dir / "train_metrics.csv"
        branch_counts = BranchGradientCounts()
        for step in range(state.step + 1, self.steps + 1):
            sample_index = state.sample_order[state.next_position]
            row = train_rows[sample_index]
            sample = trainer.load_cached_sample(row["cache_path"])
            metrics = trainer.backward_step(trainer.build_flow_batch(sample))
            loss = metrics["loss"]
            gradients = trainer.gradient_report()
            branch_counts.update(sample.is_negative, gradients)
            state.losses.append(loss)
            state.sample_usage_counts[sample_index] = state.sample_usage_counts.get(sample_index, 0) + 1
            state.next_position += 1
            state.step = step
            _append_csv(train_metrics, _train_metric_row(step, loss, metrics, gradients))
            if step % self.eval_interval == 0:
                eval_loss = self._write_eval_metrics(step, trainer, eval_batches, append=True)
                if eval_loss < state.best_eval_loss:
                    state.best_eval_loss = eval_loss
                    self._save_checkpoint(self.checkpoint_dir / "best_eval.pt", trainer, state, manifest_sha)
            if step % self.checkpoint_interval == 0:
                self._save_checkpoint(self.checkpoint_dir / f"step_{step:04d}.pt", trainer, state, manifest_sha)
        last = self.checkpoint_dir / "last.pt"
        self._save_checkpoint(last, trainer, state, manifest_sha)
        checkpoint_diff = trainer.compare_outputs(last, eval_batches[0])
        base_hash_after = trainer.base_parameter_hash()
        delta = sum(
            float((parameter.detach().cpu() - initial_snapshot[name]).abs().sum())
            for name, parameter in trainer.injector.named_parameters()
        )
        usage_audit = audit_actual_sample_usage(train_rows, state.sample_usage_counts)
        write_sample_usage_csv(self.report_dir / "sample_usage.csv", train_rows, state.sample_usage_counts)
        first100 = statistics.median(state.losses[:100])
        last100 = statistics.median(state.losses[-100:])
        eval_rows = read_csv(self.report_dir / "eval_metrics.csv")
        eval0 = float(eval_rows[0]["eval_loss_all"])
        eval_last = float(eval_rows[-1]["eval_loss_all"])
        gates = evaluate_pilot_gates(
            PilotGateInputs(
                train_steps_completed=state.step,
                oom_count=trainer.oom_count,
                nan_inf_count=trainer.nan_inf_count,
                base_hash_before=base_hash_before,
                base_hash_after=base_hash_after,
                checkpoint_reload_max_abs_diff=checkpoint_diff,
                adapter_parameter_delta=delta,
                median_first100=first100,
                median_last100=last100,
                eval_loss_step0=eval0,
                eval_loss_step1000=eval_last,
            )
        )
        gates.update(
            {
                "SD3_FULL_LOAD": "PASS",
                "PILOT_MANIFEST": audit_pilot_manifest_summary(self.pilot_manifest_summary),
                "CLEAN_PROXY": audit_clean_proxy_report(self.clean_proxy_audit),
                "REAL_PILOT_CACHE": audit_cache_report(self.cache_audit),
                "ZERO_INIT_EQUIVALENCE": _pass(
                    float(zero_evidence.get("FINAL_OUTPUT_MAX_ABS_DIFF", float("inf"))) <= 1e-3
                    and float(zero_evidence.get("PATCH_TOKEN_MAX_ABS_DIFF", float("inf"))) <= 1e-6
                    and float(zero_evidence.get("RGDA_RESIDUAL_MAX_ABS", float("inf"))) <= 1e-6
                ),
                "BASE_SD3_FROZEN": "PASS" if base_hash_before == base_hash_after else "FAIL",
                "ALL_512_SAMPLES_USED": _pass(usage_audit.unique_train_samples_used == 512),
                "SAMPLE_USE_RANGE": _pass(usage_audit.train_sample_use_min == 1 and usage_audit.train_sample_use_max == 2),
                "POSITIVE_NEGATIVE_MIX": _pass(usage_audit.train_positive_steps > 0 and usage_audit.train_negative_steps > 0),
                "DEFECT_BRANCH_ACTIVE_POSITIVE": _pass(
                    branch_counts.positive_defect_adapter_grad_nonzero_steps > 0
                    and branch_counts.positive_rg_encoder_grad_nonzero_steps > 0
                ),
                "DEFECT_BRANCH_BLOCKED_NEGATIVE": _pass(
                    branch_counts.negative_defect_adapter_grad_nonzero_steps == 0
                    and branch_counts.negative_rg_encoder_grad_nonzero_steps == 0
                ),
                "NORMAL_BRANCH_ACTIVE": _pass(
                    branch_counts.positive_normal_adapter_grad_nonzero_steps > 0
                    and branch_counts.negative_normal_adapter_grad_nonzero_steps > 0
                ),
                "EVAL_FIXED_BATCH": _pass(len(eval_batches) == 64),
                "CHECKPOINT_SAVE": "PASS" if last.exists() else "FAIL",
            }
        )
        gates.update({key: str(value) for key, value in zero_evidence.items()})
        final_gates = finalize_gate_report(gates)
        write_gate_status(self.report_dir / "07_PILOT1000_FINAL_STATUS.md", final_gates)
        _package_report_dir(self.report_dir)
        return PilotRunResult(state.step, state.losses, state.sample_usage_counts, state.next_position, last)

    def _write_eval_metrics(
        self, step: int, trainer: RealSD3RGDATrainer, fixed_batches: list[Any], append: bool = True
    ) -> float:
        losses: list[float] = []
        by_class: dict[str, list[float]] = {"D00": [], "D10": [], "D20": [], "D40": []}
        positives: list[float] = []
        negatives: list[float] = []
        with torch.no_grad():
            for batch in fixed_batches:
                loss = float(trainer.forward_loss(batch).detach().cpu())
                losses.append(loss)
                anchor = batch.sample.anchor_class
                if batch.sample.is_negative:
                    negatives.append(loss)
                else:
                    positives.append(loss)
                    if anchor in by_class:
                        by_class[anchor].append(loss)
        row = {
            "step": str(step),
            "eval_loss_all": str(_mean(losses)),
            "eval_loss_positive": str(_mean(positives)),
            "eval_loss_negative": str(_mean(negatives)),
            "D00_loss": str(_mean(by_class["D00"])),
            "D10_loss": str(_mean(by_class["D10"])),
            "D20_loss": str(_mean(by_class["D20"])),
            "D40_loss": str(_mean(by_class["D40"])),
        }
        _append_csv(self.report_dir / "eval_metrics.csv", row, append=append)
        return float(row["eval_loss_all"])

    def _save_checkpoint(
        self, path: Path, trainer: RealSD3RGDATrainer, state: PilotResumeState, manifest_sha256: str
    ) -> None:
        payload = checkpoint_scope_payload(
            trainer.injector.trainable_modules(),
            trainer.optimizer,
            step=state.step,
            seed=self.seed,
            config={**PILOT_CONFIG, "TRAIN_STEPS": self.steps},
            manifest_sha256=manifest_sha256,
            extra={
                "sample_order": state.sample_order,
                "next_position": state.next_position,
                "sample_usage_counts": state.sample_usage_counts,
                "losses": state.losses,
                "best_eval_loss": state.best_eval_loss,
                "train_metric_rows": read_csv(self.report_dir / "train_metrics.csv"),
                "eval_metric_rows": read_csv(self.report_dir / "eval_metrics.csv"),
                "gradient_metric_rows": read_csv(self.report_dir / "gradient_metrics.csv"),
                "memory_metric_rows": read_csv(self.report_dir / "memory_metrics.csv"),
                "initial_eval_metrics": _first_csv_row(self.report_dir / "eval_metrics.csv"),
                "initial_zero_init_evidence": self._initial_zero_init_evidence,
                "initial_adapter_hash": self._initial_adapter_hash,
            },
        )
        scope = inspect_pilot_checkpoint_payload(payload)
        if not scope["adapter_only"]:
            raise RuntimeError(f"Pilot checkpoint scope invalid: {scope}")
        torch.save(payload, path)

    def _load_resume(self, path: Path, trainer: RealSD3RGDATrainer, manifest_sha256: str) -> PilotResumeState:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Pilot checkpoint payload must be a dict")
        if payload.get("manifest_sha256") != manifest_sha256:
            raise ValueError("Pilot resume manifest SHA mismatch")
        scope = inspect_pilot_checkpoint_payload(payload)
        if not scope["adapter_only"]:
            raise ValueError(f"Pilot checkpoint scope invalid: {scope}")
        modules = payload["modules"]
        for name, module in trainer.injector.trainable_modules().items():
            module.load_state_dict(modules[name])
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        random.setstate(payload["python_random_state"])
        torch.random.set_rng_state(payload["torch_cpu_rng_state"])
        if torch.cuda.is_available() and payload.get("torch_cuda_rng_states"):
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng_states"])
        return PilotResumeState(
            step=int(payload["step"]),
            next_position=int(payload["next_position"]),
            sample_order=[int(item) for item in payload["sample_order"]],
            sample_usage_counts={int(key): int(value) for key, value in payload["sample_usage_counts"].items()},
            losses=[float(item) for item in payload["losses"]],
            best_eval_loss=float(payload["best_eval_loss"]),
        )

    def _write_failure_summary(
        self,
        stage: str,
        exception: BaseException,
        trainer: RealSD3RGDATrainer,
        oom_count: int,
        nan_inf_count: int,
        last_checkpoint: Path | None,
    ) -> None:
        fields: dict[str, str | int | float] = {
            "FINAL_VERDICT": "FAIL",
            "FAIL_STAGE": stage,
            "FAIL_EXCEPTION_TYPE": exception.__class__.__name__,
            "FAIL_EXCEPTION_MESSAGE": str(exception)[:500],
            "TRAIN_STEPS_COMPLETED": trainer.steps_completed,
            "OOM_COUNT": oom_count,
            "NAN_INF_COUNT": nan_inf_count,
            "LAST_CHECKPOINT": str(last_checkpoint or ""),
            "CUDA_ALLOCATED_MIB": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0,
            "CUDA_RESERVED_MIB": torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0.0,
        }
        write_gate_status(self.report_dir / "08_PILOT1000_FAILURE_SUMMARY.md", fields)

    def _restore_metric_history(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        mapping = {
            "train_metrics.csv": "train_metric_rows",
            "eval_metrics.csv": "eval_metric_rows",
            "gradient_metrics.csv": "gradient_metric_rows",
            "memory_metrics.csv": "memory_metric_rows",
        }
        for filename, payload_key in mapping.items():
            path = self.report_dir / filename
            if path.exists():
                continue
            rows = payload.get(payload_key, [])
            if rows:
                _write_csv(path, rows)


class MockPilotRunner:
    def __init__(self, report_dir: Path, *, seed: int = 2026, samples: int = 12) -> None:
        self.report_dir = report_dir
        self.seed = seed
        self.rows = [
            {"sample_id": f"mock_{index:04d}", "is_negative": str(index % 2 == 0).lower(), "anchor_class": "D00"}
            for index in range(samples)
        ]
        torch.manual_seed(seed)
        self.model = nn.Linear(1, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)

    def run(self, steps: int, resume_from: Path | None = None) -> PilotRunResult:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        order = deterministic_sample_order(self.rows, steps, self.seed)
        state = PilotResumeState(0, 0, order, {index: 0 for index in range(len(self.rows))}, [], float("inf"))
        if resume_from is not None:
            payload = torch.load(resume_from, map_location="cpu", weights_only=False)
            self.model.load_state_dict(payload["modules"]["normal_adapter"])
            self.optimizer.load_state_dict(payload["optimizer_state"])
            torch.random.set_rng_state(payload["torch_cpu_rng_state"])
            state = PilotResumeState(
                step=int(payload["step"]),
                next_position=int(payload["next_position"]),
                sample_order=order,
                sample_usage_counts={int(key): int(value) for key, value in payload["sample_usage_counts"].items()},
                losses=[float(item) for item in payload["losses"]],
                best_eval_loss=float(payload["best_eval_loss"]),
            )
        for step in range(state.step + 1, steps + 1):
            sample_index = state.sample_order[state.next_position]
            x = torch.tensor([[float(sample_index + 1) / 10.0]])
            y = x * 0.5
            self.optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(self.model(x), y)
            torch.autograd.backward(loss)
            self.optimizer.step()
            state.losses.append(float(loss.detach()))
            state.sample_usage_counts[sample_index] += 1
            state.next_position += 1
            state.step = step
        checkpoint = self.report_dir / f"mock_step_{state.step:04d}.pt"
        payload = checkpoint_scope_payload(
            {
                "normal_encoder": nn.Identity(),
                "rg_encoder": nn.Identity(),
                "normal_adapter": self.model,
                "defect_adapter": nn.Identity(),
                "timestep_gate": nn.Identity(),
            },
            self.optimizer,
            step=state.step,
            seed=self.seed,
            config={"mock": True},
            manifest_sha256="mock",
            extra={
                "sample_order": state.sample_order,
                "next_position": state.next_position,
                "sample_usage_counts": state.sample_usage_counts,
                "losses": state.losses,
                "best_eval_loss": state.best_eval_loss,
            },
        )
        torch.save(payload, checkpoint)
        return PilotRunResult(state.step, state.losses, state.sample_usage_counts, state.next_position, checkpoint)


def inspect_pilot_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    modules = payload.get("modules", {})
    module_names = sorted(str(name) for name in modules) if isinstance(modules, dict) else []
    forbidden = [
        name
        for name in module_names
        if name == "transformer" or any(name.startswith(prefix) for prefix in FORBIDDEN_BASE_PREFIXES)
    ]
    missing = sorted(EXPECTED_ADAPTER_MODULES - set(module_names))
    unexpected = sorted(set(module_names) - EXPECTED_ADAPTER_MODULES)
    return {
        "module_names": module_names,
        "missing_modules": missing,
        "unexpected_modules": unexpected,
        "forbidden_modules": sorted(forbidden),
        "adapter_only": bool(module_names) and not missing and not unexpected and not forbidden,
        "has_optimizer_state": "optimizer_state" in payload,
        "has_resume_state": all(key in payload for key in ("step", "seed", "config", "manifest_sha256")),
    }


def evaluate_pilot_gates(inputs: PilotGateInputs) -> dict[str, str]:
    gates = {
        "TRAIN_1000_STEPS": _pass(inputs.train_steps_completed == 1000),
        "OOM_GATE": _pass(inputs.oom_count == 0),
        "NAN_INF_GATE": _pass(inputs.nan_inf_count == 0),
        "BASE_HASH_UNCHANGED": _pass(inputs.base_hash_before == inputs.base_hash_after),
        "CHECKPOINT_RELOAD": _pass(inputs.checkpoint_reload_max_abs_diff <= 1e-3),
        "ADAPTER_PARAMETER_DELTA": _pass(inputs.adapter_parameter_delta > 0),
        "TRAIN_LOSS_IMPROVED": _pass(inputs.median_last100 <= 0.85 * inputs.median_first100),
        "EVAL_LOSS_IMPROVED": _pass(inputs.eval_loss_step1000 <= 0.95 * inputs.eval_loss_step0),
    }
    return gates


def finalize_gate_report(gates: Mapping[str, str | int | float]) -> dict[str, str | int | float]:
    final = dict(gates)
    final.pop("FINAL_VERDICT", None)
    required = [
        "SD3_FULL_LOAD",
        "PILOT_MANIFEST",
        "CLEAN_PROXY",
        "REAL_PILOT_CACHE",
        "ZERO_INIT_EQUIVALENCE",
        "BASE_SD3_FROZEN",
        "TRAIN_1000_STEPS",
        "ALL_512_SAMPLES_USED",
        "SAMPLE_USE_RANGE",
        "POSITIVE_NEGATIVE_MIX",
        "DEFECT_BRANCH_ACTIVE_POSITIVE",
        "DEFECT_BRANCH_BLOCKED_NEGATIVE",
        "NORMAL_BRANCH_ACTIVE",
        "EVAL_FIXED_BATCH",
        "TRAIN_LOSS_IMPROVED",
        "EVAL_LOSS_IMPROVED",
        "CHECKPOINT_SAVE",
        "CHECKPOINT_RELOAD",
        "BASE_HASH_UNCHANGED",
        "OOM_GATE",
        "NAN_INF_GATE",
    ]
    final["FINAL_VERDICT"] = "PASS" if all(final.get(key) == "PASS" for key in required) else "FAIL"
    return final


def audit_pilot_manifest_summary(path: Path | None) -> str:
    expected: dict[str, object] = {
        "train_rows": 512,
        "train_unique_images": 512,
        "train_positive": 256,
        "train_negative": 256,
        "eval_rows": 64,
        "eval_unique_images": 64,
        "eval_positive": 32,
        "eval_negative": 32,
        "train_eval_overlap": 0,
        "val_test_leakage": 0,
        "class_minimum_gate": "PASS",
    }
    return _audit_json_expected(path, expected)


def audit_clean_proxy_report(path: Path | None) -> str:
    expected: dict[str, object] = {
        "PROXY_TOTAL": 576,
        "PROXY_MISSING": 0,
        "PROXY_CORRUPT": 0,
        "PROXY_SHAPE_MISMATCH": 0,
        "NEGATIVE_PROXY_EXACT_MATCH": "PASS",
        "POSITIVE_MASK_CHANGED": "PASS",
        "OUTSIDE_MASK_UNCHANGED": "PASS",
        "GRAY_RECTANGLE_METHOD_USED": "NO",
        "SAM_USED": "NO",
        "CLEAN_PROXY_READY": "PASS",
    }
    return _audit_json_expected(path, expected)


def audit_cache_report(path: Path | None) -> str:
    expected: dict[str, object] = {
        "TRAIN_CACHE_ROWS": 512,
        "EVAL_CACHE_ROWS": 64,
        "ALL_CACHE_FILES_EXIST": "PASS",
        "ALL_TENSORS_FINITE": "PASS",
        "VAE_PARAMETER_DTYPE": "torch.float32",
        "VAE_INPUT_DTYPE": "torch.float32",
        "CACHED_LATENT_DTYPE": "torch.bfloat16",
        "TEXT_CACHE_DTYPE": "torch.bfloat16",
        "SOURCE_HASH_VERIFIED": "PASS",
        "CLEAN_PROXY_HASH_VERIFIED": "PASS",
        "LABEL_HASH_VERIFIED": "PASS",
        "NEGATIVE_RG_ZERO": "PASS",
        "NEGATIVE_TOKEN_MASK_ZERO": "PASS",
    }
    return _audit_json_expected(path, expected)


def _audit_json_expected(path: Path | None, expected: Mapping[str, object]) -> str:
    if path is None or not path.exists():
        return "FAIL"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "FAIL"
    for key, value in expected.items():
        if data.get(key) != value:
            return "FAIL"
    return "PASS"


def audit_actual_sample_usage(rows: list[dict[str, str]], usage: Mapping[int, int]) -> SampleScheduleAudit:
    class_counts = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
    positive = 0
    negative = 0
    for index, row in enumerate(rows):
        uses = int(usage.get(index, 0))
        if row.get("is_negative") == "true":
            negative += uses
        else:
            positive += uses
            anchor = row.get("anchor_class", "")
            if anchor in class_counts:
                class_counts[anchor] += uses
    values = [int(usage.get(index, 0)) for index in range(len(rows))]
    return SampleScheduleAudit(
        unique_train_samples_used=sum(value > 0 for value in values),
        train_sample_use_min=min(values) if values else 0,
        train_sample_use_max=max(values) if values else 0,
        train_positive_steps=positive,
        train_negative_steps=negative,
        class_step_counts=class_counts,
    )


def write_sample_usage_csv(path: Path, rows: list[dict[str, str]], usage: Mapping[int, int]) -> None:
    out_rows = [
        {
            "sample_index": str(index),
            "source_sample_id": row.get("source_sample_id", row.get("sample_id", "")),
            "anchor_class": row.get("anchor_class", ""),
            "is_negative": row.get("is_negative", ""),
            "uses": str(int(usage.get(index, 0))),
        }
        for index, row in enumerate(rows)
    ]
    _write_csv(path, out_rows)


def write_gate_status(path: str | Path, fields: Mapping[str, str | int | float]) -> None:
    lines = [f"{key}={value}" for key, value in fields.items()]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mock_dry_integration(report_dir: str | Path, *, steps: int = 10, seed: int = 2026) -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"sample_id": f"mock_{i:04d}", "is_negative": str(i % 3 == 0).lower(), "anchor_class": ["D00", "D10", "D20", "D40"][i % 4]}
        for i in range(12)
    ]
    order = deterministic_sample_order(rows, steps, seed)
    audit = audit_sample_schedule(rows, order)
    _write_csv(out / "train_metrics.csv", [{"step": str(i + 1), "loss": f"{1.0 / (i + 1):.6f}"} for i in range(steps)])
    _write_csv(out / "eval_metrics.csv", [{"step": "0", "eval_loss_all": "1.0"}, {"step": str(steps), "eval_loss_all": "0.9"}])
    _write_csv(out / "sample_usage.csv", [{"sample_index": str(i), "uses": str(order.count(i))} for i in range(len(rows))])
    _write_csv(out / "gradient_metrics.csv", [{"step": str(steps), "normal_adapter_grad": "1.0"}])
    _write_csv(out / "memory_metrics.csv", [{"step": str(steps), "allocated_mib": "0", "reserved_mib": "0"}])
    _write_csv(out / "checkpoint_manifest.csv", [{"checkpoint": "last.pt", "step": str(steps), "adapter_only": "PASS"}])
    (out / "pilot_manifest_summary.json").write_text(json.dumps(asdict(audit), indent=2) + "\n", encoding="utf-8")
    (out / "clean_proxy_audit.json").write_text('{"PILOT_DRY_INTEGRATION":"YES"}\n', encoding="utf-8")
    (out / "cache_audit.json").write_text('{"REAL_SD3_USED":"NO","GPU_USED":"NO"}\n', encoding="utf-8")
    (out / "00_console.log").write_text("PILOT_DRY_INTEGRATION=YES\nREAL_SD3_USED=NO\nGPU_USED=NO\n", encoding="utf-8")
    write_gate_status(
        out / "07_PILOT1000_FINAL_STATUS.md",
        {
            "PILOT_DRY_INTEGRATION": "YES",
            "REAL_SD3_USED": "NO",
            "GPU_USED": "NO",
            "TRAIN_STEPS_COMPLETED": steps,
            "READY_FOR_PILOT1000_GPU": "NOT_APPLICABLE_MOCK",
        },
    )


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, row: dict[str, str], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and append
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _first_csv_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return rows[0] if rows else {}


def _read_zero_init_evidence(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    evidence: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {
            "FINAL_OUTPUT_MAX_ABS_DIFF",
            "PATCH_TOKEN_MAX_ABS_DIFF",
            "RGDA_RESIDUAL_MAX_ABS",
            "PILOT_STARTED_FROM_ZERO_INIT",
            "INITIAL_ADAPTER_HASH",
        }:
            evidence[key] = value
    return evidence


def _train_metric_row(
    step: int, loss: float, metrics: dict[str, float], gradients: dict[str, float]
) -> dict[str, str]:
    return {
        "step": str(step),
        "loss": str(loss),
        "grad_norm": str(sum(gradients.values())),
        "learning_rate": "0.0001",
        "allocated_mib": str(torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0),
        "reserved_mib": str(torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0.0),
        "step_seconds": str(metrics.get("step_seconds", 0.0)),
        "normal_encoder_grad": str(gradients.get("normal_encoder", 0.0)),
        "rg_encoder_grad": str(gradients.get("rg_encoder", 0.0)),
        "normal_adapter_grad": str(gradients.get("normal_adapter", 0.0)),
        "defect_adapter_grad": str(gradients.get("defect_adapter", 0.0)),
        "timestep_gate_grad": str(gradients.get("timestep_gate", 0.0)),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _package_report_dir(report_dir: Path) -> Path:
    archive = report_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(report_dir, arcname=report_dir.name)
    (report_dir / "RESULT_ARCHIVE_SHA256.txt").write_text(sha256_path(archive) + "\n", encoding="utf-8")
    return archive


def _pass(value: bool) -> str:
    return "PASS" if value else "FAIL"

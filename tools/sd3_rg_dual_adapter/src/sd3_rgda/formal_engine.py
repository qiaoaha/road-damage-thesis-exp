"""Formal5000 training orchestration."""

from __future__ import annotations

import csv
import json
import random
import statistics
import tarfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sd3_rgda.cache import validate_cache_manifest
from sd3_rgda.checkpoint import load_adapter_checkpoint
from sd3_rgda.pilot_engine import (
    BranchGradientCounts,
    _evaluate_one_batch,
    _gradient_metric_row,
    _memory_metric_row,
    _pass,
    _write_csv,
    checkpoint_scope_payload,
    inspect_pilot_checkpoint_payload,
    read_csv,
    sha256_path,
    write_gate_status,
)
from sd3_rgda.real_sd3_engine import (
    FlowTrainingConfig,
    RealSD3RGDATrainer,
    build_fixed_flow_batches,
    hash_module_parameters,
    load_sd3_pipeline,
    prepare_training_scheduler,
    prepare_transformer_from_pipeline,
)

FORMAL_CONFIG: dict[str, Any] = {
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
    "TRAIN_STEPS": 5000,
    "CHECKPOINT_INTERVAL": 250,
    "EVAL_INTERVAL": 250,
    "PRECONDITION_OUTPUTS": True,
    "WEIGHTING_SCHEME": "logit_normal",
    "GRADIENT_CHECKPOINTING": True,
    "BASE_SD3_FROZEN": True,
    "LORA": False,
    "LOCAL_FILES_ONLY": True,
    "INITIALIZATION": "FRESH_ZERO_INIT_RGDA",
}


@dataclass
class FormalResumeState:
    step: int
    losses: list[float]
    best_eval_loss: float
    best_eval_step: int
    sample_usage_counts: dict[int, int]
    branch_gradient_counts: BranchGradientCounts


@dataclass(frozen=True)
class FormalRunResult:
    final_step: int
    checkpoint_path: Path
    report_dir: Path


class Formal5000Runner:
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
        self.checkpoint_dir = report_dir / "checkpoints" / "formal5000"
        self._zero_init_evidence: dict[str, str | float] = {}
        self._base_hash_before = ""

    def run(self, resume_from: Path | None = None) -> FormalRunResult:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        preflight = self._preflight_gates()
        if not all(value == "PASS" for value in preflight.values()):
            self._write_failure("PREFLIGHT_ASSET_AUDIT", RuntimeError(str(preflight)), 0, 0, 0, None)
            raise RuntimeError("FORMAL_PREFLIGHT_ASSET_AUDIT failed")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real Formal5000 training")
        train_rows = validate_cache_manifest(self.train_cache_manifest, expected_rows=1980)
        eval_rows = validate_cache_manifest(self.eval128_cache_manifest, expected_rows=128)
        val_rows = validate_cache_manifest(self.val_cache_manifest, expected_rows=424)
        schedule = read_csv(self.schedule_manifest)
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
            return self._run_with_trainer(trainer, train_rows, eval_rows, val_rows, schedule, resume_from)
        except BaseException as exc:
            self._write_failure("formal_train", exc, trainer.steps_completed, trainer.oom_count, trainer.nan_inf_count, None)
            raise

    def _run_with_trainer(
        self,
        trainer: RealSD3RGDATrainer,
        train_rows: list[dict[str, str]],
        eval_rows: list[dict[str, str]],
        val_rows: list[dict[str, str]],
        schedule: list[dict[str, str]],
        resume_from: Path | None,
    ) -> FormalRunResult:
        self._base_hash_before = trainer.base_parameter_hash()
        schedule_sha = sha256_path(self.schedule_manifest)
        manifest_sha = sha256_path(self.train_pool_manifest)
        cache_sha = sha256_path(self.train_cache_manifest)
        state = FormalResumeState(0, [], float("inf"), 0, {i: 0 for i in range(len(train_rows))}, BranchGradientCounts())
        if resume_from:
            state = self._load_resume(resume_from, trainer, schedule_sha, manifest_sha, cache_sha)
            self._restore_history(resume_from)
        fixed_eval = build_fixed_flow_batches(trainer, self.eval128_cache_manifest, self.seed)
        fixed_val = build_fixed_flow_batches(trainer, self.val_cache_manifest, self.seed)
        if not resume_from:
            first_positive = next(row for row in train_rows if row.get("is_negative") != "true")
            zero_batch = trainer.build_flow_batch(trainer.load_cached_sample(first_positive["cache_path"]))
            self._zero_init_evidence = {key: float(value) for key, value in trainer.zero_init_equivalence(zero_batch).items()}
            self._zero_init_evidence["FORMAL_STARTED_FROM_ZERO_INIT"] = "PASS"
            self._zero_init_evidence["INITIAL_ADAPTER_HASH"] = hash_module_parameters(trainer.injector)
            self._write_eval128_metrics(0, trainer, fixed_eval, append=False)
            self._write_full_val_metrics("zero_init", 0, trainer, fixed_val, append=False)
        for step in range(state.step + 1, self.steps + 1):
            item = schedule[step - 1]
            pool_index = int(item["pool_index"])
            row = train_rows[pool_index]
            if row["source_sample_id"] != item["source_sample_id"]:
                raise ValueError("SCHEDULE_EXECUTION_MISMATCH")
            sample = trainer.load_cached_sample(row["cache_path"])
            metrics = trainer.backward_step(trainer.build_flow_batch(sample))
            gradients = trainer.gradient_report()
            state.branch_gradient_counts.update(sample.is_negative, gradients)
            state.sample_usage_counts[pool_index] = state.sample_usage_counts.get(pool_index, 0) + 1
            state.losses.append(float(metrics["loss"]))
            state.step = step
            _append_csv(self.report_dir / "train_metrics.csv", _formal_train_metric_row(step, row, float(metrics["loss"]), gradients))
            _append_csv(self.report_dir / "gradient_metrics.csv", _gradient_metric_row(step, row, sample.is_negative, gradients))
            _append_csv(self.report_dir / "memory_metrics.csv", _memory_metric_row(step, metrics))
            if step % self.eval_interval == 0:
                eval_loss = self._write_eval128_metrics(step, trainer, fixed_eval, append=True)
                if eval_loss < state.best_eval_loss:
                    state.best_eval_loss = eval_loss
                    state.best_eval_step = step
                    self._save_checkpoint(self.checkpoint_dir / "best_eval.pt", trainer, state, schedule_sha, manifest_sha, cache_sha)
            if step % self.checkpoint_interval == 0:
                self._save_checkpoint(self.checkpoint_dir / f"step_{step:04d}.pt", trainer, state, schedule_sha, manifest_sha, cache_sha)
        last = self.checkpoint_dir / "last.pt"
        self._save_checkpoint(last, trainer, state, schedule_sha, manifest_sha, cache_sha)
        best = self.checkpoint_dir / "best_eval.pt"
        best_diff, last_diff = run_checkpoint_reload_evaluations(
            trainer=trainer,
            best=best,
            last=last,
            fixed_eval=fixed_eval,
            fixed_val=fixed_val,
            write_full_val=lambda tag, step, active_trainer, batches, append: self._write_full_val_metrics(
                tag, step, active_trainer, batches, append
            ),
            best_step=state.best_eval_step,
            last_step=state.step,
        )
        usage_audit = _write_usage(self.report_dir / "sample_usage.csv", train_rows, state.sample_usage_counts, schedule)
        base_after = trainer.base_parameter_hash()
        final = finalize_formal_gates(
            state=state,
            losses=state.losses,
            eval_rows=read_csv(self.report_dir / "eval128_metrics.csv"),
            full_val_rows=read_csv(self.report_dir / "full_val_metrics.csv"),
            usage_audit=usage_audit,
            branch_counts=state.branch_gradient_counts,
            base_before=self._base_hash_before,
            base_after=base_after,
            zero_evidence=self._zero_init_evidence,
            checkpoint_paths=[self.checkpoint_dir / f"step_{step:04d}.pt" for step in range(self.checkpoint_interval, self.steps + 1, self.checkpoint_interval)]
            + [best, last],
            best_diff=best_diff,
            last_diff=last_diff,
            oom_count=trainer.oom_count,
            nan_inf_count=trainer.nan_inf_count,
            preflight=self._preflight_gates(),
        )
        final.update({key: str(value) for key, value in self._zero_init_evidence.items()})
        write_gate_status(self.report_dir / "07_FORMAL5000_FINAL_STATUS.md", final)
        _package_report_dir(self.report_dir)
        return FormalRunResult(state.step, last, self.report_dir)

    def _write_eval128_metrics(self, step: int, trainer: RealSD3RGDATrainer, fixed_batches: list[Any], append: bool) -> float:
        return _write_eval_rows(self.report_dir / "eval128_metrics.csv", step, trainer, fixed_batches, append)

    def _write_full_val_metrics(self, tag: str, step: int, trainer: RealSD3RGDATrainer, batches: list[Any], append: bool) -> float:
        loss = _write_eval_rows(self.report_dir / "_tmp_full_val.csv", step, trainer, batches, append=False)
        row = read_csv(self.report_dir / "_tmp_full_val.csv")[0]
        row = {"model_tag": tag, "checkpoint_step": str(step), **{k: v for k, v in row.items() if k != "step"}}
        _append_csv(self.report_dir / "full_val_metrics.csv", row, append=append)
        (self.report_dir / "_tmp_full_val.csv").unlink(missing_ok=True)
        return loss

    def _save_checkpoint(
        self,
        path: Path,
        trainer: RealSD3RGDATrainer,
        state: FormalResumeState,
        schedule_sha: str,
        manifest_sha: str,
        cache_sha: str,
    ) -> None:
        payload = checkpoint_scope_payload(
            trainer.injector.trainable_modules(),
            trainer.optimizer,
            step=state.step,
            seed=self.seed,
            config={**FORMAL_CONFIG, "TRAIN_STEPS": self.steps},
            manifest_sha256=manifest_sha,
            extra={
                "schedule_sha256": schedule_sha,
                "cache_manifest_sha256": cache_sha,
                "losses": state.losses,
                "sample_usage_counts": state.sample_usage_counts,
                "branch_gradient_counts": state.branch_gradient_counts.__dict__,
                "best_eval_loss": state.best_eval_loss,
                "best_eval_step": state.best_eval_step,
                "train_metric_rows": read_csv(self.report_dir / "train_metrics.csv"),
                "eval_metric_rows": read_csv(self.report_dir / "eval128_metrics.csv"),
                "full_val_metric_rows": read_csv(self.report_dir / "full_val_metrics.csv"),
                "gradient_metric_rows": read_csv(self.report_dir / "gradient_metrics.csv"),
                "memory_metric_rows": read_csv(self.report_dir / "memory_metrics.csv"),
                "zero_init_evidence": self._zero_init_evidence,
                "base_hash_before": self._base_hash_before,
            },
        )
        if not inspect_pilot_checkpoint_payload(payload)["adapter_only"]:
            raise RuntimeError("Formal checkpoint scope invalid")
        torch.save(payload, path)

    def _load_resume(
        self, path: Path, trainer: RealSD3RGDATrainer, schedule_sha: str, manifest_sha: str, cache_sha: str
    ) -> FormalResumeState:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schedule_sha256") != schedule_sha or payload.get("manifest_sha256") != manifest_sha:
            raise ValueError("FORMAL_RESUME_SHA_MISMATCH")
        if payload.get("cache_manifest_sha256") != cache_sha:
            raise ValueError("FORMAL_RESUME_CACHE_SHA_MISMATCH")
        _validate_formal_config(payload.get("config", {}), self.steps)
        for name, module in trainer.injector.trainable_modules().items():
            module.load_state_dict(payload["modules"][name])
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        random.setstate(payload["python_random_state"])
        torch.random.set_rng_state(payload["torch_cpu_rng_state"])
        if torch.cuda.is_available() and payload.get("torch_cuda_rng_states"):
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng_states"])
        self._zero_init_evidence = dict(payload.get("zero_init_evidence", {}))
        self._base_hash_before = str(payload.get("base_hash_before", ""))
        return FormalResumeState(
            step=int(payload["step"]),
            losses=[float(item) for item in payload.get("losses", [])],
            best_eval_loss=float(payload.get("best_eval_loss", float("inf"))),
            best_eval_step=int(payload.get("best_eval_step", 0)),
            sample_usage_counts={int(k): int(v) for k, v in payload.get("sample_usage_counts", {}).items()},
            branch_gradient_counts=BranchGradientCounts.from_mapping(payload.get("branch_gradient_counts")),
        )

    def _restore_history(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        mapping = {
            "train_metrics.csv": "train_metric_rows",
            "eval128_metrics.csv": "eval_metric_rows",
            "full_val_metrics.csv": "full_val_metric_rows",
            "gradient_metrics.csv": "gradient_metric_rows",
            "memory_metrics.csv": "memory_metric_rows",
        }
        for filename, key in mapping.items():
            rows = payload.get(key, [])
            path = self.report_dir / filename
            if path.exists() and read_csv(path) != rows:
                raise ValueError("FORMAL_RESUME_METRIC_HISTORY_MISMATCH")
            if rows and not path.exists():
                _write_csv(path, rows)

    def _preflight_gates(self) -> dict[str, str]:
        return {
            "FORMAL_MANIFEST": audit_formal_manifest(self.manifest_summary),
            "FORMAL_SCHEDULE": audit_formal_schedule(self.schedule_audit),
            "FORMAL_CLEAN_PROXY": audit_formal_proxy(self.clean_proxy_audit),
            "FORMAL_CACHE": audit_formal_cache(self.cache_audit),
        }

    def _write_failure(
        self, stage: str, exception: BaseException, steps_completed: int, oom_count: int, nan_inf_count: int, checkpoint: Path | None
    ) -> None:
        write_gate_status(
            self.report_dir / "08_FORMAL5000_FAILURE_SUMMARY.md",
            {
                "FINAL_VERDICT": "FAIL",
                "FAIL_STAGE": stage,
                "FAIL_EXCEPTION_TYPE": exception.__class__.__name__,
                "FAIL_EXCEPTION_MESSAGE": str(exception)[:500],
                "TRAIN_STEPS_COMPLETED": steps_completed,
                "LAST_COMPLETE_CHECKPOINT": str(checkpoint or ""),
                "OOM_COUNT": oom_count,
                "NAN_INF_COUNT": nan_inf_count,
                "CUDA_ALLOCATED_MIB": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0,
                "CUDA_RESERVED_MIB": torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0.0,
                "REPORT_DIR": str(self.report_dir),
            },
        )
        _package_report_dir(self.report_dir)


def finalize_formal_gates(
    *,
    state: FormalResumeState,
    losses: list[float],
    eval_rows: list[dict[str, str]],
    full_val_rows: list[dict[str, str]],
    usage_audit: Mapping[str, str | int],
    branch_counts: BranchGradientCounts,
    base_before: str,
    base_after: str,
    zero_evidence: Mapping[str, str | float],
    checkpoint_paths: list[Path],
    best_diff: float,
    last_diff: float,
    oom_count: int,
    nan_inf_count: int,
    preflight: Mapping[str, str],
) -> dict[str, str | int | float]:
    eval0 = float(eval_rows[0]["eval_loss_all"])
    eval_last = float(eval_rows[-1]["eval_loss_all"])
    full = {row["model_tag"]: float(row["eval_loss_all"]) for row in full_val_rows}
    zero_ok = (
        float(zero_evidence.get("FINAL_OUTPUT_MAX_ABS_DIFF", float("inf"))) <= 1e-3
        and float(zero_evidence.get("PATCH_TOKEN_MAX_ABS_DIFF", float("inf"))) <= 1e-6
        and float(zero_evidence.get("RGDA_RESIDUAL_MAX_ABS", float("inf"))) <= 1e-6
        and zero_evidence.get("FORMAL_STARTED_FROM_ZERO_INIT") == "PASS"
    )
    gates: dict[str, str | int | float] = {
        "GPU_ENV": "PASS",
        "SD3_FULL_LOAD": "PASS",
        **dict(preflight),
        "ZERO_INIT_EQUIVALENCE": _pass(zero_ok),
        "BASE_SD3_FROZEN": _pass(base_before == base_after),
        "TRAIN_5000_STEPS": _pass(state.step == 5000),
        "ALL_1980_TRAIN_IMAGES_USED": usage_audit["ALL_1980_TRAIN_IMAGES_USED"],
        "POSITIVE_NEGATIVE_BALANCE": _pass(usage_audit["POSITIVE_STEPS"] == 2500 and usage_audit["NEGATIVE_STEPS"] == 2500),
        "SCHEDULE_EXECUTION_MATCH": "PASS",
        "STRATUM_USAGE_FAIR": usage_audit["STRATUM_USAGE_FAIR"],
        "DEFECT_BRANCH_ACTIVE_POSITIVE": _pass(
            branch_counts.positive_defect_adapter_grad_nonzero_steps > 0
            and branch_counts.positive_rg_encoder_grad_nonzero_steps > 0
        ),
        "DEFECT_BRANCH_BLOCKED_NEGATIVE": _pass(branch_counts.negative_defect_adapter_grad_nonzero_steps == 0 and branch_counts.negative_rg_encoder_grad_nonzero_steps == 0),
        "NORMAL_BRANCH_ACTIVE": _pass(branch_counts.positive_normal_adapter_grad_nonzero_steps > 0 and branch_counts.negative_normal_adapter_grad_nonzero_steps > 0),
        "EVAL128_FIXED_BATCH": _pass(len(eval_rows) == 21),
        "FULL_VAL_FIXED_BATCH": _pass(len(full_val_rows) == 3),
        "TRAIN_LOSS_IMPROVED": _pass(statistics.median(losses[-250:]) <= 0.90 * statistics.median(losses[:250])),
        "EVAL128_LOSS_IMPROVED": _pass(eval_last <= 0.95 * eval0),
        "BEST_EVAL128_IMPROVED": _pass(state.best_eval_loss <= 0.90 * eval0),
        "FULL_VAL_BEST_IMPROVED": _pass(full.get("best_eval", float("inf")) <= 0.95 * full.get("zero_init", 0.0)),
        "CHECKPOINT_SAVE": _pass(bool(checkpoint_paths) and all(path.exists() and path.stat().st_size > 0 for path in checkpoint_paths)),
        "BEST_CHECKPOINT_RELOAD": _pass(best_diff <= 1e-3),
        "LAST_CHECKPOINT_RELOAD": _pass(last_diff <= 1e-3),
        "BASE_HASH_UNCHANGED": _pass(base_before == base_after),
        "OOM_GATE": _pass(oom_count == 0),
        "NAN_INF_GATE": _pass(nan_inf_count == 0),
        "TRAIN_STEPS_COMPLETED": state.step,
        "OOM_COUNT": oom_count,
        "NAN_INF_COUNT": nan_inf_count,
        "BEST_EVAL_STEP": state.best_eval_step,
        "BEST_EVAL128_LOSS": state.best_eval_loss,
        "LAST_EVAL128_LOSS": eval_last,
        "FULL_VAL_ZERO_INIT_LOSS": full.get("zero_init", 0.0),
        "FULL_VAL_BEST_LOSS": full.get("best_eval", 0.0),
        "FULL_VAL_LAST_LOSS": full.get("last", 0.0),
    }
    required = [
        "GPU_ENV",
        "SD3_FULL_LOAD",
        "FORMAL_MANIFEST",
        "FORMAL_SCHEDULE",
        "FORMAL_CLEAN_PROXY",
        "FORMAL_CACHE",
        "ZERO_INIT_EQUIVALENCE",
        "BASE_SD3_FROZEN",
        "TRAIN_5000_STEPS",
        "ALL_1980_TRAIN_IMAGES_USED",
        "POSITIVE_NEGATIVE_BALANCE",
        "SCHEDULE_EXECUTION_MATCH",
        "STRATUM_USAGE_FAIR",
        "DEFECT_BRANCH_ACTIVE_POSITIVE",
        "DEFECT_BRANCH_BLOCKED_NEGATIVE",
        "NORMAL_BRANCH_ACTIVE",
        "EVAL128_FIXED_BATCH",
        "FULL_VAL_FIXED_BATCH",
        "TRAIN_LOSS_IMPROVED",
        "EVAL128_LOSS_IMPROVED",
        "BEST_EVAL128_IMPROVED",
        "FULL_VAL_BEST_IMPROVED",
        "CHECKPOINT_SAVE",
        "BEST_CHECKPOINT_RELOAD",
        "LAST_CHECKPOINT_RELOAD",
        "BASE_HASH_UNCHANGED",
        "OOM_GATE",
        "NAN_INF_GATE",
    ]
    gates["FINAL_VERDICT"] = "PASS" if all(gates.get(key) == "PASS" for key in required) else "FAIL"
    return gates


def audit_formal_manifest(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _pass(
        data.get("train_pool_rows") == 1980
        and data.get("val_full_rows") == 424
        and data.get("eval128_rows") == 128
        and data.get("train_val_overlap") == 0
        and data.get("train_test_overlap") == 0
        and data.get("val_test_overlap") == 0
        and data.get("eval_test_overlap") == 0
        and data.get("test_leakage") == 0
    )


def run_checkpoint_reload_evaluations(
    *,
    trainer: Any,
    best: Path,
    last: Path,
    fixed_eval: list[Any],
    fixed_val: list[Any],
    write_full_val: Any,
    best_step: int,
    last_step: int,
    loader: Any = load_adapter_checkpoint,
) -> tuple[float, float]:
    loader(best, trainer.injector.trainable_modules())
    best_diff = trainer.compare_outputs(best, fixed_eval[0])
    write_full_val("best_eval", best_step, trainer, fixed_val, True)
    loader(last, trainer.injector.trainable_modules())
    last_diff = trainer.compare_outputs(last, fixed_eval[0])
    write_full_val("last", last_step, trainer, fixed_val, True)
    return best_diff, last_diff


def audit_formal_schedule(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _pass(data.get("schedule_rows") == 5000 and data.get("schedule_audit") == "PASS")


def audit_formal_proxy(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _pass(data.get("PROXY_TOTAL") == 2404 and data.get("FORMAL_CLEAN_PROXY_READY") == "PASS")


def audit_formal_cache(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _pass(data.get("TRAIN_CACHE_ROWS") == 1980 and data.get("VAL_CACHE_ROWS") == 424 and data.get("CACHE_READY") == "PASS")


def write_mock_dry_integration(report_dir: Path, *, steps: int = 20, seed: int = 2026) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "train_metrics.csv", [{"step": str(i), "source_sample_id": f"mock_{i}", "loss": "1.0", "grad_norm": "0.1", "learning_rate": "0.0001"} for i in range(1, steps + 1)])
    _write_csv(report_dir / "eval128_metrics.csv", [{"step": "0", "eval_loss_all": "1.0"}, {"step": str(steps), "eval_loss_all": "0.9"}])
    _write_csv(report_dir / "full_val_metrics.csv", [{"model_tag": "zero_init", "checkpoint_step": "0", "eval_loss_all": "1.0"}, {"model_tag": "best_eval", "checkpoint_step": str(steps), "eval_loss_all": "0.9"}, {"model_tag": "last", "checkpoint_step": str(steps), "eval_loss_all": "0.9"}])
    _write_csv(report_dir / "gradient_metrics.csv", [{"step": str(i), "normal_adapter_grad": "1.0"} for i in range(1, steps + 1)])
    _write_csv(report_dir / "memory_metrics.csv", [{"step": str(i), "allocated_mib": "0", "reserved_mib": "0", "peak_allocated_mib": "0", "peak_reserved_mib": "0", "step_seconds": "0"} for i in range(1, steps + 1)])
    checkpoint_dir = report_dir / "checkpoints" / "formal5000"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "best_eval.pt").write_bytes(b"mock")
    (checkpoint_dir / "last.pt").write_bytes(b"mock")
    write_gate_status(report_dir / "07_FORMAL5000_FINAL_STATUS.md", {"FORMAL_DRY_INTEGRATION": "YES", "REAL_SD3_USED": "NO", "GPU_USED": "NO", "MOCK_STEPS_COMPLETED": steps, "MOCK_BEST_CHECKPOINT": "PASS", "MOCK_LAST_CHECKPOINT": "PASS", "MOCK_RESUME": "PASS"})
    _package_report_dir(report_dir)


def _write_eval_rows(path: Path, step: int, trainer: RealSD3RGDATrainer, batches: list[Any], append: bool) -> float:
    losses: list[float] = []
    positives: list[float] = []
    negatives: list[float] = []
    by_class: dict[str, list[float]] = {"D00": [], "D10": [], "D20": [], "D40": []}
    for batch in batches:
        loss = _evaluate_one_batch(trainer, batch)
        losses.append(loss)
        if batch.sample.is_negative:
            negatives.append(loss)
        else:
            positives.append(loss)
            if batch.sample.anchor_class in by_class:
                by_class[batch.sample.anchor_class].append(loss)
    row = {
        "step": str(step),
        "eval_loss_all": str(sum(losses) / len(losses)),
        "eval_loss_positive": str(sum(positives) / len(positives)) if positives else "0.0",
        "eval_loss_negative": str(sum(negatives) / len(negatives)) if negatives else "0.0",
        "D00_loss": str(sum(by_class["D00"]) / len(by_class["D00"])) if by_class["D00"] else "0.0",
        "D10_loss": str(sum(by_class["D10"]) / len(by_class["D10"])) if by_class["D10"] else "0.0",
        "D20_loss": str(sum(by_class["D20"]) / len(by_class["D20"])) if by_class["D20"] else "0.0",
        "D40_loss": str(sum(by_class["D40"]) / len(by_class["D40"])) if by_class["D40"] else "0.0",
    }
    _append_csv(path, row, append=append)
    return float(row["eval_loss_all"])


def _formal_train_metric_row(step: int, row: Mapping[str, str], loss: float, gradients: Mapping[str, float]) -> dict[str, str]:
    return {
        "step": str(step),
        "source_sample_id": row.get("source_sample_id", row.get("sample_id", "")),
        "loss": str(loss),
        "grad_norm": str(sum(float(value) for value in gradients.values())),
        "learning_rate": "0.0001",
    }


def _write_usage(path: Path, train_rows: list[dict[str, str]], usage: Mapping[int, int], schedule: list[dict[str, str]]) -> dict[str, str | int]:
    expected = Counter(int(row["pool_index"]) for row in schedule)
    rows = []
    for index, row in enumerate(train_rows):
        uses = int(usage.get(index, 0))
        exp = int(expected.get(index, 0))
        rows.append(
            {
                "pool_index": str(index),
                "source_sample_id": row["source_sample_id"],
                "is_negative": row["is_negative"],
                "anchor_class": row["anchor_class"],
                "stratum": row["anchor_class"] if row["is_negative"] == "false" else "NEG",
                "uses": str(uses),
                "expected_uses": str(exp),
                "usage_difference": str(uses - exp),
            }
        )
    _write_csv(path, rows)
    pos = sum(int(row["uses"]) for row in rows if row["is_negative"] == "false")
    neg = sum(int(row["uses"]) for row in rows if row["is_negative"] == "true")
    fair = all(row["usage_difference"] == "0" for row in rows)
    return {
        "ALL_1980_TRAIN_IMAGES_USED": "PASS" if all(int(row["uses"]) > 0 for row in rows) else "FAIL",
        "POSITIVE_STEPS": pos,
        "NEGATIVE_STEPS": neg,
        "STRATUM_USAGE_FAIR": "PASS" if fair else "FAIL",
    }


def _validate_formal_config(config: Mapping[str, Any], steps: int) -> None:
    expected = {**FORMAL_CONFIG, "TRAIN_STEPS": steps}
    for key in ("SEED", "TRAIN_STEPS", "LEARNING_RATE", "RESOLUTION", "DTYPE", "WEIGHTING_SCHEME", "PRECONDITION_OUTPUTS"):
        if config.get(key) != expected[key]:
            raise ValueError("FORMAL_RESUME_CORE_CONFIG_MISMATCH")


def _append_csv(path: Path, row: dict[str, str], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and append
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _package_report_dir(report_dir: Path) -> Path:
    archive = report_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(report_dir, arcname=report_dir.name)
    digest = sha256_path(archive)
    Path(str(archive) + ".sha256").write_text(f"{digest}  {archive}\n", encoding="utf-8")
    return archive

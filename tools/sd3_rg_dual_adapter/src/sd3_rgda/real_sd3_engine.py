"""Executable real SD3-RGDA validation engine."""

from __future__ import annotations

import copy
import csv
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from sd3_rgda.cache import (
    CacheBuildReport,
    CachedSD3Sample,
    cache_manifest_rows,
    load_cached_sample,
    validate_cache_manifest,
)
from sd3_rgda.checkpoint import load_adapter_checkpoint, save_adapter_checkpoint
from sd3_rgda.injector import RGDAConditionBatch, RGDAInjector, RGDAPatchHook
from sd3_rgda.losses import (
    flow_matching_target,
    sample_noisy_latent,
    sample_sd3_flow_timesteps,
    weighted_flow_matching_mse,
)
from sd3_rgda.reporting import GateResult, summarize_gates, write_gate_report
from sd3_rgda.sd3_hook import resolve_patch_embedding, temporary_forward_hook


class SD3ForwardOutput(Protocol):
    sample: torch.Tensor


@dataclass(frozen=True)
class SD3LoadStats:
    transformer_class: str
    parameter_count: int
    dtype: str
    device: str
    patch_module_name: str
    cuda_allocated_mib: float
    cuda_reserved_mib: float


@dataclass
class FlowBatch:
    sample: CachedSD3Sample
    noise: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    weighting: torch.Tensor
    noisy_latent: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class TrainingLoopSummary:
    losses: list[float]
    parameter_snapshot: dict[str, torch.Tensor]
    steps_completed: int
    peak_allocated_mib: float
    peak_reserved_mib: float


def load_sd3_pipeline(model_path: str | Path, dtype: torch.dtype) -> Any:
    from diffusers import StableDiffusion3Pipeline

    return StableDiffusion3Pipeline.from_pretrained(  # type: ignore[no-untyped-call]
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True,
    )


def prepare_transformer_from_pipeline(pipe: Any) -> tuple[nn.Module, SD3LoadStats]:
    transformer = pipe.transformer
    del pipe.vae
    del pipe.text_encoder
    del pipe.text_encoder_2
    del pipe.text_encoder_3
    del pipe.tokenizer
    del pipe.tokenizer_2
    del pipe.tokenizer_3
    torch.cuda.empty_cache()
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    transformer.to("cuda")
    transformer.train()
    patch_name, _patch_module = resolve_patch_embedding(transformer)
    first = next(transformer.parameters())
    return transformer, SD3LoadStats(
        transformer_class=transformer.__class__.__name__,
        parameter_count=sum(parameter.numel() for parameter in transformer.parameters()),
        dtype=str(first.dtype),
        device=str(first.device),
        patch_module_name=patch_name,
        cuda_allocated_mib=torch.cuda.memory_allocated() / 1024 / 1024,
        cuda_reserved_mib=torch.cuda.memory_reserved() / 1024 / 1024,
    )


def hash_module_parameters(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        raw_bytes = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(raw_bytes)
    return digest.hexdigest()


def parameter_grad_norm(parameter: nn.Parameter | torch.Tensor | None) -> float:
    if parameter is None or parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm().cpu())


def find_first_linear(module: nn.Module) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise ValueError(f"No Linear layer found in {module.__class__.__name__}")


def find_last_linear(module: nn.Module) -> nn.Linear:
    found: nn.Linear | None = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            found = child
    if found is None:
        raise ValueError(f"No Linear layer found in {module.__class__.__name__}")
    return found


class SD3RGTransformerWrapper(nn.Module):
    def __init__(self, transformer: nn.Module, injector: RGDAInjector) -> None:
        super().__init__()
        self.transformer = transformer
        self.injector = injector

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        rgda_condition: RGDAConditionBatch | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> SD3ForwardOutput:
        if rgda_condition is None:
            return cast(
                SD3ForwardOutput,
                self.transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    timestep=timestep,
                    return_dict=return_dict,
                    **kwargs,
                ),
            )
        _patch_name, patch_module = resolve_patch_embedding(self.transformer)
        hook = RGDAPatchHook(self.injector, rgda_condition)
        with temporary_forward_hook(patch_module, hook):
            output = self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                return_dict=return_dict,
                **kwargs,
            )
        if hook.calls != 1:
            raise RuntimeError(f"RGDA patch hook call count was {hook.calls}, expected 1")
        return cast(SD3ForwardOutput, output)


class RealSD3RGDATrainer:
    def __init__(
        self,
        transformer: nn.Module,
        scheduler: Any,
        token_dim: int,
        latent_channels: int,
        patch_size: int,
        dtype: torch.dtype,
    ) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.injector = RGDAInjector(token_dim=token_dim, latent_channels=latent_channels, patch_size=patch_size).to(
            self.device
        )
        self.wrapper = SD3RGTransformerWrapper(transformer, self.injector)
        modules = self.injector.trainable_modules()
        self.optimizer = torch.optim.AdamW(
            [
                *modules["normal_encoder"].parameters(),
                *modules["rg_encoder"].parameters(),
                *modules["normal_adapter"].parameters(),
                *modules["defect_adapter"].parameters(),
                *modules["timestep_gate"].parameters(),
            ],
            lr=1e-4,
        )
        self.forward_count = 0
        self.oom_count = 0
        self.nan_inf_count = 0
        self.steps_completed = 0

    def load_cached_sample(self, cache_path: str | Path) -> CachedSD3Sample:
        return load_cached_sample(cache_path, self.device, self.dtype)

    def build_flow_batch(self, sample: CachedSD3Sample) -> FlowBatch:
        noise = torch.randn_like(sample.target_latent)
        timesteps, sigmas, weighting = sample_sd3_flow_timesteps(
            self.scheduler, sample.target_latent.shape[0], self.device
        )
        noisy_latent = sample_noisy_latent(sample.target_latent, noise, sigmas)
        target = flow_matching_target(sample.target_latent, noise)
        return FlowBatch(sample, noise, timesteps, sigmas, weighting, noisy_latent, target)

    def forward_loss(self, batch: FlowBatch) -> torch.Tensor:
        condition = RGDAConditionBatch(
            pseudo_clean_latents=batch.sample.pseudo_clean_latent,
            rg_maps=batch.sample.rg_map_latent,
            token_mask=batch.sample.token_mask,
            timesteps=batch.timesteps,
        )
        prediction = self.wrapper(
            hidden_states=batch.noisy_latent,
            encoder_hidden_states=batch.sample.prompt_embeds,
            pooled_projections=batch.sample.pooled_prompt_embeds,
            timestep=batch.timesteps,
            rgda_condition=condition,
            return_dict=True,
        ).sample
        self.forward_count += 1
        if prediction.shape != batch.target.shape:
            raise ValueError(f"prediction shape {prediction.shape} != target shape {batch.target.shape}")
        if not torch.isfinite(prediction).all():
            self.nan_inf_count += 1
            raise FloatingPointError("prediction contains non-finite values")
        loss = weighted_flow_matching_mse(prediction, batch.target, batch.weighting)
        if not loss.requires_grad:
            raise RuntimeError("loss.requires_grad is false")
        return loss

    def backward_step(self, batch: FlowBatch) -> dict[str, float]:
        metrics = self.backward_only(batch)
        self.optimizer_step()
        return metrics

    def backward_only(self, batch: FlowBatch) -> dict[str, float]:
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            loss = self.forward_loss(batch)
            if not torch.isfinite(loss):
                self.nan_inf_count += 1
                raise FloatingPointError("loss is not finite")
            loss.backward()  # type: ignore[no-untyped-call]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.injector.parameters(), 1.0))
            self.assert_base_gradients_none()
        except torch.cuda.OutOfMemoryError:
            self.oom_count += 1
            raise
        return {
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
            "allocated_mib": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mib": torch.cuda.memory_reserved() / 1024 / 1024,
            "step_seconds": time.perf_counter() - started,
        }

    def optimizer_step(self) -> None:
        self.optimizer.step()
        self.steps_completed += 1

    def gradient_report(self) -> dict[str, float]:
        return {
            name: _module_grad_norm(module)
            for name, module in self.injector.trainable_modules().items()
        }

    def parameter_delta(self, before: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            name: float((parameter.detach().cpu() - before[name]).abs().sum())
            for name, parameter in self.injector.named_parameters()
            if name in before
        }

    def base_parameter_hash(self) -> str:
        return hash_module_parameters(self.transformer)

    def save_checkpoint(self, path: str | Path, metadata: dict[str, str]) -> None:
        save_adapter_checkpoint(path, self.injector.trainable_modules(), metadata)

    def load_checkpoint(self, path: str | Path) -> dict[str, str]:
        return load_adapter_checkpoint(path, self.injector.trainable_modules())

    def compare_outputs(self, checkpoint: str | Path, batch: FlowBatch) -> float:
        with torch.no_grad():
            reference_output = self.wrapper(
                hidden_states=batch.noisy_latent,
                encoder_hidden_states=batch.sample.prompt_embeds,
                pooled_projections=batch.sample.pooled_prompt_embeds,
                timestep=batch.timesteps,
                rgda_condition=RGDAConditionBatch(
                    batch.sample.pseudo_clean_latent,
                    batch.sample.rg_map_latent,
                    batch.sample.token_mask,
                    batch.timesteps,
                ),
                return_dict=True,
            ).sample.detach().clone()
        fresh = RGDAInjector(
            token_dim=self.injector.adapter.config.token_dim,
            latent_channels=batch.sample.pseudo_clean_latent.shape[1],
            patch_size=self.injector.normal_encoder.patch.kernel_size[0],
        ).to(self.device)
        load_adapter_checkpoint(checkpoint, fresh.trainable_modules())
        original = self.injector
        self.injector = fresh
        self.wrapper.injector = fresh
        try:
            with torch.no_grad():
                restored = self.wrapper(
                    hidden_states=batch.noisy_latent,
                    encoder_hidden_states=batch.sample.prompt_embeds,
                    pooled_projections=batch.sample.pooled_prompt_embeds,
                    timestep=batch.timesteps,
                    rgda_condition=RGDAConditionBatch(
                        batch.sample.pseudo_clean_latent,
                        batch.sample.rg_map_latent,
                        batch.sample.token_mask,
                        batch.timesteps,
                    ),
                    return_dict=True,
                ).sample
            return float((reference_output - restored).abs().max().cpu())
        finally:
            self.injector = original
            self.wrapper.injector = original

    def assert_base_gradients_none(self) -> None:
        bad = [name for name, parameter in self.transformer.named_parameters() if parameter.grad is not None]
        if bad:
            raise RuntimeError(f"Base transformer gradients are not None: {bad[:3]}")

    def zero_init_equivalence(self, batch: FlowBatch) -> dict[str, float]:
        with torch.no_grad():
            base_output = self.wrapper(
                hidden_states=batch.noisy_latent,
                encoder_hidden_states=batch.sample.prompt_embeds,
                pooled_projections=batch.sample.pooled_prompt_embeds,
                timestep=batch.timesteps,
                return_dict=True,
            ).sample
            injected_output = self.wrapper(
                hidden_states=batch.noisy_latent,
                encoder_hidden_states=batch.sample.prompt_embeds,
                pooled_projections=batch.sample.pooled_prompt_embeds,
                timestep=batch.timesteps,
                rgda_condition=RGDAConditionBatch(
                    batch.sample.pseudo_clean_latent,
                    batch.sample.rg_map_latent,
                    batch.sample.token_mask,
                    batch.timesteps,
                ),
                return_dict=True,
            ).sample
        residual = self.injector.last_residual
        return {
            "FINAL_OUTPUT_MAX_ABS_DIFF": float((base_output - injected_output).abs().max().cpu()),
            "RGDA_RESIDUAL_MAX_ABS": float(residual.abs().max().cpu()) if residual is not None else float("inf"),
        }

    def two_step_gradient_gate(self, batch: FlowBatch) -> dict[str, float]:
        first = self.backward_only(batch)
        first_report = self.detailed_gradient_report("FIRST")
        self.optimizer_step()
        second = self.backward_only(batch)
        second_report = self.detailed_gradient_report("SECOND")
        self.optimizer_step()
        return {
            "FIRST_LOSS": first["loss"],
            "SECOND_LOSS": second["loss"],
            **first_report,
            **second_report,
        }

    def negative_mask_gate(self, batch: FlowBatch) -> dict[str, float]:
        metrics = self.backward_step(batch)
        report = self.gradient_report()
        normal_residual = self.injector.last_normal_residual
        defect_residual = self.injector.last_defect_residual
        total_residual = self.injector.last_residual
        return {
            "LOSS": metrics["loss"],
            "TOKEN_MASK_SUM": float(batch.sample.token_mask.abs().sum().cpu()),
            "RG_MAP_SUM": float(batch.sample.rg_map_latent.abs().sum().cpu()),
            "NORMAL_RESIDUAL_MAX_ABS": _max_abs_or_inf(normal_residual),
            "DEFECT_RESIDUAL_MAX_ABS": _max_abs_or_inf(defect_residual),
            "TOTAL_RESIDUAL_MAX_ABS": _max_abs_or_inf(total_residual),
            "DEFECT_ADAPTER_GRAD": report["defect_adapter"],
            "RG_ENCODER_GRAD": report["rg_encoder"],
            "NORMAL_ADAPTER_GRAD": report["normal_adapter"],
        }

    def detailed_gradient_report(self, prefix: str) -> dict[str, float]:
        normal_final = find_last_linear(self.injector.adapter.normal_adapter)
        defect_final = find_last_linear(self.injector.adapter.defect_adapter)
        normal_early = find_first_linear(self.injector.adapter.normal_adapter)
        defect_early = find_first_linear(self.injector.adapter.defect_adapter)
        return {
            f"{prefix}_NORMAL_ADAPTER_FINAL_GRAD": parameter_grad_norm(normal_final.weight),
            f"{prefix}_DEFECT_ADAPTER_FINAL_GRAD": parameter_grad_norm(defect_final.weight),
            f"{prefix}_NORMAL_ENCODER_GRAD": _module_grad_norm(self.injector.normal_encoder),
            f"{prefix}_RG_ENCODER_GRAD": _module_grad_norm(self.injector.rg_encoder),
            f"{prefix}_NORMAL_ADAPTER_EARLY_GRAD": parameter_grad_norm(normal_early.weight),
            f"{prefix}_DEFECT_ADAPTER_EARLY_GRAD": parameter_grad_norm(defect_early.weight),
            f"{prefix}_TIMESTEP_GATE_GRAD": _module_grad_norm(self.injector.timestep_gate),
        }


def run_training_loop(
    trainer: RealSD3RGDATrainer,
    cache_manifest: str | Path,
    steps: int,
    metrics_csv: str | Path,
) -> TrainingLoopSummary:
    rows = validate_cache_manifest(cache_manifest)
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainer.injector.named_parameters()}
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    with Path(metrics_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "sample_id",
                "loss",
                "grad_norm",
                "allocated_mib",
                "reserved_mib",
                "step_seconds",
            ],
        )
        writer.writeheader()
        for step in range(steps):
            row = rows[step % len(rows)]
            batch = trainer.build_flow_batch(trainer.load_cached_sample(row["cache_path"]))
            metrics = trainer.backward_step(batch)
            losses.append(metrics["loss"])
            writer.writerow({"step": step + 1, "sample_id": row["sample_id"], **metrics})
    return TrainingLoopSummary(
        losses=losses,
        parameter_snapshot=before,
        steps_completed=len(losses),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 / 1024,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024 / 1024,
    )


def build_fixed_flow_batches(
    trainer: RealSD3RGDATrainer,
    cache_manifest: str | Path,
    base_seed: int,
) -> list[FlowBatch]:
    batches: list[FlowBatch] = []
    rows = validate_cache_manifest(cache_manifest)
    for index, row in enumerate(rows):
        with torch.random.fork_rng(devices=[trainer.device]):
            torch.manual_seed(base_seed + index)
            batches.append(trainer.build_flow_batch(trainer.load_cached_sample(row["cache_path"])))
    return batches


def run_fixed_batch_training_loop(
    trainer: RealSD3RGDATrainer,
    fixed_batches: list[FlowBatch],
    steps: int,
    metrics_csv: str | Path,
) -> TrainingLoopSummary:
    if not fixed_batches:
        raise ValueError("fixed_batches must not be empty")
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainer.injector.named_parameters()}
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    with Path(metrics_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "sample_id", "loss", "grad_norm", "allocated_mib", "reserved_mib", "step_seconds"],
        )
        writer.writeheader()
        for step in range(steps):
            batch = fixed_batches[step % len(fixed_batches)]
            metrics = trainer.backward_step(batch)
            losses.append(metrics["loss"])
            writer.writerow({"step": step + 1, "sample_id": step % len(fixed_batches), **metrics})
    return TrainingLoopSummary(
        losses=losses,
        parameter_snapshot=before,
        steps_completed=len(losses),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 / 1024,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024 / 1024,
    )


def scheduler_report(scheduler: Any) -> dict[str, object]:
    config = getattr(scheduler, "config", object())
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas
    num_train_timesteps = int(getattr(config, "num_train_timesteps", len(timesteps)))
    if len(timesteps) < num_train_timesteps or len(sigmas) < num_train_timesteps:
        raise ValueError("Scheduler timesteps/sigmas shorter than num_train_timesteps")
    return {
        "SCHEDULER_CLASS": scheduler.__class__.__name__,
        "SCHEDULER_CONFIG": str(config),
        "NUM_TRAIN_TIMESTEPS": num_train_timesteps,
        "SCHEDULER_SHIFT": getattr(config, "shift", "missing"),
        "TIMESTEP_COUNT": len(timesteps),
        "SIGMA_COUNT": len(sigmas),
    }


def write_failure_report(
    report_dir: str | Path,
    stage: str,
    exception: BaseException,
    oom_count: int,
    nan_inf_count: int,
) -> None:
    allocated = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    reserved = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    lines = [
        "FINAL_VERDICT=FAIL",
        f"FAIL_STAGE={stage}",
        f"FAIL_EXCEPTION_TYPE={exception.__class__.__name__}",
        f"FAIL_EXCEPTION_MESSAGE={str(exception)[:500]}",
        f"OOM_COUNT={oom_count}",
        f"NAN_INF_COUNT={nan_inf_count}",
        f"CUDA_ALLOCATED_MIB={allocated:.2f}",
        f"CUDA_RESERVED_MIB={reserved:.2f}",
    ]
    target = Path(report_dir) / "07_FINAL_STATUS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_real_validation(
    *,
    model_path: Path,
    smoke_manifest: Path,
    micro_manifest: Path,
    negative_manifest: Path,
    report_dir: Path,
    resolution: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    pipe = load_sd3_pipeline(model_path, dtype)
    smoke_cache = cache_manifest_rows(pipe, smoke_manifest, report_dir / "cache_smoke8", resolution, dtype, patch_size=2)
    micro_cache = cache_manifest_rows(pipe, micro_manifest, report_dir / "cache_micro4", resolution, dtype, patch_size=2)
    negative_cache = cache_manifest_rows(pipe, negative_manifest, report_dir / "cache_negative1", resolution, dtype, patch_size=2)
    scheduler = copy.deepcopy(pipe.scheduler)
    transformer, stats = prepare_transformer_from_pipeline(pipe)
    scheduler_info = scheduler_report(scheduler)
    token_dim = int(getattr(transformer.config, "caption_projection_dim", 1536))
    latent_channels = int(getattr(transformer.config, "in_channels", 16))
    patch_size = int(getattr(transformer.config, "patch_size", 2))
    trainer = RealSD3RGDATrainer(transformer, scheduler, token_dim, latent_channels, patch_size, dtype)
    base_hash_before = trainer.base_parameter_hash()
    smoke_rows = validate_cache_manifest(smoke_cache.cache_manifest)
    first_batch = trainer.build_flow_batch(trainer.load_cached_sample(smoke_rows[0]["cache_path"]))
    zero = trainer.zero_init_equivalence(first_batch)
    two_step = trainer.two_step_gradient_gate(first_batch)
    negative_rows = validate_cache_manifest(negative_cache.cache_manifest)
    negative_batch = trainer.build_flow_batch(trainer.load_cached_sample(negative_rows[0]["cache_path"]))
    negative = trainer.negative_mask_gate(negative_batch)
    gates: list[GateResult] = [
        GateResult("SD3_FULL_LOAD", True, stats.__dict__),
        GateResult(
            "REAL_CZECH_CACHE",
            _cache_report_passed(smoke_cache) and _cache_report_passed(micro_cache) and _cache_report_passed(negative_cache),
            {"CACHE_ROWS": smoke_cache.cache_rows, "UNIQUE_PROMPT_COUNT": smoke_cache.unique_prompt_count},
        ),
        GateResult("MODEL_SCHEDULER_PRESERVED", True, scheduler_info),
        GateResult("PATCH_EMBED_INJECTION", True, {"PATCH_MODULE_NAME": stats.patch_module_name}),
        GateResult(
            "ZERO_INIT_EQUIVALENCE",
            zero["FINAL_OUTPUT_MAX_ABS_DIFF"] <= 1e-3 and zero["RGDA_RESIDUAL_MAX_ABS"] <= 1e-6,
            zero,
        ),
        GateResult(
            "TWO_STEP_GRADIENT",
            two_step["FIRST_NORMAL_ADAPTER_FINAL_GRAD"] > 0
            and two_step["FIRST_DEFECT_ADAPTER_FINAL_GRAD"] > 0
            and two_step["SECOND_NORMAL_ENCODER_GRAD"] > 0
            and two_step["SECOND_RG_ENCODER_GRAD"] > 0
            and two_step["SECOND_NORMAL_ADAPTER_EARLY_GRAD"] > 0
            and two_step["SECOND_DEFECT_ADAPTER_EARLY_GRAD"] > 0
            and two_step["SECOND_TIMESTEP_GATE_GRAD"] > 0,
            two_step,
        ),
        GateResult(
            "NEGATIVE_MASK_GATE",
            negative["TOKEN_MASK_SUM"] == 0.0
            and negative["RG_MAP_SUM"] == 0.0
            and negative["DEFECT_RESIDUAL_MAX_ABS"] == 0.0
            and negative["DEFECT_ADAPTER_GRAD"] == 0.0
            and negative["RG_ENCODER_GRAD"] == 0.0
            and negative["NORMAL_ADAPTER_GRAD"] > 0,
            negative,
        ),
    ]
    smoke_summary = run_training_loop(trainer, smoke_cache.cache_manifest, 100, report_dir / "smoke100_metrics.csv")
    smoke_delta = sum(trainer.parameter_delta(smoke_summary.parameter_snapshot).values())
    gates.append(
        GateResult(
            "SMOKE100",
            smoke_summary.steps_completed == 100 and smoke_delta > 0,
            {
                "STEPS_COMPLETED": smoke_summary.steps_completed,
                "ADAPTER_PARAMETER_DELTA": smoke_delta,
                "PEAK_ALLOCATED_MIB": smoke_summary.peak_allocated_mib,
                "PEAK_RESERVED_MIB": smoke_summary.peak_reserved_mib,
            },
        )
    )
    micro_trainer = RealSD3RGDATrainer(transformer, scheduler, token_dim, latent_channels, patch_size, dtype)
    fixed_batches = build_fixed_flow_batches(micro_trainer, micro_cache.cache_manifest, seed)
    micro_summary = run_fixed_batch_training_loop(micro_trainer, fixed_batches, 500, report_dir / "micro500_metrics.csv")
    first = torch.median(torch.tensor(micro_summary.losses[:50]))
    last = torch.median(torch.tensor(micro_summary.losses[-50:]))
    checkpoint = report_dir / "rgda_checkpoint.pt"
    micro_trainer.save_checkpoint(checkpoint, {"engine": "real_sd3_rgda"})
    checkpoint_sample = micro_trainer.load_cached_sample(validate_cache_manifest(micro_cache.cache_manifest)[0]["cache_path"])
    checkpoint_diff = micro_trainer.compare_outputs(checkpoint, micro_trainer.build_flow_batch(checkpoint_sample))
    base_hash_after = trainer.base_parameter_hash()
    gates.extend(
        [
            GateResult(
                "REAL_SD3_FORWARD",
                micro_trainer.forward_count >= 500,
                {"REAL_SD3_FORWARD_COUNT": micro_trainer.forward_count},
            ),
            GateResult(
                "MICRO_FIXED_FLOW_BATCH",
                len(fixed_batches) == 4,
                {
                    "MICRO_FLOW_BATCHES_FIXED": True,
                    "MICRO_FIXED_BATCH_COUNT": len(fixed_batches),
                    "MICRO_NOISE_HASHES": [hashlib.sha256(batch.noise.detach().cpu().numpy().tobytes()).hexdigest() for batch in fixed_batches],
                    "MICRO_TIMESTEPS": [float(batch.timesteps.flatten()[0].detach().cpu()) for batch in fixed_batches],
                    "MICRO_SIGMAS": [float(batch.sigmas.flatten()[0].detach().cpu()) for batch in fixed_batches],
                },
            ),
            GateResult(
                "MICRO_OVERFIT500",
                bool(last <= first * 0.70),
                {"MEDIAN_FIRST50": float(first), "MEDIAN_LAST50": float(last)},
            ),
            GateResult("CHECKPOINT_RELOAD", checkpoint_diff <= 1e-3, {"RELOAD_OUTPUT_MAX_ABS_DIFF": checkpoint_diff}),
            GateResult(
                "BASE_HASH_UNCHANGED",
                base_hash_before == base_hash_after,
                {
                    "BASE_HASH_BEFORE": base_hash_before,
                    "BASE_HASH_AFTER": base_hash_after,
                    "BASE_PARAMETER_HASH_UNCHANGED": base_hash_before == base_hash_after,
                },
            ),
        ]
    )
    final = summarize_gates(gates, micro_trainer.oom_count + trainer.oom_count, micro_trainer.nan_inf_count + trainer.nan_inf_count)
    write_gate_report(report_dir / "07_FINAL_STATUS.md", gates, final)
    if not final.passed:
        failed = [gate.name for gate in gates if not gate.passed]
        raise RuntimeError("SD3-RGDA validation gates failed: " + ",".join(failed))


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().cpu())
    return total


def _max_abs_or_inf(tensor: torch.Tensor | None) -> float:
    return float(tensor.abs().max().cpu()) if tensor is not None else float("inf")


def _cache_report_passed(report: CacheBuildReport) -> bool:
    return (
        report.all_cache_files_exist
        and report.all_tensors_finite
        and report.no_val_test_leakage
        and report.negative_rg_map_zero
        and report.negative_token_mask_zero
        and report.vae_device_during_encoding == "cuda"
        and report.text_encoder_device_during_encoding == "cuda"
    )

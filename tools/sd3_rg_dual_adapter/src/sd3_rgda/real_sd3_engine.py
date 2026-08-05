"""Executable real SD3-RGDA validation engine."""

from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from sd3_rgda.cache import (
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
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            loss = self.forward_loss(batch)
            loss.backward()  # type: ignore[no-untyped-call]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.injector.parameters(), 1.0))
            self.assert_base_gradients_none()
            self.optimizer.step()
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
        digest = hashlib.sha256()
        for parameter in self.transformer.parameters():
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

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
        first = self.backward_step(batch)
        second = self.backward_step(batch)
        report = self.gradient_report()
        return {
            "FIRST_LOSS": first["loss"],
            "SECOND_LOSS": second["loss"],
            "normal_encoder": report["normal_encoder"],
            "rg_encoder": report["rg_encoder"],
            "normal_adapter": report["normal_adapter"],
            "defect_adapter": report["defect_adapter"],
            "timestep_gate": report["timestep_gate"],
        }

    def negative_mask_gate(self, batch: FlowBatch) -> dict[str, float]:
        metrics = self.backward_step(batch)
        report = self.gradient_report()
        residual = self.injector.last_residual
        return {
            "LOSS": metrics["loss"],
            "TOKEN_MASK_SUM": float(batch.sample.token_mask.abs().sum().cpu()),
            "RG_MAP_SUM": float(batch.sample.rg_map_latent.abs().sum().cpu()),
            "RGDA_RESIDUAL_MAX_ABS": float(residual.abs().max().cpu()) if residual is not None else float("inf"),
            "defect_adapter": report["defect_adapter"],
            "rg_encoder": report["rg_encoder"],
            "normal_adapter": report["normal_adapter"],
        }


def run_training_loop(
    trainer: RealSD3RGDATrainer,
    cache_manifest: str | Path,
    steps: int,
    metrics_csv: str | Path,
) -> tuple[list[float], dict[str, torch.Tensor]]:
    rows = validate_cache_manifest(cache_manifest)
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainer.injector.named_parameters()}
    losses: list[float] = []
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
    return losses, before


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
    transformer, stats = prepare_transformer_from_pipeline(pipe)
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler()  # type: ignore[no-untyped-call]
    token_dim = int(getattr(transformer.config, "caption_projection_dim", 1536))
    latent_channels = int(getattr(transformer.config, "in_channels", 16))
    patch_size = int(getattr(transformer.config, "patch_size", 2))
    trainer = RealSD3RGDATrainer(transformer, scheduler, token_dim, latent_channels, patch_size, dtype)
    base_hash_before = trainer.base_parameter_hash()
    smoke_rows = validate_cache_manifest(smoke_cache)
    first_batch = trainer.build_flow_batch(trainer.load_cached_sample(smoke_rows[0]["cache_path"]))
    zero = trainer.zero_init_equivalence(first_batch)
    two_step = trainer.two_step_gradient_gate(first_batch)
    negative_rows = validate_cache_manifest(negative_cache)
    negative_batch = trainer.build_flow_batch(trainer.load_cached_sample(negative_rows[0]["cache_path"]))
    negative = trainer.negative_mask_gate(negative_batch)
    gates: list[GateResult] = [
        GateResult("SD3_FULL_LOAD", True, stats.__dict__),
        GateResult("REAL_CZECH_CACHE", len(smoke_rows) == 8, {"CACHE_ROWS": len(smoke_rows)}),
        GateResult("PATCH_EMBED_INJECTION", True, {"PATCH_MODULE_NAME": stats.patch_module_name}),
        GateResult(
            "ZERO_INIT_EQUIVALENCE",
            zero["FINAL_OUTPUT_MAX_ABS_DIFF"] <= 1e-3 and zero["RGDA_RESIDUAL_MAX_ABS"] <= 1e-6,
            zero,
        ),
        GateResult(
            "TWO_STEP_GRADIENT",
            all(two_step[name] > 0 for name in ["normal_encoder", "rg_encoder", "normal_adapter", "defect_adapter", "timestep_gate"]),
            two_step,
        ),
        GateResult(
            "NEGATIVE_MASK_GATE",
            negative["TOKEN_MASK_SUM"] == 0.0
            and negative["RG_MAP_SUM"] == 0.0
            and negative["defect_adapter"] == 0.0
            and negative["rg_encoder"] == 0.0
            and negative["normal_adapter"] > 0,
            negative,
        ),
    ]
    smoke_losses, smoke_before = run_training_loop(trainer, smoke_cache, 100, report_dir / "smoke100_metrics.csv")
    smoke_delta = sum(trainer.parameter_delta(smoke_before).values())
    gates.append(
        GateResult(
            "SMOKE100",
            len(smoke_losses) == 100 and smoke_delta > 0,
            {"STEPS_COMPLETED": len(smoke_losses), "ADAPTER_PARAMETER_DELTA": smoke_delta},
        )
    )
    micro_trainer = RealSD3RGDATrainer(transformer, scheduler, token_dim, latent_channels, patch_size, dtype)
    micro_losses, _micro_before = run_training_loop(micro_trainer, micro_cache, 500, report_dir / "micro500_metrics.csv")
    first = torch.median(torch.tensor(micro_losses[:50]))
    last = torch.median(torch.tensor(micro_losses[-50:]))
    checkpoint = report_dir / "rgda_checkpoint.pt"
    micro_trainer.save_checkpoint(checkpoint, {"engine": "real_sd3_rgda"})
    checkpoint_sample = micro_trainer.load_cached_sample(validate_cache_manifest(micro_cache)[0]["cache_path"])
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


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().cpu())
    return total

"""Real SD3 loading and forward helpers for GPU validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from sd3_rgda.injector import RGDAConditionBatch, RGDAInjector, RGDAPatchHook
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


def load_real_sd3_transformer(model_path: str | Path) -> tuple[nn.Module, SD3LoadStats]:
    from diffusers import SD3Transformer2DModel

    transformer = SD3Transformer2DModel.from_pretrained(  # type: ignore[no-untyped-call]
        str(model_path),
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    transformer.to("cuda")
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    patch_name, _patch_module = resolve_patch_embedding(transformer)
    first = next(transformer.parameters())
    stats = SD3LoadStats(
        transformer_class=transformer.__class__.__name__,
        parameter_count=sum(parameter.numel() for parameter in transformer.parameters()),
        dtype=str(first.dtype),
        device=str(first.device),
        patch_module_name=patch_name,
    )
    return transformer, stats


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
        timestep_embedding: torch.Tensor | None = None,
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
                    **kwargs,
                ),
            )
        if timestep_embedding is None:
            timestep_embedding = timestep.float().reshape(timestep.shape[0], -1)
        _patch_name, patch_module = resolve_patch_embedding(self.transformer)
        hook = RGDAPatchHook(self.injector, rgda_condition, timestep_embedding)
        with temporary_forward_hook(patch_module, hook):
            output = self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                **kwargs,
            )
        if hook.calls != 1:
            raise RuntimeError(f"RGDA patch hook call count was {hook.calls}, expected 1")
        return cast(SD3ForwardOutput, output)

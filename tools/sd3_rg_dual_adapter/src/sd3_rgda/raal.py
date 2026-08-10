from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self, cast

import torch
from torch import nn


@dataclass(frozen=True)
class RAALConfig:
    enabled: bool = True
    weight: float = 0.02
    temperature: float = 1.0
    layer_indices: tuple[int, ...] = (5, 11, 17)
    eps: float = 1e-8


@dataclass
class RAALLayerStats:
    layer_index: int
    loss: torch.Tensor
    forward_count: int


@dataclass
class RAALStats:
    layer_stats: list[RAALLayerStats] = field(default_factory=list)

    @property
    def loss(self) -> torch.Tensor | None:
        if not self.layer_stats:
            return None
        return torch.stack([item.loss for item in self.layer_stats]).mean()

    @property
    def hook_call_count(self) -> int:
        return len(self.layer_stats)


def raal_parameter_count() -> int:
    return 0


def _as_bool_mask(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape [B,N] or [B,N,1], got {tuple(mask.shape)}")
    return mask.to(device=device).bool()


def compute_raal_loss_from_logits(
    logits: torch.Tensor,
    token_mask: torch.Tensor,
    defect_text_mask: torch.Tensor,
    config: RAALConfig,
) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"logits must have shape [B,H,N_img,N_txt], got {tuple(logits.shape)}")
    if not config.enabled:
        return logits.sum() * 0.0
    spatial_mask = _as_bool_mask(token_mask, logits.device)
    text_mask = _as_bool_mask(defect_text_mask, logits.device)
    if spatial_mask.shape[0] != logits.shape[0] or spatial_mask.shape[1] != logits.shape[2]:
        raise ValueError("RAAL spatial token_mask shape does not match image tokens")
    if text_mask.shape[0] == 1 and logits.shape[0] != 1:
        text_mask = text_mask.expand(logits.shape[0], -1)
    if text_mask.shape != (logits.shape[0], logits.shape[3]):
        raise ValueError("RAAL defect_text_mask shape does not match text tokens")
    if not bool(spatial_mask.any()) or not bool(text_mask.any()):
        return logits.sum() * 0.0

    losses: list[torch.Tensor] = []
    tau = max(config.temperature, config.eps)
    for batch_index in range(logits.shape[0]):
        q = spatial_mask[batch_index].float()
        q = q / q.sum().clamp_min(config.eps)
        selected = logits[batch_index, :, :, text_mask[batch_index]]
        token_logits = torch.logsumexp(selected, dim=-1) - torch.log(
            torch.tensor(float(selected.shape[-1]), device=logits.device, dtype=logits.dtype)
        )
        z = token_logits.mean(dim=0)
        log_p = torch.log_softmax(z / tau, dim=0)
        losses.append(torch.sum(q * (torch.log(q + config.eps) - log_p)))
    return torch.stack(losses).mean()


def _split_qkv(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"projected Q/K must be [B,N,C], got {tuple(tensor.shape)}")
    batch, tokens, channels = tensor.shape
    if channels % heads != 0:
        raise ValueError("projected Q/K channels must be divisible by attention heads")
    return tensor.view(batch, tokens, heads, channels // heads).transpose(1, 2)


def _maybe_norm(norm: Any, tensor: torch.Tensor) -> torch.Tensor:
    if norm is None:
        return tensor
    norm_fn = cast(Callable[[torch.Tensor], torch.Tensor], norm)
    try:
        return norm_fn(tensor)
    except TypeError:
        return norm_fn(tensor.transpose(1, 2)).transpose(1, 2)


class RAALAttentionCollector:
    def __init__(
        self,
        transformer: nn.Module,
        config: RAALConfig,
        token_mask: torch.Tensor,
        defect_text_mask: torch.Tensor,
    ) -> None:
        self.transformer = transformer
        self.config = config
        self.token_mask = token_mask
        self.defect_text_mask = defect_text_mask
        self.stats = RAALStats()
        self._handles: list[Any] = []

    def __enter__(self) -> Self:
        if not self.config.enabled:
            return self
        blocks = cast(Any, self.transformer).transformer_blocks
        if not self.config.layer_indices:
            raise ValueError("RAAL layer_indices must not be empty")
        if max(self.config.layer_indices) >= len(blocks):
            raise ValueError("RAAL layer index exceeds transformer block count")
        for layer_index in self.config.layer_indices:
            attn = blocks[layer_index].attn
            self._handles.append(attn.register_forward_pre_hook(self._make_hook(layer_index), with_kwargs=True))
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_index: int) -> Callable[[nn.Module, tuple[Any, ...], dict[str, Any]], None]:
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            encoder_hidden_states = kwargs.get(
                "encoder_hidden_states",
                args[1] if len(args) > 1 else None,
            )
            if hidden_states is None or encoder_hidden_states is None:
                raise ValueError("RAAL hook requires hidden_states and encoder_hidden_states")
            attn = cast(Any, module)
            q_img = attn.to_q(hidden_states)
            k_txt = attn.add_k_proj(encoder_hidden_states)
            heads = int(attn.heads)
            q_img = _maybe_norm(getattr(attn, "norm_q", None), _split_qkv(q_img, heads))
            k_txt = _maybe_norm(getattr(attn, "norm_added_k", None), _split_qkv(k_txt, heads))
            logits = torch.matmul(q_img, k_txt.transpose(-1, -2)) * float(attn.scale)
            loss = compute_raal_loss_from_logits(logits, self.token_mask, self.defect_text_mask, self.config)
            self.stats.layer_stats.append(
                RAALLayerStats(layer_index=layer_index, loss=loss, forward_count=self.stats.hook_call_count + 1)
            )

        return hook

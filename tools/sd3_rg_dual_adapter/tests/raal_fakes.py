from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class OffsetTokenizer:
    def __call__(self, prompt: str, **kwargs: object) -> dict[str, list[list[tuple[int, int]]]]:
        max_length = int(kwargs.get("max_length", 256))
        offsets: list[tuple[int, int]] = []
        position = 0
        for part in prompt.split(" "):
            start = prompt.find(part, position)
            end = start + len(part)
            offsets.append((start, end))
            position = end
        offsets.extend([(0, 0)] * max(0, max_length - len(offsets)))
        return {"offset_mapping": [offsets[:max_length]]}


class FakeAttention(nn.Module):
    def __init__(self, dim: int = 4, heads: int = 2) -> None:
        super().__init__()
        self.heads = heads
        self.scale = 0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.add_k_proj = nn.Linear(dim, dim, bias=False)
        self.base_weight = nn.Parameter(torch.ones(1), requires_grad=False)
        nn.init.eye_(self.to_q.weight)
        nn.init.eye_(self.add_k_proj.weight)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + encoder_hidden_states.mean() * 0.0 + self.base_weight * 0.0


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = FakeAttention()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.attn(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)


class FakeTransformer(nn.Module):
    def __init__(self, blocks: int = 20) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([FakeBlock() for _ in range(blocks)])

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)
        return hidden_states


@dataclass
class FakeSample:
    token_mask: torch.Tensor
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    pseudo_clean_latent: torch.Tensor
    rg_map_latent: torch.Tensor
    is_negative: bool
    class_ids: tuple[int, ...]


@dataclass
class FakeBatch:
    sample: FakeSample
    noisy_latent: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    target: torch.Tensor
    weighting: torch.Tensor

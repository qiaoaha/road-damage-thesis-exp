from __future__ import annotations

import torch

from sd3_rgda.raal import RAALAttentionCollector, RAALConfig


class FakeAttention(torch.nn.Module):
    def __init__(self, dim: int = 4, heads: int = 2) -> None:
        super().__init__()
        self.heads = heads
        self.scale = 0.5
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.add_k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.base_weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        torch.nn.init.eye_(self.to_q.weight)
        torch.nn.init.eye_(self.add_k_proj.weight)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + encoder_hidden_states.mean() * 0.0 + self.base_weight * 0.0


class FakeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = FakeAttention()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.attn(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)


class FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([FakeBlock() for _ in range(20)])

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)
        return hidden_states


def run() -> None:
    torch.manual_seed(2026)
    transformer = FakeTransformer()
    alpha = torch.nn.Parameter(torch.tensor(1.0))
    hidden = torch.randn(1, 4, 4)
    text = torch.randn(1, 12, 4) * alpha
    token_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    text_mask = torch.zeros(1, 12, dtype=torch.bool)
    text_mask[:, 2:5] = True
    with RAALAttentionCollector(transformer, RAALConfig(), token_mask, text_mask) as collector:
        transformer(hidden, text)
    if collector.stats.loss is None:
        raise RuntimeError("RAAL dry integration did not collect loss")
    collector.stats.loss.backward()
    if alpha.grad is None or float(alpha.grad.abs()) <= 0.0:
        raise RuntimeError("RAAL gradient did not reach auxiliary RGDA parameter")
    if transformer.transformer_blocks[5].attn.base_weight.grad is not None:
        raise RuntimeError("Frozen base attention weight received a gradient")

    negative_text_mask = torch.zeros_like(text_mask)
    with RAALAttentionCollector(transformer, RAALConfig(), torch.zeros_like(token_mask), negative_text_mask) as negative:
        transformer(hidden, text.detach())
    if negative.stats.loss is None or float(negative.stats.loss.detach()) != 0.0:
        raise RuntimeError("Negative RAAL loss is not exactly zero")
    print("RAAL_DRY_INTEGRATION=PASS")
    print("REAL_SD3_USED=NO")
    print("GPU_USED=NO")


if __name__ == "__main__":
    run()

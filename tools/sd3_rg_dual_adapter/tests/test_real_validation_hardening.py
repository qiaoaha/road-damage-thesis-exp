from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from sd3_rgda.injector import RGDAConditionBatch, RGDAInjector
from sd3_rgda.sd3_hook import temporary_forward_hook

ROOT = Path(__file__).resolve().parents[1]


def test_no_toy_gpu_validation_code() -> None:
    forbidden = [
        "token_dim = 64",
        "target = 0.05",
        "run_adapter_steps",
        "--no-load-weights",
        "base_frozen = True",
        'status = "PASS"',
        "PENDING_GPU_IMPLEMENTATION",
    ]
    paths = [*ROOT.glob("scripts/*.py"), ROOT / "run_sd3_rgda_gpu_validation.sh"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for pattern in forbidden:
        assert pattern not in text


def test_hook_cleanup_on_exception() -> None:
    module = nn.Linear(2, 2)

    def failing_hook(_module: nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"), temporary_forward_hook(module, failing_hook):
        module(torch.ones(1, 2))
    assert len(module._forward_hooks) == 0


def test_two_step_gradient_unlock() -> None:
    injector = RGDAInjector(token_dim=8, latent_channels=4, patch_size=1)
    condition = RGDAConditionBatch(
        pseudo_clean_latents=torch.randn(1, 4, 2, 2),
        rg_maps=torch.randn(1, 7, 2, 2),
        token_mask=torch.ones(1, 4, 1),
    )
    h0 = torch.randn(1, 4, 8)
    optimizer = torch.optim.AdamW(injector.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = injector(h0, condition, torch.randn(1, 16))
        output.square().mean().backward()
        optimizer.step()
    assert injector.normal_encoder.patch.weight.grad is not None
    assert injector.rg_encoder.net[0].weight.grad is not None


def test_negative_branch_zero() -> None:
    injector = RGDAInjector(token_dim=8, latent_channels=4, patch_size=1)
    condition = RGDAConditionBatch(
        pseudo_clean_latents=torch.randn(1, 4, 2, 2),
        rg_maps=torch.randn(1, 7, 2, 2),
        token_mask=torch.zeros(1, 4, 1),
    )
    h0 = torch.randn(1, 4, 8)
    output = injector(h0, condition, torch.randn(1, 16))
    residual = output - h0
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-6, rtol=0)

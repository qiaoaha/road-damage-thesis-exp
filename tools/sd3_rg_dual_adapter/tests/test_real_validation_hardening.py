from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from sd3_rgda.injector import RGDAConditionBatch, RGDAInjector
from sd3_rgda.sd3_hook import temporary_forward_hook
from sd3_rgda.timestep_gate import SinusoidalTimestepEmbedding

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


def test_cache_script_contains_real_image_loading() -> None:
    text = (ROOT / "src" / "sd3_rgda" / "cache.py").read_text(encoding="utf-8")
    assert "Image.open" in text
    assert "read_yolo_boxes" in text
    assert "build_rg_map" in text


def test_cache_script_contains_vae_encode() -> None:
    text = (ROOT / "src" / "sd3_rgda" / "cache.py").read_text(encoding="utf-8")
    assert "vae.encode" in text


def test_cache_script_contains_prompt_encode() -> None:
    text = (ROOT / "src" / "sd3_rgda" / "cache.py").read_text(encoding="utf-8")
    assert "encode_prompt" in text


def test_gpu_shell_has_no_dry_run_in_real_branch() -> None:
    text = (ROOT / "run_sd3_rgda_gpu_validation.sh").read_text(encoding="utf-8")
    assert "--dry-run" not in text
    assert "scripts/run_real_sd3_validation.py" in text


def test_real_engine_calls_wrapper_forward() -> None:
    text = (ROOT / "src" / "sd3_rgda" / "real_sd3_engine.py").read_text(encoding="utf-8")
    assert "self.wrapper(" in text


def test_real_engine_calls_loss_backward() -> None:
    text = (ROOT / "src" / "sd3_rgda" / "real_sd3_engine.py").read_text(encoding="utf-8")
    assert "loss.backward()" in text


def test_real_engine_calls_optimizer_step() -> None:
    text = (ROOT / "src" / "sd3_rgda" / "real_sd3_engine.py").read_text(encoding="utf-8")
    assert "self.optimizer.step()" in text


def test_real_engine_has_step_loop() -> None:
    text = (ROOT / "src" / "sd3_rgda" / "real_sd3_engine.py").read_text(encoding="utf-8")
    assert "for step in range(steps):" in text


def test_micro500_enforces_30_percent_drop() -> None:
    text = (ROOT / "src" / "sd3_rgda" / "real_sd3_engine.py").read_text(encoding="utf-8")
    assert "last <= first * 0.70" in text


def test_checkpoint_compares_real_outputs() -> None:
    text = (ROOT / "src" / "sd3_rgda" / "real_sd3_engine.py").read_text(encoding="utf-8")
    assert "RELOAD_OUTPUT_MAX_ABS_DIFF" in text
    assert "reference_output" in text


def test_final_status_uses_gate_results() -> None:
    text = (ROOT / "src" / "sd3_rgda" / "real_sd3_engine.py").read_text(encoding="utf-8")
    assert "GateResult" in text
    assert "summarize_gates" in text


def test_timestep_scalar_to_16d_embedding() -> None:
    embedding = SinusoidalTimestepEmbedding(embed_dim=16)
    assert embedding(torch.tensor([1.0, 2.0])).shape == (2, 16)
    assert embedding(torch.tensor([[1.0], [2.0]])).shape == (2, 16)


def test_no_entry_ready_as_final_pass() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "src" / "sd3_rgda" / "real_sd3_engine.py",
            ROOT / "scripts" / "run_real_sd3_validation.py",
            ROOT / "run_sd3_rgda_gpu_validation.sh",
        ]
    )
    forbidden = [
        "PENDING_REAL_FORWARD_LOOP",
        "READY_TO_COMPARE",
        "ENTRY_CHECK_ONLY",
        "REAL_CZECH_CACHE=ENTRY",
        "FINAL_VERDICT=SD3_RGDA_GPU_VALIDATION_FAIL",
    ]
    for pattern in forbidden:
        assert pattern not in source


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
        timesteps=torch.ones(1),
    )
    h0 = torch.randn(1, 4, 8)
    optimizer = torch.optim.AdamW(injector.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = injector(h0, condition)
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
        timesteps=torch.ones(1),
    )
    h0 = torch.randn(1, 4, 8)
    output = injector(h0, condition)
    residual = output - h0
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-6, rtol=0)

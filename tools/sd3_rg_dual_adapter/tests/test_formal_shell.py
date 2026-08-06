from __future__ import annotations

from pathlib import Path


def test_formal_shell_defaults_to_real_and_has_dry_opt_in() -> None:
    text = (Path(__file__).resolve().parents[1] / "run_sd3_rgda_formal5000.sh").read_text(encoding="utf-8")
    assert "--steps 5000" in text
    assert "--checkpoint-interval 250" in text
    assert "--eval-interval 250" in text
    assert "FORMAL_DRY_INTEGRATION" in text
    assert "--dry-integration" in text
    assert text.index("FORMAL_DRY_INTEGRATION") < text.index("--dry-integration")


def test_formal_shell_uses_pytorch_cuda_strong_gate() -> None:
    text = (Path(__file__).resolve().parents[1] / "run_sd3_rgda_formal5000.sh").read_text(encoding="utf-8")
    assert "PYTORCH_CUDA_STRONG_GATE=PASS" in text
    assert "torch.ones(1, device=\"cuda\")" in text
    assert "GPU_GATE_OVERRIDE=PYTORCH_STRONG_GATE" in text

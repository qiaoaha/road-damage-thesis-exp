from __future__ import annotations

from pathlib import Path


def test_shell_real_default() -> None:
    text = (Path(__file__).resolve().parents[1] / "run_sd3_rgda_pilot1000.sh").read_text(encoding="utf-8")
    assert 'cd "${ROOT}"' in text
    assert '--steps 1000' in text
    assert '--model-path "${MODEL_PATH}"' in text
    assert "nvidia-smi -L" in text
    assert 'PILOT_DRY_INTEGRATION:-0}" = "1"' in text


def test_shell_dry_opt_in() -> None:
    text = (Path(__file__).resolve().parents[1] / "run_sd3_rgda_pilot1000.sh").read_text(encoding="utf-8")
    assert "train_rgda_pilot1000.py" in text
    dry_index = text.index("--dry-integration")
    condition_index = text.index("PILOT_DRY_INTEGRATION")
    assert condition_index < dry_index


def test_shell_torch_gate_authoritative() -> None:
    text = (Path(__file__).resolve().parents[1] / "run_sd3_rgda_pilot1000.sh").read_text(encoding="utf-8")
    assert "PYTORCH_CUDA_STRONG_GATE=PASS" in text
    assert "torch.cuda.is_available()" in text
    assert "torch.cuda.device_count()" in text
    assert "torch.cuda.get_device_properties(0)" in text
    assert "torch.ones(1, device=\"cuda\")" in text
    assert "torch.cuda.synchronize()" in text


def test_shell_nvidia_smi_not_sole_gate() -> None:
    text = (Path(__file__).resolve().parents[1] / "run_sd3_rgda_pilot1000.sh").read_text(encoding="utf-8")
    assert "NVIDIA_SMI_RC=$?" in text
    assert "NVIDIA_SMI_TEXT_GATE=FALSE_NEGATIVE" in text
    assert "GPU_GATE_OVERRIDE=PYTORCH_STRONG_GATE" in text
    assert text.index("nvidia-smi -L") < text.index("PYTORCH_CUDA_STRONG_GATE=PASS")

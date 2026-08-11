from __future__ import annotations

from pathlib import Path


def _script() -> str:
    return (Path(__file__).resolve().parents[1] / "run_rgda_raal_formal5000.sh").read_text(encoding="utf-8")


def test_shell_prefers_project_python() -> None:
    text = _script()
    project_python = "/root/autodl-tmp/road_damage_exp/envs/sd3_bgpaste_py311/bin/python3.11"
    assert project_python in text
    assert text.index(project_python) < text.index("command -v python3.11")
    assert "PYTHON_EXECUTABLE=" in text
    assert "PYTHON_VERSION=" in text
    assert "TORCH_VERSION=" in text
    assert "TORCH_CUDA_VERSION=" in text


def test_shell_dry_skips_gpu_gate_and_real_requires_5090() -> None:
    text = _script()
    assert text.index('if [ "${RAAL_FORMAL_DRY_INTEGRATION:-0}" = "1" ]') < text.index("torch.cuda.is_available()")
    assert '"5090" not in name' in text
    assert "GPU_ENV=PASS" in text
    assert "DEPLOYED_COMMIT.txt" in text

from __future__ import annotations

from pathlib import Path


def test_gpu_shell_uses_root_workdir_and_absolute_scripts() -> None:
    text = (Path(__file__).resolve().parents[1] / "run_sd3_rgda_gpu_validation.sh").read_text(encoding="utf-8")
    assert 'cd "${ROOT}"' in text
    assert '"${ROOT}/scripts/package_gpu_validation.py"' in text
    assert '"${ROOT}/scripts/build_czech_manifest.py"' in text
    assert '"${ROOT}/scripts/run_real_sd3_validation.py"' in text
    assert "--dry-run" not in text
    assert "trap package_on_exit EXIT" in text

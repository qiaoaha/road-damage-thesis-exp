from __future__ import annotations

from pathlib import Path


def test_shell_has_exit_trap_and_no_real_branch_dry_run() -> None:
    text = (Path(__file__).resolve().parents[1] / "run_sd3_rgda_gpu_validation.sh").read_text(encoding="utf-8")
    assert "trap package_on_exit EXIT" in text
    assert "--dry-run" not in text

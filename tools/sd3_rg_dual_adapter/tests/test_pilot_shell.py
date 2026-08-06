from __future__ import annotations

from pathlib import Path


def test_pilot_shell_is_dry_integration_only_for_nogpu() -> None:
    text = (Path(__file__).resolve().parents[1] / "run_sd3_rgda_pilot1000.sh").read_text(encoding="utf-8")
    assert 'cd "${ROOT}"' in text
    assert "--dry-integration" in text
    assert "train_rgda_pilot1000.py" in text
    assert "run_real_sd3_validation.py" not in text

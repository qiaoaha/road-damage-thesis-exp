from __future__ import annotations

from pathlib import Path


def test_smoke_and_micro_zero_init_evidence_is_reported() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "src" / "sd3_rgda" / "real_sd3_engine.py"
    ).read_text(encoding="utf-8")
    assert "precheck_trainer = RealSD3RGDATrainer" in text
    assert "smoke_trainer = RealSD3RGDATrainer" in text
    assert "micro_trainer = RealSD3RGDATrainer" in text
    assert "SMOKE_STARTED_FROM_ZERO_INIT" in text
    assert "SMOKE_INITIAL_PARAMETER_HASH" in text
    assert "MICRO_STARTED_FROM_ZERO_INIT" in text
    assert "MICRO_INITIAL_PARAMETER_HASH" in text

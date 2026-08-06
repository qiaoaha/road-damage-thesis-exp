from __future__ import annotations

from pathlib import Path


def test_pilot_cache_script_declares_real_cache_contract() -> None:
    text = (Path(__file__).resolve().parents[1] / "scripts" / "cache_pilot1000_inputs.py").read_text(encoding="utf-8")
    assert "torch.cuda.is_available()" in text
    assert "Pilot1000 real cache requires CUDA" in text
    assert "train512" in text
    assert "eval64" in text
    assert "cache_pilot_manifest_rows" in text
    assert "--clean-proxy-manifest" in text

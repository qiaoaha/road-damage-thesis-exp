from __future__ import annotations

import pytest

from sd3_rgda.real_sd3_engine import GateResult, summarize_gates, write_failure_report


def test_failed_gate_is_not_passed() -> None:
    final = summarize_gates([GateResult("A", False, {})], oom_count=0, nan_inf_count=0)
    assert not final.passed


def test_failure_report_exists(tmp_path) -> None:  # type: ignore[no-untyped-def]
    write_failure_report(tmp_path, "unit", RuntimeError("boom"), oom_count=1, nan_inf_count=2)
    report = (tmp_path / "08_FAILURE_SUMMARY.md").read_text(encoding="utf-8")
    assert "FINAL_VERDICT=FAIL" in report
    assert "FAIL_STAGE=unit" in report
    assert "OOM_COUNT=1" in report
    assert "NAN_INF_COUNT=2" in report


def test_failure_report_preserves_existing_final_status(tmp_path) -> None:  # type: ignore[no-untyped-def]
    final_status = tmp_path / "07_FINAL_STATUS.md"
    original = "FINAL_VERDICT=PASS\nREAL_GATE=PASS\n"
    final_status.write_text(original, encoding="utf-8")
    path = write_failure_report(tmp_path, "load_pipeline", ValueError("nope"), oom_count=3, nan_inf_count=4)
    assert path.name == "08_FAILURE_SUMMARY.md"
    assert final_status.read_text(encoding="utf-8") == original
    failure = path.read_text(encoding="utf-8")
    assert "DETAILED_GATE_REPORT=07_FINAL_STATUS.md" in failure
    assert "DETAILED_GATE_REPORT_EXISTS=PASS" in failure


def test_failed_final_can_raise() -> None:
    final = summarize_gates([GateResult("A", False, {})], 0, 0)
    with pytest.raises(RuntimeError):
        if not final.passed:
            raise RuntimeError("SD3-RGDA validation gates failed: A")

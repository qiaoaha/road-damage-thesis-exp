from __future__ import annotations

from sd3_rgda.formal_engine import (
    audit_formal_cache,
    audit_formal_manifest,
    audit_formal_proxy,
    audit_formal_schedule,
)


def test_formal_asset_audits_fail_missing_files(tmp_path) -> None:
    assert audit_formal_manifest(tmp_path / "missing.json") == "FAIL"
    assert audit_formal_schedule(tmp_path / "missing.json") == "FAIL"
    assert audit_formal_proxy(tmp_path / "missing.json") == "FAIL"
    assert audit_formal_cache(tmp_path / "missing.json") == "FAIL"

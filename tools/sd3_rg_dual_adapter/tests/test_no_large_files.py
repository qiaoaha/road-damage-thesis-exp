from __future__ import annotations

from pathlib import Path

import pytest

from sd3_rgda.validation import assert_no_large_or_forbidden_files


def test_forbidden_suffix_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        assert_no_large_or_forbidden_files([path])


def test_project_tracked_files_small() -> None:
    root = Path(__file__).resolve().parents[1]
    files = [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]
    assert_no_large_or_forbidden_files(files)

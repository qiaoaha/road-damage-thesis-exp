from __future__ import annotations

from pathlib import Path

from sd3_rgda.pilot_manifest import DatasetCandidate, SelectedCandidate, _row_dict


def test_multilabel_anchor_assignment_preserves_selected_class(tmp_path: Path) -> None:
    candidate = DatasetCandidate(tmp_path / "a.jpg", tmp_path / "a.txt", "train", (0, 1), 2)
    row = _row_dict(0, SelectedCandidate(candidate, 1), 2026, "pilot_train")
    assert row["class_ids"] == "0 1"
    assert row["anchor_class"] == "D10"

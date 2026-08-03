from __future__ import annotations

import csv

import pytest

from sd3_rgda.dataset import assert_no_val_test_training_rows, read_manifest


def test_train_only_filters_val_test(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label", "split"])
        writer.writeheader()
        writer.writerow({"image": "a.jpg", "label": "a.txt", "split": "train"})
        writer.writerow({"image": "b.jpg", "label": "b.txt", "split": "val"})
        writer.writerow({"image": "c.jpg", "label": "c.txt", "split": "test"})
    rows = read_manifest(path, train_only=True)
    assert len(rows) == 1
    assert_no_val_test_training_rows(rows)
    with pytest.raises(ValueError):
        assert_no_val_test_training_rows(read_manifest(path, train_only=False))

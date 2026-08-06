from __future__ import annotations

from formal_test_utils import make_yolo_dataset

from sd3_rgda.formal_manifest import build_formal_manifests


def test_formal_manifest_uses_train_and_val_without_test(tmp_path) -> None:
    make_yolo_dataset(tmp_path, train=30, val=400, test=7)
    train, val, eval128, summary = build_formal_manifests(tmp_path)
    assert len(train) == 30
    assert len(val) == 400
    assert len(eval128) == 128
    assert summary.train_test_overlap == 0
    assert summary.val_test_overlap == 0
    assert summary.train_source_split_only == "PASS"

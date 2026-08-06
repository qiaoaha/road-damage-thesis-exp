from __future__ import annotations

from collections import Counter

from formal_test_utils import make_yolo_dataset

from sd3_rgda.formal_manifest import build_formal_manifests


def test_formal_eval128_balances_val_anchors(tmp_path) -> None:
    make_yolo_dataset(tmp_path, train=30, val=400, test=4)
    _train, _val, eval128, summary = build_formal_manifests(tmp_path)
    assert len(eval128) == 128
    assert sum(row["is_negative"] == "true" for row in eval128) == 64
    assert Counter(row["anchor_class"] for row in eval128 if row["anchor_class"]) == Counter({"D00": 16, "D10": 16, "D20": 16, "D40": 16})
    assert summary.eval128_positive == 64

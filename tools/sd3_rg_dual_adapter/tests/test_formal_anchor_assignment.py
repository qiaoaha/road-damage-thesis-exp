from __future__ import annotations

from formal_test_utils import make_yolo_dataset

from sd3_rgda.formal_manifest import build_formal_manifests


def test_formal_anchor_assignment_is_deterministic(tmp_path) -> None:
    make_yolo_dataset(tmp_path, train=40, val=400, test=4)
    first, _val, _eval, summary = build_formal_manifests(tmp_path, seed=2026)
    second, _val2, _eval2, summary2 = build_formal_manifests(tmp_path, seed=2026)
    assert [row["anchor_class"] for row in first] == [row["anchor_class"] for row in second]
    assert summary.train_anchor_pool_counts == summary2.train_anchor_pool_counts
    assert all(row["anchor_class"] for row in first if row["is_negative"] == "false")

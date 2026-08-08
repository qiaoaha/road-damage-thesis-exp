from __future__ import annotations

from formal_test_utils import make_yolo_dataset

from sd3_rgda.formal_manifest import (
    build_formal_manifests,
    collect_split_candidates,
    summarize_formal,
)


def test_formal_no_test_leakage(tmp_path) -> None:
    make_yolo_dataset(tmp_path, train=30, val=400, test=10)
    train, val, eval128, summary = build_formal_manifests(tmp_path)
    test_paths = {str(path.resolve()) for path in (tmp_path / "images" / "test").glob("*.jpg")}
    assert not ({row["image_path"] for row in train + val + eval128} & test_paths)
    assert summary.eval_test_overlap == 0
    assert summary.test_leakage == 0


def test_formal_test_leakage_is_derived_for_eval_rows(tmp_path) -> None:
    make_yolo_dataset(tmp_path, train=30, val=400, test=10)
    train, val, eval128, _summary = build_formal_manifests(tmp_path)
    test_candidate = collect_split_candidates(tmp_path, "test")[0]
    leaked = [{**eval128[0], "image_path": str(test_candidate.image_path.resolve())}, *eval128[1:]]
    summary = summarize_formal(tmp_path, train, val, leaked, [test_candidate])
    assert summary.eval_test_overlap == 1
    assert summary.test_leakage == 1

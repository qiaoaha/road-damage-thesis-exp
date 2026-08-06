from __future__ import annotations

from formal_test_utils import make_yolo_dataset

from sd3_rgda.formal_manifest import build_formal_manifests


def test_formal_no_test_leakage(tmp_path) -> None:
    make_yolo_dataset(tmp_path, train=30, val=400, test=10)
    train, val, eval128, summary = build_formal_manifests(tmp_path)
    test_paths = {str(path.resolve()) for path in (tmp_path / "images" / "test").glob("*.jpg")}
    assert not ({row["image_path"] for row in train + val + eval128} & test_paths)
    assert summary.test_leakage == 0

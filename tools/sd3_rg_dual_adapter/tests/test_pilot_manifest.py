from __future__ import annotations

from pathlib import Path

from PIL import Image

from sd3_rgda.pilot_manifest import collect_train_candidates, select_pilot_manifests


def _write_sample(root: Path, index: int, label: str) -> None:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (index % 255, 10, 20)).save(image_dir / f"{index:04d}.jpg")
    (label_dir / f"{index:04d}.txt").write_text(label, encoding="utf-8")


def test_pilot_manifest_balances_unique_train_eval(tmp_path: Path) -> None:
    index = 0
    for class_id in range(4):
        for _ in range(72):
            _write_sample(tmp_path, index, f"{class_id} 0.5 0.5 0.25 0.25\n")
            index += 1
    for _ in range(300):
        _write_sample(tmp_path, index, "")
        index += 1
    candidates = collect_train_candidates(tmp_path)
    train, eval_rows, summary = select_pilot_manifests(candidates)
    assert len(train) == 512
    assert len(eval_rows) == 64
    assert summary.train_unique_images == 512
    assert summary.eval_unique_images == 64
    assert summary.train_eval_overlap == 0
    assert summary.train_positive == 256
    assert summary.train_negative == 256
    assert summary.class_minimum_gate == "PASS"

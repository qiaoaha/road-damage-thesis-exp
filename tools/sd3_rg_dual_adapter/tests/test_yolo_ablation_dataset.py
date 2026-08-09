from pathlib import Path

from PIL import Image

from sd3_rgda.generation_audit import build_yolo_ablation_dataset


def test_yolo_dataset_keeps_synthetic_out_of_val_test(tmp_path: Path) -> None:
    real = tmp_path / "real"
    make_yolo(real, train=10, val=4, test=4)
    synth_i = tmp_path / "synth" / "images"
    synth_l = tmp_path / "synth" / "labels"
    synth_i.mkdir(parents=True)
    synth_l.mkdir(parents=True)
    for index in range(6):
        Image.new("RGB", (16, 16), (index, 1, 1)).save(synth_i / f"s{index}.jpg")
        (synth_l / f"s{index}.txt").write_text("", encoding="utf-8")
    audit = build_yolo_ablation_dataset(
        real,
        tmp_path / "out",
        group="rgda",
        synthetic_images=synth_i,
        synthetic_labels=synth_l,
        link_mode="copy",
        expected_real_train=10,
        expected_val=4,
        expected_test=4,
        expected_synthetic=6,
    )
    assert audit.dataset_gate == "PASS"
    assert audit.train_images == 16
    assert audit.val_images == 4
    assert audit.test_images == 4
    assert audit.synthetic_val_test_leakage == 0


def make_yolo(root: Path, *, train: int, val: int, test: int) -> None:
    for split, count in [("train", train), ("val", val), ("test", test)]:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (16, 16), (index, 0, 0)).save(root / "images" / split / f"{split}_{index}.jpg")
            (root / "labels" / split / f"{split}_{index}.txt").write_text("", encoding="utf-8")

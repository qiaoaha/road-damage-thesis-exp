from pathlib import Path

from sd3_rgda.generation_manifest import build_generation1000_rows, prompt_for_sample


def test_generation1000_manifest_uses_all_positive_and_250_negative(tmp_path: Path) -> None:
    make_czech_dataset(tmp_path)
    rows, summary = build_generation1000_rows(tmp_path, "abc", width=64, height=64)
    assert summary.manifest_gate == "PASS"
    assert summary.total == 1000
    assert summary.positive == 750
    assert summary.negative == 250
    assert summary.val_leakage == 0
    assert summary.test_leakage == 0
    assert rows[0]["seed"] == "202600000"
    assert rows[-1]["seed"] == "202600999"


def test_prompt_templates_are_deterministic() -> None:
    assert prompt_for_sample(True, "", "") == "a clean Czech road surface without visible road damage"
    assert prompt_for_sample(False, "0", "D00") == "a Czech road surface with longitudinal crack damage"
    assert prompt_for_sample(False, "0 3", "D00") == "a Czech road surface with longitudinal crack and pothole damage"


def make_czech_dataset(root: Path) -> None:
    from PIL import Image

    for split, count in [("train", 1980), ("val", 4), ("test", 4)]:
        split_offset = {"train": 0, "val": 77, "test": 155}[split]
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            image = root / "images" / split / f"{split}_{index:04d}.png"
            label = root / "labels" / split / f"{split}_{index:04d}.txt"
            img = Image.new(
                "RGB",
                (24, 24),
                ((index + split_offset) % 255, (index // 255) % 255, split_offset),
            )
            img.putpixel((index % 24, (index // 24) % 24), (index % 255, (index // 255) % 255, split_offset))
            img.save(image)
            if split == "train" and index < 750:
                label.write_text(f"{index % 4} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            elif split != "train" and index % 2:
                label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            else:
                label.write_text("", encoding="utf-8")

from __future__ import annotations

from pathlib import Path

from PIL import Image


def make_yolo_dataset(root: Path, *, train: int = 20, val: int = 12, test: int = 5) -> None:
    for split, count in [("train", train), ("val", val), ("test", test)]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = root / "images" / split / f"{split}_{index:04d}.jpg"
            label = root / "labels" / split / f"{split}_{index:04d}.txt"
            Image.new("RGB", (32, 32), (index % 255, 2, 3)).save(image)
            if index % 3 == 0:
                label.write_text("", encoding="utf-8")
            else:
                cls = index % 4
                extra = f"\n{(cls + 1) % 4} 0.4 0.4 0.2 0.2" if index % 5 == 0 else ""
                label.write_text(f"{cls} 0.5 0.5 0.25 0.25{extra}\n", encoding="utf-8")

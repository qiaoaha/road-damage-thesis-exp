from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_classes(label_path: Path) -> list[int]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            classes.append(int(line.split()[0]))
    return sorted(set(classes))


def find_train_pairs(dataset_root: Path) -> list[tuple[Path, Path, list[int]]]:
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected YOLO train dirs under {dataset_root}")
    pairs: list[tuple[Path, Path, list[int]]] = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        pairs.append((image_path, label_path, parse_classes(label_path)))
    return pairs


def choose_manifests(pairs: list[tuple[Path, Path, list[int]]]) -> dict[str, list[tuple[Path, Path, list[int]]]]:
    by_class: dict[int, list[tuple[Path, Path, list[int]]]] = {index: [] for index in range(4)}
    negatives: list[tuple[Path, Path, list[int]]] = []
    for pair in pairs:
        classes = pair[2]
        if not classes:
            negatives.append(pair)
        for class_id in classes:
            if class_id in by_class:
                by_class[class_id].append(pair)
    smoke: list[tuple[Path, Path, list[int]]] = []
    micro: list[tuple[Path, Path, list[int]]] = []
    for class_id in range(4):
        candidates = by_class[class_id]
        if not candidates:
            raise RuntimeError(f"No train sample found for class {class_id}")
        micro.append(candidates[0])
        smoke.extend(candidates[:2] if len(candidates) >= 2 else candidates * 2)
    if not negatives:
        raise RuntimeError("No negative train sample found")
    return {"gpu_smoke8.csv": smoke[:8], "micro_overfit4.csv": micro[:4], "negative_smoke1.csv": negatives[:1]}


def write_manifest(path: Path, rows: list[tuple[Path, Path, list[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "label_path",
                "class_ids",
                "box_count",
                "image_sha256",
                "label_sha256",
                "split",
            ],
        )
        writer.writeheader()
        for image_path, label_path, classes in rows:
            writer.writerow(
                {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "class_ids": " ".join(str(item) for item in classes),
                    "box_count": sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
                    if label_path.exists()
                    else 0,
                    "image_sha256": sha256_file(image_path),
                    "label_sha256": sha256_file(label_path) if label_path.exists() else hashlib.sha256(b"").hexdigest(),
                    "split": "train",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("manifests"))
    args = parser.parse_args()
    selected = choose_manifests(find_train_pairs(args.dataset_root))
    for filename, rows in selected.items():
        write_manifest(args.out_dir / filename, rows)
        print(f"{filename}=PASS rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

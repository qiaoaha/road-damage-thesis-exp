from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from sd3_rgda.cache import collect_pilot_cache_requests


def test_clean_proxy_manifest_cache_linkage(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    clean = tmp_path / "clean.png"
    label = tmp_path / "label.txt"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(image)
    Image.new("RGB", (16, 16), (4, 5, 6)).save(clean)
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    manifest = tmp_path / "clean_proxy_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "split",
                "image_path",
                "label_path",
                "clean_proxy_path",
                "image_sha256",
                "clean_proxy_sha256",
                "anchor_class",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s0",
                "split": "train",
                "image_path": str(image),
                "label_path": str(label),
                "clean_proxy_path": str(clean),
                "image_sha256": "source",
                "clean_proxy_sha256": "proxy",
                "anchor_class": "D00",
            }
        )
    request = collect_pilot_cache_requests(manifest, 16)[0]
    assert request.source_sample_id == "s0"
    assert request.anchor_class == "D00"
    assert request.clean_proxy_sha256 == "proxy"
    assert float(request.pseudo_clean_pixels.mean()) != float(request.target_pixels.mean())

from pathlib import Path

from PIL import Image

from sd3_rgda.generation_manifest import sha256_file
from sd3_rgda.inference import completed_result_is_valid


def test_resume_requires_image_label_seed_mode_and_checkpoint_sha(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    label = tmp_path / "label.txt"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(image)
    label.write_text("", encoding="utf-8")
    row = {"width": "32", "height": "32", "seed": "202600000"}
    result = {
        "status": "PASS",
        "mode": "rgda",
        "seed": "202600000",
        "rgda_checkpoint_sha256": "abc",
        "output_image_path": str(image),
        "output_label_path": str(label),
        "output_image_sha256": sha256_file(image),
        "output_label_sha256": sha256_file(label),
    }
    assert completed_result_is_valid(row, result, mode="rgda", checkpoint_sha256="abc")
    assert not completed_result_is_valid(row, {**result, "seed": "202600001"}, mode="rgda", checkpoint_sha256="abc")
    assert not completed_result_is_valid(row, result, mode="base", checkpoint_sha256="abc")

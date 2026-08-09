from pathlib import Path

from PIL import Image

from sd3_rgda.generation_manifest import sha256_file
from sd3_rgda.inference import completed_result_is_valid


def test_resume_requires_image_label_seed_mode_and_checkpoint_sha(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    label = tmp_path / "label.txt"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(image)
    label.write_text("", encoding="utf-8")
    row = {
        "generation_index": "0",
        "source_image_sha256": "srcsha",
        "prompt": "p",
        "width": "32",
        "height": "32",
        "num_inference_steps": "1",
        "guidance_scale": "1.0",
        "scheduler": "S",
        "seed": "202600000",
    }
    result = {
        "status": "PASS",
        "mode": "rgda",
        "seed": "202600000",
        "generation_index": "0",
        "source_image_sha256": "srcsha",
        "prompt": "p",
        "width": "32",
        "height": "32",
        "inference_steps": "1",
        "guidance_scale": "1.0",
        "scheduler": "S",
        "rgda_checkpoint_sha256": "abc",
        "output_image_path": str(image),
        "output_label_path": str(label),
        "output_image_sha256": sha256_file(image),
        "output_label_sha256": sha256_file(label),
    }
    assert completed_result_is_valid(row, result, mode="rgda", checkpoint_sha256="abc")
    assert not completed_result_is_valid(row, {**result, "seed": "202600001"}, mode="rgda", checkpoint_sha256="abc")
    assert not completed_result_is_valid(row, result, mode="base", checkpoint_sha256="abc")
    assert completed_result_is_valid(row, {**result, "mode": "base", "rgda_checkpoint_sha256": "NONE"}, mode="base", checkpoint_sha256="abc")

import csv
from pathlib import Path

from PIL import Image

from sd3_rgda.generation_audit import audit_generation_results
from sd3_rgda.generation_manifest import GENERATION_FIELDS, sha256_file


def test_generation_audit_checks_pairs_and_label_sha(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    results = tmp_path / "results.csv"
    image = tmp_path / "out.png"
    label = tmp_path / "out.txt"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(image)
    label.write_text("", encoding="utf-8")
    row = {
        "generation_index": "0",
        "source_sample_id": "s0",
        "source_image_path": "src.jpg",
        "source_label_path": "src.txt",
        "source_image_sha256": "srcsha",
        "source_label_sha256": sha256_file(label),
        "source_split": "train",
        "is_negative": "true",
        "anchor_class": "",
        "class_ids": "",
        "prompt": "p",
        "seed": "202600000",
        "width": "16",
        "height": "16",
        "num_inference_steps": "1",
        "guidance_scale": "1",
        "scheduler": "S",
        "base_output_filename": "b.png",
        "rgda_output_filename": "r.png",
        "rgda_checkpoint_sha256": "abc",
    }
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATION_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    fields = [
        "generation_index",
        "mode",
        "source_sample_id",
        "seed",
        "prompt",
        "source_image_sha256",
        "source_label_sha256",
        "rgda_checkpoint_sha256",
        "output_image_path",
        "output_image_sha256",
        "output_label_path",
        "output_label_sha256",
        "width",
        "height",
        "inference_steps",
        "guidance_scale",
        "scheduler",
        "generation_seconds",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "status",
        "error_type",
        "error_message",
    ]
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in ["base", "rgda"]:
            writer.writerow(
                {
                    "generation_index": row["generation_index"],
                    "mode": mode,
                    "source_sample_id": row["source_sample_id"],
                    "seed": row["seed"],
                    "prompt": row["prompt"],
                    "source_image_sha256": row["source_image_sha256"],
                    "source_label_sha256": row["source_label_sha256"],
                    "rgda_checkpoint_sha256": row["rgda_checkpoint_sha256"],
                    "output_image_path": image,
                    "output_image_sha256": sha256_file(image),
                    "output_label_path": label,
                    "output_label_sha256": sha256_file(label),
                    "width": row["width"],
                    "height": row["height"],
                    "inference_steps": row["num_inference_steps"],
                    "guidance_scale": row["guidance_scale"],
                    "scheduler": row["scheduler"],
                    "generation_seconds": "0.1",
                    "peak_allocated_mib": "0",
                    "peak_reserved_mib": "0",
                    "status": "PASS",
                    "error_type": "",
                    "error_message": "",
                }
            )
    audit = audit_generation_results(manifest, results, expected_checkpoint_sha256="abc")
    assert audit.generated_success == 2
    assert audit.base_rgda_seed_pair_match == "1/1000"
    assert audit.checkpoint_sha_match == "PASS"

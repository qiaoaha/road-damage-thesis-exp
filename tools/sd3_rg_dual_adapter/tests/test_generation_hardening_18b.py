import csv
import json
from pathlib import Path

import pytest
from PIL import Image
from test_generation_manifest import make_czech_dataset
from test_yolo_ablation_dataset import make_yolo

from sd3_rgda.generation_audit import audit_generation_results, build_yolo_ablation_dataset
from sd3_rgda.generation_manifest import GENERATION_FIELDS, build_generation1000_rows, sha256_file
from sd3_rgda.inference import validate_generation_cache_join


def test_generation_resolution_matches_formal_cache(tmp_path: Path) -> None:
    make_czech_dataset(tmp_path)
    rows, summary = build_generation1000_rows(tmp_path, "abc")
    assert summary.width == 512
    assert summary.height == 512
    assert rows[0]["width"] == "512"
    assert rows[0]["height"] == "512"


def test_real_generation_entry_no_placeholder() -> None:
    script = Path("scripts/generate_sd3_rgda1000.py").read_text(encoding="utf-8")
    assert "Real SD3 generation is intentionally staged" not in script
    assert "StableDiffusion3Pipeline.from_pretrained" in script
    assert "run_paired_generation" in script


def test_formal_cache_join_rejects_missing_and_duplicate(tmp_path: Path) -> None:
    rows = [{"source_image_sha256": "a"}, {"source_image_sha256": "b"}]
    manifest = tmp_path / "cache.csv"
    _write_cache(manifest, [("a", "a.pt")])
    with pytest.raises(ValueError, match="CACHE_MISSING=1"):
        validate_generation_cache_join(rows, manifest)
    _write_cache(manifest, [("a", "a.pt"), ("a", "a2.pt"), ("b", "b.pt")])
    with pytest.raises(ValueError, match="CACHE_DUPLICATE=1"):
        validate_generation_cache_join(rows, manifest)


def test_paired_results_expect_2000_and_checkpoint_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    results = tmp_path / "results.csv"
    rows = []
    for index in range(1000):
        image = tmp_path / f"im_{index}.png"
        label = tmp_path / f"im_{index}.txt"
        Image.new("RGB", (8, 8), (index % 255, 0, 0)).save(image)
        label.write_text("", encoding="utf-8")
        rows.append(
            {
                "generation_index": str(index),
                "source_sample_id": f"s{index}",
                "source_image_path": str(image),
                "source_label_path": str(label),
                "source_image_sha256": sha256_file(image),
                "source_label_sha256": sha256_file(label),
                "source_split": "train",
                "is_negative": "true" if index < 250 else "false",
                "anchor_class": "",
                "class_ids": "",
                "prompt": "p",
                "seed": str(202600000 + index),
                "width": "8",
                "height": "8",
                "num_inference_steps": "1",
                "guidance_scale": "1.0",
                "scheduler": "S",
                "base_output_filename": f"b{index}.png",
                "rgda_output_filename": f"r{index}.png",
                "rgda_checkpoint_sha256": "abc",
            }
        )
    _write_manifest(manifest, rows)
    _write_results(results, rows)
    audit = audit_generation_results(manifest, results, expected_checkpoint_sha256="abc")
    assert audit.result_rows == 2000
    assert audit.base_rows == 1000
    assert audit.rgda_rows == 1000
    assert audit.base_checkpoint_field_none == "1000/1000"
    assert audit.rgda_checkpoint_sha_match == "1000/1000"
    assert audit.negative_label_gate == "PASS"
    assert audit.audit_gate == "PASS"


def test_yolo_synthetic_count_not_1000_fails_and_no_double_prefix(tmp_path: Path) -> None:
    real = tmp_path / "real"
    make_yolo(real, train=10, val=4, test=4)
    synth_i = tmp_path / "synth" / "images"
    synth_l = tmp_path / "synth" / "labels"
    synth_i.mkdir(parents=True)
    synth_l.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(synth_i / "sd3rgda_x.png")
    (synth_l / "sd3rgda_x.txt").write_text("", encoding="utf-8")
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
        expected_synthetic=1000,
    )
    assert audit.dataset_gate == "FAIL"
    assert (tmp_path / "out" / "rgda" / "images" / "train" / "sd3rgda_x.png").exists()
    assert not (tmp_path / "out" / "rgda" / "images" / "train" / "sd3rgda_sd3rgda_x.png").exists()


def test_ablation_delta_math(tmp_path: Path) -> None:
    root = tmp_path / "results"
    for group, map50, map95 in [("real", 0.2, 0.1), ("sd3", 0.25, 0.12), ("rgda", 0.28, 0.13)]:
        out = root / group
        out.mkdir(parents=True)
        (out / "test_metrics.json").write_text(
            json.dumps({"precision": 0.1, "recall": 0.2, "map50": map50, "map50_95": map95, "best_pt": "best.pt"}),
            encoding="utf-8",
        )
    import sys

    from scripts.summarize_rgda_ablation import main

    old = sys.argv
    try:
        sys.argv = ["summarize", "--result-root", str(root)]
        assert main() == 0
    finally:
        sys.argv = old
    payload = json.loads((root / "final_ablation.json").read_text(encoding="utf-8"))
    assert payload["deltas"]["DELTA_RGDA_MAP50"] == pytest.approx(0.03)


def _write_cache(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_image_sha256", "cache_path"])
        writer.writeheader()
        for sha, cache_path in rows:
            writer.writerow({"source_image_sha256": sha, "cache_path": cache_path})


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_results(path: Path, rows: list[dict[str, str]]) -> None:
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
        "transformer_forward_count",
        "rgda_hook_call_count",
        "status",
        "error_type",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
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
                        "rgda_checkpoint_sha256": "NONE" if mode == "base" else "abc",
                        "output_image_path": row["source_image_path"],
                        "output_image_sha256": row["source_image_sha256"],
                        "output_label_path": row["source_label_path"],
                        "output_label_sha256": row["source_label_sha256"],
                        "width": row["width"],
                        "height": row["height"],
                        "inference_steps": row["num_inference_steps"],
                        "guidance_scale": row["guidance_scale"],
                        "scheduler": row["scheduler"],
                        "generation_seconds": "0",
                        "peak_allocated_mib": "0",
                        "peak_reserved_mib": "0",
                        "transformer_forward_count": "1",
                        "rgda_hook_call_count": "0" if mode == "base" else "1",
                        "status": "PASS",
                        "error_type": "",
                        "error_message": "",
                    }
                )

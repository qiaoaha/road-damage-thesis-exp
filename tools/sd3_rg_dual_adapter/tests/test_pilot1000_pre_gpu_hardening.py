from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts.build_clean_proxy import clean_proxy_qa_passed
from sd3_rgda.cache import collect_pilot_cache_requests
from sd3_rgda.pilot_engine import (
    BranchGradientCounts,
    Pilot1000Runner,
    audit_actual_sample_usage,
    read_csv,
    write_sample_usage_csv,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _pilot_rows(count: int = 512) -> list[dict[str, str]]:
    return [
        {"sample_id": f"s{i}", "source_sample_id": f"src{i}", "is_negative": str(i % 2 == 0).lower(), "anchor_class": "D00"}
        for i in range(count)
    ]


def _write_clean_proxy_manifest(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
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
                "label_sha256",
                "clean_proxy_sha256",
                "anchor_class",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s0",
                "split": "pilot_train",
                "image_path": str(image),
                "label_path": str(label),
                "clean_proxy_path": str(clean),
                "image_sha256": _sha256(image),
                "label_sha256": _sha256(label),
                "clean_proxy_sha256": _sha256(clean),
                "anchor_class": "D00",
            }
        )
    return manifest, image, clean, label


def test_actual_usage_not_planned_order(tmp_path: Path) -> None:
    rows = _pilot_rows(4)
    usage = {0: 2, 1: 0, 2: 0, 3: 0}
    audit = audit_actual_sample_usage(rows, usage)
    assert audit.unique_train_samples_used == 1
    assert audit.train_sample_use_min == 0
    write_sample_usage_csv(tmp_path / "sample_usage.csv", rows, usage)
    saved = read_csv(tmp_path / "sample_usage.csv")
    assert [row["uses"] for row in saved] == ["2", "0", "0", "0"]


def test_sample_use_range_gate() -> None:
    rows = _pilot_rows(512)
    passing = {index: 2 if index < 488 else 1 for index in range(512)}
    failing = {index: 2 for index in range(511)}
    assert audit_actual_sample_usage(rows, passing).train_sample_use_min == 1
    assert audit_actual_sample_usage(rows, passing).train_sample_use_max == 2
    assert audit_actual_sample_usage(rows, failing).train_sample_use_min == 0


def test_resume_does_not_duplicate_eval_step0(tmp_path: Path) -> None:
    runner = Pilot1000Runner(
        model_path=tmp_path / "model",
        train_cache_manifest=tmp_path / "train.csv",
        eval_cache_manifest=tmp_path / "eval.csv",
        report_dir=tmp_path / "report",
    )
    runner.report_dir.mkdir(parents=True)
    (runner.report_dir / "eval_metrics.csv").write_text("step,eval_loss_all\n0,1.0\n", encoding="utf-8")
    checkpoint = tmp_path / "resume.pt"
    torch.save({"eval_metric_rows": [{"step": "0", "eval_loss_all": "1.0"}]}, checkpoint)
    runner._restore_metric_history(checkpoint)
    assert [row["step"] for row in read_csv(runner.report_dir / "eval_metrics.csv")] == ["0"]


def test_resume_restores_metric_history(tmp_path: Path) -> None:
    runner = Pilot1000Runner(
        model_path=tmp_path / "model",
        train_cache_manifest=tmp_path / "train.csv",
        eval_cache_manifest=tmp_path / "eval.csv",
        report_dir=tmp_path / "report",
    )
    checkpoint = tmp_path / "resume.pt"
    torch.save(
        {
            "eval_metric_rows": [{"step": "0", "eval_loss_all": "1.0"}, {"step": "100", "eval_loss_all": "0.9"}],
            "train_metric_rows": [{"step": "1", "loss": "1.0"}],
        },
        checkpoint,
    )
    runner._restore_metric_history(checkpoint)
    assert [row["step"] for row in read_csv(runner.report_dir / "eval_metrics.csv")] == ["0", "100"]
    assert read_csv(runner.report_dir / "train_metrics.csv")[0]["step"] == "1"


def test_clean_proxy_integer_fail_counts() -> None:
    qa = {
        "PROXY_TOTAL": 576,
        "PROXY_MISSING": 0,
        "PROXY_CORRUPT": 1,
        "PROXY_SHAPE_MISMATCH": 0,
        "NEGATIVE_PROXY_EXACT_MATCH": "PASS",
        "POSITIVE_MASK_CHANGED": "PASS",
        "OUTSIDE_MASK_UNCHANGED": "PASS",
        "GRAY_RECTANGLE_METHOD_USED": "NO",
        "SAM_USED": "NO",
    }
    assert not clean_proxy_qa_passed(qa)


def test_cache_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    manifest, *_ = _write_clean_proxy_manifest(tmp_path)
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace("image_sha256", "image_sha256").replace(_sha256(tmp_path / "image.png"), "bad"), encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_IMAGE_SHA256_MISMATCH"):
        collect_pilot_cache_requests(manifest, 16)


def test_cache_rejects_proxy_hash_mismatch(tmp_path: Path) -> None:
    manifest, _image, clean, _label = _write_clean_proxy_manifest(tmp_path)
    text = manifest.read_text(encoding="utf-8").replace(_sha256(clean), "bad")
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="CLEAN_PROXY_SHA256_MISMATCH"):
        collect_pilot_cache_requests(manifest, 16)


def test_cache_preserves_pilot_split(tmp_path: Path) -> None:
    manifest, *_ = _write_clean_proxy_manifest(tmp_path)
    request = collect_pilot_cache_requests(manifest, 16)[0]
    assert request.split == "train"
    assert request.source_split == "train"
    assert request.pilot_split == "pilot_train"


def test_negative_gradient_gate_from_metrics() -> None:
    counts = BranchGradientCounts()
    counts.update(False, {"normal_adapter": 1.0, "defect_adapter": 1.0, "rg_encoder": 1.0})
    counts.update(True, {"normal_adapter": 1.0, "defect_adapter": 0.0, "rg_encoder": 0.0})
    assert counts.positive_defect_adapter_grad_nonzero_steps == 1
    assert counts.negative_defect_adapter_grad_nonzero_steps == 0
    assert counts.negative_rg_encoder_grad_nonzero_steps == 0
    assert counts.negative_normal_adapter_grad_nonzero_steps == 1


def test_failure_summary_written(tmp_path: Path) -> None:
    runner = Pilot1000Runner(
        model_path=tmp_path / "model",
        train_cache_manifest=tmp_path / "train.csv",
        eval_cache_manifest=tmp_path / "eval.csv",
        report_dir=tmp_path / "report",
    )

    class FakeTrainer:
        steps_completed = 17

    runner._write_failure_summary("unit", RuntimeError("boom"), FakeTrainer(), 0, 0, tmp_path / "last.pt")  # type: ignore[arg-type]
    report = (runner.report_dir / "08_PILOT1000_FAILURE_SUMMARY.md").read_text(encoding="utf-8")
    assert "FINAL_VERDICT=FAIL" in report
    assert "FAIL_STAGE=unit" in report
    assert "TRAIN_STEPS_COMPLETED=17" in report

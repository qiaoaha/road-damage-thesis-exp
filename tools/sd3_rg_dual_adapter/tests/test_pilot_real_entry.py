from __future__ import annotations

import json

import pytest

from scripts.train_rgda_pilot1000 import main
from sd3_rgda import pilot_engine


def test_real_pilot_entry_not_deferred(monkeypatch, tmp_path) -> None:
    calls: list[tuple[int, int]] = []

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            calls.append((kwargs["steps"], kwargs["seed"]))

        def run(self, resume_from=None) -> None:
            calls.append((1000, 2026))

    monkeypatch.setattr("scripts.train_rgda_pilot1000.Pilot1000Runner", FakeRunner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--model-path",
            str(tmp_path / "model"),
            "--train-cache-manifest",
            str(tmp_path / "train.csv"),
            "--eval-cache-manifest",
            str(tmp_path / "eval.csv"),
            "--pilot-manifest-summary",
            str(tmp_path / "manifest_summary.json"),
            "--clean-proxy-audit",
            str(tmp_path / "clean_proxy_audit.json"),
            "--cache-audit",
            str(tmp_path / "cache_audit.json"),
            "--report-dir",
            str(tmp_path / "report"),
        ],
    )
    assert main() == 0
    assert calls == [(1000, 2026), (1000, 2026)]


def test_real_entry_requires_cuda_for_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pilot_engine.torch.cuda, "is_available", lambda: False)
    manifest_summary = tmp_path / "manifest_summary.json"
    clean_proxy_audit = tmp_path / "clean_proxy_audit.json"
    cache_audit = tmp_path / "cache_audit.json"
    manifest_summary.write_text(
        json.dumps(
            {
                "train_rows": 512,
                "train_unique_images": 512,
                "train_positive": 256,
                "train_negative": 256,
                "eval_rows": 64,
                "eval_unique_images": 64,
                "eval_positive": 32,
                "eval_negative": 32,
                "train_eval_overlap": 0,
                "val_test_leakage": 0,
                "class_minimum_gate": "PASS",
            }
        ),
        encoding="utf-8",
    )
    clean_proxy_audit.write_text(
        json.dumps(
            {
                "PROXY_TOTAL": 576,
                "PROXY_MISSING": 0,
                "PROXY_CORRUPT": 0,
                "PROXY_SHAPE_MISMATCH": 0,
                "NEGATIVE_PROXY_EXACT_MATCH": "PASS",
                "POSITIVE_MASK_CHANGED": "PASS",
                "OUTSIDE_MASK_UNCHANGED": "PASS",
                "GRAY_RECTANGLE_METHOD_USED": "NO",
                "SAM_USED": "NO",
                "CLEAN_PROXY_READY": "PASS",
            }
        ),
        encoding="utf-8",
    )
    cache_audit.write_text(
        json.dumps(
            {
                "TRAIN_CACHE_ROWS": 512,
                "EVAL_CACHE_ROWS": 64,
                "ALL_CACHE_FILES_EXIST": "PASS",
                "ALL_TENSORS_FINITE": "PASS",
                "VAE_PARAMETER_DTYPE": "torch.float32",
                "VAE_INPUT_DTYPE": "torch.float32",
                "CACHED_LATENT_DTYPE": "torch.bfloat16",
                "TEXT_CACHE_DTYPE": "torch.bfloat16",
                "SOURCE_HASH_VERIFIED": "PASS",
                "CLEAN_PROXY_HASH_VERIFIED": "PASS",
                "LABEL_HASH_VERIFIED": "PASS",
                "NEGATIVE_RG_ZERO": "PASS",
                "NEGATIVE_TOKEN_MASK_ZERO": "PASS",
                "CACHE_READY": "PASS",
            }
        ),
        encoding="utf-8",
    )
    runner = pilot_engine.Pilot1000Runner(
        model_path=tmp_path / "model",
        train_cache_manifest=tmp_path / "train.csv",
        eval_cache_manifest=tmp_path / "eval.csv",
        report_dir=tmp_path / "report",
        pilot_manifest_summary=manifest_summary,
        clean_proxy_audit=clean_proxy_audit,
        cache_audit=cache_audit,
    )
    with pytest.raises(RuntimeError, match="CUDA is required"):
        runner.run()

from __future__ import annotations

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
    runner = pilot_engine.Pilot1000Runner(
        model_path=tmp_path / "model",
        train_cache_manifest=tmp_path / "train.csv",
        eval_cache_manifest=tmp_path / "eval.csv",
        report_dir=tmp_path / "report",
    )
    with pytest.raises(RuntimeError, match="CUDA is required"):
        runner.run()

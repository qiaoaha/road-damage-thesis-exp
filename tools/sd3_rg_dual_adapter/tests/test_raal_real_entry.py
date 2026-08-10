from __future__ import annotations

import sys
from pathlib import Path

from scripts import train_rgda_raal_pilot1000


def test_real_main_invokes_real_raal_pilot_runner(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, Path | None]] = []

    class Runner:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            calls.append(("init", kwargs["schedule_manifest"]))

        def run(self, resume_from: Path | None = None) -> None:
            calls.append(("run", resume_from))

    monkeypatch.setattr(train_rgda_raal_pilot1000, "RealRAALPilotRunner", Runner)
    monkeypatch.setattr(
        train_rgda_raal_pilot1000,
        "load_formal_first1000_schedule",
        lambda _path: ([{"step": "1", "pool_index": "0", "source_sample_id": "s0", "polarity": "positive"}], "s" * 64),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rgda_raal_pilot1000.py",
            "--backend",
            "real",
            "--arm",
            "r1",
            "--model-path",
            "model",
            "--train-cache-manifest",
            "train.csv",
            "--eval-cache-manifest",
            "eval.csv",
            "--schedule-manifest",
            "schedule.csv",
            "--report-dir",
            "report",
        ],
    )
    train_rgda_raal_pilot1000.main()
    assert calls == [("init", Path("schedule.csv")), ("run", None)]

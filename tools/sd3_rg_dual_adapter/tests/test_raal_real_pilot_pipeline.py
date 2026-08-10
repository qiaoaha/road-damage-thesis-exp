from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from sd3_rgda.pilot_engine import inspect_pilot_checkpoint_payload
from sd3_rgda.raal import RAALConfig
from sd3_rgda.raal_engine import FakeRAALPilotBackend, load_formal_first1000_schedule


def _schedule(path: Path, rows: int = 1000) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "pool_index", "source_sample_id", "polarity"])
        writer.writeheader()
        for step in range(1, rows + 1):
            writer.writerow(
                {
                    "step": step,
                    "pool_index": step - 1,
                    "source_sample_id": f"src-{step:04d}",
                    "polarity": "negative" if step % 2 == 0 else "positive",
                }
            )
    return path


def test_real_schedule_first1000_reuse(tmp_path: Path) -> None:
    rows, digest = load_formal_first1000_schedule(_schedule(tmp_path / "schedule_manifest.csv", 1005))
    assert len(rows) == 1000
    assert len(digest) == 64
    assert rows[0]["source_sample_id"] == "src-0001"
    assert rows[-1]["step"] == "1000"


def test_real_pilot_entry_no_placeholder() -> None:
    text = Path("scripts/train_rgda_raal_pilot1000.py").read_text(encoding="utf-8")
    assert "real training entry is prepared" not in text
    assert "--schedule-manifest" in text


def test_fake_pilot_train_eval_checkpoint_and_gate(tmp_path: Path) -> None:
    rows, digest = load_formal_first1000_schedule(_schedule(tmp_path / "schedule_manifest.csv"))
    result = FakeRAALPilotBackend(
        arm="r1",
        schedule_rows=rows[:20],
        report_dir=tmp_path / "r1",
        steps=20,
        config=RAALConfig(enabled=True, weight=0.02),
        schedule_sha=digest,
    ).run()
    assert len(result["train_rows"]) == 20
    assert result["eval_rows"][0]["step"] == "0"
    assert result["eval_rows"][-1]["positive_count"] == "64"
    assert result["eval_rows"][-1]["negative_count"] == "64"
    assert (tmp_path / "r1" / "last.pt").exists()
    assert (tmp_path / "r1" / "best_eval.pt").exists()
    assert result["gate"]["RAAL_PILOT_GATE"] == "PASS"


def test_fake_pilot_resume_state_equivalence(tmp_path: Path) -> None:
    rows, digest = load_formal_first1000_schedule(_schedule(tmp_path / "schedule_manifest.csv"))
    continuous = FakeRAALPilotBackend(arm="r1", schedule_rows=rows[:20], report_dir=tmp_path / "continuous", steps=20, schedule_sha=digest)
    continuous.run()
    split = FakeRAALPilotBackend(arm="r1", schedule_rows=rows[:10], report_dir=tmp_path / "split", steps=10, schedule_sha=digest)
    first = split.run()
    resumed = FakeRAALPilotBackend(arm="r1", schedule_rows=rows[:20], report_dir=tmp_path / "resumed", steps=20, schedule_sha=digest)
    resumed.run(first["last"])
    continuous_state = torch.load(tmp_path / "continuous" / "last.pt", map_location="cpu", weights_only=False)
    resumed_state = torch.load(tmp_path / "resumed" / "last.pt", map_location="cpu", weights_only=False)
    assert continuous_state["modules"]["normal_adapter"]["alpha"].equal(resumed_state["modules"]["normal_adapter"]["alpha"])
    assert resumed_state["step"] == 20


def test_fake_pilot_resume_config_mismatch(tmp_path: Path) -> None:
    rows, digest = load_formal_first1000_schedule(_schedule(tmp_path / "schedule_manifest.csv"))
    first = FakeRAALPilotBackend(arm="r1", schedule_rows=rows[:10], report_dir=tmp_path / "first", steps=10, schedule_sha=digest).run()
    with pytest.raises(ValueError, match="RAAL_RESUME_CONFIG_MISMATCH"):
        FakeRAALPilotBackend(arm="r0", schedule_rows=rows[:20], report_dir=tmp_path / "bad", steps=20, schedule_sha=digest).run(first["last"])


def test_r0_train_loop_has_zero_raal(tmp_path: Path) -> None:
    rows, digest = load_formal_first1000_schedule(_schedule(tmp_path / "schedule_manifest.csv"))
    result = FakeRAALPilotBackend(arm="r0", schedule_rows=rows[:20], report_dir=tmp_path / "r0", steps=20, schedule_sha=digest).run()
    assert all(row["raal_loss"] == "0.00000000" for row in result["train_rows"])
    assert all(row["raal_hook_count"] == "0" for row in result["train_rows"])


def test_negative_rows_have_zero_raal_and_no_hooks(tmp_path: Path) -> None:
    rows, digest = load_formal_first1000_schedule(_schedule(tmp_path / "schedule_manifest.csv"))
    result = FakeRAALPilotBackend(arm="r1", schedule_rows=rows[:20], report_dir=tmp_path / "r1_neg", steps=20, schedule_sha=digest).run()
    negatives = [row for row in result["train_rows"] if row["is_negative"] == "true"]
    assert negatives
    assert all(row["raal_loss"] == "0.00000000" and row["raal_hook_count"] == "0" for row in negatives)


def test_fake_checkpoint_adapter_only(tmp_path: Path) -> None:
    rows, digest = load_formal_first1000_schedule(_schedule(tmp_path / "schedule_manifest.csv"))
    result = FakeRAALPilotBackend(arm="r1", schedule_rows=rows[:20], report_dir=tmp_path / "r1_scope", steps=20, schedule_sha=digest).run()
    payload = torch.load(result["last"], map_location="cpu", weights_only=False)
    assert inspect_pilot_checkpoint_payload(payload)["adapter_only"] is True


def test_schedule_step_mismatch_rejects(tmp_path: Path) -> None:
    path = _schedule(tmp_path / "schedule_manifest.csv")
    text = path.read_text(encoding="utf-8").replace("\n2,1,", "\n4,1,", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="FORMAL_SCHEDULE_FIRST1000_STEP_MISMATCH"):
        load_formal_first1000_schedule(path)


def test_raal_shell_passes_schedule_manifest() -> None:
    text = Path("run_rgda_raal_pilot1000.sh").read_text(encoding="utf-8")
    assert "SCHEDULE_MANIFEST" in text
    assert "--schedule-manifest" in text


def test_pilot_gate_fail_math(tmp_path: Path) -> None:
    rows, digest = load_formal_first1000_schedule(_schedule(tmp_path / "schedule_manifest.csv"))
    backend = FakeRAALPilotBackend(arm="r1", schedule_rows=rows[:20], report_dir=tmp_path / "fail_gate", steps=20, schedule_sha=digest)
    gate = backend._gate(
        [{"step": "0", "raal_eval_loss_positive": "1.0"}, {"step": "20", "raal_eval_loss_positive": "0.95"}],
        [{"weighted_raal_loss": "0.1", "raal_loss": "0.1", "is_negative": "false"} for _ in range(20)],
    )
    assert gate["RAAL_PILOT_GATE"] == "FAIL"

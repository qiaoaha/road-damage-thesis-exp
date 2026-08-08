from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from sd3_rgda.formal_engine import FORMAL_CONFIG, _validate_formal_config
from sd3_rgda.pilot_engine import BranchGradientCounts


def test_formal_resume_rejects_core_config_mismatch() -> None:
    config = {**FORMAL_CONFIG, "TRAIN_STEPS": 5000, "LEARNING_RATE": 2e-4}
    with pytest.raises(ValueError, match="FORMAL_RESUME_CORE_CONFIG_MISMATCH"):
        _validate_formal_config(config, 5000)


def test_formal_resume_accepts_matching_core_config() -> None:
    _validate_formal_config({**FORMAL_CONFIG, "TRAIN_STEPS": 5000}, 5000)


class _MockFormalRunner:
    def __init__(self, report_dir: Path) -> None:
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))
        self.optimizer = torch.optim.SGD([self.weight], lr=0.1)
        self.losses: list[float] = []
        self.sample_usage: dict[int, int] = {}
        self.branch_counts = BranchGradientCounts()
        self.step = 0

    def run(self, stop_step: int, resume_from: Path | None = None) -> Path:
        if resume_from is None:
            self._write_rows("eval128_metrics.csv", [{"step": "0", "eval_loss_all": "1.0"}])
        else:
            self._load(resume_from)
            self._restore_metrics(resume_from)
        for step in range(self.step + 1, stop_step + 1):
            self.optimizer.zero_grad()
            loss = (self.weight - step / 100).pow(2).sum()
            loss.backward()
            self.optimizer.step()
            self.step = step
            self.losses.append(float(loss.detach()))
            pool_index = (step - 1) % 4
            self.sample_usage[pool_index] = self.sample_usage.get(pool_index, 0) + 1
            self.branch_counts.positive_normal_adapter_grad_nonzero_steps += 1
            self._append_row("train_metrics.csv", {"step": str(step), "loss": f"{float(loss.detach()):.8f}"})
            self._append_row("gradient_metrics.csv", {"step": str(step), "normal_adapter_grad": "1.0"})
            self._append_row("memory_metrics.csv", {"step": str(step), "allocated_mib": "0"})
        checkpoint = self.report_dir / f"step_{self.step:04d}.pt"
        torch.save(
            {
                "weight": self.weight.detach().clone(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.step,
                "losses": self.losses,
                "sample_usage": self.sample_usage,
                "branch_counts": self.branch_counts.__dict__,
                "metrics": {
                    "train_metrics.csv": self._read_rows("train_metrics.csv"),
                    "gradient_metrics.csv": self._read_rows("gradient_metrics.csv"),
                    "memory_metrics.csv": self._read_rows("memory_metrics.csv"),
                    "eval128_metrics.csv": self._read_rows("eval128_metrics.csv"),
                },
            },
            checkpoint,
        )
        return checkpoint

    def _load(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.weight.data.copy_(payload["weight"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.step = int(payload["step"])
        self.losses = [float(item) for item in payload["losses"]]
        self.sample_usage = {int(key): int(value) for key, value in payload["sample_usage"].items()}
        self.branch_counts = BranchGradientCounts.from_mapping(payload["branch_counts"])

    def _restore_metrics(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for filename, rows in payload["metrics"].items():
            self._write_rows(filename, rows)

    def _append_row(self, filename: str, row: dict[str, str]) -> None:
        path = self.report_dir / filename
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_rows(self, filename: str, rows: list[dict[str, str]]) -> None:
        path = self.report_dir / filename
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _read_rows(self, filename: str) -> list[dict[str, str]]:
        path = self.report_dir / filename
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


def test_formal_resume_behavior_equivalence(tmp_path: Path) -> None:
    continuous = _MockFormalRunner(tmp_path / "continuous")
    continuous_checkpoint = continuous.run(20)
    first_half = _MockFormalRunner(tmp_path / "first_half")
    resume_checkpoint = first_half.run(10)
    resumed = _MockFormalRunner(tmp_path / "resumed")
    resumed_checkpoint = resumed.run(20, resume_from=resume_checkpoint)

    continuous_state = torch.load(continuous_checkpoint, map_location="cpu", weights_only=False)
    resumed_state = torch.load(resumed_checkpoint, map_location="cpu", weights_only=False)
    assert torch.equal(continuous_state["weight"], resumed_state["weight"])
    assert continuous_state["optimizer_state"]["state"].keys() == resumed_state["optimizer_state"]["state"].keys()
    assert resumed.step == 20
    assert resumed.losses == continuous.losses
    assert resumed.sample_usage == continuous.sample_usage
    assert resumed.branch_counts.__dict__ == continuous.branch_counts.__dict__
    for filename in ["train_metrics.csv", "gradient_metrics.csv", "memory_metrics.csv"]:
        assert resumed._read_rows(filename) == continuous._read_rows(filename)
    assert [row["step"] for row in resumed._read_rows("eval128_metrics.csv")].count("0") == 1
    assert resumed._read_rows("train_metrics.csv")[10]["step"] == "11"

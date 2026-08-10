from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sd3_rgda.raal import RAALConfig
from sd3_rgda.raal_pilot_engine import RealRAALPilotRunner, compare_raal_pilot_arms


class Tokenizer3:
    def __call__(self, prompt: str, **kwargs: object) -> dict[str, list[list[tuple[int, int]]]]:
        max_length = int(kwargs.get("max_length", 256))
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for token in prompt.split(" "):
            start = prompt.find(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        offsets.extend([(0, 0)] * max(0, max_length - len(offsets)))
        return {"offset_mapping": [offsets[:max_length]]}


class FakePipe:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(model_max_length=77)
        self.tokenizer_3 = Tokenizer3()
        self.scheduler = object()


class FakeTransformer(nn.Linear):
    def __init__(self) -> None:
        super().__init__(1, 1)
        self.config = SimpleNamespace(caption_projection_dim=1, in_channels=1, patch_size=1)
        with torch.no_grad():
            self.weight.fill_(1.0)
            self.bias.fill_(0.0)


class FakeSample:
    def __init__(self, is_negative: bool) -> None:
        self.is_negative = is_negative


class FakeInjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normal_encoder = nn.Linear(1, 1)
        self.rg_encoder = nn.Linear(1, 1)
        self.normal_adapter = nn.Linear(1, 1)
        self.defect_adapter = nn.Linear(1, 1)
        self.timestep_gate = nn.Linear(1, 1)

    def trainable_modules(self) -> dict[str, nn.Module]:
        return {
            "normal_encoder": self.normal_encoder,
            "rg_encoder": self.rg_encoder,
            "normal_adapter": self.normal_adapter,
            "defect_adapter": self.defect_adapter,
            "timestep_gate": self.timestep_gate,
        }


class FakeTrainer:
    def __init__(self, arm: str, _config: RAALConfig, _bank: object) -> None:
        self.arm = arm
        self.transformer = FakeTransformer()
        for parameter in self.transformer.parameters():
            parameter.requires_grad_(False)
        self.injector = FakeInjector()
        self.optimizer = torch.optim.SGD(self.injector.parameters(), lr=0.01)
        self.loaded: list[str] = []

    def load_cached_sample(self, cache_path: str) -> FakeSample:
        self.loaded.append(cache_path)
        return FakeSample("neg" in cache_path)

    def build_flow_batch(self, sample: FakeSample) -> object:
        return SimpleNamespace(sample=sample)

    def forward_loss_components(self, batch: object) -> object:
        value = sum(parameter.sum() for parameter in self.injector.parameters())
        negative = bool(batch.sample.is_negative)
        flow_loss = (value + 1.0).pow(2)
        raal_loss = flow_loss * 0.0 if negative else (value + 0.5).pow(2)
        weighted = raal_loss * 0.02
        total = flow_loss + weighted
        return SimpleNamespace(
            flow_loss=flow_loss,
            raal_loss=raal_loss,
            weighted_raal_loss=weighted,
            total_loss=total,
            metrics={
                "flow_loss": float(flow_loss.detach()),
                "raal_loss": float(raal_loss.detach()),
                "weighted_raal_loss": float(weighted.detach()),
                "total_loss": float(total.detach()),
                "raal_hook_count": 0.0 if negative else 3.0,
                "inside_attention_mass": 0.75 if not negative else 0.0,
                "outside_attention_mass": 0.25 if not negative else 0.0,
                "concentration_ratio": 3.0 if not negative else 0.0,
            },
        )

    def forward_loss(self, _batch: object) -> torch.Tensor:
        return sum(parameter.sum() for parameter in self.injector.parameters()).pow(2)

    def backward_step(self, batch: object) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.forward_loss(batch)
        torch.autograd.backward(loss)
        self.optimizer.step()
        return {"loss": float(loss.detach()), "grad_norm": 0.1}

    def gradient_report(self) -> dict[str, float]:
        return {name: 0.1 for name in self.injector.trainable_modules()}

    def assert_base_gradients_none(self) -> None:
        assert all(parameter.grad is None for parameter in self.transformer.parameters())


def _manifest(path: Path, rows: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "source_sample_id",
        "image_path",
        "label_path",
        "cache_path",
        "class_ids",
        "anchor_class",
        "is_negative",
        "split",
        "source_image_sha256",
        "label_sha256",
        "clean_proxy_sha256",
        "source_split",
        "pilot_split",
        "formal_role",
        "prompt_embed_seq_len",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(rows):
            neg = index % 2 == 1
            cache_path = path.parent / f"{'neg' if neg else 'pos'}-{index}.pt"
            cache_path.write_bytes(b"cache")
            writer.writerow(
                {
                    "sample_id": str(index),
                    "source_sample_id": f"src-{index:04d}",
                    "image_path": "image.jpg",
                    "label_path": "label.txt",
                    "cache_path": str(cache_path),
                    "class_ids": "" if neg else "0",
                    "anchor_class": "" if neg else "D00",
                    "is_negative": str(neg).lower(),
                    "split": "train",
                    "source_image_sha256": "a",
                    "label_sha256": "b",
                    "clean_proxy_sha256": "c",
                    "source_split": "train",
                    "pilot_split": "train",
                    "formal_role": "train",
                    "prompt_embed_seq_len": "333",
                }
            )
    return path


def _schedule(path: Path, rows: int = 1000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "pool_index", "source_sample_id", "polarity"])
        writer.writeheader()
        for step in range(1, rows + 1):
            index = step - 1
            writer.writerow(
                {
                    "step": str(step),
                    "pool_index": str(index),
                    "source_sample_id": f"src-{index:04d}",
                    "polarity": "negative" if index % 2 == 1 else "positive",
                }
            )
    return path


def _runner(tmp_path: Path, arm: str = "r1", steps: int = 2) -> RealRAALPilotRunner:
    train_manifest = _manifest(tmp_path / "train.csv", 1980)
    eval_manifest = _manifest(tmp_path / "eval.csv", 128)
    schedule = _schedule(tmp_path / "schedule.csv")
    return RealRAALPilotRunner(
        arm=arm,
        model_path=tmp_path / "model",
        train_cache_manifest=train_manifest,
        eval_cache_manifest=eval_manifest,
        schedule_manifest=schedule,
        report_dir=tmp_path / arm,
        steps=steps,
        pipeline_loader=lambda _path, _dtype: FakePipe(),
        scheduler_preparer=lambda scheduler: scheduler,
        transformer_preparer=lambda _pipe: (FakeTransformer(), SimpleNamespace()),
        trainer_factory=lambda arm_name, config, bank: FakeTrainer(arm_name, config, bank),
        fixed_batch_builder=lambda trainer, _manifest_path, _seed: [
            trainer.build_flow_batch(FakeSample(index % 2 == 1)) for index in range(128)
        ],
    )


def test_real_runner_uses_load_sd3_pipeline_and_mask_bank(tmp_path: Path) -> None:
    called = {"loader": 0}

    def loader(_path: Path, dtype: torch.dtype) -> FakePipe:
        called["loader"] += 1
        assert dtype == torch.bfloat16
        return FakePipe()

    runner = _runner(tmp_path)
    runner.pipeline_loader = loader
    runner.run()
    assert called["loader"] == 1
    assert len(runner.mask_bank_sha) == 64


def test_real_runner_uses_1980_train_cache_and_first1000_schedule(tmp_path: Path) -> None:
    result = _runner(tmp_path).run()
    assert result.final_step == 2
    assert (tmp_path / "r1" / "eval_metrics.csv").exists()


def test_schedule_sample_identity_fail(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    text = runner.schedule_manifest.read_text(encoding="utf-8").replace("src-0000", "bad-src", 1)
    runner.schedule_manifest.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="SCHEDULE_EXECUTION_MISMATCH"):
        runner.run()


def test_real_positive_hook_count_contract(tmp_path: Path) -> None:
    _runner(tmp_path).run()
    rows = list(csv.DictReader((tmp_path / "r1" / "train_metrics.csv").open(encoding="utf-8")))
    assert rows[0]["raal_hook_count"] == "3"


def test_real_negative_raal_zero_contract(tmp_path: Path) -> None:
    _runner(tmp_path, steps=2).run()
    rows = list(csv.DictReader((tmp_path / "r1" / "train_metrics.csv").open(encoding="utf-8")))
    assert rows[1]["raal_loss"] == "0.00000000"
    assert rows[1]["raal_hook_count"] == "0"


def test_real_eval128_64_64_and_step0(tmp_path: Path) -> None:
    _runner(tmp_path).run()
    rows = list(csv.DictReader((tmp_path / "r1" / "eval_metrics.csv").open(encoding="utf-8")))
    assert rows[0]["step"] == "0"
    assert rows[0]["positive_count"] == "64"
    assert rows[0]["negative_count"] == "64"


def test_real_checkpoint_adapter_only(tmp_path: Path) -> None:
    result = _runner(tmp_path).run()
    payload = torch.load(result.last_checkpoint, map_location="cpu", weights_only=False)
    assert set(payload["modules"]) == {"normal_encoder", "rg_encoder", "normal_adapter", "defect_adapter", "timestep_gate"}


def test_best_checkpoint_by_flow_loss(tmp_path: Path) -> None:
    result = _runner(tmp_path, steps=1).run()
    assert result.best_checkpoint.exists()


def test_resume_config_mismatch(tmp_path: Path) -> None:
    result = _runner(tmp_path, steps=1).run()
    resumed = _runner(tmp_path / "other", arm="r0", steps=2)
    with pytest.raises(ValueError, match="RAAL_RESUME_CONFIG_MISMATCH"):
        resumed.run(result.last_checkpoint)


def test_resume_1_plus_1_reaches_step2(tmp_path: Path) -> None:
    first_runner = _runner(tmp_path, steps=2)
    first_runner.checkpoint_interval = 1
    first_runner.run()
    resumed_runner = RealRAALPilotRunner(
        arm="r1",
        model_path=tmp_path / "model",
        train_cache_manifest=first_runner.train_cache_manifest,
        eval_cache_manifest=first_runner.eval_cache_manifest,
        schedule_manifest=first_runner.schedule_manifest,
        report_dir=tmp_path / "resumed",
        steps=2,
        pipeline_loader=lambda _path, _dtype: FakePipe(),
        scheduler_preparer=lambda scheduler: scheduler,
        transformer_preparer=lambda _pipe: (FakeTransformer(), SimpleNamespace()),
        trainer_factory=lambda arm_name, config, bank: FakeTrainer(arm_name, config, bank),
        fixed_batch_builder=lambda trainer, _manifest_path, _seed: [
            trainer.build_flow_batch(FakeSample(index % 2 == 1)) for index in range(128)
        ],
    )
    resumed = resumed_runner.run(tmp_path / "r1" / "checkpoints" / "r1" / "step_0001.pt")
    assert resumed.final_step == 2


def test_base_hash_unchanged_status(tmp_path: Path) -> None:
    _runner(tmp_path).run()
    status = (tmp_path / "r1" / "raal_pilot_status.md").read_text(encoding="utf-8")
    assert "BASE_HASH_GATE=PASS" in status


def test_cross_arm_gate_pass(tmp_path: Path) -> None:
    r0 = tmp_path / "r0"
    r1 = tmp_path / "r1"
    r0.mkdir()
    r1.mkdir()
    _write_eval(r0 / "eval_metrics.csv", "1.0", "0.8", "1.0")
    _write_eval(r1 / "eval_metrics.csv", "1.0", "0.5", "1.0")
    _write_train(r0 / "train_metrics.csv")
    _write_train(r1 / "train_metrics.csv")
    assert compare_raal_pilot_arms(r0, r1)["RAAL_PILOT_GATE"] == "PASS"


def test_cross_arm_gate_a_fail(tmp_path: Path) -> None:
    r0 = tmp_path / "r0"
    r1 = tmp_path / "r1"
    r0.mkdir()
    r1.mkdir()
    _write_eval(r0 / "eval_metrics.csv", "1.0", "0.8", "1.0")
    _write_eval(r1 / "eval_metrics.csv", "1.0", "0.95", "1.0")
    _write_train(r0 / "train_metrics.csv")
    _write_train(r1 / "train_metrics.csv")
    assert compare_raal_pilot_arms(r0, r1)["GATE_A"] == "FAIL"


def test_cross_arm_gate_b_fail(tmp_path: Path) -> None:
    r0 = tmp_path / "r0"
    r1 = tmp_path / "r1"
    r0.mkdir()
    r1.mkdir()
    _write_eval(r0 / "eval_metrics.csv", "1.0", "0.4", "1.0")
    _write_eval(r1 / "eval_metrics.csv", "1.0", "0.5", "1.0")
    _write_train(r0 / "train_metrics.csv")
    _write_train(r1 / "train_metrics.csv")
    assert compare_raal_pilot_arms(r0, r1)["GATE_B"] == "FAIL"


def test_cross_arm_gate_c_fail(tmp_path: Path) -> None:
    r0 = tmp_path / "r0"
    r1 = tmp_path / "r1"
    r0.mkdir()
    r1.mkdir()
    _write_eval(r0 / "eval_metrics.csv", "1.0", "0.8", "1.0")
    _write_eval(r1 / "eval_metrics.csv", "1.0", "0.5", "1.2")
    _write_train(r0 / "train_metrics.csv")
    _write_train(r1 / "train_metrics.csv")
    assert compare_raal_pilot_arms(r0, r1)["GATE_C"] == "FAIL"


def _write_eval(path: Path, raal0: str, raal1000: str, flow1000: str) -> None:
    fields = ["step", "flow_eval_loss", "raal_eval_loss_positive"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"step": "0", "flow_eval_loss": "1.0", "raal_eval_loss_positive": raal0})
        writer.writerow({"step": "1000", "flow_eval_loss": flow1000, "raal_eval_loss_positive": raal1000})


def _write_train(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step"])
        writer.writeheader()
        for step in range(1, 1001):
            writer.writerow({"step": str(step)})

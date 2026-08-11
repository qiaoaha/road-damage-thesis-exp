from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sd3_rgda.raal import RAALConfig
from sd3_rgda.raal_formal_engine import RAALFormal5000Runner, write_raal_formal_dry_integration
from sd3_rgda.raal_tokens import DefectTextMaskBank


class Tokenizer3:
    def __call__(self, prompt: str, **kwargs: object) -> dict[str, list[list[tuple[int, int]]]]:
        max_length = int(kwargs.get("max_length", 256))
        offsets = []
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


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = SimpleNamespace(caption_projection_dim=1, in_channels=1, patch_size=1)


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


class FakeSample:
    def __init__(self, is_negative: bool) -> None:
        self.is_negative = is_negative
        self.class_ids = () if is_negative else (0,)
        self.anchor_class = "" if is_negative else "D00"


class FakeTrainer:
    def __init__(self, _config: RAALConfig, _bank: DefectTextMaskBank) -> None:
        self.transformer = FakeTransformer()
        self.injector = FakeInjector()
        self.optimizer = torch.optim.SGD(self.injector.parameters(), lr=0.01)
        self.loaded: list[str] = []
        self.eval_mode_calls = 0
        self.train_mode_calls = 0

    def load_cached_sample(self, cache_path: str) -> FakeSample:
        self.loaded.append(cache_path)
        return FakeSample("neg" in cache_path)

    def build_flow_batch(self, sample: FakeSample) -> object:
        return SimpleNamespace(sample=sample)

    def backward_step(self, batch: object) -> dict[str, float]:
        negative = bool(batch.sample.is_negative)
        value = sum(parameter.sum() for parameter in self.injector.parameters())
        flow = (value + 1.0).pow(2)
        raal = flow * 0.0 if negative else (value + 0.5).pow(2)
        total = flow + 0.02 * raal
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        return {
            "flow_loss": float(flow.detach()),
            "raal_loss": float(raal.detach()),
            "weighted_raal_loss": float((0.02 * raal).detach()),
            "total_loss": float(total.detach()),
            "raal_hook_count": 0.0 if negative else 3.0,
            "inside_attention_mass": 0.0 if negative else 0.7,
            "outside_attention_mass": 0.0 if negative else 0.3,
            "concentration_ratio": 0.0 if negative else 2.3,
            "layer_5_calls": 0.0 if negative else 1.0,
            "layer_11_calls": 0.0 if negative else 1.0,
            "layer_17_calls": 0.0 if negative else 1.0,
        }

    def forward_loss_components(self, batch: object, *, retain_hooks_for_backward: bool = False) -> object:
        negative = bool(batch.sample.is_negative)
        flow = torch.tensor(1.0 if negative else 0.8)
        raal = torch.tensor(0.0 if negative else 0.2)
        return SimpleNamespace(
            flow_loss=flow,
            raal_loss=raal,
            metrics={
                "inside_attention_mass": 0.0 if negative else 0.7,
                "outside_attention_mass": 0.0 if negative else 0.3,
                "concentration_ratio": 0.0 if negative else 2.3,
            },
            collector=None,
        )

    def gradient_report(self) -> dict[str, float]:
        return {name: 0.1 for name in self.injector.trainable_modules()}

    def compare_outputs(self, _checkpoint: Path, _batch: object) -> float:
        return 0.0

    def train(self) -> None:
        self.eval_mode_calls += 0


class RecordingModule(FakeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.eval_calls = 0
        self.train_calls = 0

    def eval(self) -> RecordingModule:
        self.eval_calls += 1
        return self

    def train(self, mode: bool = True) -> RecordingModule:
        if mode:
            self.train_calls += 1
        return self


def _manifest(path: Path, rows: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "source_sample_id", "cache_path", "class_ids", "anchor_class", "is_negative"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(rows):
            neg = index % 2 == 1
            cache_path = path.parent / f"{'neg' if neg else 'pos'}-{index}.pt"
            torch.save({"prompt_embeds": torch.zeros(1, 333, 4)}, cache_path)
            writer.writerow(
                {
                    "sample_id": str(index),
                    "source_sample_id": f"src-{index:04d}",
                    "cache_path": str(cache_path),
                    "class_ids": "" if neg else "0",
                    "anchor_class": "" if neg else "D00",
                    "is_negative": str(neg).lower(),
                }
            )
    return path


def _schedule(path: Path, rows: int = 5000) -> Path:
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
                    "polarity": "negative" if index % 2 else "positive",
                }
            )
    return path


def _runner(tmp_path: Path, *, steps: int = 2) -> RAALFormal5000Runner:
    train = _manifest(tmp_path / "train.csv", 1980)
    eval128 = _manifest(tmp_path / "eval128.csv", 128)
    val = _manifest(tmp_path / "val.csv", 424)
    schedule = _schedule(tmp_path / "schedule.csv")
    manifest_summary = tmp_path / "manifest_summary.json"
    manifest_summary.write_text(
        json.dumps(
            {
                "train_pool_rows": 1980,
                "val_full_rows": 424,
                "eval128_rows": 128,
                "train_val_overlap": 0,
                "train_test_overlap": 0,
                "val_test_overlap": 0,
                "eval_test_overlap": 0,
                "test_leakage": 0,
            }
        ),
        encoding="utf-8",
    )
    schedule_audit = tmp_path / "schedule_audit.json"
    schedule_audit.write_text(json.dumps({"schedule_rows": 5000, "schedule_audit": "PASS"}), encoding="utf-8")
    proxy_audit = tmp_path / "clean_proxy_audit.json"
    proxy_audit.write_text(json.dumps({"PROXY_TOTAL": 2404, "FORMAL_CLEAN_PROXY_READY": "PASS"}), encoding="utf-8")
    cache_audit = tmp_path / "cache_audit.json"
    cache_audit.write_text(json.dumps({"TRAIN_CACHE_ROWS": 1980, "VAL_CACHE_ROWS": 424, "CACHE_READY": "PASS"}), encoding="utf-8")
    runner = RAALFormal5000Runner(
        model_path=tmp_path / "model",
        train_cache_manifest=train,
        eval128_cache_manifest=eval128,
        val_cache_manifest=val,
        train_pool_manifest=train,
        val_full_manifest=val,
        eval128_manifest=eval128,
        schedule_manifest=schedule,
        manifest_summary=manifest_summary,
        schedule_audit=schedule_audit,
        clean_proxy_audit=proxy_audit,
        cache_audit=cache_audit,
        report_dir=tmp_path / "report",
        steps=steps,
        checkpoint_interval=1,
        eval_interval=1,
        pipeline_loader=lambda _path, _dtype: FakePipe(),
        scheduler_preparer=lambda scheduler: scheduler,
        transformer_preparer=lambda _pipe: (FakeTransformer(), SimpleNamespace()),
        trainer_factory=lambda config, bank: FakeTrainer(config, bank),
        fixed_batch_builder=lambda trainer, _manifest_path, _seed: [
            trainer.build_flow_batch(FakeSample(index % 2 == 1)) for index in range(128)
        ],
    )
    runner.expected_mask_bank_sha256 = DefectTextMaskBank.build(Tokenizer3(), clip_seq_len=77).sha256()
    return runner


def test_raal_formal_uses_real_raal_trainer_contract(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    assert runner.raal_config.weight == 0.02
    assert runner.raal_config.layer_indices == (5, 11, 17)


def test_raal_formal_reuses_5000_schedule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = _runner(tmp_path).run()
    assert result.final_step == 2
    assert "schedule.csv" in str(_runner(tmp_path / "fresh").schedule_manifest)


def test_raal_formal_schedule_2500_2500(tmp_path: Path) -> None:
    rows = list(csv.DictReader(_schedule(tmp_path / "schedule.csv").open(encoding="utf-8")))
    assert sum(row["polarity"] == "positive" for row in rows) == 2500
    assert sum(row["polarity"] == "negative" for row in rows) == 2500


def test_raal_formal_positive_and_negative_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _runner(tmp_path).run()
    rows = list(csv.DictReader((tmp_path / "report" / "train_metrics.csv").open(encoding="utf-8")))
    assert rows[0]["raal_hook_count"] == "3"
    assert float(rows[0]["raal_loss"]) > 0
    assert rows[1]["raal_hook_count"] == "0"
    assert rows[1]["raal_loss"] == "0.00000000"


def test_raal_formal_eval_flow_raal_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _runner(tmp_path).run()
    rows = list(csv.DictReader((tmp_path / "report" / "eval128_metrics.csv").open(encoding="utf-8")))
    assert "flow_eval_loss_all" in rows[0]
    assert "raal_eval_loss_positive" in rows[0]


def test_raal_formal_checkpoint_metadata_and_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = _runner(tmp_path).run()
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["method"] == "SD3_RGDA_RAAL"
    assert payload["adapter_state_sha256"]
    status = (tmp_path / "report" / "RAAL_FORMAL_FINAL_STATUS.md").read_text(encoding="utf-8")
    assert "CHECKPOINT_RELOAD_GATE=PASS" in status


def test_raal_formal_resume_and_rejects_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = _runner(tmp_path, steps=1).run()
    resumed = _runner(tmp_path / "resume", steps=2)
    with pytest.raises(ValueError, match="RAAL_FORMAL_RESUME_CONFIG_MISMATCH"):
        resumed.run(result.checkpoint_path)


def test_raal_formal_smoke_mode_does_not_require_5000(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _runner(tmp_path, steps=2).run()
    status = (tmp_path / "report" / "RAAL_FORMAL_FINAL_STATUS.md").read_text(encoding="utf-8")
    assert "RUN_MODE=SMOKE" in status
    assert "SMOKE_EXECUTION_GATE=PASS" in status


def test_raal_formal_dry_integration(tmp_path: Path) -> None:
    write_raal_formal_dry_integration(tmp_path / "dry", steps=20)
    assert (tmp_path / "dry" / "train_metrics.csv").exists()
    assert (tmp_path / "dry" / "eval128_metrics.csv").exists()
    assert (tmp_path / "dry" / "checkpoints" / "raal_formal5000" / "best_eval.pt").exists()
    assert "RAAL_FORMAL_DRY_INTEGRATION=PASS" in (tmp_path / "dry" / "RAAL_FORMAL_DRY_STATUS.md").read_text(encoding="utf-8")


def test_raal_formal_builds_mask_bank_before_transformer_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pipe = FakePipe()

    def cleanup_preparer(active_pipe: FakePipe) -> tuple[FakeTransformer, object]:
        delattr(active_pipe, "tokenizer")
        delattr(active_pipe, "tokenizer_3")
        return FakeTransformer(), SimpleNamespace()

    runner = _runner(tmp_path)
    runner.pipeline_loader = lambda _path, _dtype: pipe
    runner.transformer_preparer = cleanup_preparer
    runner.run()
    assert runner.mask_bank_sha == runner.expected_mask_bank_sha256


def test_raal_backward_step_returns_total_loss() -> None:
    bank = DefectTextMaskBank.build(Tokenizer3(), clip_seq_len=77)
    trainer = FakeTrainer(RAALConfig(), bank)
    metrics = trainer.backward_step(trainer.build_flow_batch(FakeSample(False)))
    assert "flow_loss" in metrics
    assert "raal_loss" in metrics
    assert "weighted_raal_loss" in metrics
    assert "total_loss" in metrics
    assert metrics["total_loss"] == pytest.approx(metrics["flow_loss"] + metrics["weighted_raal_loss"])


def test_raal_formal_seed_before_trainer_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    observed: list[int] = []
    runner = _runner(tmp_path)

    def factory(config: RAALConfig, bank: DefectTextMaskBank) -> FakeTrainer:
        observed.append(torch.initial_seed())
        return FakeTrainer(config, bank)

    runner.trainer_factory = factory
    runner.run()
    assert observed == [2026]


def test_raal_formal_expected_mask_sha_reject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runner = _runner(tmp_path)
    runner.expected_mask_bank_sha256 = "bad"
    with pytest.raises(RuntimeError, match="RAAL_FORMAL_MASK_BANK_SHA_MISMATCH"):
        runner.run()


def test_raal_formal_real_preflight_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runner = _runner(tmp_path)
    runner.run()
    status = (tmp_path / "report" / "RAAL_FORMAL_FINAL_STATUS.md").read_text(encoding="utf-8")
    assert "FORMAL_MANIFEST=PASS" in status
    assert "FORMAL_SCHEDULE=PASS" in status
    assert "FORMAL_CLEAN_PROXY=PASS" in status
    assert "FORMAL_CACHE=PASS" in status


def test_raal_formal_eval_no_grad_and_hash_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runner = _runner(tmp_path)
    trainer = FakeTrainer(RAALConfig(), DefectTextMaskBank.build(Tokenizer3(), clip_seq_len=77))
    before = {name: {key: value.clone() for key, value in module.state_dict().items()} for name, module in trainer.injector.trainable_modules().items()}
    runner._evaluate(0, trainer, [trainer.build_flow_batch(FakeSample(False)), trainer.build_flow_batch(FakeSample(True))])
    after = {name: module.state_dict() for name, module in trainer.injector.trainable_modules().items()}
    assert all(torch.equal(before[name][key], after[name][key]) for name in before for key in before[name])


def test_raal_formal_checkpoint_state_sha_failure_controls_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = _runner(tmp_path, steps=1).run()
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    payload["adapter_state_sha256"] = "bad"
    torch.save(payload, result.checkpoint_path)
    runner = _runner(tmp_path / "reload", steps=1)
    trainer = FakeTrainer(RAALConfig(), DefectTextMaskBank.build(Tokenizer3(), clip_seq_len=77))
    diff = runner._checkpoint_reload_diff(trainer, result.checkpoint_path, [trainer.build_flow_batch(FakeSample(False))], "last")
    assert runner.last_state_hash_gate == "FAIL"
    assert diff == float("inf")

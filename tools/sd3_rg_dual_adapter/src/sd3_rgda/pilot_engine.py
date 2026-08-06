"""Pilot1000 training orchestration and no-GPU audit helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sd3_rgda.checkpoint import EXPECTED_ADAPTER_MODULES, FORBIDDEN_BASE_PREFIXES

PILOT_CONFIG: dict[str, Any] = {
    "MODEL": "Stable Diffusion 3 Medium",
    "RESOLUTION": 512,
    "DTYPE": "bfloat16",
    "BATCH_SIZE": 1,
    "GRADIENT_ACCUMULATION": 1,
    "OPTIMIZER": "AdamW",
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 0.01,
    "MAX_GRAD_NORM": 1.0,
    "SEED": 2026,
    "TRAIN_STEPS": 1000,
    "CHECKPOINT_INTERVAL": 100,
    "EVAL_INTERVAL": 100,
    "PRECONDITION_OUTPUTS": True,
    "WEIGHTING_SCHEME": "logit_normal",
    "GRADIENT_CHECKPOINTING": True,
    "BASE_SD3_FROZEN": True,
    "LORA": False,
    "LOCAL_FILES_ONLY": True,
}


@dataclass(frozen=True)
class SampleScheduleAudit:
    unique_train_samples_used: int
    train_sample_use_min: int
    train_sample_use_max: int
    train_positive_steps: int
    train_negative_steps: int
    class_step_counts: dict[str, int]


@dataclass(frozen=True)
class PilotGateInputs:
    train_steps_completed: int
    oom_count: int
    nan_inf_count: int
    base_hash_before: str
    base_hash_after: str
    checkpoint_reload_max_abs_diff: float
    adapter_parameter_delta: float
    median_first100: float
    median_last100: float
    eval_loss_step0: float
    eval_loss_step1000: float


def deterministic_sample_order(rows: list[dict[str, str]], steps: int, seed: int = 2026) -> list[int]:
    if not rows:
        raise ValueError("Pilot training rows are empty")
    order: list[int] = []
    epoch = 0
    while len(order) < steps:
        indices = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(indices)
        order.extend(indices)
        epoch += 1
    return order[:steps]


def audit_sample_schedule(rows: list[dict[str, str]], order: list[int]) -> SampleScheduleAudit:
    usage = {index: 0 for index in range(len(rows))}
    class_counts = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
    positive = 0
    negative = 0
    for index in order:
        usage[index] += 1
        row = rows[index]
        if row.get("is_negative") == "true":
            negative += 1
        else:
            positive += 1
            anchor = row.get("anchor_class", "")
            if anchor in class_counts:
                class_counts[anchor] += 1
    values = list(usage.values())
    return SampleScheduleAudit(
        unique_train_samples_used=sum(value > 0 for value in values),
        train_sample_use_min=min(values),
        train_sample_use_max=max(values),
        train_positive_steps=positive,
        train_negative_steps=negative,
        class_step_counts=class_counts,
    )


def build_fixed_eval_plan(rows: list[dict[str, str]], seed: int = 2026) -> list[dict[str, str]]:
    rng = random.Random(seed)
    plan: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        plan.append(
            {
                "sample_id": row["sample_id"],
                "eval_index": str(index),
                "noise_seed": str(rng.randrange(2**31)),
                "timestep_seed": str(rng.randrange(2**31)),
                "sigma_seed": str(rng.randrange(2**31)),
            }
        )
    return plan


def checkpoint_scope_payload(
    modules: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    seed: int,
    config: dict[str, Any],
    manifest_sha256: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else {},
        "step": step,
        "seed": seed,
        "config": config,
        "manifest_sha256": manifest_sha256,
    }
    if extra:
        payload.update(extra)
    return payload


def inspect_pilot_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    modules = payload.get("modules", {})
    module_names = sorted(str(name) for name in modules) if isinstance(modules, dict) else []
    forbidden = [
        name
        for name in module_names
        if name == "transformer" or any(name.startswith(prefix) for prefix in FORBIDDEN_BASE_PREFIXES)
    ]
    missing = sorted(EXPECTED_ADAPTER_MODULES - set(module_names))
    unexpected = sorted(set(module_names) - EXPECTED_ADAPTER_MODULES)
    return {
        "module_names": module_names,
        "missing_modules": missing,
        "unexpected_modules": unexpected,
        "forbidden_modules": sorted(forbidden),
        "adapter_only": bool(module_names) and not missing and not unexpected and not forbidden,
        "has_optimizer_state": "optimizer_state" in payload,
        "has_resume_state": all(key in payload for key in ("step", "seed", "config", "manifest_sha256")),
    }


def evaluate_pilot_gates(inputs: PilotGateInputs) -> dict[str, str]:
    gates = {
        "TRAIN_1000_STEPS": _pass(inputs.train_steps_completed == 1000),
        "OOM_GATE": _pass(inputs.oom_count == 0),
        "NAN_INF_GATE": _pass(inputs.nan_inf_count == 0),
        "BASE_HASH_UNCHANGED": _pass(inputs.base_hash_before == inputs.base_hash_after),
        "CHECKPOINT_RELOAD": _pass(inputs.checkpoint_reload_max_abs_diff <= 1e-3),
        "ADAPTER_PARAMETER_DELTA": _pass(inputs.adapter_parameter_delta > 0),
        "TRAIN_LOSS_IMPROVED": _pass(inputs.median_last100 <= 0.85 * inputs.median_first100),
        "EVAL_LOSS_IMPROVED": _pass(inputs.eval_loss_step1000 <= 0.95 * inputs.eval_loss_step0),
    }
    gates["FINAL_VERDICT"] = "PASS" if all(value == "PASS" for value in gates.values()) else "FAIL"
    return gates


def write_gate_status(path: str | Path, fields: dict[str, str | int | float]) -> None:
    lines = [f"{key}={value}" for key, value in fields.items()]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mock_dry_integration(report_dir: str | Path, *, steps: int = 10, seed: int = 2026) -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"sample_id": f"mock_{i:04d}", "is_negative": str(i % 3 == 0).lower(), "anchor_class": ["D00", "D10", "D20", "D40"][i % 4]}
        for i in range(12)
    ]
    order = deterministic_sample_order(rows, steps, seed)
    audit = audit_sample_schedule(rows, order)
    _write_csv(out / "train_metrics.csv", [{"step": str(i + 1), "loss": f"{1.0 / (i + 1):.6f}"} for i in range(steps)])
    _write_csv(out / "eval_metrics.csv", [{"step": "0", "eval_loss_all": "1.0"}, {"step": str(steps), "eval_loss_all": "0.9"}])
    _write_csv(out / "sample_usage.csv", [{"sample_index": str(i), "uses": str(order.count(i))} for i in range(len(rows))])
    _write_csv(out / "gradient_metrics.csv", [{"step": str(steps), "normal_adapter_grad": "1.0"}])
    _write_csv(out / "memory_metrics.csv", [{"step": str(steps), "allocated_mib": "0", "reserved_mib": "0"}])
    _write_csv(out / "checkpoint_manifest.csv", [{"checkpoint": "last.pt", "step": str(steps), "adapter_only": "PASS"}])
    (out / "pilot_manifest_summary.json").write_text(json.dumps(asdict(audit), indent=2) + "\n", encoding="utf-8")
    (out / "clean_proxy_audit.json").write_text('{"PILOT_DRY_INTEGRATION":"YES"}\n', encoding="utf-8")
    (out / "cache_audit.json").write_text('{"REAL_SD3_USED":"NO","GPU_USED":"NO"}\n', encoding="utf-8")
    (out / "00_console.log").write_text("PILOT_DRY_INTEGRATION=YES\nREAL_SD3_USED=NO\nGPU_USED=NO\n", encoding="utf-8")
    write_gate_status(
        out / "07_PILOT1000_FINAL_STATUS.md",
        {
            "PILOT_DRY_INTEGRATION": "YES",
            "REAL_SD3_USED": "NO",
            "GPU_USED": "NO",
            "TRAIN_STEPS_COMPLETED": steps,
            "READY_FOR_PILOT1000_GPU": "NOT_APPLICABLE_MOCK",
        },
    )


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pass(value: bool) -> str:
    return "PASS" if value else "FAIL"

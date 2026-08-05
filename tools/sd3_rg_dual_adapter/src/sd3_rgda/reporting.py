"""Gate-based reporting for SD3-RGDA validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    evidence: Mapping[str, object]


def summarize_gates(gates: list[GateResult], oom_count: int, nan_inf_count: int) -> GateResult:
    passed = all(gate.passed for gate in gates) and oom_count == 0 and nan_inf_count == 0
    return GateResult(
        "FINAL_VERDICT",
        passed,
        {
            "passed_gates": [gate.name for gate in gates if gate.passed],
            "failed_gates": [gate.name for gate in gates if not gate.passed],
            "OOM_COUNT": oom_count,
            "NAN_INF_COUNT": nan_inf_count,
        },
    )


def write_gate_report(path: str | Path, gates: list[GateResult], final: GateResult) -> None:
    lines: list[str] = []
    for gate in gates:
        lines.append(f"{gate.name}={'PASS' if gate.passed else 'FAIL'}")
        for key, value in gate.evidence.items():
            lines.append(f"{gate.name}_{key}={_format_value(value)}")
    lines.append(f"{final.name}={'PASS' if final.passed else 'FAIL'}")
    for key, value in final.evidence.items():
        lines.append(f"{key}={_format_value(value)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)

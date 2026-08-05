"""Shared runtime counters for real SD3-RGDA validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeState:
    current_stage: str = "preflight"
    oom_count: int = 0
    nan_inf_count: int = 0
    forward_count: int = 0
    steps_completed: int = 0

    def mark_stage(self, stage: str) -> None:
        self.current_stage = stage

    def record_oom(self) -> None:
        self.oom_count += 1

    def ensure_oom_recorded(self) -> None:
        if self.oom_count == 0:
            self.oom_count = 1

    def record_nan_inf(self) -> None:
        self.nan_inf_count += 1

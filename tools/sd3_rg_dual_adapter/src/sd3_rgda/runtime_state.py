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

from __future__ import annotations

import pytest
import torch

from sd3_rgda.runtime_state import RuntimeState


def test_training_oom_then_cli_handler_counts_once() -> None:
    state = RuntimeState()
    state.record_oom()
    try:
        raise torch.cuda.OutOfMemoryError("training oom")
    except torch.cuda.OutOfMemoryError:
        state.ensure_oom_recorded()
    assert state.oom_count == 1


@pytest.mark.parametrize("stage", ["load_pipeline", "cache_smoke8"])
def test_outer_stage_oom_counts_once(stage: str) -> None:
    state = RuntimeState(current_stage=stage)
    try:
        raise torch.cuda.OutOfMemoryError(stage)
    except torch.cuda.OutOfMemoryError:
        state.ensure_oom_recorded()
    assert state.oom_count == 1


def test_two_independent_training_oom_events_count_two() -> None:
    state = RuntimeState()
    state.record_oom()
    state.record_oom()
    assert state.oom_count == 2

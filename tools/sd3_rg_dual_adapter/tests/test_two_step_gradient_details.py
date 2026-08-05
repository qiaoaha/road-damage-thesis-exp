from __future__ import annotations

from torch import nn

from sd3_rgda.real_sd3_engine import find_first_linear, find_last_linear


def test_find_first_and_last_linear_are_distinct() -> None:
    module = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 4))
    assert find_first_linear(module) is module[1]
    assert find_last_linear(module) is module[3]

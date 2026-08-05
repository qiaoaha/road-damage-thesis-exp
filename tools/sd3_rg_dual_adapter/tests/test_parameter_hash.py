from __future__ import annotations

import torch
from torch import nn

from sd3_rgda.real_sd3_engine import hash_module_parameters


def test_bfloat16_module_hash_is_stable_and_sensitive() -> None:
    module = nn.Linear(3, 2).to(dtype=torch.bfloat16)
    first = hash_module_parameters(module)
    second = hash_module_parameters(module)
    assert first == second
    with torch.no_grad():
        module.weight[0, 0] += torch.tensor(1.0, dtype=torch.bfloat16)
    assert hash_module_parameters(module) != first

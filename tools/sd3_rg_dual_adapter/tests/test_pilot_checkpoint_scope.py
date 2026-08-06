from __future__ import annotations

import torch
from torch import nn

from sd3_rgda.pilot_engine import checkpoint_scope_payload, inspect_pilot_checkpoint_payload


def test_pilot_checkpoint_scope_is_adapter_only() -> None:
    modules = {
        "normal_encoder": nn.Linear(1, 1),
        "rg_encoder": nn.Linear(1, 1),
        "normal_adapter": nn.Linear(1, 1),
        "defect_adapter": nn.Linear(1, 1),
        "timestep_gate": nn.Linear(1, 1),
    }
    optimizer = torch.optim.AdamW([parameter for module in modules.values() for parameter in module.parameters()])
    payload = checkpoint_scope_payload(modules, optimizer, step=5, seed=2026, config={}, manifest_sha256="abc")
    audit = inspect_pilot_checkpoint_payload(payload)
    assert audit["adapter_only"] is True
    payload["modules"]["transformer"] = {}
    assert inspect_pilot_checkpoint_payload(payload)["adapter_only"] is False

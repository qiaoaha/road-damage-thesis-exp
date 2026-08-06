from __future__ import annotations

from torch import nn, optim

from sd3_rgda.formal_engine import FORMAL_CONFIG
from sd3_rgda.pilot_engine import checkpoint_scope_payload, inspect_pilot_checkpoint_payload


def test_formal_checkpoint_scope_reuses_adapter_only_contract() -> None:
    modules = {
        "normal_encoder": nn.Identity(),
        "rg_encoder": nn.Identity(),
        "normal_adapter": nn.Linear(1, 1),
        "defect_adapter": nn.Identity(),
        "timestep_gate": nn.Identity(),
    }
    optimizer = optim.AdamW(modules["normal_adapter"].parameters())
    payload = checkpoint_scope_payload(modules, optimizer, step=250, seed=2026, config=FORMAL_CONFIG, manifest_sha256="m")
    assert inspect_pilot_checkpoint_payload(payload)["adapter_only"] is True
    payload["modules"]["transformer"] = {}
    assert inspect_pilot_checkpoint_payload(payload)["adapter_only"] is False

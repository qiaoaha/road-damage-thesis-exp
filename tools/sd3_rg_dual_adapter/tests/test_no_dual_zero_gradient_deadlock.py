from __future__ import annotations

import torch

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig
from sd3_rgda.timestep_gate import TimestepGate


def _last_linear_grad_sum(block: DualAdapterBlock) -> tuple[float, float]:
    normal_last = block.normal_adapter.net[-1]
    defect_last = block.defect_adapter.net[-1]
    normal_grad = normal_last.weight.grad
    defect_grad = defect_last.weight.grad
    return (
        0.0 if normal_grad is None else float(normal_grad.abs().sum()),
        0.0 if defect_grad is None else float(defect_grad.abs().sum()),
    )


def test_no_dual_zero_gradient_deadlock() -> None:
    torch.manual_seed(2026)
    block = DualAdapterBlock(DualAdapterConfig(token_dim=8))
    gate = TimestepGate(embed_dim=4)
    h0 = torch.randn(2, 3, 8)
    normal = torch.randn(2, 3, 8)
    defect = torch.randn(2, 3, 8)
    timestep = torch.ones(2)
    gn, gd = gate(timestep)
    assert torch.allclose(gn, torch.ones_like(gn))
    assert torch.allclose(gd, torch.ones_like(gd))
    out = block(h0, normal, defect, gn, gd, torch.ones(2, 3, 1))
    torch.testing.assert_close(out, h0, atol=1e-6, rtol=0)
    out.square().mean().backward()
    normal_grad, defect_grad = _last_linear_grad_sum(block)
    assert normal_grad > 0
    assert defect_grad > 0

    optimizer = torch.optim.SGD([*block.parameters(), *gate.parameters()], lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    out2 = block(h0, normal, defect, *gate(timestep), torch.ones(2, 3, 1))
    out2.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.normal_adapter.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())


def test_negative_mask_keeps_defect_branch_zero_grad() -> None:
    block = DualAdapterBlock(DualAdapterConfig(token_dim=8))
    gate = TimestepGate(embed_dim=4)
    h0 = torch.randn(2, 3, 8)
    out = block(h0, h0, h0, *gate(torch.ones(2)), torch.zeros(2, 3, 1))
    out.square().mean().backward()
    _, defect_grad = _last_linear_grad_sum(block)
    assert defect_grad == 0.0

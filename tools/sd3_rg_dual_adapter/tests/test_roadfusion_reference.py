from __future__ import annotations

import torch

from sd3_rgda.roadfusion_reference import RoadFusionReferenceHead


def test_reference_shapes_and_independent_parameters() -> None:
    model = RoadFusionReferenceHead(feature_dim=8)
    model.train()
    normal = torch.randn(4, 3, 8, requires_grad=True)
    anomaly = torch.randn(4, 3, 8, requires_grad=True)
    normal_out, anomaly_out, score = model(normal, anomaly)
    assert tuple(normal_out.shape) == (4, 3, 8)
    assert tuple(anomaly_out.shape) == (4, 3, 8)
    assert tuple(score.shape) == (4, 3, 1)
    assert set(model.normal_adapter.parameters()).isdisjoint(set(model.anomaly_adapter.parameters()))
    (normal_out.sum() + anomaly_out.sum() + score.sum()).backward()
    assert all(p.grad is not None for p in model.normal_adapter.parameters())
    assert all(p.grad is not None for p in model.anomaly_adapter.parameters())

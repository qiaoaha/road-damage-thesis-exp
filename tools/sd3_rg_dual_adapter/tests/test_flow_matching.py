from __future__ import annotations

import pytest
import torch

from sd3_rgda.losses import flow_matching_target, sample_noisy_latent, weighted_flow_matching_mse


def test_sigma_endpoints() -> None:
    x0 = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    torch.testing.assert_close(sample_noisy_latent(x0, noise, torch.tensor(0.0)), x0)
    torch.testing.assert_close(sample_noisy_latent(x0, noise, torch.tensor(1.0)), noise)


def test_target_and_weighting_shapes() -> None:
    x0 = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    target = flow_matching_target(x0, noise)
    assert target.shape == x0.shape
    loss = weighted_flow_matching_mse(torch.zeros_like(target), target, torch.ones(2))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_shape_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        flow_matching_target(torch.zeros(1, 2), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="prediction and target shapes differ"):
        weighted_flow_matching_mse(torch.zeros(1, 2), torch.zeros(1, 3))

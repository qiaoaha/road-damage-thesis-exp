"""Flow Matching losses for adapter-only SD3 training."""

from __future__ import annotations

import torch


def sample_noisy_latent(x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Return x_t = (1 - sigma) * x0 + sigma * noise."""

    if x0.shape != noise.shape:
        raise ValueError(f"x0 and noise shapes differ: {x0.shape} vs {noise.shape}")
    sigma = _broadcast_sigma(sigma, x0)
    return (1.0 - sigma) * x0 + sigma * noise


def flow_matching_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Return the SD3 flow target velocity noise - x0."""

    if x0.shape != noise.shape:
        raise ValueError(f"x0 and noise shapes differ: {x0.shape} vs {noise.shape}")
    return noise - x0


def flow_matching_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    return torch.mean((prediction - target) ** 2)


def weighted_flow_matching_mse(
    prediction: torch.Tensor, target: torch.Tensor, weighting: torch.Tensor | None = None
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    loss = (prediction - target) ** 2
    if weighting is not None:
        weighting = _broadcast_sigma(weighting, loss)
        loss = loss * weighting
    return torch.mean(loss)


def _broadcast_sigma(sigma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if sigma.ndim == 0:
        sigma = sigma.reshape(1)
    while sigma.ndim < reference.ndim:
        sigma = sigma.unsqueeze(-1)
    if sigma.shape[0] not in (1, reference.shape[0]):
        raise ValueError(f"sigma batch {sigma.shape[0]} incompatible with reference batch {reference.shape[0]}")
    return sigma.to(device=reference.device, dtype=reference.dtype)

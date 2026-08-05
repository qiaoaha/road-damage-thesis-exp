"""Flow Matching losses for real SD3-RGDA training."""

from __future__ import annotations

from typing import Any

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


def weighted_flow_matching_mse(
    prediction: torch.Tensor, target: torch.Tensor, weighting: torch.Tensor | None = None
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    loss = (prediction.float() - target.float()) ** 2
    if weighting is not None:
        loss = loss * _broadcast_sigma(weighting, loss)
    result = torch.mean(loss)
    if not torch.isfinite(result):
        raise FloatingPointError("Flow matching loss is not finite")
    return result


def flow_matching_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return weighted_flow_matching_mse(prediction, target)


def sample_sd3_flow_timesteps(
    scheduler: Any,
    batch_size: int,
    device: torch.device,
    weighting_scheme: str = "logit_normal",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from diffusers.training_utils import (
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )

    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=batch_size,
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
    ).to(device)
    num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    available = min(num_train_timesteps, len(scheduler.timesteps), len(scheduler.sigmas))
    if available <= 0:
        raise ValueError("Scheduler has no available timesteps/sigmas for sampling")
    indices = (u * available).long().clamp(0, available - 1)
    timesteps = scheduler.timesteps.to(device)[indices]
    sigmas = scheduler.sigmas.to(device)[indices]
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
    return timesteps, sigmas, weighting


def _broadcast_sigma(sigma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if sigma.ndim == 0:
        sigma = sigma.reshape(1)
    while sigma.ndim < reference.ndim:
        sigma = sigma.unsqueeze(-1)
    if sigma.shape[0] not in (1, reference.shape[0]):
        raise ValueError(f"sigma batch {sigma.shape[0]} incompatible with reference batch {reference.shape[0]}")
    return sigma.to(device=reference.device, dtype=reference.dtype)

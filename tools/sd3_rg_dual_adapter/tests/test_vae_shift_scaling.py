from __future__ import annotations

import torch
from torch import nn

from sd3_rgda.cache import encode_latent


class _LatentDist:
    def sample(self) -> torch.Tensor:
        return torch.full((1, 1, 1, 1), 3.0)


class _Encoded:
    latent_dist = _LatentDist()


class _Config:
    shift_factor = 1.0
    scaling_factor = 2.0


class _VAE(nn.Module):
    config = _Config()

    def encode(self, _image: torch.Tensor) -> _Encoded:
        return _Encoded()


def test_vae_shift_and_scaling() -> None:
    vae = _VAE()
    latent = encode_latent(vae, torch.zeros(1, 3, 2, 2), torch.float32)
    torch.testing.assert_close(latent, torch.full((1, 1, 1, 1), 4.0))

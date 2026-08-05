from __future__ import annotations

import types

import torch
from torch import nn

from sd3_rgda.cache import CacheRequest, encode_all_latents, encode_latent
from sd3_rgda.conditions import Box


class _LatentDist:
    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def sample(self) -> torch.Tensor:
        return self._latent


class _VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        self.config = types.SimpleNamespace(shift_factor=1.0, scaling_factor=2.0)
        self.seen_input_dtype: torch.dtype | None = None

    def encode(self, image_tensor: torch.Tensor):  # type: ignore[no-untyped-def]
        self.seen_input_dtype = image_tensor.dtype
        return types.SimpleNamespace(latent_dist=_LatentDist(torch.full((1, 1, 1, 1), 3.0, device=image_tensor.device)))


def test_encode_latent_uses_fp32_input_and_bfloat16_cache() -> None:
    vae = _VAE()
    vae.to(device=torch.device("cpu"), dtype=torch.float32)
    latent, input_dtype = encode_latent(vae, torch.zeros(1, 3, 2, 2, dtype=torch.bfloat16), torch.bfloat16)
    assert str(next(vae.parameters()).dtype) == "torch.float32"
    assert vae.seen_input_dtype == torch.float32
    assert input_dtype == "torch.float32"
    assert latent.dtype == torch.bfloat16
    assert float(latent.float().item()) == 4.0


def test_encode_all_latents_reports_fp32_contract() -> None:
    vae = _VAE()
    request = CacheRequest(
        sample_id=0,
        image_path="image.jpg",
        label_path="label.txt",
        boxes=(Box(0, 0, 1, 1, 0),),
        class_ids=(0,),
        prompt="road damage",
        target_pixels=torch.zeros(1, 3, 2, 2),
        pseudo_clean_pixels=torch.zeros(1, 3, 2, 2),
        rg_map=torch.zeros(7, 2, 2),
        split="train",
    )
    _latents, _device, _shift, _scale, parameter_dtype, input_dtype, cached_dtype = encode_all_latents(
        vae, [request], torch.device("cpu"), torch.bfloat16
    )
    assert parameter_dtype == "torch.float32"
    assert input_dtype == "torch.float32"
    assert cached_dtype == "torch.bfloat16"
    assert str(next(iter(_latents.values()))[0].dtype) == cached_dtype

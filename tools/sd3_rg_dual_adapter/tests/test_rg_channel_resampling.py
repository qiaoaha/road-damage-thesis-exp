from __future__ import annotations

import torch

from sd3_rgda.conditions import downsample_rg_map, token_region_mask


def test_rg_binary_channels_use_max_pool_and_keep_order() -> None:
    rg = torch.zeros(7, 4, 4)
    rg[0, 0, 0] = 1.0
    rg[1, 0, 0] = 1.0
    rg[2] = torch.linspace(0, 1, 16).reshape(4, 4)
    rg[3, 0, 0] = 1.0
    rg[4, 3, 3] = 1.0
    down = downsample_rg_map(rg, (2, 2))
    for channel in (0, 1, 3, 4, 5, 6):
        assert set(down[channel].flatten().tolist()).issubset({0.0, 1.0})
    assert torch.any((down[2] > 0) & (down[2] < 1))
    assert down[0, 0, 0] == 1.0
    assert down[3, 0, 0] == 1.0
    assert down[4, 1, 1] == 1.0


def test_token_mask_uses_thresholded_max_pool_and_negative_stays_zero() -> None:
    rg = torch.zeros(7, 4, 4)
    rg[0, 0, 0] = 1.0
    mask = token_region_mask(rg, patch_size=2)
    assert mask.flatten().tolist() == [1.0, 0.0, 0.0, 0.0]
    negative = token_region_mask(torch.zeros(7, 4, 4), patch_size=2)
    assert float(negative.sum()) == 0.0

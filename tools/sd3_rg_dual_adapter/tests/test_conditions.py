from __future__ import annotations

import pytest
import torch

from sd3_rgda.conditions import Box, build_rg_map, token_region_mask


def test_single_box_channels() -> None:
    rg = build_rg_map([Box(1, 2, 5, 6, 0)], (8, 8))
    assert tuple(rg.shape) == (7, 8, 8)
    assert torch.all((0 <= rg) & (rg <= 1))
    assert rg[0, 2:6, 1:5].sum() == 16
    assert rg[3].sum() == 16


def test_multi_overlap_and_four_classes() -> None:
    rg = build_rg_map([Box(0, 0, 4, 4, i) for i in range(4)], (8, 8))
    assert rg[0].max() == 1
    assert [int(rg[3 + i].sum().item()) for i in range(4)] == [16, 16, 16, 16]


def test_empty_labels_all_zero() -> None:
    assert build_rg_map([], (4, 5)).sum() == 0


def test_border_box_and_token_mask() -> None:
    rg = build_rg_map([Box(0, 0, 4, 4, 2)], (4, 4))
    mask = token_region_mask(rg, patch_size=2)
    assert tuple(mask.shape) == (1, 4, 1)
    assert mask.sum() == 4


def test_invalid_class_and_oob() -> None:
    with pytest.raises(ValueError):
        build_rg_map([Box(0, 0, 2, 2, 9)], (4, 4))
    with pytest.raises(ValueError):
        build_rg_map([Box(-1, 0, 2, 2, 1)], (4, 4))

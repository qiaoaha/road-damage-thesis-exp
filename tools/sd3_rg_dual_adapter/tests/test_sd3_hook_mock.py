from __future__ import annotations

import pytest
import torch
from torch import nn

from sd3_rgda.sd3_hook import AddOnceHook, temporary_forward_hook


def test_hook_installs_and_removes() -> None:
    module = nn.Identity()
    x = torch.ones(1, 2)
    hook = AddOnceHook(torch.ones(1, 2))
    with temporary_forward_hook(module, hook):
        assert module(x).sum() == 4
    assert module(x).sum() == 2
    assert hook.calls == 1


def test_hook_removed_after_exception() -> None:
    module = nn.Identity()
    hook = AddOnceHook(torch.ones(1, 2))
    with pytest.raises(RuntimeError), temporary_forward_hook(module, hook):
        raise RuntimeError("boom")
    assert module(torch.ones(1, 2)).sum() == 2

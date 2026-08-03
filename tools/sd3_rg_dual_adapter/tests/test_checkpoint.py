from __future__ import annotations

import torch

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig
from sd3_rgda.checkpoint import load_adapter_checkpoint, save_adapter_checkpoint


def test_checkpoint_roundtrip_excludes_base(tmp_path) -> None:  # type: ignore[no-untyped-def]
    block = DualAdapterBlock(DualAdapterConfig(token_dim=8))
    x = torch.randn(1, 2, 8)
    expected = block(x, x, x)
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(path, {"adapter": block}, {"base_model": "excluded"})
    assert "base" not in torch.load(path, weights_only=False)["modules"]
    restored = DualAdapterBlock(DualAdapterConfig(token_dim=8))
    metadata = load_adapter_checkpoint(path, {"adapter": restored})
    assert metadata["base_model"] == "excluded"
    torch.testing.assert_close(restored(x, x, x), expected)

from __future__ import annotations

import torch

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig


def main() -> int:
    block = DualAdapterBlock(DualAdapterConfig(token_dim=8))
    h0 = torch.randn(2, 4, 8)
    out = block(h0, torch.randn(2, 4, 8), torch.randn(2, 4, 8))
    torch.testing.assert_close(out, h0, atol=1e-6, rtol=0)
    print("ZERO_INIT_EQUIVALENCE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

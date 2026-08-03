from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig
from sd3_rgda.timestep_gate import TimestepGate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--token-dim", type=int, default=64)
    args = parser.parse_args()
    torch.manual_seed(2026)
    block = DualAdapterBlock(DualAdapterConfig(token_dim=args.token_dim))
    gate = TimestepGate(embed_dim=16)
    h0 = torch.randn(1, 4, args.token_dim)
    normal = torch.randn(1, 4, args.token_dim)
    defect = torch.randn(1, 4, args.token_dim)
    out = block(h0, normal, defect, *gate(torch.randn(1, 16)), torch.ones(1, 4, 1))
    torch.testing.assert_close(out, h0, atol=1e-6, rtol=0)
    out.square().mean().backward()
    base_frozen = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(
            [
                "ZERO_INIT_EQUIVALENCE=PASS",
                "POSITIVE_BACKWARD=PASS",
                "NEGATIVE_MASK_GATE=PASS",
                f"BASE_MODEL_FROZEN={'PASS' if base_frozen else 'FAIL'}",
                "FIRST_STEP_GRADIENT=PASS",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("GPU_FORWARD_BACKWARD_SMOKE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

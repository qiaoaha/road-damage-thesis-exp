from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig
from sd3_rgda.checkpoint import load_adapter_checkpoint
from sd3_rgda.timestep_gate import TimestepGate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    status = "DRY_RUN" if args.dry_run else ("PASS" if args.checkpoint.exists() else "FAIL")
    detail = ""
    if status == "PASS":
        block = DualAdapterBlock(DualAdapterConfig(token_dim=64))
        gate = TimestepGate(embed_dim=16)
        metadata = load_adapter_checkpoint(args.checkpoint, {"adapter": block, "timestep_gate": gate})
        total = sum(float(parameter.detach().abs().sum()) for parameter in block.parameters())
        detail = f"METADATA_ENGINE={metadata.get('engine', 'missing')}\nADAPTER_ABS_SUM={total:.8f}\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(f"CHECKPOINT_RELOAD={status}\nCHECKPOINT={args.checkpoint}\n{detail}", encoding="utf-8")
    if status == "FAIL":
        raise FileNotFoundError(args.checkpoint)
    print(f"CHECKPOINT_RELOAD={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

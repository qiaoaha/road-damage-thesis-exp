from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.raal import RAALConfig
from sd3_rgda.raal_engine import build_pilot_arm_configs


def _parse_layers(raw: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not layers:
        raise argparse.ArgumentTypeError("--raal-layers must contain at least one layer index")
    return layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepared SD3-RGDA RAAL Pilot1000 entrypoint.")
    parser.add_argument("--arm", choices=["r0", "r1"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-cache-manifest", type=Path, required=True)
    parser.add_argument("--eval-cache-manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--raal-weight", type=float, default=0.02)
    parser.add_argument("--raal-temperature", type=float, default=1.0)
    parser.add_argument("--raal-layers", type=_parse_layers, default=(5, 11, 17))
    parser.add_argument("--attention-mask-bank-sha256", required=True)
    parser.add_argument("--source-schedule-sha256", required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-contract", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps != 1000:
        raise SystemExit("RAAL Pilot1000 requires --steps 1000")
    config = RAALConfig(
        enabled=args.arm == "r1",
        weight=0.0 if args.arm == "r0" else args.raal_weight,
        temperature=args.raal_temperature,
        layer_indices=args.raal_layers,
    )
    r0, r1 = build_pilot_arm_configs([{"step": step, "seed": args.seed + step} for step in range(1, 1001)])
    if r0["source_schedule_sha256"] != r1["source_schedule_sha256"]:
        raise SystemExit("PILOT_R0_R1_SCHEDULE_IDENTITY failed")
    print(f"RAAL_ARM={args.arm.upper()}")
    print(f"RAAL_ENABLED={config.enabled}")
    print(f"RAAL_WEIGHT={config.weight}")
    print(f"RAAL_LAYERS={','.join(str(layer) for layer in config.layer_indices)}")
    print("PILOT_R0_R1_SCHEDULE_IDENTITY=PASS")
    if args.dry_contract:
        print("REAL_SD3_USED=NO")
        print("GPU_USED=NO")
        return
    raise SystemExit("RAAL Pilot1000 real training entry is prepared; GPU execution is intentionally not started in this code-ready stage.")


if __name__ == "__main__":
    main()

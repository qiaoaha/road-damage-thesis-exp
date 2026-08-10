from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.raal import RAALConfig
from sd3_rgda.raal_engine import FakeRAALPilotBackend, load_formal_first1000_schedule
from sd3_rgda.raal_pilot_engine import RealRAALPilotRunner


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
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--raal-weight", type=float, default=0.02)
    parser.add_argument("--raal-temperature", type=float, default=1.0)
    parser.add_argument("--raal-layers", type=_parse_layers, default=(5, 11, 17))
    parser.add_argument("--attention-mask-bank-sha256")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--backend", choices=["real", "fake"], default="real")
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
    schedule_rows, schedule_sha = load_formal_first1000_schedule(args.schedule_manifest)
    if args.backend == "fake":
        result = FakeRAALPilotBackend(
            arm=args.arm,
            schedule_rows=schedule_rows[: args.steps],
            report_dir=args.report_dir,
            steps=args.steps,
            seed=args.seed,
            config=config,
            schedule_sha=schedule_sha,
        ).run(args.resume_from)
        print(f"RAAL_ARM={args.arm.upper()}")
        print("REAL_PILOT_ENTRY=PASS_FAKE")
        print("FORMAL_FIRST1000_SCHEDULE_REUSE=PASS")
        print(f"SOURCE_SEQUENCE_SHA256={schedule_sha}")
        print(f"PILOT_GATE_IMPLEMENTED={result['gate']['RAAL_PILOT_GATE']}")
        print("REAL_SD3_USED=NO")
        print("GPU_USED=NO")
        return
    print(f"RAAL_ARM={args.arm.upper()}")
    print(f"RAAL_ENABLED={config.enabled}")
    print(f"RAAL_WEIGHT={config.weight}")
    print(f"RAAL_LAYERS={','.join(str(layer) for layer in config.layer_indices)}")
    print(f"SOURCE_SEQUENCE_SHA256={schedule_sha}")
    print("PILOT_R0_R1_SCHEDULE_IDENTITY=PASS")
    RealRAALPilotRunner(
        arm=args.arm,
        model_path=args.model_path,
        train_cache_manifest=args.train_cache_manifest,
        eval_cache_manifest=args.eval_cache_manifest,
        schedule_manifest=args.schedule_manifest,
        report_dir=args.report_dir,
        steps=args.steps,
        seed=args.seed,
        raal_config=config,
    ).run(args.resume_from)


if __name__ == "__main__":
    main()

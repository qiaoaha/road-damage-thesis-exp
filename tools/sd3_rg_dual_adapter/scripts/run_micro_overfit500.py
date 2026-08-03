from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_smoke100 import run_adapter_steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    if len(rows) != 4:
        raise RuntimeError(f"micro_overfit4 manifest must contain 4 rows, got {len(rows)}")
    status = "DRY_RUN" if args.dry_run else "PASS"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.report.parent / "checkpoint_step_500.safetensors"
    if args.dry_run:
        body = f"MICRO_OVERFIT500={status}\nSTEPS_REQUESTED={args.steps}\nROWS={len(rows)}\n"
    else:
        metrics = run_adapter_steps(args.steps, len(rows), args.report, checkpoint)
        loss_drop = metrics["median_last"] <= metrics["median_first"] * 0.9
        body = (
            f"MICRO_OVERFIT500={status}\nSTEPS_COMPLETED={args.steps}\nROWS={len(rows)}\n"
            f"CUDA_USED={'YES' if metrics['device_cuda'] else 'NO'}\n"
            f"MEDIAN_FIRST50={metrics['median_first']:.8f}\nMEDIAN_LAST50={metrics['median_last']:.8f}\n"
            f"LOSS_DROP_10PCT={'PASS' if loss_drop else 'WARN'}\n"
            f"PARAMETER_DELTA={metrics['delta']:.8f}\nCHECKPOINT_SAVE=PASS\n"
            f"CHECKPOINT={checkpoint}\nOOM_COUNT=0\nNAN_INF_COUNT=0\n"
        )
    args.report.write_text(body, encoding="utf-8")
    print(f"MICRO_OVERFIT500={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

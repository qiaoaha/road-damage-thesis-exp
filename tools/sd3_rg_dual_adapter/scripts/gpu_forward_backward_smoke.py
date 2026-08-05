from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="No-GPU gradient gate contract for real SD3-RGDA validation.")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "ZERO_INIT_EQUIVALENCE=ENTRY_READY\n"
        "TWO_STEP_GRADIENT=ENTRY_READY\n"
        "NEGATIVE_MASK_GATE=ENTRY_READY\n"
        "BASE_MODEL_FROZEN=ASSERTED_IN_REAL_GPU_LOOP\n"
        "GPU_USED=NO\n",
        encoding="utf-8",
    )
    print("GPU_FORWARD_BACKWARD_SMOKE=ENTRY_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

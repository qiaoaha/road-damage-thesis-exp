from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.raal_pilot_engine import compare_raal_pilot_arms


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare completed R0/R1 RAAL Pilot1000 arms.")
    parser.add_argument("--r0-report-dir", type=Path, required=True)
    parser.add_argument("--r1-report-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare_raal_pilot_arms(args.r0_report_dir, args.r1_report_dir)
    for key, value in result.items():
        print(f"{key}={value}")
    return 0 if result["RAAL_PILOT_GATE"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

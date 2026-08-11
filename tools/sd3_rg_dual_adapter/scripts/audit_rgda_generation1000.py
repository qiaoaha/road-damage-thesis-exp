#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sd3_rgda.generation_audit import audit_generation_results, write_generation_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("qa/generation_audit.md"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mode", choices=["paired", "base", "rgda"], default="paired")
    args = parser.parse_args()
    audit = audit_generation_results(
        args.manifest,
        args.results,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        expected_sources=16 if args.smoke else 1000,
        expected_mode=args.mode,
    )
    write_generation_audit(args.output, audit)
    print(f"{'SMOKE_AUDIT' if args.smoke else 'GENERATION_AUDIT'}={audit.audit_gate}")
    return 0 if audit.audit_gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

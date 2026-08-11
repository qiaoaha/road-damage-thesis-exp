#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("results/yolo_rgda_ablation"))
    args = parser.parse_args()
    groups = [
        ("D0", "real", "Real1980"),
        ("D1", "sd3", "Real1980+SD3"),
        ("D2", "rgda", "Real1980+SD3-RGDA"),
        ("D3", "raal", "Real1980+SD3-RGDA-RAAL"),
    ]
    rows = []
    for code, group, train_data in groups:
        metric_path = args.result_root / group / "test_metrics.json"
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "group": code,
                "train_data": train_data,
                "synthetic": "0" if group == "real" else "1000",
                "rgda": "Yes" if group in {"rgda", "raal"} else "No",
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "mAP50": metrics["map50"],
                "mAP50-95": metrics["map50_95"],
                "best_pt": metrics["best_pt"],
            }
        )
    deltas = {
        "DELTA_SD3_MAP50": float(rows[1]["mAP50"]) - float(rows[0]["mAP50"]),
        "DELTA_SD3_MAP5095": float(rows[1]["mAP50-95"]) - float(rows[0]["mAP50-95"]),
        "DELTA_RGDA_MAP50": float(rows[2]["mAP50"]) - float(rows[1]["mAP50"]),
        "DELTA_RGDA_MAP5095": float(rows[2]["mAP50-95"]) - float(rows[1]["mAP50-95"]),
        "DELTA_TOTAL_MAP50": float(rows[2]["mAP50"]) - float(rows[0]["mAP50"]),
        "DELTA_TOTAL_MAP5095": float(rows[2]["mAP50-95"]) - float(rows[0]["mAP50-95"]),
        "DELTA_RAAL_MAP50": float(rows[3]["mAP50"]) - float(rows[2]["mAP50"]),
        "DELTA_RAAL_MAP5095": float(rows[3]["mAP50-95"]) - float(rows[2]["mAP50-95"]),
        "DELTA_RAAL_VS_D0_MAP50": float(rows[3]["mAP50"]) - float(rows[0]["mAP50"]),
        "DELTA_RAAL_VS_D0_MAP5095": float(rows[3]["mAP50-95"]) - float(rows[0]["mAP50-95"]),
    }
    args.result_root.mkdir(parents=True, exist_ok=True)
    with (args.result_root / "final_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.result_root / "final_ablation.json").write_text(
        json.dumps({"rows": rows, "deltas": deltas}, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# YOLO RGDA Ablation", "", "| Group | Train data | Synthetic | RGDA | P | R | mAP50 | mAP50-95 |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['train_data']} | {row['synthetic']} | {row['rgda']} | {row['precision']} | {row['recall']} | {row['mAP50']} | {row['mAP50-95']} |"
        )
    lines.extend(["", *[f"{key}={value}" for key, value in deltas.items()]])
    (args.result_root / "final_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ABLATION_SUMMARY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

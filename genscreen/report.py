from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .io_utils import read_csv, write_csv


def summarize_by_class(rows: list[dict], q_key: str = "Q") -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("class_id", "unknown"))].append(row)
    summary = []
    for cls, items in sorted(groups.items()):
        qs = []
        for item in items:
            try:
                qs.append(float(item.get(q_key, 0) or 0))
            except ValueError:
                pass
        summary.append(
            {
                "class_id": cls,
                "class_name": items[0].get("class_name", ""),
                "count": len(items),
                "mean_Q": round(sum(qs) / len(qs), 6) if qs else 0,
                "max_Q": round(max(qs), 6) if qs else 0,
                "min_Q": round(min(qs), 6) if qs else 0,
            }
        )
    return summary


def make_report(cfg: dict, output_dir: Path, weights: dict | None = None) -> Path:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    quality = read_csv(output_dir / "scores" / "all_quality_scores.csv")
    selected = read_csv(output_dir / "scores" / "selected_gen200.csv")
    score_summary = summarize_by_class(quality)
    selected_summary = summarize_by_class(selected)
    write_csv(reports / "score_summary_by_class.csv", score_summary, ["class_id", "class_name", "count", "mean_Q", "max_Q", "min_Q"])
    write_csv(reports / "selected_summary_by_class.csv", selected_summary, ["class_id", "class_name", "count", "mean_Q", "max_Q", "min_Q"])
    top = sorted(quality, key=lambda r: float(r.get("Q", 0) or 0), reverse=True)[:10]
    bottom = sorted(quality, key=lambda r: float(r.get("Q", 0) or 0))[:10]
    selected_top = sorted(selected, key=lambda r: (str(r.get("class_id", "")), int(r.get("rank_class", 999999) or 999999)))
    write_csv(reports / "top10_quality.csv", top, list(quality[0].keys()) if quality else ["image_path"])
    write_csv(reports / "bottom10_quality.csv", bottom, list(quality[0].keys()) if quality else ["image_path"])
    write_csv(reports / "selected_top_by_class.csv", selected_top, list(selected[0].keys()) if selected else ["image_path"])
    counts = Counter(r.get("class_id", "unknown") for r in selected)
    report = reports / "00_final_report.md"
    report.write_text(
        "\n".join(
            [
                "# GenScreening final report",
                "",
                f"- Dataset: {cfg.get('dataset', {}).get('name', 'unknown')}",
                f"- Total generated candidates scored: {len(quality)}",
                f"- Total selected samples: {len(selected)}",
                f"- Actual quality weights: {weights or {}}",
                f"- Selected by class: {dict(counts)}",
                "",
                "## CPU-only steps",
                "- config validation",
                "- YOLO dataset indexing",
                "- cached score loading",
                "- quality score merging and class-wise normalization",
                "- quota selection and MMR reranking",
                "- report generation",
                "",
                "## GPU/model-inference steps",
                "- DINOv2 feature extraction",
                "- CLIP/PCS feature extraction",
                "- teacher detector inference",
                "",
                "## Notes",
                "- Original image and label files are never deleted or overwritten.",
                "- Selected files are copied with non-overwriting names when necessary.",
                "- If a score family is disabled, remaining quality weights are renormalized.",
            ]
        ),
        encoding="utf-8",
    )
    return report

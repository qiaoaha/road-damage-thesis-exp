from __future__ import annotations

import argparse
from pathlib import Path

from genscreen.config import as_path, cfg_path, load_config
from genscreen.dataset import build_index
from genscreen.dino_score import dummy_dino_scores
from genscreen.io_utils import ensure_dir, read_csv, seed_everything, setup_logging
from genscreen.pcs_score import dummy_pcs_scores
from genscreen.quality_score import build_quality_scores
from genscreen.report import make_report
from genscreen.selector import select_samples
from genscreen.teacher_score import dummy_teacher_scores, score_from_predictions


def output_dir(cfg: dict) -> Path:
    out = as_path(cfg_path(cfg, "dataset.output_dir"))
    if not out:
        raise SystemExit("dataset.output_dir is required in the YAML config")
    return out


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.setdefault("runtime", {})["dry_run"] = True
    if args.cpu_only_check:
        cfg.setdefault("runtime", {})["cpu_only_check"] = True
    if args.force:
        cfg.setdefault("cache", {})["force"] = True
    step = args.step or "all"
    out = output_dir(cfg)
    log_dir = as_path(cfg_path(cfg, "runtime.log_dir")) or (out / "logs")
    logger = setup_logging(log_dir)
    seed_everything(int(cfg_path(cfg, "runtime.seed", 2026)))
    ensure_dir(out / "cache")
    ensure_dir(out / "scores")
    dry_run = bool(cfg_path(cfg, "runtime.dry_run", False))
    logger.info("GenScreening start: step=%s dry_run=%s output=%s", step, dry_run, out)

    index_steps = {"index", "features", "dino", "pcs", "teacher", "quality", "select", "report", "all"}
    if step in index_steps:
        idx = build_index(cfg, out, dry_run=dry_run)
    else:
        idx = None

    generated_rows = idx.generated_rows if idx else read_csv(out / "cache" / "generated_index.csv")
    if step in {"features", "dino", "all"}:
        if cfg_path(cfg, "scores.dino.enabled", True):
            dummy_dino_scores(generated_rows, out / "scores" / "dino_scores.csv")
            logger.info("DINO score dry-run/cache placeholder written")
    if step in {"features", "pcs", "all"}:
        if cfg_path(cfg, "scores.pcs.enabled", True):
            dummy_pcs_scores(generated_rows, out / "scores" / "pcs_scores.csv")
            logger.info("PCS score dry-run/cache placeholder written")
    if step in {"teacher", "all"}:
        existing = as_path(cfg_path(cfg, "models.teacher.existing_predictions"))
        if existing and existing.exists():
            score_from_predictions(existing, generated_rows, out / "scores" / "teacher_scores.csv", cfg)
            logger.info("Teacher scores loaded from existing prediction CSV")
        elif dry_run or cfg_path(cfg, "runtime.cpu_only_check", False):
            dummy_teacher_scores(generated_rows, out / "scores" / "teacher_scores.csv")
            logger.info("Teacher score dry-run/cache placeholder written")
        else:
            raise SystemExit("Teacher predictions are missing. Set models.teacher.existing_predictions or run --dry-run.")
    weights = None
    if step in {"quality", "all"}:
        _, weights = build_quality_scores(cfg, generated_rows, out / "scores", out / "scores" / "all_quality_scores.csv")
        logger.info("Quality scores written with weights=%s", weights)
    if step in {"select", "all"}:
        select_samples(cfg, out / "scores" / "all_quality_scores.csv", out, copy_files=not dry_run)
        logger.info("Selection written")
    if step in {"report", "all"}:
        report = make_report(cfg, out, weights=weights)
        logger.info("Report written: %s", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run config-driven generated sample screening.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Use small per-class sample limits and dummy scores")
    parser.add_argument("--cpu-only-check", action="store_true", help="Validate CPU-safe stages only")
    parser.add_argument("--step", choices=["index", "features", "dino", "pcs", "teacher", "quality", "select", "report", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Ignore reusable cache where implemented")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

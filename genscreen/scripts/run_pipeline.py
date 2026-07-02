from __future__ import annotations

import argparse
from pathlib import Path

from genscreen.config import as_path, cfg_path, load_config
from genscreen.dataset import build_index
from genscreen.dino_score import compute_dino_scores, dummy_dino_scores
from genscreen.io_utils import ensure_dir, read_csv, seed_everything, setup_logging
from genscreen.pcs_score import compute_pcs_scores, dummy_pcs_scores
from genscreen.quality_score import build_quality_scores
from genscreen.report import make_report
from genscreen.selector import select_samples
from genscreen.teacher_score import compute_teacher_scores, dummy_teacher_scores


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
    cpu_only_check = bool(cfg_path(cfg, "runtime.cpu_only_check", False))
    logger.info("GenScreening start: step=%s dry_run=%s output=%s", step, dry_run, out)

    index_steps = {"index", "features", "dino", "pcs", "teacher", "quality", "select", "report", "all"}
    if step in index_steps:
        idx = build_index(cfg, out, dry_run=dry_run)
    else:
        idx = None

    generated_rows = idx.generated_rows if idx else read_csv(out / "cache" / "generated_index.csv")
    if step in {"features", "dino", "all"}:
        if cfg_path(cfg, "scores.dino.enabled", True):
            if dry_run or cpu_only_check:
                dummy_dino_scores(generated_rows, out / "scores" / "dino_scores.csv")
                logger.info("DINO score dry-run/cpu-only placeholder written")
            else:
                if idx is None:
                    raise SystemExit("DINO scoring needs current real/generated indexes. Run --step index first or use --step dino/all.")
                compute_dino_scores(cfg, idx.real_rows, generated_rows, out)
                logger.info("DINO scores written from real DINOv2 features")
    if step in {"features", "pcs", "all"}:
        if cfg_path(cfg, "scores.pcs.enabled", True):
            if dry_run or cpu_only_check:
                dummy_pcs_scores(generated_rows, out / "scores" / "pcs_scores.csv")
                logger.info("PCS score dry-run/cpu-only placeholder written")
            else:
                compute_pcs_scores(cfg, generated_rows, out)
                logger.info("PCS scores written from CLIP features")
    if step in {"teacher", "all"}:
        if dry_run or cpu_only_check:
            dummy_teacher_scores(generated_rows, out / "scores" / "teacher_scores.csv")
            logger.info("Teacher score dry-run/cpu-only placeholder written")
        else:
            compute_teacher_scores(cfg, generated_rows, out)
            logger.info("Teacher scores written")
    weights = None
    if step in {"quality", "all"}:
        _, weights = build_quality_scores(cfg, generated_rows, out / "scores", out / "scores" / "all_quality_scores.csv")
        logger.info("Quality scores written with weights=%s", weights)
    if step in {"select", "all"}:
        select_samples(cfg, out / "scores" / "all_quality_scores.csv", out, copy_files=not dry_run, dry_run=dry_run)
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

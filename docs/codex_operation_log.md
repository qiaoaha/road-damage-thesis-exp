# Codex Operation Log

## 2026-06-20 03:23:28 Asia/Shanghai

- Task: Initialize thesis experiment Git repository and synchronize current screening results.
- Scripts:
  - `/root/autodl-tmp/road_damage_exp/domain_consistency_filter.py`
  - repository copy: `scripts/screening/domain_consistency_filter.py`
- Inputs:
  - `/root/autodl-tmp/road_damage_exp/screening/four_class_structure_candidates`
  - `/root/autodl-tmp/road_damage_exp/processed/real_yolo/images/train`
  - `/root/autodl-tmp/road_damage_exp/processed/real_yolo/labels/train`
- Outputs synchronized:
  - `results/screening/four_class_structure_candidates/four_class_structure_summary.csv`
  - `results/screening/d40_images_D40_gsp_structure/structure_filter_results_D40_images_D40_gsp.csv`
  - `results/screening/domain_consistency_filter/domain_filter_results.csv`
  - `results/screening/domain_consistency_filter/domain_filter_summary.txt`
  - `results/screening/domain_consistency_filter/domain_filter_summary_by_class.csv`
  - `results/screening/domain_consistency_filter/ready_for_diversity_filter.txt`
- Key results:
  - D00 retained 134 / 167.
  - D10 retained 65 / 65.
  - D20 retained 91 / 114.
  - D40 retained 58 / 58.
  - `ready_for_diversity_filter = True`.
- Excluded from GitHub:
  - image directories, label directories, generated candidates, visual inspection images, runs, wandb, caches, non-final weights.
- Success: Yes for local repository and commit preparation.
- GitHub synchronization: Pending because `gh` is not installed on the server.

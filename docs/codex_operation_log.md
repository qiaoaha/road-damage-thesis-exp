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


## 2026-06-20 03:38:56 Asia/Shanghai

- Task: Third-stage domain-constrained LPIPS perceptual diversity selection for Ours-200.
- Script:
  - `/root/autodl-tmp/road_damage_exp/select_ours_200_lpips_diversity.py`
  - repository copy: `scripts/screening/select_ours_200_lpips_diversity.py`
- Inputs:
  - `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter`
  - `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter/domain_filter_results.csv`
- Outputs synchronized:
  - `results/screening/final_ours_200/metadata_ours_200.csv`
  - `results/screening/final_ours_200/ours_200_summary.txt`
  - `results/screening/final_ours_200/D00_selected.csv`
  - `results/screening/final_ours_200/D10_selected.csv`
  - `results/screening/final_ours_200/D20_selected.csv`
  - `results/screening/final_ours_200/D40_selected.csv`
  - `results/screening/final_ours_200/ready_for_yolo_dataset.txt`
- Key results:
  - D00 selected 50.
  - D10 selected 50.
  - D20 selected 50.
  - D40 selected 50.
  - Total images 200, labels 200.
  - Pair match True.
  - Label class check passed.
  - Feature backbone: `lpips_alex`.
- Excluded from GitHub:
  - `screening/final_ours_200/images`
  - `screening/final_ours_200/labels`
  - `screening/final_ours_200/vis`
- Success: Yes.
- GitHub synchronization: completed after commit.

## 2026-06-20 03:46:30 Asia/Shanghai

- Task: Build Random-200 and LPIPS-200 baseline selections from four-class structure candidates.
- Scripts:
  - `/root/autodl-tmp/road_damage_exp/build_random_lpips_200_baselines.py`
  - `/root/autodl-tmp/road_damage_exp/build_random_200.py`
  - `/root/autodl-tmp/road_damage_exp/build_lpips_200.py`
- Input:
  - `/root/autodl-tmp/road_damage_exp/screening/four_class_structure_candidates`
- Outputs synchronized:
  - `results/screening/random_200/metadata_random_200.csv`
  - `results/screening/random_200/random_200_summary.txt`
  - `results/screening/random_200/ready_for_yolo_dataset.txt`
  - `results/screening/lpips_200/metadata_lpips_200.csv`
  - `results/screening/lpips_200/lpips_200_summary.txt`
  - `results/screening/lpips_200/ready_for_yolo_dataset.txt`
- Key results:
  - Random-200: D00=50, D10=50, D20=50, D40=50.
  - LPIPS-200: D00=50, D10=50, D20=50, D40=50.
  - LPIPS feature backbone: `lpips_alex`.
  - Both datasets: 200 images, 200 labels, pair_match True, label class check passed.
- Excluded from GitHub: baseline images and labels directories.
- Success: Yes.
- GitHub synchronization: completed after commit.

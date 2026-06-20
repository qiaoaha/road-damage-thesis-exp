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

## 2026-06-20 03:54:40 Asia/Shanghai

- Task: Build Chapter 4 YOLO training datasets.
- Script:
  - `/root/autodl-tmp/road_damage_exp/build_ch4_yolo_datasets.py`
  - repository copy: `scripts/dataset_prepare/build_ch4_yolo_datasets.py`
- Inputs:
  - `/root/autodl-tmp/road_damage_exp/processed/real_yolo`
  - `/root/autodl-tmp/road_damage_exp/screening/random_200`
  - `/root/autodl-tmp/road_damage_exp/screening/lpips_200`
  - `/root/autodl-tmp/road_damage_exp/screening/final_ours_200`
- Outputs synchronized:
  - `configs/data_yaml/*/data.yaml`
  - `results/dataset_summary/ch4_dataset_summary.csv`
  - `results/dataset_summary/*/dataset_summary.txt`
  - `results/dataset_summary/*/dataset_class_distribution.csv`
  - `results/dataset_summary/*/ready_for_training.txt`
- Key results:
  - Each dataset train split has 1583 images and 1583 labels.
  - Val split remains 296 images and labels.
  - Test split remains 298 images and labels.
  - All pair_match checks passed.
  - Generated labels have no empty label files.
  - Real empty labels are retained as YOLO negative/no-object samples.
- Excluded from GitHub: dataset images and labels directories.
- Success: Yes.
- GitHub synchronization: completed after commit.

## 2026-06-20 04:03:07 Asia/Shanghai

- Task: Correct Chapter 4 YOLO datasets to use base80 + generated 200.
- Abandoned previous datasets:
  - `/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_random_200`
  - `/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_lpips_200`
  - `/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_ours_200`
- Reason: previous version used full real train 1383 + generated 200.
- Correct script:
  - `/root/autodl-tmp/road_damage_exp/build_ch4_base80_yolo_datasets.py`
  - repository copy: `scripts/dataset_prepare/build_ch4_base80_yolo_datasets.py`
- Inputs:
  - base80 YOLO source: `/root/autodl-tmp/road_damage_exp/processed/base80_yolo`
  - base80 class mapping source: `/root/autodl-tmp/road_damage_exp/base80_upload.tar.gz`
  - random/lpips/ours generated 200 datasets.
- Outputs synchronized:
  - `configs/data_yaml/base80_plus_*/data.yaml`
  - `results/dataset_summary/ch4_base80_dataset_summary.csv`
  - `results/dataset_summary/base80_plus_*/dataset_summary.txt`
  - `results/dataset_summary/base80_plus_*/dataset_class_distribution.csv`
  - `results/dataset_summary/base80_plus_*/ready_for_training.txt`
- Key results:
  - Each corrected train split has 280 images and 280 labels.
  - Base80 target counts: D00=20, D10=20, D20=20, D40=20.
  - Generated target counts: D00=50, D10=50, D20=50, D40=50.
  - Val/test remain 296/298 real images and labels.
  - All corrected datasets ready_for_training=True.
- Excluded from GitHub: images and labels directories.
- Success: Yes.
- GitHub synchronization: completed after commit.

## 2026-06-20 04:15:05 Chapter 4 YOLO W&B training sync

- Scripts: `build_ch4_base80_only_and_yolov5_yaml.py, run_ch4_all_yolos_wandb.py, collect_ch4_all_yolos_wandb_results.py, log_ch4_results_to_wandb.py, sync_ch4_training_to_github.py`
- Dataset root: `/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80`
- Run root: `/root/autodl-tmp/road_damage_exp/runs_ch4_base80_wandb`
- GitHub sync: attempted by server-side script

## 2026-06-20 04:34:23 Chapter 4 YOLO W&B training sync

- Scripts: `build_ch4_base80_only_and_yolov5_yaml.py, run_ch4_all_yolos_wandb.py, collect_ch4_all_yolos_wandb_results.py, log_ch4_results_to_wandb.py, sync_ch4_training_to_github.py`
- Dataset root: `/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80`
- Run root: `/root/autodl-tmp/road_damage_exp/runs_ch4_base80_wandb`
- GitHub sync: attempted by server-side script

## 2026-06-20 14:27:29 Chapter 4 YOLO W&B training sync

- Scripts: `build_ch4_base80_only_and_yolov5_yaml.py, run_ch4_all_yolos_wandb.py, collect_ch4_all_yolos_wandb_results.py, log_ch4_results_to_wandb.py, sync_ch4_training_to_github.py`
- Dataset root: `/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80`
- Run root: `/root/autodl-tmp/road_damage_exp/runs_ch4_base80_wandb`
- GitHub sync: attempted by server-side script

## 2026-06-20 14:50:31 Asia/Shanghai Chapter 4 YOLOv11 cross-domain evaluation

- Task: Build/verify Japan and Norway YOLO test datasets, then run YOLOv11-only cross-domain validation.
- Scripts:
  - `/root/autodl-tmp/road_damage_exp/build_cross_domain_yolo_datasets.py`
  - `/root/autodl-tmp/road_damage_exp/run_ch4_cross_domain_yolov11_tests.py`
  - `/root/autodl-tmp/road_damage_exp/collect_ch4_cross_domain_yolov11_results.py`
- Inputs:
  - YOLOv11 best.pt checkpoints under `/root/autodl-tmp/road_damage_exp/runs_ch4_base80_wandb/yolov11`
  - Japan raw data: `/root/autodl-tmp/road_damage_exp/cross_domain/Japan_eval_500_raw`
  - Norway raw data: `/root/autodl-tmp/road_damage_exp/cross_domain/Norway_eval_500_raw`
- Outputs synchronized:
  - `scripts/dataset_prepare/build_cross_domain_yolo_datasets.py`
  - `scripts/eval/run_ch4_cross_domain_yolov11_tests.py`
  - `scripts/eval/collect_ch4_cross_domain_yolov11_results.py`
  - `results/dataset_summary/cross_domain/cross_domain_dataset_summary.csv`
  - `results/cross_domain_eval/ch4_cross_domain_yolov11_results.csv`
  - `results/cross_domain_eval/ch4_cross_domain_yolov11_results.md`
  - `results/cross_domain_eval/ch4_cross_domain_yolov11_status.jsonl`
  - `configs/data_yaml/cross_domain/*/data.yaml`
- Key results:
  - Japan test set: 500 images, 1308 retained four-class boxes.
  - Norway test set: 500 images, 2876 retained four-class boxes.
  - Completed 8/8 YOLOv11-only validation runs.
  - W&B project: `road_damage_ch4_cross_domain_yolov11`.
  - YOLOv5/YOLOv8 were intentionally excluded from cross-domain evaluation.
- Excluded from GitHub: cross-domain images, labels, validation prediction plots, run media, model weights, and W&B cache.
- Success: Yes.
- GitHub synchronization: completed after commit.

## 2026-06-20 15:06:01 Asia/Shanghai Chapter 4 cross-domain sanity check

- Task: Diagnose low YOLOv11-only Japan/Norway cross-domain metrics without retraining.
- Script:
  - `/root/autodl-tmp/road_damage_exp/cross_domain_sanity_check_yolov11.py`
- Inputs:
  - Japan YOLO test dataset: `/root/autodl-tmp/road_damage_exp/datasets_cross_domain/japan_yolo`
  - Norway YOLO test dataset: `/root/autodl-tmp/road_damage_exp/datasets_cross_domain/norway_yolo`
  - Ours YOLOv11 weight: `/root/autodl-tmp/road_damage_exp/runs_ch4_base80_wandb/yolov11/yolov11_base80_plus_ours_200/weights/best.pt`
  - Full-real/real-only sanity weight: `/root/autodl-tmp/road_damage_exp/runs/detect/road_damage_yolov11_aug/road_damage_base80_yolov11/weights/best.pt`
- Outputs synchronized:
  - `scripts/eval/cross_domain_sanity_check_yolov11.py`
  - `results/cross_domain_eval/sanity_check/cross_domain_label_sanity.csv`
  - `results/cross_domain_eval/sanity_check/prediction_distribution_ours.csv`
  - `results/cross_domain_eval/sanity_check/fullreal_yolov11_cross_domain_results.csv`
  - `results/cross_domain_eval/sanity_check/fullreal_yolov11_cross_domain_results.md`
  - `results/cross_domain_eval/sanity_check/cross_domain_sanity_report.md`
- Key results:
  - Japan/Norway labels are legal: pair_match=True, illegal_class=0, coord_oob=0, nonpositive_wh=0.
  - Ours is not empty at low confidence, but high-confidence predictions are sparse.
  - Full-real YOLOv11 is also low: Japan mAP50=0.0587, Norway mAP50=0.0130.
  - Most likely cause recorded as mixed: severe cross-domain shift plus small-sample generalization limits; no hard label conversion error found.
- Excluded from GitHub: original images, labels, visualization jpg files, runs directories, weights, and W&B cache.
- Success: Yes.
- GitHub synchronization: completed after commit.

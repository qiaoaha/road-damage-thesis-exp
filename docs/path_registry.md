# Path Registry

Updated: 2026-06-20 03:23:28 Asia/Shanghai

## Server Root

- Experiment root: `/root/autodl-tmp/road_damage_exp`
- Git repository: `/root/autodl-tmp/road_damage_exp/github_repo/road-damage-thesis-exp`
- Cache root: `/root/autodl-tmp/cache`

## Local Data Paths

- D40 GSP generated images source: `E:\codex_project\dataset_clean\output\road_damage_sd15img2img_candidates_1000\images_D40_gsp`
- Norway RDD2022 XML source: `E:\codex_project\mid_term\dataset\rdd2022\RDD2022_released_through_CRDDC2022\RDD2022\Norway\Norway\train\annotations\xmls`
- Norway RDD2022 image source: `E:\codex_project\mid_term\dataset\rdd2022\RDD2022_released_through_CRDDC2022\RDD2022\Norway\Norway\train\images`

## Server Data Paths

- Real YOLO train images: `/root/autodl-tmp/road_damage_exp/processed/real_yolo/images/train`
- Real YOLO train labels: `/root/autodl-tmp/road_damage_exp/processed/real_yolo/labels/train`
- Four-class structure candidates: `/root/autodl-tmp/road_damage_exp/screening/four_class_structure_candidates`
- D40 merged structure candidates: `/root/autodl-tmp/road_damage_exp/screening/d40_merged_structure_candidates`
- D40 GSP source images: `/root/autodl-tmp/road_damage_exp/generated_candidates/images_D40_gsp`

## Screening Outputs

- Four-class structure summary: `/root/autodl-tmp/road_damage_exp/screening/four_class_structure_candidates/four_class_structure_summary.csv`
- D40 GSP structure results: `/root/autodl-tmp/road_damage_exp/screening/d40_images_D40_gsp_structure/structure_filter_results_D40_images_D40_gsp.csv`
- Domain consistency output root: `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter`
- Domain filter results: `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter/domain_filter_results.csv`
- Domain filter summary: `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter/domain_filter_summary.txt`
- Domain filter per-class summary: `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter/domain_filter_summary_by_class.csv`
- Domain visualization directory, not uploaded: `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter/vis`
- Ready flag: `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter/ready_for_diversity_filter.txt`

## Weights

- YOLOv11 structure screening best.pt: `/root/autodl-tmp/road_damage_exp/runs/detect/road_damage_yolov11_aug/road_damage_base80_yolov11/weights/best.pt`

Do not upload full datasets, generated image directories, full training runs, wandb caches, or non-final weights.


## Ours-200 Final Selection

Updated: 2026-06-20 03:38:56 Asia/Shanghai

- Ours-200 output root: `/root/autodl-tmp/road_damage_exp/screening/final_ours_200`
- Ours-200 images, not uploaded: `/root/autodl-tmp/road_damage_exp/screening/final_ours_200/images`
- Ours-200 labels, not uploaded: `/root/autodl-tmp/road_damage_exp/screening/final_ours_200/labels`
- Ours-200 metadata: `/root/autodl-tmp/road_damage_exp/screening/final_ours_200/metadata_ours_200.csv`
- Ours-200 summary: `/root/autodl-tmp/road_damage_exp/screening/final_ours_200/ours_200_summary.txt`
- Ours-200 visualization directory, not uploaded: `/root/autodl-tmp/road_damage_exp/screening/final_ours_200/vis`
- Ours-200 ready flag: `/root/autodl-tmp/road_damage_exp/screening/final_ours_200/ready_for_yolo_dataset.txt`

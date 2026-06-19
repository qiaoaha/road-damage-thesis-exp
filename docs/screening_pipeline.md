# Screening Pipeline

Updated: 2026-06-20 03:23:28 Asia/Shanghai

## Stage 1: Structure Correctness Filtering

- Detector: YOLOv11 best.pt from `/root/autodl-tmp/road_damage_exp/runs/detect/road_damage_yolov11_aug/road_damage_base80_yolov11/weights/best.pt`.
- Structure candidates were organized into `/root/autodl-tmp/road_damage_exp/screening/four_class_structure_candidates`.
- Current counts: D00=167, D10=65, D20=114, D40=58.
- D40 required extra GSP generation. The D40 GSP structure result CSV is tracked under `results/screening/d40_images_D40_gsp_structure/`.

## Stage 2: Road-Surface Domain Consistency

- Script: `scripts/screening/domain_consistency_filter.py`.
- Real road prototype source: `/root/autodl-tmp/road_damage_exp/processed/real_yolo/images/train` and `/root/autodl-tmp/road_damage_exp/processed/real_yolo/labels/train`.
- Disease boxes are masked using YOLO labels; features are extracted from background road regions only.
- Feature groups: Lab mean/std, HSV V mean/std, LBP histogram, GLCM contrast/homogeneity/energy, Laplacian variance.
- Output root: `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter`.
- Retention rules: D00 top 134, D10 all 65, D20 top 91, D40 all 58.
- `ready_for_diversity_filter = True`.

## Stage 3: LPIPS Perceptual Diversity Selection

Pending. The next step should select final Ours-200 samples from the domain-consistent pool, targeting 50 images per class.

## Baselines

Pending.

- Random-200: class-balanced random selection baseline.
- LPIPS-200: diversity-only selection baseline.
- Ours-200: structure + domain consistency + LPIPS diversity selection.


## Stage 3 Completed: LPIPS Perceptual Diversity Selection

Updated: 2026-06-20 03:38:56 Asia/Shanghai

- Script: `scripts/screening/select_ours_200_lpips_diversity.py`.
- Input: `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter`.
- Domain CSV: `/root/autodl-tmp/road_damage_exp/screening/domain_consistency_filter/domain_filter_results.csv`.
- Output: `/root/autodl-tmp/road_damage_exp/screening/final_ours_200`.
- Feature backbone: `lpips_alex`.
- Method: Greedy Farthest Point Sampling with Quality Score.
- Final counts: D00=50, D10=50, D20=50, D40=50.
- Validation: 200 images, 200 labels, pair matched, label class IDs passed.
- `ready_for_yolo_dataset = True`.

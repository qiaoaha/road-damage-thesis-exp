# Chapter 4 Cross-Domain YOLOv11 Results

- Status source: `/root/autodl-tmp/road_damage_exp/runs_ch4_cross_domain_yolov11/ch4_cross_domain_yolov11_status.jsonl`
- Dataset summary: `/root/autodl-tmp/road_damage_exp/datasets_cross_domain/cross_domain_dataset_summary.csv`
- Results CSV: `/root/autodl-tmp/road_damage_exp/runs_ch4_cross_domain_yolov11/ch4_cross_domain_yolov11_results.csv`
- Model family: YOLOv11 only
- Source train domain: China
- Target test domains: Japan, Norway

| target_domain | dataset_variant | selector_method | precision | recall | mAP50 | mAP50-95 | status | wandb |
|---|---|---|---:|---:|---:|---:|---|---|
| Japan | base80_only | none | 0.142 | 0.0786 | 0.0451 | 0.0165 | completed | True |
| Japan | base80_plus_random_200 | random | 0.165 | 0.049 | 0.0398 | 0.0154 | completed | True |
| Japan | base80_plus_lpips_200 | lpips_diversity_only | 0.171 | 0.0929 | 0.0575 | 0.0227 | completed | True |
| Japan | base80_plus_ours_200 | structure_domain_lpips | 0.167 | 0.0696 | 0.0464 | 0.0175 | completed | True |
| Norway | base80_only | none | 0.0228 | 0.0752 | 0.00812 | 0.00272 | completed | True |
| Norway | base80_plus_random_200 | random | 0.0976 | 0.0268 | 0.0118 | 0.0052 | completed | True |
| Norway | base80_plus_lpips_200 | lpips_diversity_only | 0.0232 | 0.0463 | 0.00631 | 0.00296 | completed | True |
| Norway | base80_plus_ours_200 | structure_domain_lpips | 0.0375 | 0.059 | 0.0112 | 0.00465 | completed | True |

## Notes

Only the four YOLOv11 `best.pt` checkpoints from the China-domain Chapter 4 main experiment are evaluated here. YOLOv5 and YOLOv8 are intentionally excluded from cross-domain validation.

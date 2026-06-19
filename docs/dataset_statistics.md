# Dataset Statistics

Updated: 2026-06-20 03:23:28 Asia/Shanghai

## Four-Class Structure Candidates

```csv
class,class_name,image_count,label_count,pair_match,source_dirs,ready_for_domain_filter
D00,D00_longitudinal_crack,167,167,True,/root/autodl-tmp/road_damage_exp/screening/sd15_candidates_1000_structure_adaptive,True
D10,D10_transverse_crack,65,65,True,/root/autodl-tmp/road_damage_exp/screening/sd15_candidates_1000_structure_adaptive,True
D20,D20_alligator_crack,114,114,True,/root/autodl-tmp/road_damage_exp/screening/sd15_candidates_1000_structure_adaptive,True
D40,D40_pothole,58,58,True,/root/autodl-tmp/road_damage_exp/screening/d40_merged_structure_candidates,True
```

## Domain Consistency Filter Output

```csv
class,input_count,kept_count,output_image_count,output_label_count,pair_match,domain_score_mean,domain_score_min,domain_score_max
D00,167,134,134,134,True,0.7672027945518494,0.6718384027481079,0.8285989165306091
D10,65,65,65,65,True,0.7898675203323364,0.7138465046882629,0.861896276473999
D20,114,91,91,91,True,0.7718469500541687,0.6390396952629089,0.8528225421905518
D40,58,58,58,58,True,0.7393344640731812,0.3949102163314819,0.938222110271454
```

## D40 GSP Structure Screening

- Total images: 1581

- Structure pass: 47

```csv
reason,count
no_detection,724
class_mismatch,696
low_conf,114
pass,47
```

## Notes

- Full image and label directories are not tracked by GitHub. Only summaries and metadata are tracked.


## Final Ours-200

Updated: 2026-06-20 03:38:56 Asia/Shanghai

```csv
class_name,selected_count
D00,50
D10,50
D20,50
D40,50
```

- Total images: 200
- Total labels: 200
- Pair match: True
- Label class check: passed
- Feature backbone: `lpips_alex`
- Purpose: final Chapter 4 Ours-200 augmentation set after structure filtering, domain consistency filtering, and LPIPS diversity selection.

## Random-200 Baseline

Updated: 2026-06-20 03:46:30 Asia/Shanghai

```csv
class_name,selected_count
D00,50
D10,50
D20,50
D40,50
```

- Total images: 200
- Total labels: 200
- Pair match: True
- Label class check: passed
- Seed: 42

## LPIPS-200 Baseline

Updated: 2026-06-20 03:46:30 Asia/Shanghai

```csv
class_name,selected_count
D00,50
D10,50
D20,50
D40,50
```

- Total images: 200
- Total labels: 200
- Pair match: True
- Label class check: passed
- Feature backbone: `lpips_alex`

## Chapter 4 YOLO Training Datasets

Updated: 2026-06-20 03:54:40 Asia/Shanghai

```csv
dataset_name,split,image_count,label_count,bbox_count,D00_bbox,D10_bbox,D20_bbox,D40_bbox,pair_match,empty_real_label_count,empty_aug_label_count,data_yaml_path
real_plus_random_200,train,1583,1583,3493,1945,830,507,211,True,30,0,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_random_200/data.yaml
real_plus_random_200,val,296,296,718,401,161,118,38,True,5,0,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_random_200/data.yaml
real_plus_random_200,test,298,298,720,422,177,79,42,True,8,0,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_random_200/data.yaml
real_plus_lpips_200,train,1583,1583,3508,1956,825,513,214,True,30,0,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_lpips_200/data.yaml
real_plus_lpips_200,val,296,296,718,401,161,118,38,True,5,0,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_lpips_200/data.yaml
real_plus_lpips_200,test,298,298,720,422,177,79,42,True,8,0,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_lpips_200/data.yaml
real_plus_ours_200,train,1583,1583,3493,1942,823,514,214,True,30,0,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_ours_200/data.yaml
real_plus_ours_200,val,296,296,718,401,161,118,38,True,5,0,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_ours_200/data.yaml
real_plus_ours_200,test,298,298,720,422,177,79,42,True,8,0,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4/real_plus_ours_200/data.yaml
```

Note: empty real labels are retained as YOLO negative/no-object samples to preserve the original train/val/test splits. Generated augmentation labels have `empty_aug_label_count = 0`.

## Correction: Chapter 4 Base80 YOLO Datasets

Updated: 2026-06-20 04:03:07 Asia/Shanghai

Old full-real-plus-generated datasets are abandoned for the Chapter 4 main experiment. Correct datasets use 80 real base images plus 200 generated images.

Base80 target class counts are read from `base80_upload.tar.gz` class directories: D00=20, D10=20, D20=20, D40=20. Labels are matched from `processed/base80_yolo`.

```csv
dataset_name,split,image_count,label_count,empty_label_count,bbox_count,D00_bbox,D10_bbox,D20_bbox,D40_bbox,base80_image_count,generated_image_count,pair_match,data_yaml_path,ready_for_training
base80_plus_random_200,train,280,280,0,420,137,105,94,84,80,200,True,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80/base80_plus_random_200/data.yaml,True
base80_plus_random_200,val,296,296,5,718,401,161,118,38,0,0,True,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80/base80_plus_random_200/data.yaml,True
base80_plus_random_200,test,298,298,8,720,422,177,79,42,0,0,True,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80/base80_plus_random_200/data.yaml,True
base80_plus_lpips_200,train,280,280,0,435,148,100,100,87,80,200,True,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80/base80_plus_lpips_200/data.yaml,True
base80_plus_lpips_200,val,296,296,5,718,401,161,118,38,0,0,True,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80/base80_plus_lpips_200/data.yaml,True
base80_plus_lpips_200,test,298,298,8,720,422,177,79,42,0,0,True,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80/base80_plus_lpips_200/data.yaml,True
base80_plus_ours_200,train,280,280,0,420,134,98,101,87,80,200,True,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80/base80_plus_ours_200/data.yaml,True
base80_plus_ours_200,val,296,296,5,718,401,161,118,38,0,0,True,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80/base80_plus_ours_200/data.yaml,True
base80_plus_ours_200,test,298,298,8,720,422,177,79,42,0,0,True,/root/autodl-tmp/road_damage_exp/datasets_yolo_ch4_base80/base80_plus_ours_200/data.yaml,True
```

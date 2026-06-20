# Chapter 4 YOLO all-model results

Output root: `/root/autodl-tmp/road_damage_exp/runs_ch4_base80_wandb`

|model|dataset|status|batch|best.pt|mAP50|mAP50-95|
|---|---|---:|---:|---|---:|---:|
|yolov5|base80_only|skipped_existing|16|True|0.29719|0.11223|
|yolov5|base80_plus_random_200|skipped_existing|16|True|0.5627|0.24361|
|yolov5|base80_plus_lpips_200|skipped_existing|16|True|0.51634|0.2238|
|yolov5|base80_plus_ours_200|skipped_existing|16|True|0.39651|0.16149|
|yolov8|base80_only|completed|16|True|0.49467|0.24397|
|yolov8|base80_plus_random_200|completed|16|True|0.56795|0.28017|
|yolov8|base80_plus_lpips_200|completed|16|True|0.54925|0.27843|
|yolov8|base80_plus_ours_200|completed|16|True|0.54694|0.27176|
|yolov11|base80_only|completed|16|True|0.50405|0.25545|
|yolov11|base80_plus_random_200|completed|16|True|0.54119|0.25785|
|yolov11|base80_plus_lpips_200|completed|16|True|0.52708|0.26393|
|yolov11|base80_plus_ours_200|completed|16|True|0.53334|0.24431|

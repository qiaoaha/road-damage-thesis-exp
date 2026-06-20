# Cross-Domain Sanity Check Report

## Label Sanity

| domain | images | labels | empty | boxes | c0 | c1 | c2 | c3 | illegal_class | coord_oob | nonpositive_wh | area_mean | boxes_per_image_mean | pair_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Japan | 500 | 500 | 0 | 1308 | 306 | 372 | 402 | 228 | 0 | 0 | 0 | 0.046572493858246175 | 2.616 | True |
| Norway | 500 | 500 | 0 | 2876 | 1931 | 438 | 258 | 249 | 0 | 0 | 0 | 0.006974578982947844 | 5.752 | True |

## Visualization Artifacts

- Japan GT contact sheet: `/root/autodl-tmp/road_damage_exp/runs_ch4_cross_domain_yolov11/sanity_check/gt_vis/japan_gt_contact_sheet.jpg`
- Norway GT contact sheet: `/root/autodl-tmp/road_damage_exp/runs_ch4_cross_domain_yolov11/sanity_check/gt_vis/norway_gt_contact_sheet.jpg`
- Japan Ours prediction sheet: `/root/autodl-tmp/road_damage_exp/runs_ch4_cross_domain_yolov11/sanity_check/pred_vis/japan_ours_pred_contact_sheet.jpg`
- Norway Ours prediction sheet: `/root/autodl-tmp/road_damage_exp/runs_ch4_cross_domain_yolov11/sanity_check/pred_vis/norway_ours_pred_contact_sheet.jpg`
- Japan GT + Pred compare: `/root/autodl-tmp/road_damage_exp/runs_ch4_cross_domain_yolov11/sanity_check/pred_vis/japan_gt_pred_compare.jpg`
- Norway GT + Pred compare: `/root/autodl-tmp/road_damage_exp/runs_ch4_cross_domain_yolov11/sanity_check/pred_vis/norway_gt_pred_compare.jpg`

## Ours Prediction Distribution

| domain | pred_boxes conf>=0.001 | conf>=0.05 | conf>=0.25 | per_image_mean | conf_mean | conf_median | conf_max | c0 | c1 | c2 | c3 | top_class_ratio | almost_no_predictions | severe_class_bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Japan | 10636 | 676 | 172 | 21.272 | 0.01786248748439906 | 0.0026133707724511623 | 0.9054341912269592 | 1348 | 1381 | 6530 | 1377 | 0.6139526137645731 | False | False |
| Norway | 5104 | 321 | 88 | 10.208 | 0.01829310426939035 | 0.002403815626166761 | 0.887553334236145 | 2080 | 1154 | 924 | 946 | 0.40752351097178685 | False | False |

## Full-Real YOLOv11 Sanity Val

| domain | P | R | mAP50 | mAP50-95 | status |
|---|---:|---:|---:|---:|---|
| Japan | 0.196 | 0.0977 | 0.0587 | 0.0238 | completed |
| Norway | 0.176 | 0.0311 | 0.013 | 0.00519 | completed |

## Judgement

1. Japan / Norway labels normal: `True`.
2. Bbox values normal: `True`.
3. Class mapping suspicious: `False`.
4. GT visualization files were generated successfully; manual inspection should use the contact sheets listed above.
5. Ours almost no predictions at conf>=0.001: `False`. However, high-confidence predictions are sparse: Japan conf>=0.25 has 172 boxes, Norway conf>=0.25 has 88 boxes.
6. Full-real YOLOv11 cross-domain results also very low: `True`.
7. Most likely current cause: `D. ??????????????? + ???????????????????????????`.

Note: This is a sanity check only. No model was retrained and no experiment conclusion was written here.

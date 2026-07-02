# GenScreening GPU Real Smoke Test Report

日期：2026-07-03  
工具包路径：`/root/autodl-tmp/road_damage_exp/tools/gen_screening/`  
输出目录：`/root/autodl-tmp/road_damage_exp/reports/gen_screening_real_smoke_test/`

## 1. 测试范围

本次只做小规模真实评分验证，不跑全量、不训练、不覆盖正式输出。

小样本输入：

- real：每类 3 张，共 12 张
- generated：每类 3 张，共 12 张
- selected：8 张

小样本配置：

```text
/root/autodl-tmp/road_damage_exp/reports/gen_screening_real_smoke_test/small_real_smoke.yaml
```

## 2. 执行命令

```bash
cd /root/autodl-tmp/road_damage_exp/tools/gen_screening
/root/miniconda3/envs/yolo5090/bin/python -m compileall -q genscreen scripts
/root/miniconda3/bin/python -m genscreen.scripts.run_pipeline --config configs/road_damage_japan.yaml --dry-run
/root/miniconda3/envs/yolo5090/bin/python -m genscreen.scripts.run_pipeline --config /root/autodl-tmp/road_damage_exp/reports/gen_screening_real_smoke_test/small_real_smoke.yaml --step all
```

## 3. 环境

- GPU 环境：`/root/miniconda3/envs/yolo5090/bin/python`
- Torch：`2.11.0+cu128`
- CUDA：可用，`torch.cuda.is_available() = True`
- Transformers：可用
- Ultralytics：`8.4.67`

## 4. 关键结果

| 问题 | 结论 | 证据 |
|---|---|---|
| 1. DINOv2 是否真实加载成功？ | 是 | 日志显示 `DINO scores written from real DINOv2 features`；使用 `/root/autodl-tmp/models/dinov2/dinov2_vits14_pretrain.pth` |
| 2. DINOv2 feature npz 是否生成？ | 是 | `cache/real_dino_features.npz` 和 `cache/generated_dino_features.npz` 均存在 |
| 3. `S_DINO` 是否不是 dummy？ | 是 | `dino_scores.csv` 有 12 行，首行 `S_DINO=0.709007`、`dino_topk_used=3`，来自真实 DINO feature TopK |
| 4. CLIP 是否真实加载成功？ | 是 | 日志显示 `PCS scores written from CLIP features`；使用 `/root/autodl-tmp/models/clip/ViT-B-16.pt` |
| 5. PCS 是否生成 `S_IT`、`S_stable`、`S_PCS`？ | 是 | `pcs_scores.csv` 有 12 行，首行 `S_IT=0.646545`、`S_stable=0.994749`、`S_PCS=0.785827` |
| 6. Teacher 评分是读取 CSV 还是 YOLO adapter 推理？ | 读取 CSV | 使用映射后的 `teacher_predictions_smoke.csv`，来源为已有 Japan teacher predictions；未跑 D-FINE/YOLO 推理 |
| 7. MMR 是否使用真实 DINO 特征？ | 是 | `selected_gen200.csv` 有 8 行，其中 4 行 `max_similarity_to_selected > 0`，相似度来自 `generated_dino_features.npz` |
| 8. 小规模 `selected_gen200.csv` 是否生成？ | 是 | `scores/selected_gen200.csv` 存在，8 行 |
| 9. 哪一步失败？ | 首次 CLIP 兼容失败，已修复并重跑通过 | 首次导入 SDS CLIP 变体导致 `CLIP.encode_image() missing H and W`；修正为优先 OpenAI CLIP 后通过 |

## 5. 生成文件

```text
cache/real_dino_features.npz
cache/generated_dino_features.npz
cache/pcs_features.npz
scores/dino_scores.csv
scores/pcs_scores.csv
scores/teacher_scores.csv
scores/all_quality_scores.csv
scores/selected_gen200.csv
reports/00_final_report.md
reports/score_summary_by_class.csv
reports/selected_summary_by_class.csv
reports/top10_quality.csv
reports/bottom10_quality.csv
reports/selected_top_by_class.csv
selected/images/
selected/labels/
```

## 6. 输出形状与样例

DINO feature：

```text
real_dino_features.feature shape = (12, 384)
generated_dino_features.feature shape = (12, 384)
```

PCS feature：

```text
pcs_features.image_feature shape = (12, 512)
pcs_features.text_feature shape = (12, 512)
```

`dino_scores.csv` 首行：

```text
S_DINO=0.709007
dino_topk_mean=0.709007
dino_topk_used=3
```

`pcs_scores.csv` 首行：

```text
S_IT=0.646545
S_stable=0.994749
S_PCS=0.785827
num_augmentations=2
```

`selected_gen200.csv` MMR 核验：

```text
selected_rows = 8
nonzero_similarity_rows = 4
similarities = ['0.0', '0.0', '0.0', '0.0', '0.873632', '0.675762', '0.665622', '0.91683']
```

## 7. 注意事项

1. 本次没有跑全量 1000/1500 张。
2. 本次没有启动任何训练。
3. D-FINE adapter 仍未实现；本次按要求优先使用已有 teacher prediction CSV。
4. `configs/road_damage_japan.yaml --dry-run` 按用户指定执行过；该配置自身输出目录是 `/root/autodl-tmp/road_damage_exp/reports/gen_screening/japan`。
5. 真实 smoke 输出使用独立目录 `/root/autodl-tmp/road_damage_exp/reports/gen_screening_real_smoke_test/`。

## 8. 总结

GPU 小规模真实评分验证通过。DINOv2 真实特征、TopK `S_DINO`、CLIP/PCS、Teacher CSV scoring、Q(x)、MMR 真实 DINO similarity、selected 输出和 final report 均已在 12 张 real + 12 张 generated 的小样本上完成验证。

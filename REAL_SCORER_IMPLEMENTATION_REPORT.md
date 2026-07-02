# Real Scorer Implementation Report

日期：2026-07-03

## 1. 已替换的 dummy/半实现模块

| 模块 | 当前状态 |
|---|---|
| `genscreen/features.py` | 新增 DINOv2/CLIP 本地模型加载、device 选择、批量图像/文本编码、L2 normalize、弱扰动、`.npz` 特征缓存工具 |
| `genscreen/dino_score.py` | 新增真实 DINOv2 特征提取、`real_dino_features.npz` / `generated_dino_features.npz` 缓存、同类 TopK 余弦相似度评分 |
| `genscreen/pcs_score.py` | 新增 CLIP 图文一致性、弱扰动稳定性、`pcs_features.npz` 缓存 |
| `genscreen/teacher_score.py` | 保留已有 CSV 模式，新增 ultralytics YOLO 推理 adapter，新增 class_probs margin 优先路径 |
| `genscreen/mmr.py` | 新增从 `generated_dino_features.npz` 读取真实 DINO feature similarity |
| `genscreen/selector.py` | `selected_gen200.csv` 新增 `max_similarity_to_selected` 字段 |
| `genscreen/scripts/run_pipeline.py` | `--step dino/pcs/teacher/features/all` 现在可进入真实 scorer 路径；`dry-run` 和 `cpu-only-check` 仍使用 fallback |

## 2. 保留的 fallback

未删除 dry-run dummy fallback：

- `dummy_dino_scores()`
- `dummy_pcs_scores()`
- `dummy_teacher_scores()`
- MMR dry-run dummy similarity

正式运行下，如果 MMR 开启但缺少 `generated_dino_features.npz`，会报错；dry-run 下允许 dummy similarity。

## 3. DINOv2 是否真实可运行

代码层面已实现真实 DINOv2 路径：

1. 从 `models.dinov2.model_dir` 读取本地模型。
2. 使用 `transformers.AutoImageProcessor` 和 `AutoModel`，`local_files_only=True`。
3. 支持 `device=auto/cpu/cuda`。
4. 输出：
   - `cache/real_dino_features.npz`
   - `cache/generated_dino_features.npz`
   - `scores/dino_scores.csv`

本机实测状态：未执行真实 DINOv2 推理。原因是当前本机 `torch 2.12.1+cpu`，`cuda_available=False`，且未提供本地 DINOv2 模型目录。

## 4. PCS 是否真实可运行

代码层面已实现真实 PCS/CLIP 路径：

1. 从 `models.clip.model_dir` 读取本地 CLIP 模型。
2. prompt 从 `classes.prompts` 读取。
3. 执行 brightness、contrast、小 resize/crop、light blur 弱扰动。
4. 计算：
   - `S_IT`
   - `S_stable`
   - `S_PCS`
5. 输出：
   - `cache/pcs_features.npz`
   - `scores/pcs_scores.csv`

本机实测状态：未执行真实 CLIP 推理。原因是当前本机为 CPU-only 且未提供本地 CLIP 模型目录。

## 5. Teacher adapter 支持类型

| 类型 | 状态 | 说明 |
|---|---|---|
| `existing_predictions` CSV | 已支持并已测试 | 可读取 `image_path,class_id,conf,x1,y1,x2,y2`，也支持 `prob_0,prob_1,...` |
| `type: yolo` | 已实现，未实测权重 | 使用 `ultralytics.YOLO`，输出 `cache/teacher_predictions.csv` |
| `type: dfine` | 接口保留，清晰报错 | 当前要求配置 `models.teacher.existing_predictions` |

Teacher scoring 输出字段：

```text
image_path,class_id,S_conf,S_margin,p_max_same,p_mean_same,p_max_wrong,num_same,num_wrong,num_all,margin_source
```

## 6. MMR 是否使用真实 DINO 特征

已实现。

当 `mmr.enabled=true` 时，`selector.py` 会通过 `mmr.load_similarity_features()` 读取：

```text
output_dir/cache/generated_dino_features.npz
```

相似度公式：

```text
Sim(x_i,x_j)=((z_i^T z_j)+1)/2
```

如果正式运行缺少 DINO 特征，会报错；dry-run 或 `mmr.allow_dummy_similarity=true` 才允许 dummy similarity。

## 7. 不开 GPU 能跑哪些步骤

不开 GPU 或没有模型时可跑：

- `--step index`
- `--cpu-only-check`
- `--dry-run`
- 使用已有 `teacher_predictions.csv` 的 teacher scoring
- disabled DINO/PCS 后的 quality/select/report
- 使用已有 score CSV 的 quality/select/report

不开 GPU 不能真实跑：

- DINOv2 feature extraction
- CLIP/PCS feature extraction
- YOLO teacher inference 如果权重/依赖/设备不可用

## 8. 开 GPU 后全量命令

先在 YAML 中配置真实路径：

```yaml
models:
  dinov2:
    model_dir: /abs/path/to/dinov2
    device: auto
  clip:
    model_dir: /abs/path/to/clip
    device: auto
  teacher:
    type: yolo
    weights: /abs/path/to/yolo_teacher.pt
    device: auto
```

全量运行：

```bash
python -m genscreen.scripts.run_pipeline --config configs/road_damage_japan.yaml --step all
```

分阶段运行：

```bash
python -m genscreen.scripts.run_pipeline --config configs/road_damage_japan.yaml --step index
python -m genscreen.scripts.run_pipeline --config configs/road_damage_japan.yaml --step dino
python -m genscreen.scripts.run_pipeline --config configs/road_damage_japan.yaml --step pcs
python -m genscreen.scripts.run_pipeline --config configs/road_damage_japan.yaml --step teacher
python -m genscreen.scripts.run_pipeline --config configs/road_damage_japan.yaml --step quality
python -m genscreen.scripts.run_pipeline --config configs/road_damage_japan.yaml --step select
python -m genscreen.scripts.run_pipeline --config configs/road_damage_japan.yaml --step report
```

## 9. 已通过测试

```powershell
python -m compileall -q genscreen scripts
```

结果：`COMPILEALL_PASS`

```powershell
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --cpu-only-check
```

结果：`CPU_ONLY_PASS`

```powershell
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --dry-run
```

结果：`DRY_RUN_PASS`

小规模真实图片 + 已有 teacher prediction CSV：

```powershell
python -m genscreen.scripts.run_pipeline --config $env:TEMP\genscreen_small_real_test\small.yaml --step all
```

结果：`SMALL_REAL_SCORER_PASS`

验证到的真实链路：

- YOLO 索引
- 主类别识别
- teacher CSV 读取
- class_probs margin
- disabled score 权重重分配
- selected images/labels 复制
- final report 输出

## 10. 仍需用户提供的模型/缓存

全量真实 scorer 需要：

1. 本地 DINOv2 模型目录：`models.dinov2.model_dir`
2. 本地 CLIP 模型目录：`models.clip.model_dir`
3. YOLO teacher 权重：`models.teacher.weights`
4. 或已有 teacher CSV：`models.teacher.existing_predictions`
5. 真实 YOLO 数据集路径和 generated candidate 路径

## 11. 结论

本轮已把核心 scorer 从纯 dummy/半实现推进为真实可运行代码路径，并保留 dry-run fallback。由于当前本机缺 GPU 和本地模型目录，真实 DINOv2/PCS/YOLO 推理尚未实测；但已有 teacher CSV 的真实评分、Q(x)、筛选、复制和报告链路已通过小样本验证。

# GenScreening 完整性检查报告

检查时间：2026-07-03  
补全复查时间：2026-07-03  
本地路径：`E:\codex_project\mid_term\tools\gen_screening`  
远端路径：`/root/autodl-tmp/road_damage_exp/tools/gen_screening/`  
当前本地最新 commit：`b9b580e Add GitHub publishing helpers`  
说明：用户提到的 `a64679c Add config-driven gen screening toolkit` 是核心工具包初始提交；当前 HEAD 另包含 GitHub 发布辅助文件，不影响工具逻辑。

## 0. 代码补全后更新结论

本轮已将原先的 dummy/半实现评分模块补为“真实实现 + dry-run fallback”结构：

- DINOv2：已接入 `transformers.AutoImageProcessor` / `AutoModel` 的本地模型加载、特征提取、L2 normalize、`.npz` 缓存、同类真实图 TopK 余弦相似度评分。
- PCS/CLIP：已接入 `transformers.CLIPProcessor` / `CLIPModel` 的本地模型加载、prompt 图文相似度、弱扰动稳定性、`pcs_features.npz` 缓存。
- Teacher：已保留已有 prediction CSV 模式，并新增 ultralytics YOLO adapter；D-FINE 明确报错并要求配置 `existing_predictions`。
- Margin：已支持 `prob_0,prob_1,...` 或 JSON class_probs 优先路径；无 class_probs 时使用 fallback margin。
- MMR：已优先读取 `cache/generated_dino_features.npz` 的真实 DINO 特征计算 similarity；dry-run 可继续用 dummy similarity。

当前仍需用户提供本地模型/缓存才能做全量真实评分：

- `models.dinov2.model_dir`
- `models.clip.model_dir`
- `models.teacher.weights` 或 `models.teacher.existing_predictions`

本机验证环境为 CPU-only：`torch 2.12.1+cpu`，`cuda_available=False`。因此本轮没有执行真实 DINOv2/PCS/YOLO GPU 推理，只验证了编译、cpu-only、dry-run，以及“小规模真实图片 + 已有 teacher prediction CSV”的真实 scorer 链路。

## 1. 文件完整性表

| 项目 | 本地状态 | 远端状态 | 结论 |
|---|---:|---:|---|
| `configs/` | 存在 | 存在 | 通过 |
| `genscreen/` | 存在 | 存在 | 通过 |
| `scripts/` | 存在 | 存在 | 通过 |
| `README.md` | 存在 | 存在 | 通过 |
| `requirements.txt` | 存在 | 存在 | 通过 |
| `run_genscreen.sh` | 存在 | 存在 | 通过 |

## 2. 核心模块完整性表

| 模块 | 状态 | 说明 |
|---|---:|---|
| `genscreen/config.py` | 存在 | 支持 YAML 读取、default.yaml 深度合并、点路径读取 |
| `genscreen/dataset.py` | 存在 | 支持 YOLO 标签读取、图片索引、主类别判断 |
| `genscreen/features.py` | 存在 | 已实现 DINOv2/CLIP 模型加载、device 选择、批量编码、弱扰动、特征缓存工具 |
| `genscreen/dino_score.py` | 存在 | 已实现真实 DINOv2 特征缓存与同类 TopK 域一致性评分；保留 dry-run dummy |
| `genscreen/pcs_score.py` | 存在 | 已实现 CLIP 图文一致性和弱扰动稳定性评分；保留 dry-run dummy |
| `genscreen/teacher_score.py` | 存在 | 支持已有 CSV、class_probs margin、ultralytics YOLO adapter；D-FINE 明确报错 |
| `genscreen/quality_score.py` | 存在 | 支持类别内 min-max、disabled score 权重重分配、Q(x) 合成 |
| `genscreen/mmr.py` | 存在 | 优先使用 `generated_dino_features.npz` 真实 DINO 特征；dry-run 可 dummy similarity |
| `genscreen/selector.py` | 存在 | 支持类别配额、配额不足重分配、selected images/labels 复制 |
| `genscreen/report.py` | 存在 | 支持 final report 和 summary CSV 输出 |
| `genscreen/io_utils.py` | 存在 | 支持日志、CSV、图片枚举、安全复制 |
| `genscreen/visualization.py` | 存在 | 空占位模块，未实现可视化 |

## 3. 命令入口完整性表

| 入口 | 状态 | 说明 |
|---|---:|---|
| `scripts/check_dataset.py` | 存在 | 包装 `genscreen.scripts.run_pipeline.main`，非独立专用逻辑 |
| `scripts/extract_features.py` | 存在 | 包装主入口，未实现独立特征提取 CLI |
| `scripts/compute_scores.py` | 存在 | 包装主入口，未实现独立评分 CLI |
| `scripts/select_samples.py` | 存在 | 包装主入口 |
| `scripts/make_report.py` | 存在 | 包装主入口 |
| `scripts/run_pipeline.py` | 存在 | 包装主入口 |
| `genscreen/scripts/run_pipeline.py` | 存在 | 实际 argparse 总入口 |

## 4. 功能完整性表

| 功能 | 状态 | 证据/说明 |
|---|---:|---|
| YOLO 数据集索引 | 已实现 | `dataset.py` 读取图片、匹配 `.txt` 标签，输出 `real_index.csv` / `generated_index.csv` |
| 主类别识别 | 已实现 | 单类直接取该类，多类取最大 bbox 面积类别；无标签为 `unknown` |
| DINOv2 域一致性评分 | 已实现，未实测模型 | 已接本地 DINOv2 模型、`.npz` 缓存、同类 TopK、真实余弦相似度；本机缺模型/GPU未跑真实推理 |
| PCS 图文语义一致性评分 | 已实现，未实测模型 | 已接本地 CLIP、prompt、弱扰动、S_IT/S_stable/S_PCS；本机缺模型/GPU未跑真实推理 |
| Teacher Confidence 评分 | 已实现 | 已有 CSV 模式和 YOLO adapter；CSV scorer 已用小样本通过 |
| Margin 类别判别评分 | 已实现 | 已支持 class_probs 优先路径和 fallback margin |
| 类别内 min-max 归一化 | 已实现 | `normalize_by_class()` 按 `class_id` 分组，常数类默认 0.5 |
| 综合质量评分 Q(x) | 已实现 | `build_quality_scores()` 合并 norm 分数，按配置权重计算 |
| disabled score 自动重分配权重 | 已实现 | `active_weights()` 仅保留 enabled score 并归一化 |
| 类别配额筛选 | 已实现 | `quota_map()` 和 `select_samples()` 支持固定配额与不足重分配 |
| MMR 多样性重排序 | 已实现，未实测 DINO 特征 | 优先使用真实 generated DINO feature similarity；无特征时正式运行报错，dry-run 允许 dummy |
| selected images/labels 复制 | 已实现 | `safe_copy()` 避免覆盖；dry-run 时不复制 |
| score CSV 输出 | 已实现 | 输出 `dino_scores.csv`、`pcs_scores.csv`、`teacher_scores.csv`、`all_quality_scores.csv`、`selected_gen200.csv` |
| final report 输出 | 已实现 | 输出 `reports/00_final_report.md` 与多份 summary CSV |

## 5. 配置与通用性检查

| 要求 | 状态 | 说明 |
|---|---:|---|
| YAML 配置驱动 | 已实现 | 路径、类别、权重、配额、开关从 YAML 读取 |
| 不写死 China/Japan/Czech | 基本满足 | 核心代码未写死；但示例配置含 `road_damage_china.yaml` / `road_damage_japan.yaml` |
| 不写死 D00/D10/D20/D40 | 基本满足 | 核心代码未写死；`default.yaml` 默认示例包含这四类，符合原需求允许默认给出 |
| dry-run | 已实现 | `--dry-run` 限制每类样本数，并使用 dummy scores |
| cpu-only-check | 已实现基础模式 | CLI 接收参数并使用 placeholder，不启动模型推理 |
| cache 复用 | 部分实现 | DINO `.npz`、PCS `.npz`、teacher prediction CSV 均支持复用；更细粒度缓存校验尚未实现 |
| 无 GPU 时不启动模型推理 | 当前满足 | 因为真实模型推理尚未实现；需要在未来接入 DINO/CLIP/teacher 时保持此约束 |

## 6. 已通过测试

### 本地编译测试

命令：

```powershell
python -m compileall -q genscreen scripts
```

结果：通过，返回 `COMPILEALL_PASS`。

### 本地 dry-run 测试

命令：

```powershell
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --dry-run
```

结果：通过，返回 `DRY_RUN_PASS`。

实际日志显示：

```text
GenScreening start: step=all dry_run=True output=\abs\path\output\gen_screening_result
DINO score dry-run/cache placeholder written
PCS score dry-run/cache placeholder written
Teacher score dry-run/cache placeholder written
Quality scores written with weights={'dino': 0.3, 'pcs': 0.3, 'teacher_conf': 0.25, 'margin': 0.15}
Selection written
Report written: \abs\path\output\gen_screening_result\reports\00_final_report.md
```

注意：`configs/template.yaml` 使用占位路径，因此该 dry-run 产生 0 个候选样本；它证明 CLI 和空输入链路不崩溃，但不证明真实数据筛选效果。

### 本轮新增测试

编译：

```powershell
python -m compileall -q genscreen scripts
```

结果：`COMPILEALL_PASS`

CPU-only：

```powershell
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --cpu-only-check
```

结果：`CPU_ONLY_PASS`

Dry-run：

```powershell
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --dry-run
```

结果：`DRY_RUN_PASS`

小规模真实图片 + 已有 teacher prediction CSV：

```powershell
python -m genscreen.scripts.run_pipeline --config $env:TEMP\genscreen_small_real_test\small.yaml --step all
```

结果：`SMALL_REAL_SCORER_PASS`

该测试验证：

- YOLO 图片/标签索引
- 主类别识别
- 读取已有 teacher prediction CSV
- `prob_0/prob_1` class_probs margin 优先路径
- disabled DINO/PCS 后权重自动重分配
- selected images/labels 复制
- report 输出

### 远端编译测试

命令：

```bash
cd /root/autodl-tmp/road_damage_exp/tools/gen_screening
/root/miniconda3/bin/python -m compileall -q genscreen scripts
```

结果：通过，返回 `REMOTE_COMPILEALL_PASS`。

## 7. 未实现/半实现功能

1. DINOv2 真实实现已写入，但未在本机完成模型推理实测，因为缺少本地 DINOv2 模型路径且当前 torch 为 CPU-only。
2. PCS/CLIP 真实实现已写入，但未在本机完成模型推理实测，因为缺少本地 CLIP 模型路径且当前 torch 为 CPU-only。
3. YOLO teacher adapter 已实现，但未实测真实 YOLO 权重推理，因为未提供权重路径。
4. D-FINE adapter 仍未实现；当前会清晰报错并要求使用 `existing_predictions`。
5. `visualization.py` 仍为空占位。
6. `check_dataset.py`、`extract_features.py`、`compute_scores.py` 等脚本仍主要是主入口包装；阶段能力由 `--step` 实现。

## 8. 是否可以用于任意 YOLO 数据集

结论：可以用于任意 YOLO 数据集做索引、已有评分/teacher CSV 评分、Q(x)、配额筛选和报告生成；在提供 DINOv2/CLIP/YOLO 模型或缓存后，可进入完整真实评分流程。

限制：本机未提供真实模型路径，因此尚未验证实际 DINOv2/PCS 模型推理效果。

## 9. 是否可以用于 China/Japan 数据集

结论：可以作为配置模板和通用筛图工具用于 China/Japan 数据集；需要补齐配置里的真实数据路径、DINOv2/CLIP 模型路径、YOLO teacher 权重或 teacher prediction CSV。

已存在：

- `configs/road_damage_china.yaml`
- `configs/road_damage_japan.yaml`

仍需在配置中填对真实路径、模型路径、teacher prediction CSV 或 YOLO teacher 权重。

## 10. 全量运行还缺哪些模型或缓存

全量真实运行至少需要：

1. DINOv2 模型目录或权重路径。
2. CLIP 模型目录或权重路径。
3. Teacher detector 权重，例如 D-FINE 或 YOLO。
4. 或者已有 teacher prediction CSV，字段至少包含 `image_path,class_id,conf,x1,y1,x2,y2`。
5. 真实 real/generated YOLO 图片与标签目录。
6. 可复用 DINO/PCS 特征缓存：`real_dino_features.npz`、`generated_dino_features.npz`、`pcs_features.npz`。
7. 使用真实 DINO 特征的 MMR similarity 依赖 `generated_dino_features.npz`。

## 11. 总体结论

当前 `gen_screening` 是一个结构完整、CLI 可运行、已补入真实 scorer 代码路径的通用筛图工具包。

它已经完成：

- 配置驱动框架
- YOLO 数据索引
- 主类别识别
- 评分 CSV 合并
- 类别内归一化
- Q(x) 计算
- 类别配额
- 真实 DINO/PCS/YOLO scorer 入口和缓存
- MMR 真实 DINO similarity 入口
- selected CSV / report 输出

它仍需要用户提供本地模型、权重或缓存来完成全量真实运行；本机 CPU-only 环境未验证实际 DINOv2/PCS/YOLO 模型推理。

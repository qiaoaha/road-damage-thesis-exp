# GenScreening

GenScreening is a config-driven toolkit for selecting high-quality generated samples from any YOLO-format detection dataset. It does not hard-code countries, classes, model paths, thresholds, or output paths.

## What it does

The pipeline indexes real and generated YOLO images, computes or loads score families, normalizes scores per class, merges them into a final quality score, applies class quotas, reranks with MMR diversity, copies selected images and labels without overwriting originals, and writes CSV plus Markdown reports.

Quality score:

```text
Q(x) = 0.30 * norm_DINO + 0.30 * norm_PCS + 0.25 * norm_teacher_conf + 0.15 * norm_margin
```

If a score family is disabled, the remaining weights are renormalized to sum to 1.

## Data format

Use YOLO detection labels:

```text
class_id x_center y_center width height
```

The dataset paths are read from YAML:

- `dataset.real_images_dir`
- `dataset.real_labels_dir`
- `dataset.generated_images_dir`
- `dataset.generated_labels_dir`
- `dataset.output_dir`

Supported image suffixes are `jpg`, `jpeg`, `png`, `bmp`, and `webp`.

## Quick start

```bash
pip install -r requirements.txt
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --dry-run
```

## CPU-only check

```bash
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --cpu-only-check --dry-run
```

CPU-safe stages:

- config validation
- dataset indexing
- cached score loading
- score merge and class-wise min-max normalization
- quota selection
- MMR reranking
- report generation

GPU/model-inference stages:

- DINOv2 feature extraction
- CLIP/PCS feature extraction
- teacher detector inference

## Step commands

```bash
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --step index
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --step features
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --step dino
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --step pcs
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --step teacher
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --step quality
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --step select
python -m genscreen.scripts.run_pipeline --config configs/template.yaml --step report
```

## Reusing existing teacher predictions

Set:

```yaml
models:
  teacher:
    existing_predictions: /abs/path/teacher_predictions.csv
```

Expected CSV columns:

```text
image_path,class_id,conf,x1,y1,x2,y2,optional_class_probs
```

When class probabilities are absent, margin uses:

```text
S_margin = clip(p_max_same - p_max_wrong + 1, 0, 2) / 2
```

## Output files

The configured `dataset.output_dir` contains:

- `cache/real_index.csv`
- `cache/generated_index.csv`
- `scores/dino_scores.csv`
- `scores/pcs_scores.csv`
- `scores/teacher_scores.csv`
- `scores/all_quality_scores.csv`
- `scores/selected_gen200.csv`
- `selected/images/`
- `selected/labels/`
- `reports/00_final_report.md`
- `reports/score_summary_by_class.csv`
- `reports/selected_summary_by_class.csv`
- `reports/top10_quality.csv`
- `reports/bottom10_quality.csv`
- `reports/selected_top_by_class.csv`

Original data is never deleted or overwritten.

## China and Japan examples

Edit paths in `configs/road_damage_china.yaml` and run:

```bash
python -m genscreen.scripts.run_pipeline --config configs/road_damage_china.yaml --dry-run
```

Edit paths in `configs/road_damage_japan.yaml` and run:

```bash
python -m genscreen.scripts.run_pipeline --config configs/road_damage_japan.yaml --dry-run
```

For a new dataset, copy `configs/template.yaml`, update dataset paths, class names, prompts, quotas, models, and weights, then run the same command.

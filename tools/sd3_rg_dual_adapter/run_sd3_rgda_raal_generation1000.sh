#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

: "${D2_GENERATION_MANIFEST:?D2_GENERATION_MANIFEST is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${RAAL_CHECKPOINT:?RAAL_CHECKPOINT is required}"
: "${RAAL_CHECKPOINT_SHA256:?RAAL_CHECKPOINT_SHA256 is required}"
: "${FORMAL_CACHE_MANIFEST:?FORMAL_CACHE_MANIFEST is required}"
: "${RAAL_FORMAL_COMMIT:?RAAL_FORMAL_COMMIT is required}"

MANIFEST_DIR="${MANIFEST_DIR:-${ROOT}/manifests/generation1000_raal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/road_damage_exp/generated/sd3_rgda_raal_generation1000}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-/root/autodl-tmp/road_damage_exp/generated/sd3_rgda_raal_generation1000_smoke}"
YOLO_OUTPUT_ROOT="${YOLO_OUTPUT_ROOT:-/root/autodl-tmp/road_damage_exp/datasets/yolo_rgda_ablation}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/yolo_rgda_ablation}"
STOP_AFTER_SMOKE="${STOP_AFTER_SMOKE:-0}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${MANIFEST_DIR}"

python "${ROOT}/scripts/derive_raal_generation1000_manifest.py" \
  --d2-manifest "${D2_GENERATION_MANIFEST}" \
  --output "${MANIFEST_DIR}/generation1000_raal.csv" \
  --formal-commit "${RAAL_FORMAL_COMMIT}" \
  --checkpoint-path "${RAAL_CHECKPOINT}" \
  --checkpoint-sha256 "${RAAL_CHECKPOINT_SHA256}" \
  --dataset-root "${DATASET_ROOT}" \
  --best-flow-eval-step "${RAAL_BEST_FLOW_EVAL_STEP:-}" \
  --best-flow-eval-loss "${RAAL_BEST_FLOW_EVAL_LOSS:-}"

python "${ROOT}/scripts/generate_sd3_rgda1000.py" \
  --manifest "${MANIFEST_DIR}/generation1000_raal.csv" \
  --formal-cache-manifest "${FORMAL_CACHE_MANIFEST}" \
  --output-root "${SMOKE_OUTPUT_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --checkpoint "${RAAL_CHECKPOINT}" \
  --checkpoint-sha256 "${RAAL_CHECKPOINT_SHA256}" \
  --mode rgda \
  --smoke \
  --resume

python "${ROOT}/scripts/audit_rgda_generation1000.py" \
  --manifest "${MANIFEST_DIR}/generation1000_raal.csv" \
  --results "${SMOKE_OUTPUT_ROOT}/generation_results.csv" \
  --checkpoint-sha256 "${RAAL_CHECKPOINT_SHA256}" \
  --output "${SMOKE_OUTPUT_ROOT}/qa/generation_audit.md" \
  --mode rgda \
  --smoke || exit 1

if [[ "${STOP_AFTER_SMOKE}" == "1" ]]; then
  exit 0
fi

python "${ROOT}/scripts/generate_sd3_rgda1000.py" \
  --manifest "${MANIFEST_DIR}/generation1000_raal.csv" \
  --formal-cache-manifest "${FORMAL_CACHE_MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --checkpoint "${RAAL_CHECKPOINT}" \
  --checkpoint-sha256 "${RAAL_CHECKPOINT_SHA256}" \
  --mode rgda \
  --resume

python "${ROOT}/scripts/audit_rgda_generation1000.py" \
  --manifest "${MANIFEST_DIR}/generation1000_raal.csv" \
  --results "${OUTPUT_ROOT}/generation_results.csv" \
  --checkpoint-sha256 "${RAAL_CHECKPOINT_SHA256}" \
  --output "${OUTPUT_ROOT}/qa/generation_audit.md" \
  --mode rgda

python "${ROOT}/scripts/build_yolo_rgda_ablation_dataset.py" \
  --real-dataset-root "${DATASET_ROOT}" \
  --output-root "${YOLO_OUTPUT_ROOT}" \
  --group raal \
  --synthetic-images "${OUTPUT_ROOT}/rgda/images" \
  --synthetic-labels "${OUTPUT_ROOT}/rgda/labels"

python "${ROOT}/scripts/run_yolo_rgda_ablation.py" \
  --group raal \
  --data "${YOLO_OUTPUT_ROOT}/raal/data.yaml" \
  --config "${ROOT}/configs/yolo_czech_ablation.yaml"

python "${ROOT}/scripts/summarize_rgda_ablation.py" --result-root "${RESULT_ROOT}"

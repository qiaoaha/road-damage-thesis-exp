#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${RGDA_CHECKPOINT:?RGDA_CHECKPOINT is required}"
: "${RGDA_CHECKPOINT_SHA256:?RGDA_CHECKPOINT_SHA256 is required}"

MANIFEST_DIR="${MANIFEST_DIR:-${ROOT}/manifests/generation1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/road_damage_exp/generated/sd3_rgda_generation1000}"

python "${ROOT}/scripts/build_rgda_generation1000_manifest.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${MANIFEST_DIR}" \
  --checkpoint-sha256 "${RGDA_CHECKPOINT_SHA256}"

python "${ROOT}/scripts/generate_sd3_rgda1000.py" \
  --manifest "${MANIFEST_DIR}/generation1000.csv" \
  --output-root "${OUTPUT_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --checkpoint "${RGDA_CHECKPOINT}" \
  --checkpoint-sha256 "${RGDA_CHECKPOINT_SHA256}" \
  --mode paired \
  --resume

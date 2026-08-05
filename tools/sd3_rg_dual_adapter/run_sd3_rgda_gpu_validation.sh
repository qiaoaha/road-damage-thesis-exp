#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/road_damage_exp/models/stable-diffusion-3-medium-diffusers}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/road_damage_exp/datasets/Czech_YOLO_seed2026}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/road_damage_exp/reports/sd3_rg_dual_adapter_gpu_validation}"
MANIFEST_DIR="${ROOT}/manifests"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PIP_NO_INDEX=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${REPORT_DIR}" "${MANIFEST_DIR}"

package_on_exit() {
  status=$?
  "${PYTHON}" "${ROOT}/scripts/package_gpu_validation.py" \
    --report-dir "${REPORT_DIR}" \
    --archive "${REPORT_DIR}.tar.gz" || true
  exit "${status}"
}

trap package_on_exit EXIT

"${PYTHON}" "${ROOT}/scripts/build_czech_manifest.py" --dataset-root "${DATASET_ROOT}" --out-dir "${MANIFEST_DIR}"

if ! nvidia-smi -L | grep -q "NVIDIA GeForce RTX 5090"; then
  echo "GPU_ENV=FAIL" > "${REPORT_DIR}/07_FINAL_STATUS.md"
  echo "Expected NVIDIA GeForce RTX 5090" >> "${REPORT_DIR}/07_FINAL_STATUS.md"
  exit 2
fi

"${PYTHON}" "${ROOT}/scripts/run_real_sd3_validation.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-root "${DATASET_ROOT}" \
  --smoke-manifest "${MANIFEST_DIR}/gpu_smoke8.csv" \
  --micro-manifest "${MANIFEST_DIR}/micro_overfit4.csv" \
  --negative-manifest "${MANIFEST_DIR}/negative_smoke1.csv" \
  --report-dir "${REPORT_DIR}" \
  --resolution 512 \
  --dtype bfloat16 \
  --seed 2026

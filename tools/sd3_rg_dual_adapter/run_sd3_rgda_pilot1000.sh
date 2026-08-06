#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/road_damage_exp/models/stable-diffusion-3-medium-diffusers}"
CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/road_damage_exp/cache/sd3_rgda_pilot1000}"
PILOT_MANIFEST_SUMMARY="${PILOT_MANIFEST_SUMMARY:-${ROOT}/manifests/pilot1000/manifest_summary.json}"
CLEAN_PROXY_AUDIT="${CLEAN_PROXY_AUDIT:-${ROOT}/pilot_assets/clean_proxy/clean_proxy_audit.json}"
CACHE_AUDIT="${CACHE_AUDIT:-${CACHE_ROOT}/cache_audit.json}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/road_damage_exp/reports/sd3_rgda_pilot1000_${RUN_TS}}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PIP_NO_INDEX=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${REPORT_DIR}"

package_on_exit() {
  status=$?
  if [ -d "${REPORT_DIR}" ]; then
    tar -C "$(dirname "${REPORT_DIR}")" -czf "${REPORT_DIR}.tar.gz" "$(basename "${REPORT_DIR}")" || true
    sha256sum "${REPORT_DIR}.tar.gz" > "${REPORT_DIR}.tar.gz.sha256" || true
  fi
  exit "${status}"
}

trap package_on_exit EXIT

if [ "${PILOT_DRY_INTEGRATION:-0}" = "1" ]; then
  "${PYTHON}" "${ROOT}/scripts/train_rgda_pilot1000.py" \
    --report-dir "${REPORT_DIR}" \
    --dry-integration \
    --dry-steps 10
else
  nvidia-smi -L | grep -q "NVIDIA GeForce RTX 5090"
  "${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("TORCH_CUDA_GATE=FAIL")
if torch.cuda.device_count() < 1:
    raise SystemExit("CUDA_DEVICE_COUNT_GATE=FAIL")
name = torch.cuda.get_device_name(0)
if "RTX 5090" not in name:
    raise SystemExit("CUDA_DEVICE_NAME_GATE=FAIL:" + name)
PY
  "${PYTHON}" "${ROOT}/scripts/train_rgda_pilot1000.py" \
    --model-path "${MODEL_PATH}" \
    --train-cache-manifest "${CACHE_ROOT}/train512/cache_manifest.csv" \
    --eval-cache-manifest "${CACHE_ROOT}/eval64/cache_manifest.csv" \
    --pilot-manifest-summary "${PILOT_MANIFEST_SUMMARY}" \
    --clean-proxy-audit "${CLEAN_PROXY_AUDIT}" \
    --cache-audit "${CACHE_AUDIT}" \
    --report-dir "${REPORT_DIR}" \
    --steps 1000 \
    --seed 2026 \
    --checkpoint-interval 100 \
    --eval-interval 100
fi

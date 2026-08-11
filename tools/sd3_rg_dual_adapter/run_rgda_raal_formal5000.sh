#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

resolve_python() {
  if [ -n "${PYTHON:-}" ]; then
    printf '%s\n' "${PYTHON}"
    return 0
  fi
  if command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  if [ -x /root/autodl-tmp/road_damage_exp/envs/sd3_bgpaste_py311/bin/python3.11 ]; then
    printf '%s\n' /root/autodl-tmp/road_damage_exp/envs/sd3_bgpaste_py311/bin/python3.11
    return 0
  fi
  echo "PYTHON_RESOLUTION=FAIL" >&2
  return 127
}

PYTHON="$(resolve_python)"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/road_damage_exp/models/stable-diffusion-3-medium-diffusers}"
CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/road_damage_exp/cache/sd3_rgda_formal5000}"
FORMAL_MANIFEST_ROOT="${FORMAL_MANIFEST_ROOT:-/root/autodl-tmp/road_damage_exp/tools/sd3_rg_dual_adapter_formal5000/manifests/formal5000}"
FORMAL_PROXY_AUDIT="${FORMAL_PROXY_AUDIT:-/root/autodl-tmp/road_damage_exp/tools/sd3_rg_dual_adapter_formal5000/formal_assets/clean_proxy/clean_proxy_audit.json}"
FORMAL_CACHE_AUDIT="${FORMAL_CACHE_AUDIT:-${CACHE_ROOT}/cache_audit.json}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/road_damage_exp/reports/sd3_rgda_raal_formal5000_${RUN_TS}}"
STEPS="${STEPS:-5000}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PIP_NO_INDEX=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${REPORT_DIR}"

if [ "${RAAL_FORMAL_DRY_INTEGRATION:-0}" = "1" ]; then
  "${PYTHON}" "${ROOT}/scripts/train_rgda_raal_formal5000.py" \
    --report-dir "${REPORT_DIR}" \
    --dry-integration \
    --dry-steps "${DRY_STEPS:-20}"
else
  "${PYTHON}" "${ROOT}/scripts/train_rgda_raal_formal5000.py" \
    --model-path "${MODEL_PATH}" \
    --train-cache-manifest "${CACHE_ROOT}/train1980/cache_manifest.csv" \
    --eval128-cache-manifest "${CACHE_ROOT}/eval128_cache_manifest.csv" \
    --val-cache-manifest "${CACHE_ROOT}/val424/cache_manifest.csv" \
    --train-pool-manifest "${FORMAL_MANIFEST_ROOT}/train_pool1980.csv" \
    --val-full-manifest "${FORMAL_MANIFEST_ROOT}/val_full424.csv" \
    --eval128-manifest "${FORMAL_MANIFEST_ROOT}/eval128.csv" \
    --schedule-manifest "${FORMAL_MANIFEST_ROOT}/schedule5000.csv" \
    --manifest-summary "${FORMAL_MANIFEST_ROOT}/manifest_summary.json" \
    --schedule-audit "${FORMAL_MANIFEST_ROOT}/schedule_audit.json" \
    --clean-proxy-audit "${FORMAL_PROXY_AUDIT}" \
    --cache-audit "${FORMAL_CACHE_AUDIT}" \
    --report-dir "${REPORT_DIR}" \
    --steps "${STEPS}" \
    --seed 2026 \
    --checkpoint-interval 250 \
    --eval-interval 250 \
    --raal-weight 0.02 \
    --raal-temperature 1.0 \
    --raal-layers 5,11,17
fi

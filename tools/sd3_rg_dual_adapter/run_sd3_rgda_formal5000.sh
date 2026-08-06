#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/road_damage_exp/models/stable-diffusion-3-medium-diffusers}"
CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/road_damage_exp/cache/sd3_rgda_formal5000}"
FORMAL_MANIFEST_ROOT="${FORMAL_MANIFEST_ROOT:-${ROOT}/manifests/formal5000}"
FORMAL_PROXY_AUDIT="${FORMAL_PROXY_AUDIT:-${ROOT}/formal_assets/clean_proxy/clean_proxy_audit.json}"
FORMAL_CACHE_AUDIT="${FORMAL_CACHE_AUDIT:-${CACHE_ROOT}/cache_audit.json}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/road_damage_exp/reports/sd3_rgda_formal5000_${RUN_TS}}"

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

if [ "${FORMAL_DRY_INTEGRATION:-0}" = "1" ]; then
  "${PYTHON}" "${ROOT}/scripts/train_rgda_formal5000.py" \
    --report-dir "${REPORT_DIR}" \
    --dry-integration \
    --dry-steps 20
else
  set +e
  NVIDIA_SMI_OUTPUT="$(nvidia-smi -L 2>&1)"
  NVIDIA_SMI_RC=$?
  set -e
  echo "NVIDIA_SMI_RC=${NVIDIA_SMI_RC}"
  echo "NVIDIA_SMI_OUTPUT=${NVIDIA_SMI_OUTPUT}"
  "${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("TORCH_CUDA_GATE=FAIL")
if torch.cuda.device_count() < 1:
    raise SystemExit("CUDA_DEVICE_COUNT_GATE=FAIL")
name = torch.cuda.get_device_name(0)
if "RTX 5090" not in name:
    raise SystemExit("CUDA_DEVICE_NAME_GATE=FAIL:" + name)
props = torch.cuda.get_device_properties(0)
if props.total_memory < 30 * 1024**3:
    raise SystemExit("CUDA_MEMORY_GATE=FAIL:" + str(props.total_memory))
probe = torch.ones(1, device="cuda")
torch.cuda.synchronize()
del probe
print("PYTORCH_CUDA_STRONG_GATE=PASS")
PY
  if printf '%s' "${NVIDIA_SMI_OUTPUT}" | grep -q "NVIDIA GeForce RTX 5090"; then
    echo "GPU_ENV=PASS"
  else
    echo "GPU_ENV=PASS"
    echo "NVIDIA_SMI_TEXT_GATE=FALSE_NEGATIVE"
    echo "GPU_GATE_OVERRIDE=PYTORCH_STRONG_GATE"
  fi
  "${PYTHON}" "${ROOT}/scripts/train_rgda_formal5000.py" \
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
    --steps 5000 \
    --seed 2026 \
    --checkpoint-interval 250 \
    --eval-interval 250
fi

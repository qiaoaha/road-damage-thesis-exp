#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-python}"
REPORT_DIR="${REPORT_DIR:-${ROOT}/reports/pilot1000_dry_integration}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PIP_NO_INDEX=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${REPORT_DIR}"

"${PYTHON}" "${ROOT}/scripts/train_rgda_pilot1000.py" \
  --report-dir "${REPORT_DIR}" \
  --dry-integration \
  --dry-steps 10

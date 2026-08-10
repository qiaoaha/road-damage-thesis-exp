#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"

python scripts/train_rgda_raal_pilot1000.py \
  --arm r0 \
  --model-path "$MODEL_PATH" \
  --train-cache-manifest "$TRAIN_CACHE_MANIFEST" \
  --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
  --report-dir "$REPORT_DIR/raal_r0" \
  --steps 1000 \
  --raal-weight 0.0 \
  --raal-layers 5,11,17 \
  --attention-mask-bank-sha256 "$ATTENTION_MASK_BANK_SHA256" \
  --source-schedule-sha256 "$SOURCE_SCHEDULE_SHA256"

python scripts/train_rgda_raal_pilot1000.py \
  --arm r1 \
  --model-path "$MODEL_PATH" \
  --train-cache-manifest "$TRAIN_CACHE_MANIFEST" \
  --eval-cache-manifest "$EVAL_CACHE_MANIFEST" \
  --report-dir "$REPORT_DIR/raal_r1" \
  --steps 1000 \
  --raal-weight 0.02 \
  --raal-layers 5,11,17 \
  --attention-mask-bank-sha256 "$ATTENTION_MASK_BANK_SHA256" \
  --source-schedule-sha256 "$SOURCE_SCHEDULE_SHA256"

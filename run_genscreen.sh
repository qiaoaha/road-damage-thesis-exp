#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m genscreen.scripts.run_pipeline "$@"

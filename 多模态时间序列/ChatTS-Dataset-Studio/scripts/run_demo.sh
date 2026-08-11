#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/make_demo_data.py" \
    --output "$PROJECT_ROOT/demo_data"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m chatts_dataset_studio serve \
    -c "$PROJECT_ROOT/configs/demo.yaml" "$@"

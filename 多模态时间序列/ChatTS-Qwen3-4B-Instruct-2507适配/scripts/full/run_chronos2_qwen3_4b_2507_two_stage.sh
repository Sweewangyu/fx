#!/usr/bin/env bash
set -Eeuo pipefail

# Thin container-side entry point for a fresh Qwen3-4B-Instruct-2507 run.
# The base must first be converted by prepare_qwen3_chatts_base.py.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
MODEL_PATH="${MODEL_PATH:-/share/airesearch/data/finiverse/model/ChatTS-Qwen3-4B-Instruct-2507}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/share/airesearch/data/finiverse/output/ChatTS-msxf-4B-Instruct-2507}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/full/run_chronos2_qwen3_4b_2507_two_stage.sh [options]

Options:
  --model-path PATH       Prepared ChatTS-Qwen3-4B-Instruct-2507 directory
  --output-root PATH      Stage1, final model, and log output root
  --dataset-dir PATH      Dataset snapshot containing dataset_info.json
  --chronos2-path PATH    Local Chronos-2 directory
  --preflight-only        Validate configuration without training
  -h, --help

Training hyperparameters and dataset names use the existing pipeline's
environment variables. Dataset Studio's training.env can be sourced first.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --model-path) MODEL_PATH="${2:?missing value}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:?missing value}"; shift 2 ;;
        --dataset-dir) DATASET_DIR="${2:?missing value}"; shift 2 ;;
        --chronos2-path) CHRONOS2_MODEL_PATH="${2:?missing value}"; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -f "$MODEL_PATH/CHATTS_BASE_MANIFEST.json" ]] || {
    echo "MODEL_PATH is not a prepared ChatTS 4B base: $MODEL_PATH" >&2
    echo "Run scripts/download_prepare_qwen3_4b_instruct_2507.sh first." >&2
    exit 1
}
"$PYTHON_BIN" - "$MODEL_PATH/CHATTS_BASE_MANIFEST.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "architecture": "Qwen3TSForCausalLM",
    "hidden_size": 2560,
    "num_hidden_layers": 36,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"Prepared 4B manifest mismatch for {key}: {payload.get(key)!r} != {value!r}")
PY

export PROJECT_ROOT MODEL_PATH OUTPUT_ROOT CHRONOS2_MODEL_PATH PYTHON_BIN
export DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/data}"
export PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
export STAGE1_TIMESERIES_SFT_LR="${STAGE1_TIMESERIES_SFT_LR:-3e-5}"
export STAGE2_TIMESERIES_SFT_LR="${STAGE2_TIMESERIES_SFT_LR:-1e-5}"
export S1_LR="${S1_LR:-1e-5}"
export S2_LR="${S2_LR:-1e-5}"
export SEED="${SEED:-42}"
export KEEP_STAGE1="${KEEP_STAGE1:-1}"

exec bash "$PROJECT_ROOT/scripts/full/run_chronos2_best_two_stage.sh"

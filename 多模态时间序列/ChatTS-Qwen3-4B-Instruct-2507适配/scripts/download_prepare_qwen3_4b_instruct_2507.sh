#!/usr/bin/env bash
set -Eeuo pipefail

# Run this once on a machine that can access huggingface.co. The resulting
# ChatTS-compatible directory can then be copied to the offline cluster.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
REVISION="${REVISION:-main}"
RAW_MODEL_DIR="${RAW_MODEL_DIR:-/share/airesearch/data/finiverse/model/Qwen3-4B-Instruct-2507}"
CHATTS_TEMPLATE="${CHATTS_TEMPLATE:-/share/airesearch/data/finiverse/model/ChatTS-Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/airesearch/data/finiverse/model/ChatTS-Qwen3-4B-Instruct-2507}"
WEIGHT_MODE="${WEIGHT_MODE:-hardlink}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/download_prepare_qwen3_4b_instruct_2507.sh [options]

Options:
  --raw-model-dir PATH      Downloaded raw Qwen directory
  --chatts-template PATH    Existing official ChatTS-Qwen3-8B directory
  --output-dir PATH         New ChatTS-compatible 4B base directory
  --revision REVISION       Hugging Face revision, tag, or commit hash
  --weight-mode MODE        hardlink (default) or copy
  --skip-download           Convert an already downloaded raw directory
  -h, --help

All values can also be supplied as uppercase environment variables.
EOF
}

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
while (( $# > 0 )); do
    case "$1" in
        --raw-model-dir) RAW_MODEL_DIR="${2:?missing value}"; shift 2 ;;
        --chatts-template) CHATTS_TEMPLATE="${2:?missing value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?missing value}"; shift 2 ;;
        --revision) REVISION="${2:?missing value}"; shift 2 ;;
        --weight-mode) WEIGHT_MODE="${2:?missing value}"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$WEIGHT_MODE" in hardlink|copy) ;; *) echo "WEIGHT_MODE must be hardlink or copy." >&2; exit 2 ;; esac
case "$SKIP_DOWNLOAD" in 0|1) ;; *) echo "SKIP_DOWNLOAD must be 0 or 1." >&2; exit 2 ;; esac
[[ -d "$CHATTS_TEMPLATE" ]] || { echo "ChatTS template not found: $CHATTS_TEMPLATE" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists: $OUTPUT_DIR" >&2; exit 2; }

if (( SKIP_DOWNLOAD == 0 )); then
    mkdir -p "$RAW_MODEL_DIR"
    if command -v hf >/dev/null 2>&1; then
        hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$RAW_MODEL_DIR"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$MODEL_ID" --revision "$REVISION" --local-dir "$RAW_MODEL_DIR"
    else
        echo "Neither 'hf' nor 'huggingface-cli' is installed." >&2
        echo "Install huggingface_hub on the Internet-connected machine first." >&2
        exit 1
    fi
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/prepare_qwen3_chatts_base.py" \
    --qwen-checkpoint "$RAW_MODEL_DIR" \
    --chatts-template "$CHATTS_TEMPLATE" \
    --output-dir "$OUTPUT_DIR" \
    --weight-mode "$WEIGHT_MODE"

echo "Prepared model: $OUTPUT_DIR"

#!/usr/bin/env bash
set -Eeuo pipefail

# Small structured-reasoning diagnostic run. It preserves every invalid retry
# and the final valid/invalid response without changing answer scoring.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS/ChatTS-main}"
TSRBENCH_ROOT="${TSRBENCH_ROOT:-/workspace/TSRBench}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/ckpt}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TRACE_RUN_ROOT="${TRACE_RUN_ROOT:-${TSRBENCH_ROOT}/evaluation/trace_runs/${MODEL_NAME}/${RUN_ID}}"

# event_prediction is a useful default because every item is an MCQ and the
# generated <think> path lets us inspect why the model selected the letter.
DATASETS="${DATASETS:-event_prediction}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
SAMPLE_INDICES="${SAMPLE_INDICES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRACE_RUN_ROOT}/results}"
TRACE_OUTPUT="${TRACE_OUTPUT:-${TRACE_RUN_ROOT}/inference_trace.jsonl}"
TRACE_REPORT="${TRACE_REPORT:-${TRACE_RUN_ROOT}/trace_report.md}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PROJECT_ROOT TSRBENCH_ROOT MODEL_PATH MODEL_NAME DATASETS
export MAX_SAMPLES SAMPLE_INDICES OUTPUT_ROOT TRACE_OUTPUT PYTHON_BIN
export PROMPT_MODE="${PROMPT_MODE:-json_reasoning}"
export FORCE_INFERENCE="${FORCE_INFERENCE:-1}"
# Full-benchmark strict accuracy is misleading for a five-row subset; the
# generated Markdown report calculates correctness only over the traced rows.
export RUN_EVALUATION="${RUN_EVALUATION:-0}"
export NUM_GPUS="${NUM_GPUS:-2}"
export NUM_GPUS_PER_PROCESS="${NUM_GPUS_PER_PROCESS:-2}"
export REQUEST_CHUNK_SIZE="${REQUEST_CHUNK_SIZE:-1}"
export SEED="${SEED:-42}"

mkdir -p "$TRACE_RUN_ROOT"

echo "Trace run root: $TRACE_RUN_ROOT"
echo "Datasets:       $DATASETS"
echo "Samples:        ${SAMPLE_INDICES:-first $MAX_SAMPLES}"

bash "$PROJECT_ROOT/scripts/run_chatts_tsrbench.sh"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/render_tsrbench_trace.py" \
    "$TRACE_OUTPUT" \
    --output "$TRACE_REPORT"

echo "Raw retry trace: $TRACE_OUTPUT"
echo "Readable report: $TRACE_REPORT"

#!/usr/bin/env bash
set -Eeuo pipefail

# Run one finalized Chronos-2 ChatTS checkpoint sequentially on four benchmark
# suites.  Each suite exclusively receives all eight visible GPUs before the
# next suite starts.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS/ChatTS-main}"
SEED="${SEED:-42}"
MODEL_PATH="${MODEL_PATH:-/share/airesearch/data/finiverse/output/ChatTS-msxf-8B-datav1/best_seed${SEED}}"
MODEL_NAME="${MODEL_NAME:-chatts-msxf-8B-datav1-seed${SEED}}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TSRBENCH_ROOT="${TSRBENCH_ROOT:-/share/airesearch/data/finiverse/TSRBench-dataset}"
TSRBENCH_DATASET_ROOT="${TSRBENCH_DATASET_ROOT:-${TSRBENCH_ROOT}}"
TINYBENCH_DATASET_ROOT="${TINYBENCH_DATASET_ROOT:-/share/airesearch/data/finiverse/tyb}"
TS_HAYSTACK_ROOT="${TS_HAYSTACK_ROOT:-/workspace/TS-Haystack}"
TIMESERIESEXAM_ROOT="${TIMESERIESEXAM_ROOT:-/workspace/TimeSeriesExam}"
TIMESERIESEXAM_DATA_FILE="${TIMESERIESEXAM_DATA_FILE:-${TIMESERIESEXAM_ROOT}/output/round_3_folder/qa_dataset.json}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/share/airesearch/data/finiverse/evaluation/all-benchmarks/${MODEL_NAME}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
STATUS_FILE="${STATUS_FILE:-${OUTPUT_ROOT}/benchmark_status.tsv}"
SUMMARY_FILE="${SUMMARY_FILE:-${OUTPUT_ROOT}/all_benchmarks_summary.md}"
MANIFEST_FILE="${MANIFEST_FILE:-${OUTPUT_ROOT}/run_manifest.json}"

FORCE_EVAL="${FORCE_EVAL:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
OFFLINE="${OFFLINE:-1}"
REQUIRE_TRAINING_MARKER="${REQUIRE_TRAINING_MARKER:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# All suites run one after another on these eight devices.  The three
# time-series runners use four replicated two-GPU engines for throughput;
# tinyBenchmarks uses one eight-way tensor-parallel engine.
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-8}"
TS_GPUS_PER_PROCESS="${TS_GPUS_PER_PROCESS:-2}"

TSR_PROMPT_MODE="${TSR_PROMPT_MODE:-answer_only}"
TSR_MAX_MODEL_LEN="${TSR_MAX_MODEL_LEN:-12288}"
TSR_MAX_NEW_TOKENS="${TSR_MAX_NEW_TOKENS:-8}"
TSR_BATCH_SIZE="${TSR_BATCH_SIZE:-16}"
TSR_REQUEST_CHUNK_SIZE="${TSR_REQUEST_CHUNK_SIZE:-128}"

TINY_MAX_MODEL_LEN="${TINY_MAX_MODEL_LEN:-6000}"
TINY_REQUEST_CHUNK_SIZE="${TINY_REQUEST_CHUNK_SIZE:-16}"
TINY_GPU_MEMORY_UTILIZATION="${TINY_GPU_MEMORY_UTILIZATION:-0.70}"

HAYSTACK_MAX_MODEL_LEN="${HAYSTACK_MAX_MODEL_LEN:-40960}"
HAYSTACK_MAX_NEW_TOKENS="${HAYSTACK_MAX_NEW_TOKENS:-500}"
HAYSTACK_BATCH_SIZE="${HAYSTACK_BATCH_SIZE:-1}"
HAYSTACK_REQUEST_CHUNK_SIZE="${HAYSTACK_REQUEST_CHUNK_SIZE:-8}"

EXAM_MAX_MODEL_LEN="${EXAM_MAX_MODEL_LEN:-8192}"
EXAM_MAX_NEW_TOKENS="${EXAM_MAX_NEW_TOKENS:-1024}"
EXAM_BATCH_SIZE="${EXAM_BATCH_SIZE:-8}"
EXAM_REQUEST_CHUNK_SIZE="${EXAM_REQUEST_CHUNK_SIZE:-64}"

for flag_name in FORCE_EVAL PREFLIGHT_ONLY OFFLINE REQUIRE_TRAINING_MARKER; do
    flag_value="${!flag_name}"
    [[ "$flag_value" == "0" || "$flag_value" == "1" ]] || {
        echo "$flag_name must be 0 or 1, got: $flag_value" >&2
        exit 2
    }
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer." >&2; exit 2; }
[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || { echo "MAX_SAMPLES must be non-negative." >&2; exit 2; }
[[ "$EVAL_NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_NUM_GPUS must be positive." >&2; exit 2; }
[[ "$TS_GPUS_PER_PROCESS" =~ ^[1-9][0-9]*$ ]] || { echo "TS_GPUS_PER_PROCESS must be positive." >&2; exit 2; }
(( EVAL_NUM_GPUS == 8 )) || { echo "This runner is fixed to eight-GPU evaluation." >&2; exit 2; }
(( EVAL_NUM_GPUS % TS_GPUS_PER_PROCESS == 0 )) || {
    echo "EVAL_NUM_GPUS must be divisible by TS_GPUS_PER_PROCESS." >&2
    exit 2
}
[[ "$MODEL_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "MODEL_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
}

require_dir() {
    local label="$1" path="$2"
    [[ -d "$path" ]] || { echo "$label directory not found: $path" >&2; exit 1; }
}

require_file() {
    local label="$1" path="$2"
    [[ -f "$path" ]] || { echo "$label file not found: $path" >&2; exit 1; }
}

require_dir "ChatTS project" "$PROJECT_ROOT"
require_dir "Chronos-2 backbone" "$CHRONOS2_MODEL_PATH"
require_dir "TSRBench dataset root" "$TSRBENCH_DATASET_ROOT"
require_dir "tinyBenchmarks dataset root" "$TINYBENCH_DATASET_ROOT"
require_file "TS-Haystack registry" "$TS_HAYSTACK_ROOT/src/datasets/registry.py"
require_dir "TS-Haystack data" "$TS_HAYSTACK_ROOT/data"
require_file "TimeSeriesExam concepts" "$TIMESERIESEXAM_ROOT/evaluate/concepts.py"
require_file "TimeSeriesExam dataset" "$TIMESERIESEXAM_DATA_FILE"
require_file "Encoder inspector" "$PROJECT_ROOT/scripts/inspect_chatts_ts_encoder_checkpoints.py"
require_file "TSRBench runner" "$PROJECT_ROOT/scripts/run_chatts_tsrbench.sh"
require_file "tinyBenchmarks runner" "$PROJECT_ROOT/scripts/run_chatts_tinybenchmarks_mcq.sh"
require_file "TS-Haystack runner" "$PROJECT_ROOT/scripts/run_chatts_ts_haystack.sh"
require_file "TimeSeriesExam runner" "$PROJECT_ROOT/scripts/run_chatts_timeseriesexam.sh"

tsr_probe="$(find "$TSRBENCH_DATASET_ROOT" -type f -name perception.jsonl -print -quit)"
[[ -n "$tsr_probe" ]] || {
    echo "Cannot find perception.jsonl under $TSRBENCH_DATASET_ROOT" >&2
    exit 1
}

AVAILABLE_GPUS="${AVAILABLE_GPUS_OVERRIDE:-$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')}"
if (( AVAILABLE_GPUS < 8 )); then
    echo "Sequential evaluation requires 8 visible GPUs, but PyTorch sees $AVAILABLE_GPUS." >&2
    exit 1
fi

"$PYTHON_BIN" - "$AVAILABLE_GPUS" "$EVAL_GPUS" <<'PY'
import sys

available = int(sys.argv[1])
mask = sys.argv[2]
try:
    ids = [int(item) for item in mask.split(",")]
except ValueError as exc:
    raise SystemExit(f"Invalid EVAL_GPUS mask: {mask}") from exc
if len(ids) != 8 or len(set(ids)) != 8:
    raise SystemExit(f"EVAL_GPUS must contain exactly eight distinct GPUs: {mask}")
for gpu_id in ids:
    if gpu_id < 0 or gpu_id >= available:
        raise SystemExit(f"GPU {gpu_id} is outside 0..{available - 1}")
PY

if [[ -f "$MODEL_PATH/config.json" ]]; then
    if [[ "$REQUIRE_TRAINING_MARKER" == "1" ]]; then
        require_file "Training completion marker" "$MODEL_PATH/TRAINING_COMPLETE.json"
    fi
elif [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "Note: final model does not exist yet; model validation is deferred until training completes: $MODEL_PATH"
else
    echo "Final model config not found: $MODEL_PATH/config.json" >&2
    exit 1
fi

echo "============================================================"
echo " ChatTS four-suite sequential eight-GPU evaluation"
echo " Model:          $MODEL_PATH"
echo " Model name:     $MODEL_NAME"
echo " Encoder:        chronos2 ($CHRONOS2_MODEL_PATH)"
echo " Seed:           $SEED"
echo " GPU allocation: $EVAL_GPUS (exclusive for each suite)"
echo " TS engines:     $((EVAL_NUM_GPUS / TS_GPUS_PER_PROCESS)) x ${TS_GPUS_PER_PROCESS}-GPU"
echo " Max samples:    $MAX_SAMPLES (0 means full benchmark)"
echo " Output:         $OUTPUT_ROOT"
echo "============================================================"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "Evaluation preflight passed. No files were changed."
    exit 0
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
LOG_ROOT="$(cd "$LOG_ROOT" && pwd)"

export CUDA_VISIBLE_DEVICES="$EVAL_GPUS"
export CHATTS_TS_ENCODER_TYPE=chronos2
export CHATTS_CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH"
export CHATTS_VLLM_SEED="$SEED"
export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ "$OFFLINE" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

run_tsrbench() {
    env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        TSRBENCH_ROOT="$TSRBENCH_ROOT" \
        DATASET_ROOT="$TSRBENCH_DATASET_ROOT" \
        OUTPUT_ROOT="$OUTPUT_ROOT/tsrbench" \
        MODEL_PATH="$MODEL_PATH" \
        MODEL_NAME="$MODEL_NAME" \
        DATASETS=all \
        PROMPT_MODE="$TSR_PROMPT_MODE" \
        NUM_GPUS="$EVAL_NUM_GPUS" \
        NUM_GPUS_PER_PROCESS="$TS_GPUS_PER_PROCESS" \
        BATCH_SIZE="$TSR_BATCH_SIZE" \
        REQUEST_CHUNK_SIZE="$TSR_REQUEST_CHUNK_SIZE" \
        MAX_SAMPLES="$MAX_SAMPLES" \
        MAX_NEW_TOKENS="$TSR_MAX_NEW_TOKENS" \
        CHATTS_VLLM_MAX_MODEL_LEN="$TSR_MAX_MODEL_LEN" \
        MAX_PROCESSED_INPUT_TOKENS="$((TSR_MAX_MODEL_LEN - TSR_MAX_NEW_TOKENS))" \
        ENABLE_THINKING=0 \
        SEED="$SEED" \
        FORCE_INFERENCE="$FORCE_EVAL" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/scripts/run_chatts_tsrbench.sh"
}

run_tinybenchmarks() {
    local -a command=(
        bash "$PROJECT_ROOT/scripts/run_chatts_tinybenchmarks_mcq.sh"
        --model "$MODEL_NAME=$MODEL_PATH"
        --baseline "$MODEL_NAME"
    )
    if [[ "$FORCE_EVAL" == "1" ]]; then
        command+=(--force)
    fi
    env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        DATASET_ROOT="$TINYBENCH_DATASET_ROOT" \
        OUTPUT_ROOT="$OUTPUT_ROOT/tinybenchmarks" \
        TASKS_CSV="tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande" \
        NUM_GPUS="$EVAL_NUM_GPUS" \
        REQUEST_CHUNK_SIZE="$TINY_REQUEST_CHUNK_SIZE" \
        GPU_MEMORY_UTILIZATION="$TINY_GPU_MEMORY_UTILIZATION" \
        CHATTS_VLLM_MAX_MODEL_LEN="$TINY_MAX_MODEL_LEN" \
        MAX_SAMPLES="$MAX_SAMPLES" \
        SEED="$SEED" \
        OFFLINE="$OFFLINE" \
        PYTHON_BIN="$PYTHON_BIN" \
        "${command[@]}"
}

run_ts_haystack() {
    env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        TS_HAYSTACK_ROOT="$TS_HAYSTACK_ROOT" \
        OUTPUT_ROOT="$OUTPUT_ROOT/ts_haystack" \
        MODEL_PATH="$MODEL_PATH" \
        MODEL_NAME="$MODEL_NAME" \
        DATASETS=all \
        TASKS=all \
        CONTEXT_LENGTHS=all \
        SPLIT=test \
        NUM_GPUS="$EVAL_NUM_GPUS" \
        NUM_GPUS_PER_PROCESS="$TS_GPUS_PER_PROCESS" \
        BATCH_SIZE="$HAYSTACK_BATCH_SIZE" \
        REQUEST_CHUNK_SIZE="$HAYSTACK_REQUEST_CHUNK_SIZE" \
        CHATTS_VLLM_MAX_MODEL_LEN="$HAYSTACK_MAX_MODEL_LEN" \
        MAX_NEW_TOKENS="$HAYSTACK_MAX_NEW_TOKENS" \
        MAX_PROCESSED_INPUT_TOKENS="$((HAYSTACK_MAX_MODEL_LEN - HAYSTACK_MAX_NEW_TOKENS))" \
        MAX_SAMPLES="$MAX_SAMPLES" \
        TEMPERATURE=0.0 \
        SEED="$SEED" \
        ENABLE_THINKING=0 \
        FORCE_INFERENCE="$FORCE_EVAL" \
        SCORE_ONLY=0 \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/scripts/run_chatts_ts_haystack.sh"
}

run_timeseriesexam() {
    env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        TIMESERIESEXAM_ROOT="$TIMESERIESEXAM_ROOT" \
        DATA_FILE_PATH="$TIMESERIESEXAM_DATA_FILE" \
        OUTPUT_ROOT="$OUTPUT_ROOT/timeseriesexam" \
        MODEL_PATH="$MODEL_PATH" \
        MODEL_NAME="$MODEL_NAME" \
        NUM_GPUS="$EVAL_NUM_GPUS" \
        NUM_GPUS_PER_PROCESS="$TS_GPUS_PER_PROCESS" \
        BATCH_SIZE="$EXAM_BATCH_SIZE" \
        REQUEST_CHUNK_SIZE="$EXAM_REQUEST_CHUNK_SIZE" \
        CHATTS_VLLM_MAX_MODEL_LEN="$EXAM_MAX_MODEL_LEN" \
        MAX_NEW_TOKENS="$EXAM_MAX_NEW_TOKENS" \
        MAX_PROCESSED_INPUT_TOKENS="$((EXAM_MAX_MODEL_LEN - EXAM_MAX_NEW_TOKENS))" \
        MAX_SAMPLES="$MAX_SAMPLES" \
        TEMPERATURE=0.0 \
        SEED="$SEED" \
        ADD_QUESTION_HINT=1 \
        ADD_CONCEPTS=1 \
        ADD_EXAMPLES=1 \
        ENABLE_THINKING=0 \
        FORCE_INFERENCE="$FORCE_EVAL" \
        SCORE_ONLY=0 \
        OFFLINE="$OFFLINE" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/scripts/run_chatts_timeseriesexam.sh"
}

SUITE_NAMES=()
SUITE_STATUSES=()
SUITE_CODES=()
SUITE_OUTPUTS=()
SUITE_LOGS=()
FAILED_SUITES=0

run_step() {
    local name="$1" suite_output="$2"
    shift 2
    local log_file="$LOG_ROOT/${name}.log"
    local status exit_code
    echo
    echo "==================== starting $name ===================="
    echo "Exclusive GPUs: $EVAL_GPUS"
    echo "Log: $log_file"
    if "$@" 2>&1 | tee "$log_file"; then
        status=PASS
        exit_code=0
    else
        exit_code=$?
        status=FAIL
        FAILED_SUITES=$((FAILED_SUITES + 1))
    fi
    SUITE_NAMES+=("$name")
    SUITE_STATUSES+=("$status")
    SUITE_CODES+=("$exit_code")
    SUITE_OUTPUTS+=("$suite_output")
    SUITE_LOGS+=("$log_file")
    echo "==================== $name: $status ===================="
}

run_step tsrbench "$OUTPUT_ROOT/tsrbench" run_tsrbench
run_step tinybenchmarks "$OUTPUT_ROOT/tinybenchmarks" run_tinybenchmarks
run_step ts_haystack "$OUTPUT_ROOT/ts_haystack" run_ts_haystack
run_step timeseriesexam "$OUTPUT_ROOT/timeseriesexam" run_timeseriesexam

{
    printf 'suite\tgpus\tstatus\texit_code\toutput_dir\tlog_file\n'
    for index in "${!SUITE_NAMES[@]}"; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${SUITE_NAMES[$index]}" \
            "$EVAL_GPUS" \
            "${SUITE_STATUSES[$index]}" \
            "${SUITE_CODES[$index]}" \
            "${SUITE_OUTPUTS[$index]}" \
            "${SUITE_LOGS[$index]}"
    done
} > "$STATUS_FILE"

{
    echo "# ChatTS all-benchmark evaluation"
    echo
    echo "- Model: \`$MODEL_PATH\`"
    echo "- Seed: \`$SEED\`"
    echo "- Encoder: \`chronos2\`"
    echo "- Scheduling: sequential, eight exclusive GPUs per suite"
    echo "- Full benchmark: \`$([[ "$MAX_SAMPLES" == "0" ]] && echo yes || echo no)\`"
    echo
    echo "| Suite | GPUs | Status | Exit code | Output | Log |"
    echo "|---|---:|---:|---:|---|---|"
    for index in "${!SUITE_NAMES[@]}"; do
        printf '| %s | %s | %s | %s | `%s` | `%s` |\n' \
            "${SUITE_NAMES[$index]}" \
            "$EVAL_GPUS" \
            "${SUITE_STATUSES[$index]}" \
            "${SUITE_CODES[$index]}" \
            "${SUITE_OUTPUTS[$index]}" \
            "${SUITE_LOGS[$index]}"
    done
} > "$SUMMARY_FILE"

STATUS_FILE="$STATUS_FILE" \
MANIFEST_FILE="$MANIFEST_FILE" \
MODEL_PATH="$MODEL_PATH" \
MODEL_NAME="$MODEL_NAME" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SEED="$SEED" \
MAX_SAMPLES="$MAX_SAMPLES" \
FORCE_EVAL="$FORCE_EVAL" \
"$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(os.environ["STATUS_FILE"])
with status_path.open(encoding="utf-8", newline="") as stream:
    suites = list(csv.DictReader(stream, delimiter="\t"))
payload = {
    "status": "pass" if all(item["status"] == "PASS" for item in suites) else "fail",
    "model_path": os.environ["MODEL_PATH"],
    "model_name": os.environ["MODEL_NAME"],
    "ts_encoder_type": "chronos2",
    "seed": int(os.environ["SEED"]),
    "max_samples": int(os.environ["MAX_SAMPLES"]),
    "force_eval": os.environ["FORCE_EVAL"] == "1",
    "scheduling": "sequential",
    "exclusive_gpus_per_suite": 8,
    "output_root": os.environ["OUTPUT_ROOT"],
    "suites": suites,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
path = Path(os.environ["MANIFEST_FILE"])
temporary = path.with_suffix(path.suffix + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
temporary.replace(path)
PY

echo
echo "==================== benchmark status ===================="
column -t -s $'\t' "$STATUS_FILE" 2>/dev/null || cat "$STATUS_FILE"
echo "Summary:  $SUMMARY_FILE"
echo "Manifest: $MANIFEST_FILE"
echo "Outputs:  $OUTPUT_ROOT"
echo "=========================================================="

if (( FAILED_SUITES > 0 )); then
    exit 1
fi

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
METRICS_FILE="${METRICS_FILE:-${OUTPUT_ROOT}/metrics.json}"

# Comma-separated subset.  The default preserves the historical four-suite
# order. RUN_ID is metadata only and does not prevent reuse of an otherwise
# identical fingerprinted result.
BENCHMARKS="${BENCHMARKS:-tsrbench,tinybenchmarks,ts_haystack,timeseriesexam}"
RUN_ID="${RUN_ID:-manual}"
EVAL_PROTOCOL_HASH="${EVAL_PROTOCOL_HASH:-}"
DATA_VERSION="${DATA_VERSION:-}"
DATASET_SNAPSHOT_HASH="${DATASET_SNAPSHOT_HASH:-}"

FORCE_EVAL="${FORCE_EVAL:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
OFFLINE="${OFFLINE:-1}"
REQUIRE_TRAINING_MARKER="${REQUIRE_TRAINING_MARKER:-1}"
# Training -> evaluation preflight intentionally permits a model that has not
# been produced yet.  Standalone evaluation sets this flag to 1 so preflight
# validates the user-selected checkpoint immediately instead of deferring it.
REQUIRE_MODEL_ON_PREFLIGHT="${REQUIRE_MODEL_ON_PREFLIGHT:-0}"
MODEL_COMPLETION_MARKER="${MODEL_COMPLETION_MARKER:-TRAINING_COMPLETE.json}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# All suites run one after another on these eight devices.  The three
# time-series runners use four replicated two-GPU engines for throughput;
# tinyBenchmarks uses one eight-way tensor-parallel engine.
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-8}"
TS_GPUS_PER_PROCESS="${TS_GPUS_PER_PROCESS:-2}"

TSR_PROMPT_MODE="${TSR_PROMPT_MODE:-answer_only}"
TSR_MAX_MODEL_LEN="${TSR_MAX_MODEL_LEN:-12288}"
TSR_REQUEST_CHUNK_SIZE="${TSR_REQUEST_CHUNK_SIZE:-128}"
case "$TSR_PROMPT_MODE" in
    official)
        TSR_MAX_NEW_TOKENS="${TSR_MAX_NEW_TOKENS:-512}"
        TSR_BATCH_SIZE="${TSR_BATCH_SIZE:-1}"
        TSR_TEMPERATURE=1.0
        TSR_MAX_RETRIES=10
        TSR_MAX_INPUT_TOKENS=8000
        ;;
    json_reasoning)
        TSR_MAX_NEW_TOKENS="${TSR_MAX_NEW_TOKENS:-256}"
        TSR_BATCH_SIZE="${TSR_BATCH_SIZE:-1}"
        TSR_TEMPERATURE=0.0
        TSR_MAX_RETRIES=1
        TSR_MAX_INPUT_TOKENS=8000
        ;;
    answer_only)
        TSR_MAX_NEW_TOKENS="${TSR_MAX_NEW_TOKENS:-8}"
        TSR_BATCH_SIZE="${TSR_BATCH_SIZE:-16}"
        TSR_TEMPERATURE=0.0
        TSR_MAX_RETRIES=0
        TSR_MAX_INPUT_TOKENS=0
        ;;
    *)
        echo "TSR_PROMPT_MODE must be answer_only, official, or json_reasoning." >&2
        exit 2
        ;;
esac

TINY_MAX_MODEL_LEN="${TINY_MAX_MODEL_LEN:-6000}"
TINY_REQUEST_CHUNK_SIZE="${TINY_REQUEST_CHUNK_SIZE:-16}"
TINY_GPU_MEMORY_UTILIZATION="${TINY_GPU_MEMORY_UTILIZATION:-0.70}"
TINY_DATA_PARTITION="${TINY_DATA_PARTITION:-all}"
TINY_PARTITION_SEED="${TINY_PARTITION_SEED:-42}"
TINY_DTYPE=auto
TINY_FORGETTING_THRESHOLD_PP=5.0

HAYSTACK_MAX_MODEL_LEN="${HAYSTACK_MAX_MODEL_LEN:-40960}"
HAYSTACK_MAX_NEW_TOKENS="${HAYSTACK_MAX_NEW_TOKENS:-500}"
HAYSTACK_BATCH_SIZE="${HAYSTACK_BATCH_SIZE:-1}"
HAYSTACK_REQUEST_CHUNK_SIZE="${HAYSTACK_REQUEST_CHUNK_SIZE:-8}"
HAYSTACK_SPLIT="${HAYSTACK_SPLIT:-test}"

EXAM_MAX_MODEL_LEN="${EXAM_MAX_MODEL_LEN:-8192}"
EXAM_MAX_NEW_TOKENS="${EXAM_MAX_NEW_TOKENS:-1024}"
EXAM_BATCH_SIZE="${EXAM_BATCH_SIZE:-8}"
EXAM_REQUEST_CHUNK_SIZE="${EXAM_REQUEST_CHUNK_SIZE:-64}"
EXAM_MAX_CONCEPTS=3

for flag_name in \
    FORCE_EVAL PREFLIGHT_ONLY OFFLINE REQUIRE_TRAINING_MARKER \
    REQUIRE_MODEL_ON_PREFLIGHT; do
    flag_value="${!flag_name}"
    [[ "$flag_value" == "0" || "$flag_value" == "1" ]] || {
        echo "$flag_name must be 0 or 1, got: $flag_value" >&2
        exit 2
    }
done
case "$MODEL_COMPLETION_MARKER" in
    TRAINING_COMPLETE.json|STAGE1_COMPLETE.json) ;;
    *)
        echo "MODEL_COMPLETION_MARKER must be TRAINING_COMPLETE.json or STAGE1_COMPLETE.json." >&2
        exit 2
        ;;
esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer." >&2; exit 2; }
[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || { echo "MAX_SAMPLES must be non-negative." >&2; exit 2; }
[[ -z "$DATA_VERSION" || "$DATA_VERSION" =~ ^datav[0-9]+$ ]] || {
    echo "DATA_VERSION must be empty or use the canonical datavN form." >&2
    exit 2
}
[[ -z "$DATASET_SNAPSHOT_HASH" || "$DATASET_SNAPSHOT_HASH" =~ ^[0-9a-fA-F]{64}$ ]] || {
    echo "DATASET_SNAPSHOT_HASH must be empty or a 64-character SHA256 hex digest." >&2
    exit 2
}
[[ "$EVAL_NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_NUM_GPUS must be positive." >&2; exit 2; }
[[ "$TS_GPUS_PER_PROCESS" =~ ^[1-9][0-9]*$ ]] || { echo "TS_GPUS_PER_PROCESS must be positive." >&2; exit 2; }
[[ "$TINY_PARTITION_SEED" =~ ^[0-9]+$ ]] || { echo "TINY_PARTITION_SEED must be non-negative." >&2; exit 2; }
[[ "$TSR_MAX_RETRIES" =~ ^[0-9]+$ ]] || { echo "TSR_MAX_RETRIES must be non-negative." >&2; exit 2; }
[[ "$TSR_MAX_INPUT_TOKENS" =~ ^[0-9]+$ ]] || { echo "TSR_MAX_INPUT_TOKENS must be non-negative." >&2; exit 2; }
[[ "$TSR_TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "TSR_TEMPERATURE must be a non-negative number." >&2; exit 2; }
[[ "$TINY_DTYPE" == "auto" ]] || { echo "TINY_DTYPE is fixed to auto for this protocol." >&2; exit 2; }
[[ "$TINY_FORGETTING_THRESHOLD_PP" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "TINY_FORGETTING_THRESHOLD_PP must be a non-negative number." >&2
    exit 2
}
[[ "$EXAM_MAX_CONCEPTS" =~ ^[1-9][0-9]*$ ]] || { echo "EXAM_MAX_CONCEPTS must be positive." >&2; exit 2; }
case "$TINY_DATA_PARTITION" in
    all|search-dev|final-test) ;;
    *) echo "TINY_DATA_PARTITION must be all, search-dev, or final-test." >&2; exit 2 ;;
esac
case "$HAYSTACK_SPLIT" in
    train|validation|test) ;;
    *) echo "HAYSTACK_SPLIT must be train, validation, or test." >&2; exit 2 ;;
esac
(( EVAL_NUM_GPUS == 8 )) || { echo "This runner is fixed to eight-GPU evaluation." >&2; exit 2; }
(( EVAL_NUM_GPUS % TS_GPUS_PER_PROCESS == 0 )) || {
    echo "EVAL_NUM_GPUS must be divisible by TS_GPUS_PER_PROCESS." >&2
    exit 2
}
[[ "$MODEL_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "MODEL_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "RUN_ID may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
}

SELECTED_SUITES=()
selected_suite_csv=","
IFS=',' read -r -a requested_suites <<< "$BENCHMARKS"
for requested_suite in "${requested_suites[@]}"; do
    requested_suite="${requested_suite//[[:space:]]/}"
    case "$requested_suite" in
        tsrbench|timeseriesexam|ts_haystack|tinybenchmarks) ;;
        "") echo "BENCHMARKS contains an empty suite name: $BENCHMARKS" >&2; exit 2 ;;
        *) echo "Unsupported benchmark '$requested_suite' in BENCHMARKS=$BENCHMARKS" >&2; exit 2 ;;
    esac
    [[ "$selected_suite_csv" != *",$requested_suite,"* ]] || {
        echo "Duplicate benchmark '$requested_suite' in BENCHMARKS=$BENCHMARKS" >&2
        exit 2
    }
    SELECTED_SUITES+=("$requested_suite")
    selected_suite_csv+="$requested_suite,"
done

suite_selected() {
    local wanted="$1" selected
    for selected in "${SELECTED_SUITES[@]}"; do
        [[ "$selected" == "$wanted" ]] && return 0
    done
    return 1
}

require_dir() {
    local label="$1" path="$2"
    [[ -d "$path" ]] || { echo "$label directory not found: $path" >&2; exit 1; }
}

require_file() {
    local label="$1" path="$2"
    [[ -f "$path" ]] || { echo "$label file not found: $path" >&2; exit 1; }
}

require_nonempty_file() {
    local label="$1" path="$2"
    [[ -s "$path" ]] || { echo "$label is missing or empty: $path" >&2; exit 1; }
}

require_model_weights() {
    "$PYTHON_BIN" - "$1" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
patterns = (
    "pytorch_model*.bin",
    "model*.safetensors",
    "adapter_model*.bin",
    "adapter_model*.safetensors",
)
weights = {
    path
    for pattern in patterns
    for path in root.glob(pattern)
    if path.is_file() and path.stat().st_size > 0
}
if not weights:
    raise SystemExit(f"No non-empty model weights found in {root}")
print(f"Validated {len(weights)} non-empty model weight file(s) in {root}")
PY
}

require_dir "ChatTS project" "$PROJECT_ROOT"
require_dir "Chronos-2 backbone" "$CHRONOS2_MODEL_PATH"
require_file "Encoder inspector" "$PROJECT_ROOT/scripts/inspect_chatts_ts_encoder_checkpoints.py"
ARTIFACT_HELPER="$PROJECT_ROOT/scripts/chatts_benchmark_artifacts.py"
require_file "Benchmark artifact helper" "$ARTIFACT_HELPER"
require_file "vLLM model adapter" "$PROJECT_ROOT/chatts/vllm/chatts_vllm.py"

if suite_selected tsrbench; then
    require_dir "TSRBench dataset root" "$TSRBENCH_DATASET_ROOT"
    require_file "TSRBench runner" "$PROJECT_ROOT/scripts/run_chatts_tsrbench.sh"
    require_file "TSRBench evaluator" "$PROJECT_ROOT/scripts/evaluate_tsrbench.py"
    require_file "TSRBench inference module" "$PROJECT_ROOT/chatts/utils/inference_tsrbench_vllm.py"
    require_file "TSRBench JSON response validator" "$PROJECT_ROOT/chatts/utils/tsrbench_trace.py"
    tsr_probe="$(find "$TSRBENCH_DATASET_ROOT" -type f -name perception.jsonl -print -quit)"
    [[ -n "$tsr_probe" ]] || {
        echo "Cannot find perception.jsonl under $TSRBENCH_DATASET_ROOT" >&2
        exit 1
    }
fi
if suite_selected tinybenchmarks; then
    require_dir "tinyBenchmarks dataset root" "$TINYBENCH_DATASET_ROOT"
    require_file "tinyBenchmarks runner" "$PROJECT_ROOT/scripts/run_chatts_tinybenchmarks_mcq.sh"
    require_file "tinyBenchmarks inference module" "$PROJECT_ROOT/chatts/utils/inference_tinybenchmarks_mcq_vllm.py"
fi
if suite_selected ts_haystack; then
    require_file "TS-Haystack registry" "$TS_HAYSTACK_ROOT/src/datasets/registry.py"
    require_dir "TS-Haystack data" "$TS_HAYSTACK_ROOT/data"
    require_file "TS-Haystack runner" "$PROJECT_ROOT/scripts/run_chatts_ts_haystack.sh"
    require_file "TS-Haystack evaluator" "$PROJECT_ROOT/scripts/evaluate_ts_haystack.py"
    require_file "TS-Haystack inference module" "$PROJECT_ROOT/chatts/utils/inference_ts_haystack_vllm.py"
fi
if suite_selected timeseriesexam; then
    require_file "TimeSeriesExam concepts" "$TIMESERIESEXAM_ROOT/evaluate/concepts.py"
    require_file "TimeSeriesExam dataset" "$TIMESERIESEXAM_DATA_FILE"
    require_file "TimeSeriesExam runner" "$PROJECT_ROOT/scripts/run_chatts_timeseriesexam.sh"
    require_file "TimeSeriesExam evaluator" "$PROJECT_ROOT/scripts/evaluate_timeseriesexam.py"
    require_file "TimeSeriesExam inference module" "$PROJECT_ROOT/chatts/utils/inference_timeseriesexam_vllm.py"
fi

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
        require_nonempty_file "Declared model completion marker" "$MODEL_PATH/$MODEL_COMPLETION_MARKER"
        if [[ "$MODEL_COMPLETION_MARKER" == "STAGE1_COMPLETE.json" ]]; then
            require_nonempty_file "Stage1 best-model manifest" "$MODEL_PATH/best_model_manifest.json"
        fi
    fi
    require_model_weights "$MODEL_PATH"
elif [[ "$PREFLIGHT_ONLY" == "1" && "$REQUIRE_MODEL_ON_PREFLIGHT" == "0" ]]; then
    echo "Note: final model does not exist yet; model validation is deferred until training completes: $MODEL_PATH"
else
    echo "Final model config not found: $MODEL_PATH/config.json" >&2
    exit 1
fi

echo "============================================================"
echo " ChatTS four-suite sequential eight-GPU evaluation"
echo " Model:          $MODEL_PATH"
echo " Model name:     $MODEL_NAME"
echo " Run ID:         $RUN_ID"
echo " Data version:   ${DATA_VERSION:-<none>}"
echo " Encoder:        chronos2 ($CHRONOS2_MODEL_PATH)"
echo " Seed:           $SEED"
echo " GPU allocation: $EVAL_GPUS (exclusive for each suite)"
echo " TS engines:     $((EVAL_NUM_GPUS / TS_GPUS_PER_PROCESS)) x ${TS_GPUS_PER_PROCESS}-GPU"
echo " Max samples:    $MAX_SAMPLES (0 means full benchmark)"
echo " Benchmarks:     ${SELECTED_SUITES[*]}"
echo " Protocol hash:  ${EVAL_PROTOCOL_HASH:-derived from code and arguments}"
echo " Model marker:   $MODEL_COMPLETION_MARKER"
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
# Explicit zeroes matter: a long-lived container may otherwise leak offline
# flags from a previous run into an OFFLINE=0 protocol.
export HF_HUB_OFFLINE="$OFFLINE"
export TRANSFORMERS_OFFLINE="$OFFLINE"

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
        TEMPERATURE="$TSR_TEMPERATURE" \
        MAX_RETRIES="$TSR_MAX_RETRIES" \
        MAX_INPUT_TOKENS="$TSR_MAX_INPUT_TOKENS" \
        CHATTS_VLLM_MAX_MODEL_LEN="$TSR_MAX_MODEL_LEN" \
        MAX_PROCESSED_INPUT_TOKENS="$((TSR_MAX_MODEL_LEN - TSR_MAX_NEW_TOKENS))" \
        ENABLE_THINKING=0 \
        TS_ENCODER_TYPE=chronos2 \
        CHATTS_TS_ENCODER_TYPE=chronos2 \
        CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH" \
        CHATTS_CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH" \
        SEED="$SEED" \
        FORCE_INFERENCE="${SUITE_FORCE_EVAL:-$FORCE_EVAL}" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/scripts/run_chatts_tsrbench.sh"
}

run_tinybenchmarks() {
    local -a command=(
        bash "$PROJECT_ROOT/scripts/run_chatts_tinybenchmarks_mcq.sh"
        --model "$MODEL_NAME=$MODEL_PATH"
        --baseline "$MODEL_NAME"
    )
    if [[ "${SUITE_FORCE_EVAL:-$FORCE_EVAL}" == "1" ]]; then
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
        DTYPE="$TINY_DTYPE" \
        MAX_SAMPLES="$MAX_SAMPLES" \
        SUMMARY_ONLY=0 \
        ALLOW_SIZE_MISMATCH=0 \
        FORGETTING_THRESHOLD_PP="$TINY_FORGETTING_THRESHOLD_PP" \
        CHATTS_TS_ENCODER_TYPE=chronos2 \
        CHATTS_CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH" \
        SEED="$SEED" \
        OFFLINE="$OFFLINE" \
        DATA_PARTITION="$TINY_DATA_PARTITION" \
        PARTITION_SEED="$TINY_PARTITION_SEED" \
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
        SPLIT="$HAYSTACK_SPLIT" \
        NUM_GPUS="$EVAL_NUM_GPUS" \
        NUM_GPUS_PER_PROCESS="$TS_GPUS_PER_PROCESS" \
        BATCH_SIZE="$HAYSTACK_BATCH_SIZE" \
        REQUEST_CHUNK_SIZE="$HAYSTACK_REQUEST_CHUNK_SIZE" \
        CHATTS_VLLM_MAX_MODEL_LEN="$HAYSTACK_MAX_MODEL_LEN" \
        MAX_NEW_TOKENS="$HAYSTACK_MAX_NEW_TOKENS" \
        MAX_PROCESSED_INPUT_TOKENS="$((HAYSTACK_MAX_MODEL_LEN - HAYSTACK_MAX_NEW_TOKENS))" \
        MAX_SAMPLES="$MAX_SAMPLES" \
        TEMPERATURE=0.0 \
        TS_ENCODER_TYPE=chronos2 \
        CHATTS_TS_ENCODER_TYPE=chronos2 \
        CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH" \
        CHATTS_CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH" \
        SEED="$SEED" \
        ENABLE_THINKING=0 \
        FORCE_INFERENCE="${SUITE_FORCE_EVAL:-$FORCE_EVAL}" \
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
        MAX_CONCEPTS="$EXAM_MAX_CONCEPTS" \
        TS_ENCODER_TYPE=chronos2 \
        CHATTS_TS_ENCODER_TYPE=chronos2 \
        CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH" \
        CHATTS_CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH" \
        SEED="$SEED" \
        ADD_QUESTION_HINT=1 \
        ADD_CONCEPTS=1 \
        ADD_EXAMPLES=1 \
        ENABLE_THINKING=0 \
        FORCE_INFERENCE="${SUITE_FORCE_EVAL:-$FORCE_EVAL}" \
        SCORE_ONLY=0 \
        OFFLINE="$OFFLINE" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/scripts/run_chatts_timeseriesexam.sh"
}

suite_summary_file() {
    local name="$1" suite_output="$2"
    case "$name" in
        tsrbench) echo "$suite_output/tsrbench_summary_${MODEL_NAME}.json" ;;
        tinybenchmarks) echo "$suite_output/$MODEL_NAME/metrics.json" ;;
        ts_haystack) echo "$suite_output/ts_haystack_summary_${MODEL_NAME}.json" ;;
        timeseriesexam)
            echo "$suite_output/${MODEL_NAME}_query_hint_concepts_examples/timeseriesexam_summary_${MODEL_NAME}.json"
            ;;
        *) echo "Unknown suite: $name" >&2; return 2 ;;
    esac
}

suite_artifact() {
    local action="$1" name="$2" suite_output="$3"
    local summary_file
    summary_file="$(suite_summary_file "$name" "$suite_output")"
    local -a command=(
        "$PYTHON_BIN" "$ARTIFACT_HELPER" "$action"
        --suite "$name"
        --model-path "$MODEL_PATH"
        --model-name "$MODEL_NAME"
        --model-component "$CHRONOS2_MODEL_PATH"
        --eval-protocol-hash "$EVAL_PROTOCOL_HASH"
        --data-version "$DATA_VERSION"
        --dataset-snapshot-hash "$DATASET_SNAPSHOT_HASH"
        --protocol-file "$PROJECT_ROOT/scripts/run_all_chatts_benchmarks.sh"
        --protocol-file "$PROJECT_ROOT/chatts/vllm/chatts_vllm.py"
        --protocol-file "$PROJECT_ROOT/chatts/utils/llm_utils.py"
        --protocol "encoder=chronos2"
        --protocol "seed=$SEED"
        --protocol "max_samples=$MAX_SAMPLES"
        --protocol "eval_num_gpus=$EVAL_NUM_GPUS"
        --protocol "ts_gpus_per_process=$TS_GPUS_PER_PROCESS"
    )

    case "$name" in
        tsrbench)
            command+=(
                --data-path "$TSRBENCH_DATASET_ROOT"
                --protocol-file "$PROJECT_ROOT/scripts/run_chatts_tsrbench.sh"
                --protocol-file "$PROJECT_ROOT/scripts/evaluate_tsrbench.py"
                --protocol-file "$PROJECT_ROOT/chatts/utils/inference_tsrbench_vllm.py"
                --protocol-file "$PROJECT_ROOT/chatts/utils/tsrbench_trace.py"
                --protocol "datasets=all"
                --protocol "prompt_mode=$TSR_PROMPT_MODE"
                --protocol "max_model_len=$TSR_MAX_MODEL_LEN"
                --protocol "max_new_tokens=$TSR_MAX_NEW_TOKENS"
                --protocol "max_processed_input_tokens=$((TSR_MAX_MODEL_LEN - TSR_MAX_NEW_TOKENS))"
                --protocol "batch_size=$TSR_BATCH_SIZE"
                --protocol "request_chunk_size=$TSR_REQUEST_CHUNK_SIZE"
                --protocol "temperature=$TSR_TEMPERATURE"
                --protocol "max_retries=$TSR_MAX_RETRIES"
                --protocol "max_input_tokens=$TSR_MAX_INPUT_TOKENS"
                --protocol "ts_encoder_type=chronos2"
                --protocol "enable_thinking=0"
            )
            ;;
        tinybenchmarks)
            command+=(
                --data-path "$TINYBENCH_DATASET_ROOT"
                --protocol-file "$PROJECT_ROOT/scripts/run_chatts_tinybenchmarks_mcq.sh"
                --protocol-file "$PROJECT_ROOT/scripts/summarize_tinybenchmarks_mcq.py"
                --protocol-file "$PROJECT_ROOT/chatts/utils/inference_tinybenchmarks_mcq_vllm.py"
                --protocol "tasks=tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande"
                --protocol "max_model_len=$TINY_MAX_MODEL_LEN"
                --protocol "request_chunk_size=$TINY_REQUEST_CHUNK_SIZE"
                --protocol "gpu_memory_utilization=$TINY_GPU_MEMORY_UTILIZATION"
                --protocol "dtype=$TINY_DTYPE"
                --protocol "summary_only=0"
                --protocol "allow_size_mismatch=0"
                --protocol "forgetting_threshold_pp=$TINY_FORGETTING_THRESHOLD_PP"
            )
            if [[ "$TINY_DATA_PARTITION" != "all" ]]; then
                command+=(
                    --protocol "data_partition=$TINY_DATA_PARTITION"
                    --protocol "partition_seed=$TINY_PARTITION_SEED"
                )
            fi
            ;;
        ts_haystack)
            command+=(
                --data-path "$TS_HAYSTACK_ROOT/data"
                --data-path "$TS_HAYSTACK_ROOT/src/datasets/registry.py"
                --protocol-file "$PROJECT_ROOT/scripts/run_chatts_ts_haystack.sh"
                --protocol-file "$PROJECT_ROOT/scripts/evaluate_ts_haystack.py"
                --protocol-file "$PROJECT_ROOT/chatts/utils/inference_ts_haystack_vllm.py"
                --protocol-file "$TS_HAYSTACK_ROOT/src"
                --protocol "datasets=all"
                --protocol "tasks=all"
                --protocol "context_lengths=all"
                --protocol "split=$HAYSTACK_SPLIT"
                --protocol "max_model_len=$HAYSTACK_MAX_MODEL_LEN"
                --protocol "max_new_tokens=$HAYSTACK_MAX_NEW_TOKENS"
                --protocol "batch_size=$HAYSTACK_BATCH_SIZE"
                --protocol "request_chunk_size=$HAYSTACK_REQUEST_CHUNK_SIZE"
                --protocol "temperature=0.0"
                --protocol "ts_encoder_type=chronos2"
                --protocol "enable_thinking=0"
            )
            ;;
        timeseriesexam)
            command+=(
                --data-path "$TIMESERIESEXAM_DATA_FILE"
                --data-path "$TIMESERIESEXAM_ROOT/evaluate/concepts.py"
                --protocol-file "$PROJECT_ROOT/scripts/run_chatts_timeseriesexam.sh"
                --protocol-file "$PROJECT_ROOT/scripts/evaluate_timeseriesexam.py"
                --protocol-file "$PROJECT_ROOT/chatts/utils/inference_timeseriesexam_vllm.py"
                --protocol "add_question_hint=1"
                --protocol "add_concepts=1"
                --protocol "add_examples=1"
                --protocol "max_model_len=$EXAM_MAX_MODEL_LEN"
                --protocol "max_new_tokens=$EXAM_MAX_NEW_TOKENS"
                --protocol "batch_size=$EXAM_BATCH_SIZE"
                --protocol "request_chunk_size=$EXAM_REQUEST_CHUNK_SIZE"
                --protocol "temperature=0.0"
                --protocol "max_concepts=$EXAM_MAX_CONCEPTS"
                --protocol "ts_encoder_type=chronos2"
                --protocol "enable_thinking=0"
            )
            ;;
        *) echo "Unknown suite: $name" >&2; return 2 ;;
    esac

    case "$action" in
        cache-status) command+=(--manifest "$suite_output/.chatts_benchmark_manifest.json") ;;
        write-suite-manifest)
            command+=(
                --output-dir "$suite_output"
                --summary-file "$summary_file"
                --run-id "$RUN_ID"
            )
            ;;
        *) echo "Unknown artifact action: $action" >&2; return 2 ;;
    esac
    "${command[@]}"
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
    local status exit_code cache_message cache_code
    echo
    echo "==================== starting $name ===================="
    echo "Exclusive GPUs: $EVAL_GPUS"
    echo "Log: $log_file"
    if [[ "$FORCE_EVAL" != "1" ]]; then
        if cache_message="$(suite_artifact cache-status "$name" "$suite_output" 2>&1)"; then
            echo "$cache_message"
            echo "[$RUN_ID] Reused fingerprint-matched $name result." | tee -a "$log_file"
            status=CACHED
            exit_code=0
        else
            cache_code=$?
            echo "$cache_message"
            if (( cache_code != 1 )); then
                echo "Cannot compute the $name cache fingerprint." | tee "$log_file" >&2
                status=FAIL
                exit_code="$cache_code"
                FAILED_SUITES=$((FAILED_SUITES + 1))
            fi
        fi
    fi

    if [[ -z "${status:-}" ]]; then
        # A cache miss forces the child runner too, preventing its legacy
        # path-only cache from reusing stale predictions.
        if SUITE_FORCE_EVAL=1 "$@" 2>&1 | tee "$log_file"; then
            if suite_artifact write-suite-manifest "$name" "$suite_output" 2>&1 | tee -a "$log_file"; then
                status=PASS
                exit_code=0
            else
                exit_code=$?
                status=FAIL
                FAILED_SUITES=$((FAILED_SUITES + 1))
            fi
        else
            exit_code=$?
            status=FAIL
            FAILED_SUITES=$((FAILED_SUITES + 1))
        fi
    fi

    SUITE_NAMES+=("$name")
    SUITE_STATUSES+=("$status")
    SUITE_CODES+=("$exit_code")
    SUITE_OUTPUTS+=("$suite_output")
    SUITE_LOGS+=("$log_file")
    echo "==================== $name: $status ===================="
}

for selected_suite in "${SELECTED_SUITES[@]}"; do
    case "$selected_suite" in
        tsrbench) run_step tsrbench "$OUTPUT_ROOT/tsrbench" run_tsrbench ;;
        tinybenchmarks) run_step tinybenchmarks "$OUTPUT_ROOT/tinybenchmarks" run_tinybenchmarks ;;
        ts_haystack) run_step ts_haystack "$OUTPUT_ROOT/ts_haystack" run_ts_haystack ;;
        timeseriesexam) run_step timeseriesexam "$OUTPUT_ROOT/timeseriesexam" run_timeseriesexam ;;
    esac
done

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
    echo "- Run ID: \`$RUN_ID\`"
    echo "- Seed: \`$SEED\`"
    echo "- Encoder: \`chronos2\`"
    echo "- Benchmarks: \`${SELECTED_SUITES[*]}\`"
    echo "- Evaluation protocol: \`${EVAL_PROTOCOL_HASH:-derived SHA256}\`"
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

AGGREGATE_ARGS=()
for index in "${!SUITE_NAMES[@]}"; do
    AGGREGATE_ARGS+=(
        --suite-manifest
        "${SUITE_NAMES[$index]}=${SUITE_OUTPUTS[$index]}/.chatts_benchmark_manifest.json"
    )
done

aggregate_code=0
"$PYTHON_BIN" "$ARTIFACT_HELPER" aggregate \
    --status-file "$STATUS_FILE" \
    "${AGGREGATE_ARGS[@]}" \
    --metrics-file "$METRICS_FILE" \
    --run-manifest-file "$MANIFEST_FILE" \
    --run-id "$RUN_ID" \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --seed "$SEED" \
    --max-samples "$MAX_SAMPLES" \
    --force-eval "$FORCE_EVAL" \
    --output-root "$OUTPUT_ROOT" \
    --eval-protocol-hash "$EVAL_PROTOCOL_HASH" \
    --data-version "$DATA_VERSION" \
    --dataset-snapshot-hash "$DATASET_SNAPSHOT_HASH" || aggregate_code=$?
if (( aggregate_code != 0 && FAILED_SUITES == 0 )); then
    echo "Metric aggregation failed with exit code $aggregate_code." >&2
    FAILED_SUITES=$((FAILED_SUITES + 1))
fi

echo
echo "==================== benchmark status ===================="
column -t -s $'\t' "$STATUS_FILE" 2>/dev/null || cat "$STATUS_FILE"
echo "Summary:  $SUMMARY_FILE"
echo "Manifest: $MANIFEST_FILE"
echo "Metrics:  $METRICS_FILE"
echo "Outputs:  $OUTPUT_ROOT"
echo "=========================================================="

if (( FAILED_SUITES > 0 )); then
    exit 1
fi

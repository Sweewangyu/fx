#!/usr/bin/env bash
set -Eeuo pipefail

# Run one ChatTS checkpoint sequentially on all four evaluation suites:
# TSRBench, tinyBenchmarks MCQ, TS-Haystack, and TimeSeriesExam.
#
# The four child runners retain their own official prompt/scoring behavior.
# This orchestrator only validates paths, isolates output/log directories,
# passes one consistent checkpoint/encoder configuration, and records status.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS/ChatTS-main}"
MODEL_PATH="${MODEL_PATH:-/share/airesearch/data/finiverse/output/ChatTS-Qwen3-1.7B-PR-onestep/onestep_chronos2_lr1e-5}"
MODEL_NAME="${MODEL_NAME:-chatts-1.7B-msxf-datav1}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/share/airesearch/data/finiverse/model/Qwen3-1.7B}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# This evaluation run is intentionally fixed to Chronos-2.  Only its local
# backbone path is configurable; no checkpoint-type guessing is performed.
readonly ENCODER_TYPE="chronos2"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-/workspace/chronos2}"

# Benchmark code/data roots.
TSRBENCH_ROOT="${TSRBENCH_ROOT:-/share/airesearch/data/finiverse/TSRBench-dataset}"
TSRBENCH_DATASET_ROOT="${TSRBENCH_DATASET_ROOT:-${TSRBENCH_ROOT}}"
TINYBENCH_DATASET_ROOT="${TINYBENCH_DATASET_ROOT:-/share/airesearch/data/finiverse/tyb}"
TS_HAYSTACK_ROOT="${TS_HAYSTACK_ROOT:-/workspace/TS-Haystack}"
TIMESERIESEXAM_ROOT="${TIMESERIESEXAM_ROOT:-/workspace/TimeSeriesExam}"
TIMESERIESEXAM_DATA_FILE="${TIMESERIESEXAM_DATA_FILE:-${TIMESERIESEXAM_ROOT}/output/round_3_folder/qa_dataset.json}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/share/airesearch/data/finiverse/evaluation/all-benchmarks/${MODEL_NAME}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"

# Suite switches.
RUN_TSRBENCH="${RUN_TSRBENCH:-1}"
RUN_TINYBENCH="${RUN_TINYBENCH:-1}"
RUN_TS_HAYSTACK="${RUN_TS_HAYSTACK:-1}"
RUN_TIMESERIESEXAM="${RUN_TIMESERIESEXAM:-1}"
RUN_TINY_BASELINE="${RUN_TINY_BASELINE:-1}"

# Resume by default. FORCE_ALL=1 deletes nothing, but asks every child runner
# to overwrite/recompute its selected result files.
FORCE_ALL="${FORCE_ALL:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
OFFLINE="${OFFLINE:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

# GPU policy: the three time-series suites use replicated vLLM engines;
# tinyBenchmarks uses one tensor-parallel engine per checkpoint, sequentially.
AVAILABLE_GPUS="${AVAILABLE_GPUS_OVERRIDE:-$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')}"
GENERAL_NUM_GPUS="${GENERAL_NUM_GPUS:-${AVAILABLE_GPUS}}"
if [[ -n "${GENERAL_GPUS_PER_MODEL:-}" ]]; then
    GENERAL_GPUS_PER_MODEL="$GENERAL_GPUS_PER_MODEL"
elif (( GENERAL_NUM_GPUS >= 2 && GENERAL_NUM_GPUS % 2 == 0 )); then
    GENERAL_GPUS_PER_MODEL=2
else
    GENERAL_GPUS_PER_MODEL=1
fi
TINY_NUM_GPUS="${TINY_NUM_GPUS:-1}"

# Per-suite runtime controls.
TSR_PROMPT_MODE="${TSR_PROMPT_MODE:-answer_only}"
TSR_MAX_MODEL_LEN="${TSR_MAX_MODEL_LEN:-12288}"
case "$TSR_PROMPT_MODE" in
    answer_only)
        TSR_MAX_NEW_TOKENS="${TSR_MAX_NEW_TOKENS:-8}"
        TSR_BATCH_SIZE="${TSR_BATCH_SIZE:-16}"
        ;;
    official)
        TSR_MAX_NEW_TOKENS="${TSR_MAX_NEW_TOKENS:-512}"
        TSR_BATCH_SIZE="${TSR_BATCH_SIZE:-1}"
        ;;
    *)
        echo "TSR_PROMPT_MODE must be answer_only or official, got: $TSR_PROMPT_MODE" >&2
        exit 2
        ;;
esac
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

for flag_name in RUN_TSRBENCH RUN_TINYBENCH RUN_TS_HAYSTACK RUN_TIMESERIESEXAM RUN_TINY_BASELINE FORCE_ALL CONTINUE_ON_ERROR OFFLINE PREFLIGHT_ONLY; do
    flag_value="${!flag_name}"
    if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
        echo "$flag_name must be 0 or 1, got: $flag_value" >&2
        exit 2
    fi
done

if [[ ! "$MODEL_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "MODEL_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
if (( AVAILABLE_GPUS < 1 )); then
    echo "PyTorch sees no CUDA GPU." >&2
    exit 1
fi
if (( GENERAL_NUM_GPUS < 1 || GENERAL_NUM_GPUS > AVAILABLE_GPUS )); then
    echo "GENERAL_NUM_GPUS=$GENERAL_NUM_GPUS is incompatible with AVAILABLE_GPUS=$AVAILABLE_GPUS." >&2
    exit 2
fi
if (( GENERAL_GPUS_PER_MODEL < 1 || GENERAL_NUM_GPUS % GENERAL_GPUS_PER_MODEL != 0 )); then
    echo "GENERAL_NUM_GPUS must be divisible by GENERAL_GPUS_PER_MODEL." >&2
    exit 2
fi
if (( TINY_NUM_GPUS < 1 || TINY_NUM_GPUS > AVAILABLE_GPUS )); then
    echo "TINY_NUM_GPUS=$TINY_NUM_GPUS is incompatible with AVAILABLE_GPUS=$AVAILABLE_GPUS." >&2
    exit 2
fi

require_dir() {
    local label="$1"
    local path="$2"
    if [[ ! -d "$path" ]]; then
        echo "$label directory not found: $path" >&2
        exit 1
    fi
}

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "$path" ]]; then
        echo "$label file not found: $path" >&2
        exit 1
    fi
}

require_dir "ChatTS project" "$PROJECT_ROOT"
require_file "ChatTS model config" "$MODEL_PATH/config.json"
require_file "encoder inspector" "$PROJECT_ROOT/scripts/inspect_chatts_ts_encoder_checkpoints.py"

require_dir "Chronos-2 backbone" "$CHRONOS2_MODEL_PATH"

if [[ "$RUN_TSRBENCH" == "1" ]]; then
    require_dir "TSRBench dataset root" "$TSRBENCH_DATASET_ROOT"
    require_file "TSRBench runner" "$PROJECT_ROOT/scripts/run_chatts_tsrbench.sh"
    tsr_probe="$(find "$TSRBENCH_DATASET_ROOT" -type f -name perception.jsonl -print -quit)"
    if [[ -z "$tsr_probe" ]]; then
        echo "Cannot find perception.jsonl under TSRBENCH_DATASET_ROOT=$TSRBENCH_DATASET_ROOT" >&2
        exit 1
    fi
fi

if [[ "$RUN_TINYBENCH" == "1" ]]; then
    require_dir "tinyBenchmarks dataset root" "$TINYBENCH_DATASET_ROOT"
    require_file "tinyBenchmarks runner" "$PROJECT_ROOT/scripts/run_chatts_tinybenchmarks_mcq.sh"
    if [[ "$RUN_TINY_BASELINE" == "1" ]]; then
        require_file "base model config" "$BASE_MODEL_PATH/config.json"
    fi
fi

if [[ "$RUN_TS_HAYSTACK" == "1" ]]; then
    require_file "TS-Haystack registry" "$TS_HAYSTACK_ROOT/src/datasets/registry.py"
    require_dir "TS-Haystack data" "$TS_HAYSTACK_ROOT/data"
    require_file "TS-Haystack runner" "$PROJECT_ROOT/scripts/run_chatts_ts_haystack.sh"
fi

if [[ "$RUN_TIMESERIESEXAM" == "1" ]]; then
    require_file "TimeSeriesExam concepts" "$TIMESERIESEXAM_ROOT/evaluate/concepts.py"
    require_file "TimeSeriesExam dataset" "$TIMESERIESEXAM_DATA_FILE"
    require_file "TimeSeriesExam runner" "$PROJECT_ROOT/scripts/run_chatts_timeseriesexam.sh"
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
LOG_ROOT="$(cd "$LOG_ROOT" && pwd)"

export CHATTS_TS_ENCODER_TYPE="$ENCODER_TYPE"
export CHATTS_CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ "$OFFLINE" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

STEP_NAMES=()
STEP_RESULTS=()
FAILED_STEPS=0

print_summary() {
    echo
    echo "==================== benchmark status ===================="
    local index
    for index in "${!STEP_NAMES[@]}"; do
        printf '%-24s %s\n' "${STEP_NAMES[$index]}" "${STEP_RESULTS[$index]}"
    done
    echo "Logs:    $LOG_ROOT"
    echo "Outputs: $OUTPUT_ROOT"
    echo "=========================================================="
}

run_step() {
    local name="$1"
    shift
    local log_file="$LOG_ROOT/${name}.log"
    local status

    echo
    echo "==================== starting $name ===================="
    echo "Log: $log_file"
    if "$@" 2>&1 | tee "$log_file"; then
        status="PASS"
    else
        local exit_code=$?
        status="FAIL(exit=$exit_code)"
        FAILED_STEPS=$((FAILED_STEPS + 1))
    fi
    STEP_NAMES+=("$name")
    STEP_RESULTS+=("$status")
    echo "==================== $name: $status ===================="

    if [[ "$status" != "PASS" && "$CONTINUE_ON_ERROR" != "1" ]]; then
        print_summary
        exit 1
    fi
}

echo "=========================================================="
echo " ChatTS all-benchmark evaluation"
echo " Model:              $MODEL_PATH"
echo " Model name:         $MODEL_NAME"
echo " Encoder:            $ENCODER_TYPE"
echo " Available GPUs:     $AVAILABLE_GPUS"
echo " TS suite GPUs:      $GENERAL_NUM_GPUS ($GENERAL_GPUS_PER_MODEL per engine)"
echo " tinyBench GPUs:     $TINY_NUM_GPUS"
echo " Force recompute:    $FORCE_ALL"
echo " Continue on error:  $CONTINUE_ON_ERROR"
echo " Output root:        $OUTPUT_ROOT"
echo "=========================================================="

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "Preflight passed. No benchmark was started."
    exit 0
fi

if [[ "$RUN_TSRBENCH" == "1" ]]; then
    run_step "01_tsrbench" env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        TSRBENCH_ROOT="$TSRBENCH_ROOT" \
        DATASET_ROOT="$TSRBENCH_DATASET_ROOT" \
        OUTPUT_ROOT="$OUTPUT_ROOT/tsrbench" \
        MODEL_PATH="$MODEL_PATH" \
        MODEL_NAME="$MODEL_NAME" \
        DATASETS=all \
        PROMPT_MODE="$TSR_PROMPT_MODE" \
        NUM_GPUS="$GENERAL_NUM_GPUS" \
        NUM_GPUS_PER_PROCESS="$GENERAL_GPUS_PER_MODEL" \
        BATCH_SIZE="$TSR_BATCH_SIZE" \
        REQUEST_CHUNK_SIZE="$TSR_REQUEST_CHUNK_SIZE" \
        MAX_SAMPLES=0 \
        MAX_NEW_TOKENS="$TSR_MAX_NEW_TOKENS" \
        CHATTS_VLLM_MAX_MODEL_LEN="$TSR_MAX_MODEL_LEN" \
        MAX_PROCESSED_INPUT_TOKENS="$((TSR_MAX_MODEL_LEN - TSR_MAX_NEW_TOKENS))" \
        ENABLE_THINKING=0 \
        FORCE_INFERENCE="$FORCE_ALL" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/scripts/run_chatts_tsrbench.sh"
fi

if [[ "$RUN_TINYBENCH" == "1" ]]; then
    tiny_command=(bash "$PROJECT_ROOT/scripts/run_chatts_tinybenchmarks_mcq.sh")
    if [[ "$RUN_TINY_BASELINE" == "1" ]]; then
        tiny_command+=(--model "base=$BASE_MODEL_PATH")
    fi
    tiny_command+=(--model "chatts=$MODEL_PATH")
    if [[ "$RUN_TINY_BASELINE" == "1" ]]; then
        tiny_command+=(--baseline base)
    else
        tiny_command+=(--baseline chatts)
    fi
    if [[ "$FORCE_ALL" == "1" ]]; then
        tiny_command+=(--force)
    fi

    run_step "02_tinybenchmarks" env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        DATASET_ROOT="$TINYBENCH_DATASET_ROOT" \
        OUTPUT_ROOT="$OUTPUT_ROOT/tinybenchmarks" \
        TASKS_CSV="tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande" \
        NUM_GPUS="$TINY_NUM_GPUS" \
        REQUEST_CHUNK_SIZE="$TINY_REQUEST_CHUNK_SIZE" \
        GPU_MEMORY_UTILIZATION="$TINY_GPU_MEMORY_UTILIZATION" \
        CHATTS_VLLM_MAX_MODEL_LEN="$TINY_MAX_MODEL_LEN" \
        MAX_SAMPLES=0 \
        OFFLINE="$OFFLINE" \
        PYTHON_BIN="$PYTHON_BIN" \
        "${tiny_command[@]}"
fi

if [[ "$RUN_TS_HAYSTACK" == "1" ]]; then
    run_step "03_ts_haystack" env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        TS_HAYSTACK_ROOT="$TS_HAYSTACK_ROOT" \
        OUTPUT_ROOT="$OUTPUT_ROOT/ts_haystack" \
        MODEL_PATH="$MODEL_PATH" \
        MODEL_NAME="$MODEL_NAME" \
        DATASETS=all \
        TASKS=all \
        CONTEXT_LENGTHS=all \
        SPLIT=test \
        NUM_GPUS="$GENERAL_NUM_GPUS" \
        NUM_GPUS_PER_PROCESS="$GENERAL_GPUS_PER_MODEL" \
        BATCH_SIZE="$HAYSTACK_BATCH_SIZE" \
        REQUEST_CHUNK_SIZE="$HAYSTACK_REQUEST_CHUNK_SIZE" \
        CHATTS_VLLM_MAX_MODEL_LEN="$HAYSTACK_MAX_MODEL_LEN" \
        MAX_NEW_TOKENS="$HAYSTACK_MAX_NEW_TOKENS" \
        MAX_PROCESSED_INPUT_TOKENS="$((HAYSTACK_MAX_MODEL_LEN - HAYSTACK_MAX_NEW_TOKENS))" \
        MAX_SAMPLES=0 \
        ENABLE_THINKING=0 \
        FORCE_INFERENCE="$FORCE_ALL" \
        SCORE_ONLY=0 \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/scripts/run_chatts_ts_haystack.sh"
fi

if [[ "$RUN_TIMESERIESEXAM" == "1" ]]; then
    run_step "04_timeseriesexam" env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        TIMESERIESEXAM_ROOT="$TIMESERIESEXAM_ROOT" \
        DATA_FILE_PATH="$TIMESERIESEXAM_DATA_FILE" \
        OUTPUT_ROOT="$OUTPUT_ROOT/timeseriesexam" \
        MODEL_PATH="$MODEL_PATH" \
        MODEL_NAME="$MODEL_NAME" \
        NUM_GPUS="$GENERAL_NUM_GPUS" \
        NUM_GPUS_PER_PROCESS="$GENERAL_GPUS_PER_MODEL" \
        BATCH_SIZE="$EXAM_BATCH_SIZE" \
        REQUEST_CHUNK_SIZE="$EXAM_REQUEST_CHUNK_SIZE" \
        CHATTS_VLLM_MAX_MODEL_LEN="$EXAM_MAX_MODEL_LEN" \
        MAX_NEW_TOKENS="$EXAM_MAX_NEW_TOKENS" \
        MAX_PROCESSED_INPUT_TOKENS="$((EXAM_MAX_MODEL_LEN - EXAM_MAX_NEW_TOKENS))" \
        MAX_SAMPLES=0 \
        ADD_QUESTION_HINT=1 \
        ADD_CONCEPTS=1 \
        ADD_EXAMPLES=1 \
        ENABLE_THINKING=0 \
        FORCE_INFERENCE="$FORCE_ALL" \
        SCORE_ONLY=0 \
        OFFLINE="$OFFLINE" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/scripts/run_chatts_timeseriesexam.sh"
fi

print_summary

echo
echo "Generated summary files:"
find "$OUTPUT_ROOT" -type f \( -name '*summary*.json' -o -name '*summary*.csv' -o -name '*summary*.md' \) -print | sort

if (( FAILED_STEPS > 0 )); then
    exit 1
fi

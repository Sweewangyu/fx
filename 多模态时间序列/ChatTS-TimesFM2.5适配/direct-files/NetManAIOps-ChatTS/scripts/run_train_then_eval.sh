#!/usr/bin/env bash
set -Eeuo pipefail

# Host-side one-click entrypoint.  This script intentionally does not use -it:
# docker exec exits only after each container-side stage has completed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_PYTHON_BIN="${HOST_PYTHON_BIN:-python3}"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/../configs/train_eval_chronos2.yaml}"
CONFIG_LOADER="${CONFIG_LOADER:-${SCRIPT_DIR}/load_train_eval_config.py}"

command -v "$HOST_PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Host Python not found: $HOST_PYTHON_BIN" >&2
    exit 1
}
[[ -f "$CONFIG_LOADER" ]] || { echo "Configuration loader not found: $CONFIG_LOADER" >&2; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "Configuration file not found: $CONFIG_FILE" >&2; exit 1; }

# The loader only emits a fixed whitelist of shell assignments and quotes every
# value. Variables already exported by the caller take precedence over YAML.
CONFIG_EXPORTS="$("$HOST_PYTHON_BIN" "$CONFIG_LOADER" "$CONFIG_FILE")"
eval "$CONFIG_EXPORTS"

TRAIN_CONTAINER="${TRAIN_CONTAINER:-chatts}"
EVAL_CONTAINER="${EVAL_CONTAINER:-ragas}"
TRAIN_PROJECT_ROOT="${TRAIN_PROJECT_ROOT:-/workspace/ChatTS-Training}"
EVAL_PROJECT_ROOT="${EVAL_PROJECT_ROOT:-/workspace/ChatTS/ChatTS-main}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${TRAIN_PROJECT_ROOT}/scripts/full/run_chronos2_best_two_stage.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${EVAL_PROJECT_ROOT}/scripts/run_all_chatts_benchmarks.sh}"

SHARED_ROOT="${SHARED_ROOT:-/share/airesearch/data/finiverse}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${SHARED_ROOT}/model/ChatTS-Qwen3-8B}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${SHARED_ROOT}/output/ChatTS-msxf-8B-datav1}"
SEED="${SEED:-42}"
S1_LR="${S1_LR:-1e-5}"
S2_LR="${S2_LR:-1e-5}"
STAGE1_OUT="${STAGE1_OUT:-${TRAIN_OUTPUT_ROOT}/.stage1_seed${SEED}_s1lr_${S1_LR}}"
FINAL_MODEL_PATH="${FINAL_MODEL_PATH:-${TRAIN_OUTPUT_ROOT}/best_seed${SEED}}"
MODEL_NAME="${MODEL_NAME:-chatts-msxf-8B-datav1-seed${SEED}}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${SHARED_ROOT}/evaluation/all-benchmarks/${MODEL_NAME}}"

TRAIN_CHRONOS2_MODEL_PATH="${TRAIN_CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
EVAL_CHRONOS2_MODEL_PATH="${EVAL_CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
DATASET_DIR="${DATASET_DIR:-${TRAIN_PROJECT_ROOT}/data}"
KEEP_STAGE1="${KEEP_STAGE1:-0}"
# Backward-compatible default. Stage1 mode finalizes and retains the Stage1
# model, then evaluates that exact checkpoint using its Stage1 marker.
PIPELINE_MODE="${PIPELINE_MODE:-full}"
if [[ "$PIPELINE_MODE" == "stage1" ]]; then
    EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-$STAGE1_OUT}"
    MODEL_COMPLETION_MARKER="${MODEL_COMPLETION_MARKER:-STAGE1_COMPLETE.json}"
else
    EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-$FINAL_MODEL_PATH}"
    MODEL_COMPLETION_MARKER="${MODEL_COMPLETION_MARKER:-TRAINING_COMPLETE.json}"
fi

TSRBENCH_ROOT="${TSRBENCH_ROOT:-${SHARED_ROOT}/TSRBench-dataset}"
TINYBENCH_DATASET_ROOT="${TINYBENCH_DATASET_ROOT:-${SHARED_ROOT}/tyb}"
TS_HAYSTACK_ROOT="${TS_HAYSTACK_ROOT:-/workspace/TS-Haystack}"
TIMESERIESEXAM_ROOT="${TIMESERIESEXAM_ROOT:-/workspace/TimeSeriesExam}"
TIMESERIESEXAM_DATA_FILE="${TIMESERIESEXAM_DATA_FILE:-${TIMESERIESEXAM_ROOT}/output/round_3_folder/qa_dataset.json}"
BENCHMARKS="${BENCHMARKS:-tsrbench,tinybenchmarks,ts_haystack,timeseriesexam}"
RUN_ID="${RUN_ID:-train-eval-${MODEL_NAME}}"
EVAL_PROTOCOL_HASH="${EVAL_PROTOCOL_HASH:-}"
HAYSTACK_SPLIT="${HAYSTACK_SPLIT:-test}"
TINY_DATA_PARTITION="${TINY_DATA_PARTITION:-all}"
TINY_PARTITION_SEED="${TINY_PARTITION_SEED:-42}"

FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OFFLINE="${OFFLINE:-1}"
DATA_VERSION="${DATA_VERSION:-}"
DATASET_SNAPSHOT_HASH="${DATASET_SNAPSHOT_HASH:-}"
TRAINING_RECIPE_HASH="${TRAINING_RECIPE_HASH:-}"
TRIAL_ID="${TRIAL_ID:-}"
TRIAL_CONFIG_HASH="${TRIAL_CONFIG_HASH:-}"

# Safe benchmark tuning controls supported by run_all_chatts_benchmarks.sh.
# Keep these defaults synchronized with that runner so older YAML files retain
# exactly the same evaluation protocol.
TSR_PROMPT_MODE="${TSR_PROMPT_MODE:-answer_only}"
TSR_MAX_MODEL_LEN="${TSR_MAX_MODEL_LEN:-12288}"
TSR_REQUEST_CHUNK_SIZE="${TSR_REQUEST_CHUNK_SIZE:-128}"
case "$TSR_PROMPT_MODE" in
    official)
        TSR_MAX_NEW_TOKENS="${TSR_MAX_NEW_TOKENS:-512}"
        TSR_BATCH_SIZE="${TSR_BATCH_SIZE:-1}"
        ;;
    json_reasoning)
        TSR_MAX_NEW_TOKENS="${TSR_MAX_NEW_TOKENS:-256}"
        TSR_BATCH_SIZE="${TSR_BATCH_SIZE:-1}"
        ;;
    answer_only)
        TSR_MAX_NEW_TOKENS="${TSR_MAX_NEW_TOKENS:-8}"
        TSR_BATCH_SIZE="${TSR_BATCH_SIZE:-16}"
        ;;
    *)
        echo "TSR_PROMPT_MODE must be answer_only, official, or json_reasoning." >&2
        exit 2
        ;;
esac
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

TRAIN_PARAMETER_NAMES=(
    DEEPSPEED_INCLUDE MASTER_PORT
    STAGE1_TIMESERIES_SFT_LR STAGE1_DATASETS STAGE1_INTERLEAVE_PROBS
    STAGE1_MIX_STRATEGY STAGE1_NUM_TRAIN_EPOCHS STAGE1_MAX_STEPS
    STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE STAGE1_GRADIENT_ACCUMULATION_STEPS
    STAGE1_LR_SCHEDULER_TYPE STAGE1_WARMUP_RATIO STAGE1_LOGGING_STEPS
    STAGE1_SAVE_STEPS STAGE1_EVAL_STEPS STAGE1_VAL_SIZE
    STAGE1_PER_DEVICE_EVAL_BATCH_SIZE STAGE1_CUTOFF_LEN
    STAGE1_PREPROCESSING_NUM_WORKERS
    STAGE2_TIMESERIES_SFT_LR STAGE2_DATASETS STAGE2_INTERLEAVE_PROBS
    STAGE2_MIX_STRATEGY STAGE2_NUM_TRAIN_EPOCHS STAGE2_MAX_STEPS
    STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE STAGE2_GRADIENT_ACCUMULATION_STEPS
    STAGE2_LR_SCHEDULER_TYPE STAGE2_WARMUP_RATIO STAGE2_LOGGING_STEPS
    STAGE2_SAVE_STEPS STAGE2_EVAL_STEPS STAGE2_VAL_SIZE
    STAGE2_PER_DEVICE_EVAL_BATCH_SIZE STAGE2_CUTOFF_LEN
    STAGE2_PREPROCESSING_NUM_WORKERS
)
# Keep this array non-empty for the Bash 3.2 shipped by macOS, where expanding
# an empty array under `set -u` can raise an unbound-variable error.
TRAIN_PARAMETER_ENV=(-e "PIPELINE_MODE=$PIPELINE_MODE")
for parameter_name in "${TRAIN_PARAMETER_NAMES[@]}"; do
    if [[ -n "${!parameter_name+x}" ]]; then
        TRAIN_PARAMETER_ENV+=(-e "$parameter_name=${!parameter_name}")
    fi
done

for flag_name in FORCE_TRAIN PREFLIGHT_ONLY KEEP_STAGE1; do
    flag_value="${!flag_name}"
    [[ "$flag_value" == "0" || "$flag_value" == "1" ]] || {
        echo "$flag_name must be 0 or 1, got: $flag_value" >&2
        exit 2
    }
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer." >&2; exit 2; }
case "$PIPELINE_MODE" in
    full|stage1) ;;
    *) echo "PIPELINE_MODE must be full or stage1, got: $PIPELINE_MODE" >&2; exit 2 ;;
esac
[[ -n "$DATASET_DIR" ]] || { echo "DATASET_DIR must not be empty." >&2; exit 2; }
if [[ -n "$DATASET_SNAPSHOT_HASH" && ! "$DATASET_SNAPSHOT_HASH" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "DATASET_SNAPSHOT_HASH must be empty or a 64-character hexadecimal SHA256." >&2
    exit 2
fi
if [[ -n "$TRAINING_RECIPE_HASH" && ! "$TRAINING_RECIPE_HASH" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "TRAINING_RECIPE_HASH must be empty or a 64-character hexadecimal SHA256." >&2
    exit 2
fi
if [[ -n "$DATA_VERSION" && ! "$DATA_VERSION" =~ ^datav[0-9]+$ ]]; then
    echo "DATA_VERSION must be empty or use the canonical datavN form." >&2
    exit 2
fi
if [[ -n "$TRIAL_ID" && ! "$TRIAL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "TRIAL_ID may contain only letters, numbers, dot, underscore, and dash." >&2
    exit 2
fi
if [[ -n "$TRIAL_CONFIG_HASH" && ! "$TRIAL_CONFIG_HASH" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "TRIAL_CONFIG_HASH must be empty or a 64-character hexadecimal SHA256." >&2
    exit 2
fi
for flag_name in FORCE_EVAL OFFLINE; do
    flag_value="${!flag_name}"
    [[ "$flag_value" == "0" || "$flag_value" == "1" ]] || {
        echo "$flag_name must be 0 or 1, got: $flag_value" >&2
        exit 2
    }
done
[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || { echo "MAX_SAMPLES must be non-negative." >&2; exit 2; }
[[ "$TINY_PARTITION_SEED" =~ ^[0-9]+$ ]] || {
    echo "TINY_PARTITION_SEED must be a non-negative integer." >&2
    exit 2
}
[[ -n "$BENCHMARKS" ]] || { echo "BENCHMARKS must not be empty." >&2; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "RUN_ID may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
}
if [[ -n "$EVAL_PROTOCOL_HASH" && ! "$EVAL_PROTOCOL_HASH" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "EVAL_PROTOCOL_HASH must be empty or a 64-character hexadecimal SHA256." >&2
    exit 2
fi
case "$MODEL_COMPLETION_MARKER" in
    TRAINING_COMPLETE.json|STAGE1_COMPLETE.json) ;;
    *) echo "MODEL_COMPLETION_MARKER must be TRAINING_COMPLETE.json or STAGE1_COMPLETE.json." >&2; exit 2 ;;
esac
if [[ "$PIPELINE_MODE" == "stage1" && "$MODEL_COMPLETION_MARKER" != "STAGE1_COMPLETE.json" ]]; then
    echo "Stage1 evaluation requires MODEL_COMPLETION_MARKER=STAGE1_COMPLETE.json." >&2
    exit 2
fi
if [[ "$PIPELINE_MODE" == "full" && "$MODEL_COMPLETION_MARKER" != "TRAINING_COMPLETE.json" ]]; then
    echo "Full evaluation requires MODEL_COMPLETION_MARKER=TRAINING_COMPLETE.json." >&2
    exit 2
fi
if [[ "$PIPELINE_MODE" == "stage1" && "$EVAL_MODEL_PATH" != "$STAGE1_OUT" ]]; then
    echo "Stage1 evaluation model must exactly match STAGE1_OUT." >&2
    exit 2
fi
if [[ "$PIPELINE_MODE" == "full" && "$EVAL_MODEL_PATH" != "$FINAL_MODEL_PATH" ]]; then
    echo "Full evaluation model must exactly match FINAL_MODEL_PATH." >&2
    exit 2
fi
case "$HAYSTACK_SPLIT" in
    train|validation|test) ;;
    *) echo "HAYSTACK_SPLIT must be train, validation, or test." >&2; exit 2 ;;
esac
case "$TINY_DATA_PARTITION" in
    all|search-dev|final-test) ;;
    *) echo "TINY_DATA_PARTITION must be all, search-dev, or final-test." >&2; exit 2 ;;
esac
for parameter_name in \
    TSR_MAX_MODEL_LEN TSR_MAX_NEW_TOKENS TSR_BATCH_SIZE TSR_REQUEST_CHUNK_SIZE \
    TINY_MAX_MODEL_LEN TINY_REQUEST_CHUNK_SIZE \
    HAYSTACK_MAX_MODEL_LEN HAYSTACK_MAX_NEW_TOKENS HAYSTACK_BATCH_SIZE \
    HAYSTACK_REQUEST_CHUNK_SIZE EXAM_MAX_MODEL_LEN EXAM_MAX_NEW_TOKENS \
    EXAM_BATCH_SIZE EXAM_REQUEST_CHUNK_SIZE; do
    parameter_value="${!parameter_name}"
    [[ "$parameter_value" =~ ^[1-9][0-9]*$ ]] || {
        echo "$parameter_name must be a positive integer, got: $parameter_value" >&2
        exit 2
    }
done
(( TSR_MAX_MODEL_LEN > TSR_MAX_NEW_TOKENS )) || {
    echo "TSR_MAX_MODEL_LEN must be larger than TSR_MAX_NEW_TOKENS." >&2
    exit 2
}
(( HAYSTACK_MAX_MODEL_LEN > HAYSTACK_MAX_NEW_TOKENS )) || {
    echo "HAYSTACK_MAX_MODEL_LEN must be larger than HAYSTACK_MAX_NEW_TOKENS." >&2
    exit 2
}
(( EXAM_MAX_MODEL_LEN > EXAM_MAX_NEW_TOKENS )) || {
    echo "EXAM_MAX_MODEL_LEN must be larger than EXAM_MAX_NEW_TOKENS." >&2
    exit 2
}
"$HOST_PYTHON_BIN" - "$TINY_GPU_MEMORY_UTILIZATION" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("TINY_GPU_MEMORY_UTILIZATION must be numeric.") from exc
if not math.isfinite(value) or not 0.0 < value <= 1.0:
    raise SystemExit("TINY_GPU_MEMORY_UTILIZATION must be in (0, 1].")
PY

command -v docker >/dev/null 2>&1 || {
    echo "Docker CLI is unavailable to the ChatTS control plane." >&2
    echo "Run Dataset Studio and this pipeline on the Docker host; training and evaluation remain in their existing containers." >&2
    echo "Do not install Docker or mount the Docker Socket into the training container just to run this pipeline." >&2
    exit 1
}

require_running_container() {
    local container="$1"
    local running
    running="$(docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null || true)"
    [[ "$running" == "true" ]] || {
        echo "Docker container is not running: $container" >&2
        exit 1
    }
}

require_container_path() {
    local container="$1" kind="$2" path="$3"
    if [[ "$kind" == "file" ]]; then
        docker exec "$container" test -f "$path" || {
            echo "$container cannot see required file: $path" >&2
            exit 1
        }
    else
        docker exec "$container" test -d "$path" || {
            echo "$container cannot see required directory: $path" >&2
            exit 1
        }
    fi
}

require_container_gpus() {
    local container="$1"
    local count
    count="$(docker exec "$container" python -c 'import torch; print(torch.cuda.device_count())')"
    [[ "$count" =~ ^[0-9]+$ ]] || { echo "Could not read GPU count in $container." >&2; exit 1; }
    if (( count < 8 )); then
        echo "$container requires 8 visible GPUs, but PyTorch sees $count." >&2
        exit 1
    fi
    echo "$container: PyTorch sees $count GPUs"
}

require_container_model_weights() {
    local container="$1" model_path="$2"
    docker exec "$container" python -c '
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
' "$model_path"
}

require_running_container "$TRAIN_CONTAINER"
require_container_path "$TRAIN_CONTAINER" dir "$SHARED_ROOT"
require_container_path "$TRAIN_CONTAINER" file "$TRAIN_SCRIPT"
require_container_gpus "$TRAIN_CONTAINER"
require_running_container "$EVAL_CONTAINER"
require_container_path "$EVAL_CONTAINER" dir "$SHARED_ROOT"
require_container_path "$EVAL_CONTAINER" file "$EVAL_SCRIPT"
require_container_gpus "$EVAL_CONTAINER"

echo "============================================================"
echo " ChatTS host pipeline: $([[ "$PIPELINE_MODE" == "stage1" ]] && echo 'Stage1 -> evaluate' || echo 'Stage1 -> Stage2 -> evaluate')"
echo " Configuration:       $CONFIG_FILE"
echo " Training container:  $TRAIN_CONTAINER"
echo " Evaluation container:$EVAL_CONTAINER"
echo " Pipeline mode:       $PIPELINE_MODE"
echo " Seed:                $SEED"
echo " Dataset directory:   $DATASET_DIR"
echo " Evaluation model:    $EVAL_MODEL_PATH"
echo " Completion marker:   $MODEL_COMPLETION_MARKER"
echo " Evaluation output:   $EVAL_OUTPUT_ROOT"
echo " Benchmarks:          $BENCHMARKS"
echo " Run ID:              $RUN_ID"
echo " Trial ID:            ${TRIAL_ID:-<none>}"
echo " Smoke sample limit:  $MAX_SAMPLES (0 means full evaluation)"
echo "============================================================"

run_training() {
    docker exec \
        "${TRAIN_PARAMETER_ENV[@]}" \
        -e PROJECT_ROOT="$TRAIN_PROJECT_ROOT" \
        -e MODEL_PATH="$BASE_MODEL_PATH" \
        -e OUTPUT_ROOT="$TRAIN_OUTPUT_ROOT" \
        -e STAGE1_OUT="$STAGE1_OUT" \
        -e FINAL_MODEL_PATH="$FINAL_MODEL_PATH" \
        -e CHRONOS2_MODEL_PATH="$TRAIN_CHRONOS2_MODEL_PATH" \
        -e DATASET_DIR="$DATASET_DIR" \
        -e DATA_VERSION="$DATA_VERSION" \
        -e DATASET_SNAPSHOT_HASH="$DATASET_SNAPSHOT_HASH" \
        -e TRAINING_RECIPE_HASH="$TRAINING_RECIPE_HASH" \
        -e TRIAL_ID="$TRIAL_ID" \
        -e TRIAL_CONFIG_HASH="$TRIAL_CONFIG_HASH" \
        -e KEEP_STAGE1="$KEEP_STAGE1" \
        -e SEED="$SEED" \
        -e S1_LR="$S1_LR" \
        -e S2_LR="$S2_LR" \
        -e FORCE_TRAIN="$FORCE_TRAIN" \
        -e PREFLIGHT_ONLY="$PREFLIGHT_ONLY" \
        "$TRAIN_CONTAINER" \
        bash "$TRAIN_SCRIPT"
}

run_evaluation() {
    docker exec \
        -e PROJECT_ROOT="$EVAL_PROJECT_ROOT" \
        -e MODEL_PATH="$EVAL_MODEL_PATH" \
        -e MODEL_NAME="$MODEL_NAME" \
        -e OUTPUT_ROOT="$EVAL_OUTPUT_ROOT" \
        -e CHRONOS2_MODEL_PATH="$EVAL_CHRONOS2_MODEL_PATH" \
        -e TSRBENCH_ROOT="$TSRBENCH_ROOT" \
        -e TSRBENCH_DATASET_ROOT="$TSRBENCH_ROOT" \
        -e TINYBENCH_DATASET_ROOT="$TINYBENCH_DATASET_ROOT" \
        -e TS_HAYSTACK_ROOT="$TS_HAYSTACK_ROOT" \
        -e TIMESERIESEXAM_ROOT="$TIMESERIESEXAM_ROOT" \
        -e TIMESERIESEXAM_DATA_FILE="$TIMESERIESEXAM_DATA_FILE" \
        -e BENCHMARKS="$BENCHMARKS" \
        -e RUN_ID="$RUN_ID" \
        -e EVAL_PROTOCOL_HASH="$EVAL_PROTOCOL_HASH" \
        -e HAYSTACK_SPLIT="$HAYSTACK_SPLIT" \
        -e TINY_DATA_PARTITION="$TINY_DATA_PARTITION" \
        -e TINY_PARTITION_SEED="$TINY_PARTITION_SEED" \
        -e DATA_VERSION="$DATA_VERSION" \
        -e DATASET_SNAPSHOT_HASH="$DATASET_SNAPSHOT_HASH" \
        -e TSR_PROMPT_MODE="$TSR_PROMPT_MODE" \
        -e TSR_MAX_MODEL_LEN="$TSR_MAX_MODEL_LEN" \
        -e TSR_MAX_NEW_TOKENS="$TSR_MAX_NEW_TOKENS" \
        -e TSR_BATCH_SIZE="$TSR_BATCH_SIZE" \
        -e TSR_REQUEST_CHUNK_SIZE="$TSR_REQUEST_CHUNK_SIZE" \
        -e TINY_MAX_MODEL_LEN="$TINY_MAX_MODEL_LEN" \
        -e TINY_REQUEST_CHUNK_SIZE="$TINY_REQUEST_CHUNK_SIZE" \
        -e TINY_GPU_MEMORY_UTILIZATION="$TINY_GPU_MEMORY_UTILIZATION" \
        -e HAYSTACK_MAX_MODEL_LEN="$HAYSTACK_MAX_MODEL_LEN" \
        -e HAYSTACK_MAX_NEW_TOKENS="$HAYSTACK_MAX_NEW_TOKENS" \
        -e HAYSTACK_BATCH_SIZE="$HAYSTACK_BATCH_SIZE" \
        -e HAYSTACK_REQUEST_CHUNK_SIZE="$HAYSTACK_REQUEST_CHUNK_SIZE" \
        -e EXAM_MAX_MODEL_LEN="$EXAM_MAX_MODEL_LEN" \
        -e EXAM_MAX_NEW_TOKENS="$EXAM_MAX_NEW_TOKENS" \
        -e EXAM_BATCH_SIZE="$EXAM_BATCH_SIZE" \
        -e EXAM_REQUEST_CHUNK_SIZE="$EXAM_REQUEST_CHUNK_SIZE" \
        -e SEED="$SEED" \
        -e FORCE_EVAL="$FORCE_EVAL" \
        -e PREFLIGHT_ONLY="$PREFLIGHT_ONLY" \
        -e REQUIRE_TRAINING_MARKER=1 \
        -e MODEL_COMPLETION_MARKER="$MODEL_COMPLETION_MARKER" \
        -e MAX_SAMPLES="$MAX_SAMPLES" \
        -e OFFLINE="$OFFLINE" \
        "$EVAL_CONTAINER" \
        bash "$EVAL_SCRIPT"
}

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "Running training-container preflight..."
    run_training
    echo "Running evaluation-container preflight..."
    run_evaluation
    echo "All host/container preflight checks passed. No training or evaluation was started."
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') | Starting/reusing training"
run_training

if [[ "$PIPELINE_MODE" == "stage1" ]]; then
    require_container_path "$TRAIN_CONTAINER" file "$STAGE1_OUT/STAGE1_COMPLETE.json"
    require_container_path "$TRAIN_CONTAINER" file "$STAGE1_OUT/config.json"
    require_container_path "$TRAIN_CONTAINER" file "$STAGE1_OUT/best_model_manifest.json"
    require_container_model_weights "$TRAIN_CONTAINER" "$STAGE1_OUT"
else
    require_container_path "$TRAIN_CONTAINER" file "$FINAL_MODEL_PATH/TRAINING_COMPLETE.json"
    require_container_path "$TRAIN_CONTAINER" file "$FINAL_MODEL_PATH/config.json"
    require_container_model_weights "$TRAIN_CONTAINER" "$FINAL_MODEL_PATH"
fi

READY_MARKER="$EVAL_MODEL_PATH/$MODEL_COMPLETION_MARKER"
require_container_path "$TRAIN_CONTAINER" file "$READY_MARKER"
require_container_path "$TRAIN_CONTAINER" file "$EVAL_MODEL_PATH/config.json"
require_container_model_weights "$TRAIN_CONTAINER" "$EVAL_MODEL_PATH"

# This is the conclusive shared-volume check: the evaluation container must see
# the exact model and atomic completion marker just written by training.
require_container_path "$EVAL_CONTAINER" file "$READY_MARKER"
require_container_path "$EVAL_CONTAINER" file "$EVAL_MODEL_PATH/config.json"
require_container_model_weights "$EVAL_CONTAINER" "$EVAL_MODEL_PATH"

echo "$(date '+%Y-%m-%d %H:%M:%S') | Training gate passed; starting sequential eight-GPU evaluation"
run_evaluation

# A zero exit from the evaluator is necessary but not sufficient for a durable
# pipeline result. Require all aggregate artifacts before declaring success.
require_container_path "$EVAL_CONTAINER" file "$EVAL_OUTPUT_ROOT/benchmark_status.tsv"
require_container_path "$EVAL_CONTAINER" file "$EVAL_OUTPUT_ROOT/all_benchmarks_summary.md"
require_container_path "$EVAL_CONTAINER" file "$EVAL_OUTPUT_ROOT/metrics.json"

echo "============================================================"
echo " Pipeline completed successfully"
echo " Evaluated model:   $EVAL_MODEL_PATH"
echo " Evaluation output: $EVAL_OUTPUT_ROOT"
echo " Status table:      $EVAL_OUTPUT_ROOT/benchmark_status.tsv"
echo " Summary:           $EVAL_OUTPUT_ROOT/all_benchmarks_summary.md"
echo " Metrics:           $EVAL_OUTPUT_ROOT/metrics.json"
echo "============================================================"

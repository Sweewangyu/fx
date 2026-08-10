#!/usr/bin/env bash
set -Eeuo pipefail

# Host-side one-click entrypoint.  This script intentionally does not use -it:
# docker exec exits only after each container-side stage has completed.

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
FINAL_MODEL_PATH="${FINAL_MODEL_PATH:-${TRAIN_OUTPUT_ROOT}/best_seed${SEED}}"
MODEL_NAME="${MODEL_NAME:-chatts-msxf-8B-datav1-seed${SEED}}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${SHARED_ROOT}/evaluation/all-benchmarks/${MODEL_NAME}}"

TRAIN_CHRONOS2_MODEL_PATH="${TRAIN_CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
EVAL_CHRONOS2_MODEL_PATH="${EVAL_CHRONOS2_MODEL_PATH:-/workspace/chronos2}"

TSRBENCH_ROOT="${TSRBENCH_ROOT:-${SHARED_ROOT}/TSRBench-dataset}"
TINYBENCH_DATASET_ROOT="${TINYBENCH_DATASET_ROOT:-${SHARED_ROOT}/tyb}"
TS_HAYSTACK_ROOT="${TS_HAYSTACK_ROOT:-/workspace/TS-Haystack}"
TIMESERIESEXAM_ROOT="${TIMESERIESEXAM_ROOT:-/workspace/TimeSeriesExam}"
TIMESERIESEXAM_DATA_FILE="${TIMESERIESEXAM_DATA_FILE:-${TIMESERIESEXAM_ROOT}/output/round_3_folder/qa_dataset.json}"

FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OFFLINE="${OFFLINE:-1}"

for flag_name in FORCE_TRAIN FORCE_EVAL PREFLIGHT_ONLY OFFLINE; do
    flag_value="${!flag_name}"
    [[ "$flag_value" == "0" || "$flag_value" == "1" ]] || {
        echo "$flag_name must be 0 or 1, got: $flag_value" >&2
        exit 2
    }
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer." >&2; exit 2; }
[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || { echo "MAX_SAMPLES must be non-negative." >&2; exit 2; }

command -v docker >/dev/null 2>&1 || { echo "docker command not found on the host." >&2; exit 1; }

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

require_running_container "$TRAIN_CONTAINER"
require_running_container "$EVAL_CONTAINER"
require_container_path "$TRAIN_CONTAINER" dir "$SHARED_ROOT"
require_container_path "$EVAL_CONTAINER" dir "$SHARED_ROOT"
require_container_path "$TRAIN_CONTAINER" file "$TRAIN_SCRIPT"
require_container_path "$EVAL_CONTAINER" file "$EVAL_SCRIPT"
require_container_gpus "$TRAIN_CONTAINER"
require_container_gpus "$EVAL_CONTAINER"

echo "============================================================"
echo " ChatTS host pipeline: train -> evaluate"
echo " Training container:  $TRAIN_CONTAINER"
echo " Evaluation container:$EVAL_CONTAINER"
echo " Seed:                $SEED"
echo " Final model:         $FINAL_MODEL_PATH"
echo " Evaluation output:   $EVAL_OUTPUT_ROOT"
echo " Smoke sample limit:  $MAX_SAMPLES (0 means full evaluation)"
echo "============================================================"

run_training() {
    docker exec \
        -e PROJECT_ROOT="$TRAIN_PROJECT_ROOT" \
        -e MODEL_PATH="$BASE_MODEL_PATH" \
        -e OUTPUT_ROOT="$TRAIN_OUTPUT_ROOT" \
        -e FINAL_MODEL_PATH="$FINAL_MODEL_PATH" \
        -e CHRONOS2_MODEL_PATH="$TRAIN_CHRONOS2_MODEL_PATH" \
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
        -e MODEL_PATH="$FINAL_MODEL_PATH" \
        -e MODEL_NAME="$MODEL_NAME" \
        -e OUTPUT_ROOT="$EVAL_OUTPUT_ROOT" \
        -e CHRONOS2_MODEL_PATH="$EVAL_CHRONOS2_MODEL_PATH" \
        -e TSRBENCH_ROOT="$TSRBENCH_ROOT" \
        -e TSRBENCH_DATASET_ROOT="$TSRBENCH_ROOT" \
        -e TINYBENCH_DATASET_ROOT="$TINYBENCH_DATASET_ROOT" \
        -e TS_HAYSTACK_ROOT="$TS_HAYSTACK_ROOT" \
        -e TIMESERIESEXAM_ROOT="$TIMESERIESEXAM_ROOT" \
        -e TIMESERIESEXAM_DATA_FILE="$TIMESERIESEXAM_DATA_FILE" \
        -e SEED="$SEED" \
        -e FORCE_EVAL="$FORCE_EVAL" \
        -e PREFLIGHT_ONLY="$PREFLIGHT_ONLY" \
        -e REQUIRE_TRAINING_MARKER=1 \
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

READY_MARKER="$FINAL_MODEL_PATH/TRAINING_COMPLETE.json"
require_container_path "$TRAIN_CONTAINER" file "$READY_MARKER"
require_container_path "$TRAIN_CONTAINER" file "$FINAL_MODEL_PATH/config.json"

# This is the conclusive shared-volume check: the evaluation container must see
# the exact model and atomic completion marker just written by training.
require_container_path "$EVAL_CONTAINER" file "$READY_MARKER"
require_container_path "$EVAL_CONTAINER" file "$FINAL_MODEL_PATH/config.json"

echo "$(date '+%Y-%m-%d %H:%M:%S') | Training gate passed; starting sequential eight-GPU evaluation"
run_evaluation

echo "============================================================"
echo " Pipeline completed successfully"
echo " Final model:       $FINAL_MODEL_PATH"
echo " Evaluation output: $EVAL_OUTPUT_ROOT"
echo " Status table:      $EVAL_OUTPUT_ROOT/benchmark_status.tsv"
echo " Summary:           $EVAL_OUTPUT_ROOT/all_benchmarks_summary.md"
echo "============================================================"

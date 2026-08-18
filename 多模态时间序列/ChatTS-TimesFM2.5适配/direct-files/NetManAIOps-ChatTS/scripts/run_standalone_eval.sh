#!/usr/bin/env bash
set -Eeuo pipefail

# Host-side standalone evaluation entrypoint.  It never enters or starts the
# training container; the selected checkpoint must already be visible inside
# the existing evaluation container.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_PYTHON_BIN="${HOST_PYTHON_BIN:-python3}"
CONFIG_FILE="${CONFIG_FILE:-${1:-}}"
CONFIG_LOADER="${CONFIG_LOADER:-${SCRIPT_DIR}/load_studio_evaluation_config.py}"

[[ -n "$CONFIG_FILE" ]] || {
    echo "Usage: CONFIG_FILE=/absolute/job.resolved.yaml bash $0" >&2
    exit 2
}
command -v "$HOST_PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Host Python not found: $HOST_PYTHON_BIN" >&2
    exit 1
}
[[ -f "$CONFIG_FILE" ]] || { echo "Configuration file not found: $CONFIG_FILE" >&2; exit 1; }
[[ -f "$CONFIG_LOADER" ]] || { echo "Configuration loader not found: $CONFIG_LOADER" >&2; exit 1; }

RESOLVED_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/chatts-eval-env.XXXXXX")"
cleanup() {
    rm -f "$RESOLVED_ENV_FILE"
}
trap cleanup EXIT

"$HOST_PYTHON_BIN" "$CONFIG_LOADER" "$CONFIG_FILE" > "$RESOLVED_ENV_FILE"

SEEN_ENV_NAMES=" "
while IFS= read -r assignment || [[ -n "$assignment" ]]; do
    [[ -n "$assignment" && "$assignment" == *=* ]] || {
        echo "Malformed standalone evaluation loader output" >&2
        exit 2
    }
    name="${assignment%%=*}"
    case "$name" in
        BATCH_ID|BENCHMARKS|CHATTS_EVALUATION_DIR|CHATTS_EVAL_SIF_IMAGE|CHATTS_HOST_CHRONOS2_PATH|CHATTS_HOST_TIMESERIESEXAM_PATH|CHATTS_HOST_TINYBENCH_PATH|CHATTS_HOST_TSRBENCH_PATH|CHATTS_HOST_TS_HAYSTACK_PATH|EVAL_CHRONOS2_MODEL_PATH|EVAL_CONTAINER|EVAL_MODEL_PATH|EVAL_OUTPUT_ROOT|EVAL_PROJECT_ROOT|EVAL_PROTOCOL_HASH|EVAL_SCRIPT|EXAM_BATCH_SIZE|EXAM_MAX_MODEL_LEN|EXAM_MAX_NEW_TOKENS|EXAM_REQUEST_CHUNK_SIZE|FORCE_EVAL|HAYSTACK_BATCH_SIZE|HAYSTACK_MAX_MODEL_LEN|HAYSTACK_MAX_NEW_TOKENS|HAYSTACK_REQUEST_CHUNK_SIZE|HAYSTACK_SPLIT|MAX_SAMPLES|MODEL_NAME|OFFLINE|PREFLIGHT_ONLY|RUN_ID|SEED|TASK_TYPE|TIMESERIESEXAM_DATA_FILE|TIMESERIESEXAM_ROOT|TINYBENCH_DATASET_ROOT|TINY_DATA_PARTITION|TINY_GPU_MEMORY_UTILIZATION|TINY_MAX_MODEL_LEN|TINY_PARTITION_SEED|TINY_REQUEST_CHUNK_SIZE|TRIAL_CONFIG_HASH|TRIAL_ID|TSRBENCH_ROOT|TSR_BATCH_SIZE|TSR_MAX_MODEL_LEN|TSR_MAX_NEW_TOKENS|TSR_PROMPT_MODE|TSR_REQUEST_CHUNK_SIZE|TS_HAYSTACK_ROOT) ;;
        *)
            echo "Unexpected standalone evaluation environment name: $name" >&2
            exit 2
            ;;
    esac
    [[ "$SEEN_ENV_NAMES" != *" $name "* ]] || {
        echo "Duplicate standalone evaluation environment name: $name" >&2
        exit 2
    }
    SEEN_ENV_NAMES+="$name "
    export "$assignment"
done < "$RESOLVED_ENV_FILE"

for required_name in \
    TASK_TYPE SEED FORCE_EVAL PREFLIGHT_ONLY MAX_SAMPLES OFFLINE TRIAL_ID \
    TRIAL_CONFIG_HASH EVAL_CONTAINER EVAL_PROJECT_ROOT EVAL_SCRIPT \
    EVAL_MODEL_PATH MODEL_NAME EVAL_OUTPUT_ROOT EVAL_CHRONOS2_MODEL_PATH \
    TSRBENCH_ROOT TINYBENCH_DATASET_ROOT TS_HAYSTACK_ROOT TIMESERIESEXAM_ROOT \
    TIMESERIESEXAM_DATA_FILE BENCHMARKS RUN_ID EVAL_PROTOCOL_HASH \
    HAYSTACK_SPLIT TINY_DATA_PARTITION TINY_PARTITION_SEED TSR_PROMPT_MODE \
    TSR_MAX_MODEL_LEN TSR_MAX_NEW_TOKENS TSR_BATCH_SIZE TSR_REQUEST_CHUNK_SIZE \
    TINY_MAX_MODEL_LEN TINY_REQUEST_CHUNK_SIZE TINY_GPU_MEMORY_UTILIZATION \
    HAYSTACK_MAX_MODEL_LEN HAYSTACK_MAX_NEW_TOKENS HAYSTACK_BATCH_SIZE \
    HAYSTACK_REQUEST_CHUNK_SIZE EXAM_MAX_MODEL_LEN EXAM_MAX_NEW_TOKENS \
    EXAM_BATCH_SIZE EXAM_REQUEST_CHUNK_SIZE; do
    [[ "$SEEN_ENV_NAMES" == *" $required_name "* ]] || {
        echo "Loader did not emit required environment: $required_name" >&2
        exit 2
    }
done

command -v docker >/dev/null 2>&1 || {
    echo "Docker CLI is unavailable to the ChatTS control plane." >&2
    exit 1
}
running="$(docker inspect --format '{{.State.Running}}' "$EVAL_CONTAINER" 2>/dev/null || true)"
[[ "$running" == "true" ]] || {
    echo "Docker evaluation container is not running: $EVAL_CONTAINER" >&2
    exit 1
}
docker exec "$EVAL_CONTAINER" test -d "$EVAL_PROJECT_ROOT" || {
    echo "$EVAL_CONTAINER cannot see evaluation project: $EVAL_PROJECT_ROOT" >&2
    exit 1
}
docker exec "$EVAL_CONTAINER" test -f "$EVAL_SCRIPT" || {
    echo "$EVAL_CONTAINER cannot see evaluation script: $EVAL_SCRIPT" >&2
    exit 1
}

echo "============================================================"
echo " ChatTS standalone Docker evaluation"
echo " Trial:                $TRIAL_ID"
echo " Batch:                ${BATCH_ID:-<none>}"
echo " Evaluation container: $EVAL_CONTAINER"
echo " Model:                $EVAL_MODEL_PATH"
echo " Model name:           $MODEL_NAME"
echo " Evaluation output:    $EVAL_OUTPUT_ROOT"
echo " Benchmarks:           $BENCHMARKS"
echo " Protocol hash:        $EVAL_PROTOCOL_HASH"
echo " Preflight only:       $PREFLIGHT_ONLY"
echo "============================================================"

EVALUATION_ENV=(
    -e PROJECT_ROOT="$EVAL_PROJECT_ROOT"
    -e EVAL_SCRIPT="$EVAL_SCRIPT"
    -e MODEL_PATH="$EVAL_MODEL_PATH"
    -e MODEL_NAME="$MODEL_NAME"
    -e OUTPUT_ROOT="$EVAL_OUTPUT_ROOT"
    -e LOG_ROOT="$EVAL_OUTPUT_ROOT/logs"
    -e STATUS_FILE="$EVAL_OUTPUT_ROOT/benchmark_status.tsv"
    -e SUMMARY_FILE="$EVAL_OUTPUT_ROOT/all_benchmarks_summary.md"
    -e MANIFEST_FILE="$EVAL_OUTPUT_ROOT/run_manifest.json"
    -e METRICS_FILE="$EVAL_OUTPUT_ROOT/metrics.json"
    -e CHRONOS2_MODEL_PATH="$EVAL_CHRONOS2_MODEL_PATH"
    -e TSRBENCH_ROOT="$TSRBENCH_ROOT"
    -e TSRBENCH_DATASET_ROOT="$TSRBENCH_ROOT"
    -e TINYBENCH_DATASET_ROOT="$TINYBENCH_DATASET_ROOT"
    -e TS_HAYSTACK_ROOT="$TS_HAYSTACK_ROOT"
    -e TIMESERIESEXAM_ROOT="$TIMESERIESEXAM_ROOT"
    -e TIMESERIESEXAM_DATA_FILE="$TIMESERIESEXAM_DATA_FILE"
    -e BENCHMARKS="$BENCHMARKS"
    -e RUN_ID="$RUN_ID"
    -e EVAL_PROTOCOL_HASH="$EVAL_PROTOCOL_HASH"
    -e HAYSTACK_SPLIT="$HAYSTACK_SPLIT"
    -e TINY_DATA_PARTITION="$TINY_DATA_PARTITION"
    -e TINY_PARTITION_SEED="$TINY_PARTITION_SEED"
    -e TSR_PROMPT_MODE="$TSR_PROMPT_MODE"
    -e TSR_MAX_MODEL_LEN="$TSR_MAX_MODEL_LEN"
    -e TSR_MAX_NEW_TOKENS="$TSR_MAX_NEW_TOKENS"
    -e TSR_BATCH_SIZE="$TSR_BATCH_SIZE"
    -e TSR_REQUEST_CHUNK_SIZE="$TSR_REQUEST_CHUNK_SIZE"
    -e TINY_MAX_MODEL_LEN="$TINY_MAX_MODEL_LEN"
    -e TINY_REQUEST_CHUNK_SIZE="$TINY_REQUEST_CHUNK_SIZE"
    -e TINY_GPU_MEMORY_UTILIZATION="$TINY_GPU_MEMORY_UTILIZATION"
    -e HAYSTACK_MAX_MODEL_LEN="$HAYSTACK_MAX_MODEL_LEN"
    -e HAYSTACK_MAX_NEW_TOKENS="$HAYSTACK_MAX_NEW_TOKENS"
    -e HAYSTACK_BATCH_SIZE="$HAYSTACK_BATCH_SIZE"
    -e HAYSTACK_REQUEST_CHUNK_SIZE="$HAYSTACK_REQUEST_CHUNK_SIZE"
    -e EXAM_MAX_MODEL_LEN="$EXAM_MAX_MODEL_LEN"
    -e EXAM_MAX_NEW_TOKENS="$EXAM_MAX_NEW_TOKENS"
    -e EXAM_BATCH_SIZE="$EXAM_BATCH_SIZE"
    -e EXAM_REQUEST_CHUNK_SIZE="$EXAM_REQUEST_CHUNK_SIZE"
    -e SEED="$SEED"
    -e DATA_VERSION=
    -e DATASET_SNAPSHOT_HASH=
    -e FORCE_EVAL="$FORCE_EVAL"
    -e SUITE_FORCE_EVAL="$FORCE_EVAL"
    -e PREFLIGHT_ONLY="$PREFLIGHT_ONLY"
    -e REQUIRE_TRAINING_MARKER=0
    -e REQUIRE_MODEL_ON_PREFLIGHT=1
    -e MODEL_COMPLETION_MARKER=TRAINING_COMPLETE.json
    -e MAX_SAMPLES="$MAX_SAMPLES"
    -e OFFLINE="$OFFLINE"
    -e EVAL_GPUS=0,1,2,3,4,5,6,7
    -e EVAL_NUM_GPUS=8
    -e TS_GPUS_PER_PROCESS=2
)

docker exec \
    --workdir "$EVAL_PROJECT_ROOT" \
    "${EVALUATION_ENV[@]}" \
    "$EVAL_CONTAINER" \
    bash -c '
        set -Eeuo pipefail
        unset AVAILABLE_GPUS_OVERRIDE
        if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
            exec bash "$EVAL_SCRIPT"
        fi
        command -v flock >/dev/null 2>&1 || {
            echo "Evaluation image must provide flock for output locking." >&2
            exit 1
        }
        lock_path="${OUTPUT_ROOT}.chatts-evaluation.lock"
        mkdir -p "$(dirname "$lock_path")"
        exec 9>>"$lock_path"
        if ! flock -n 9; then
            echo "Another evaluation is already writing this protocol output." >&2
            echo "Output: $OUTPUT_ROOT" >&2
            echo "Lock:   $lock_path" >&2
            exit 75
        fi
        bash "$EVAL_SCRIPT"
        test -f "$OUTPUT_ROOT/benchmark_status.tsv"
        test -f "$OUTPUT_ROOT/all_benchmarks_summary.md"
        test -f "$OUTPUT_ROOT/metrics.json"
    '

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "Standalone Docker evaluation preflight passed; no artifacts were created."
else
    echo "Standalone Docker evaluation completed: $EVAL_OUTPUT_ROOT"
fi

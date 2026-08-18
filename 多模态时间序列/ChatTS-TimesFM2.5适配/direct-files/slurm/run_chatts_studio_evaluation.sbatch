#!/usr/bin/env bash
# CHATTS_STUDIO_SBATCH_API=1
# CHATTS_STUDIO_EVALUATION_SBATCH_API=1
#SBATCH --job-name=chatts-eval
#SBATCH --partition=xj-h100-ib-2
#SBATCH --output=log/%j-chatts-eval.out
#SBATCH --error=log/%j-chatts-eval.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --account=nichemodel

set -Eeuo pipefail

# Dataset Studio invocation contract:
#   sbatch ... run_chatts_studio_evaluation.sbatch <resolved.yaml> <studio-job-id>
#
# This allocation performs evaluation only.  No training image, trainer, data
# snapshot, or training completion marker is required.
[[ "$#" -eq 2 ]] || {
    echo "Usage: sbatch $0 <immutable-resolved.yaml> <studio-job-id>" >&2
    exit 2
}
FROZEN_CONFIG="$1"
STUDIO_JOB_ID="$2"
[[ "$FROZEN_CONFIG" == /* && -f "$FROZEN_CONFIG" ]] || {
    echo "Frozen Studio configuration must be an existing absolute path: $FROZEN_CONFIG" >&2
    exit 2
}
[[ "$STUDIO_JOB_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
    echo "Invalid Studio job id: $STUDIO_JOB_ID" >&2
    exit 2
}
SLURM_JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is required; submit this file with sbatch}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "Invalid SLURM_JOB_ID: $SLURM_JOB_ID" >&2
    exit 2
}

# Slurm executes a spool copy, therefore locate repositories from the submit
# directory rather than BASH_SOURCE.  ChatTS is normally a sibling of
# ChatTS-Training; deployments with a different layout export
# CHATTS_EVALUATION_DIR in the Studio service environment.
DEFAULT_TRAINING_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CHATTS_TRAINING_DIR="${CHATTS_TRAINING_DIR:-$DEFAULT_TRAINING_DIR}"
CHATTS_HOST_ROOT="${CHATTS_HOST_ROOT:-$HOME}"
CHATTS_EVALUATION_DIR="${CHATTS_EVALUATION_DIR:-$(dirname "$CHATTS_TRAINING_DIR")/ChatTS}"
CHATTS_SIF_IMAGE="${CHATTS_SIF_IMAGE:-$CHATTS_HOST_ROOT/chatts_v1.sif}"
CHATTS_HOST_PYTHON_BIN="${CHATTS_HOST_PYTHON_BIN:-python3}"
CHATTS_SRUN_BIN="${CHATTS_SRUN_BIN:-srun}"
CHATTS_SINGULARITY_BIN="${CHATTS_SINGULARITY_BIN:-singularity}"
CHATTS_FLOCK_BIN="${CHATTS_FLOCK_BIN:-flock}"
CHATTS_SHARED_HOST_PATH="${CHATTS_SHARED_HOST_PATH:-/share}"
CHATTS_SHARED_CONTAINER_PATH="${CHATTS_SHARED_CONTAINER_PATH:-/share}"

CONFIG_LOADER="$CHATTS_EVALUATION_DIR/scripts/load_studio_evaluation_config.py"
[[ -d "$CHATTS_EVALUATION_DIR" ]] || {
    echo "Required host evaluation project not found: $CHATTS_EVALUATION_DIR" >&2
    exit 1
}
[[ -f "$CONFIG_LOADER" ]] || {
    echo "Required standalone evaluation loader not found: $CONFIG_LOADER" >&2
    exit 1
}
[[ -d "$CHATTS_SHARED_HOST_PATH" ]] || {
    echo "Required shared host path not found: $CHATTS_SHARED_HOST_PATH" >&2
    exit 1
}
for required_command in \
    "$CHATTS_HOST_PYTHON_BIN" "$CHATTS_SRUN_BIN" \
    "$CHATTS_SINGULARITY_BIN" "$CHATTS_FLOCK_BIN"; do
    command -v "$required_command" >/dev/null 2>&1 || {
        echo "Required command not found: $required_command" >&2
        exit 1
    }
done

if [[ -n "${SINGULARITY_TMPDIR:-}" && -z "${APPTAINER_TMPDIR:-}" ]]; then
    export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
    unset SINGULARITY_TMPDIR
fi

JOB_TMP="${CHATTS_JOB_TMP_ROOT:-/tmp}/chatts_eval_${SLURM_JOB_ID}_${STUDIO_JOB_ID}"
JOB_HOME="$JOB_TMP/home"
TRITON_CACHE="$JOB_TMP/triton"
HF_CACHE="$JOB_TMP/huggingface"
RESOLVED_ENV_FILE="$JOB_TMP/resolved-evaluation.env"
mkdir -p "$JOB_TMP" "$JOB_HOME" "$TRITON_CACHE" "$HF_CACHE"

"$CHATTS_HOST_PYTHON_BIN" \
    "$CONFIG_LOADER" \
    "$FROZEN_CONFIG" \
    --expected-job-id "$STUDIO_JOB_ID" \
    > "$RESOLVED_ENV_FILE"

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
    TRIAL_CONFIG_HASH EVAL_PROJECT_ROOT EVAL_SCRIPT EVAL_MODEL_PATH MODEL_NAME \
    EVAL_OUTPUT_ROOT EVAL_CHRONOS2_MODEL_PATH TSRBENCH_ROOT \
    TINYBENCH_DATASET_ROOT TS_HAYSTACK_ROOT TIMESERIESEXAM_ROOT \
    TIMESERIESEXAM_DATA_FILE BENCHMARKS RUN_ID EVAL_PROTOCOL_HASH \
    HAYSTACK_SPLIT TINY_DATA_PARTITION TINY_PARTITION_SEED TSR_PROMPT_MODE \
    TSR_MAX_MODEL_LEN TSR_MAX_NEW_TOKENS TSR_BATCH_SIZE TSR_REQUEST_CHUNK_SIZE \
    TINY_MAX_MODEL_LEN TINY_REQUEST_CHUNK_SIZE TINY_GPU_MEMORY_UTILIZATION \
    HAYSTACK_MAX_MODEL_LEN HAYSTACK_MAX_NEW_TOKENS HAYSTACK_BATCH_SIZE \
    HAYSTACK_REQUEST_CHUNK_SIZE EXAM_MAX_MODEL_LEN EXAM_MAX_NEW_TOKENS \
    EXAM_BATCH_SIZE EXAM_REQUEST_CHUNK_SIZE CHATTS_EVALUATION_DIR; do
    [[ "$SEEN_ENV_NAMES" == *" $required_name "* ]] || {
        echo "Loader did not emit required environment: $required_name" >&2
        exit 2
    }
done

[[ -d "$CHATTS_EVALUATION_DIR" ]] || {
    echo "Resolved host evaluation project not found: $CHATTS_EVALUATION_DIR" >&2
    exit 1
}
[[ -f "$CHATTS_EVALUATION_DIR/scripts/run_all_chatts_benchmarks.sh" ]] || {
    echo "Evaluation runner not found under host project: $CHATTS_EVALUATION_DIR" >&2
    exit 1
}

# ragas.sif defaults to the directory containing the training image, exactly as
# used on the target cluster.  An explicit server-side Slurm setting still wins.
CHATTS_EVAL_SIF_IMAGE="${CHATTS_EVAL_SIF_IMAGE:-$(dirname "$CHATTS_SIF_IMAGE")/ragas.sif}"
[[ -f "$CHATTS_EVAL_SIF_IMAGE" ]] || {
    echo "Required evaluation SIF not found: $CHATTS_EVAL_SIF_IMAGE" >&2
    exit 1
}
CHATTS_HOST_CHRONOS2_PATH="${CHATTS_HOST_CHRONOS2_PATH:-$CHATTS_HOST_ROOT/chronos2}"
[[ -d "$CHATTS_HOST_CHRONOS2_PATH" ]] || {
    echo "Required host Chronos-2 directory not found: $CHATTS_HOST_CHRONOS2_PATH" >&2
    exit 1
}

require_shared_container_path() {
    local label="$1" value="$2" root="${CHATTS_SHARED_CONTAINER_PATH%/}"
    case "$value" in
        "$root"/*) ;;
        *)
            echo "$label must be inside CHATTS_SHARED_CONTAINER_PATH=$root: $value" >&2
            exit 2
            ;;
    esac
}
require_shared_container_path EVAL_MODEL_PATH "$EVAL_MODEL_PATH"
require_shared_container_path EVAL_OUTPUT_ROOT "$EVAL_OUTPUT_ROOT"

map_shared_host_path() {
    local value="$1" container_root="${CHATTS_SHARED_CONTAINER_PATH%/}"
    case "$value" in
        "$container_root"/*)
            printf '%s/%s\n' "${CHATTS_SHARED_HOST_PATH%/}" "${value#"$container_root"/}"
            ;;
        *)
            printf '%s/%s\n' "${CHATTS_HOST_ROOT%/}" "$(basename "$value")"
            ;;
    esac
}

suite_selected() {
    local wanted="$1"
    case ",${BENCHMARKS}," in
        *",${wanted},"*) return 0 ;;
        *) return 1 ;;
    esac
}

CHATTS_HOST_TSRBENCH_PATH="${CHATTS_HOST_TSRBENCH_PATH:-$(map_shared_host_path "$TSRBENCH_ROOT")}"
CHATTS_HOST_TINYBENCH_PATH="${CHATTS_HOST_TINYBENCH_PATH:-$(map_shared_host_path "$TINYBENCH_DATASET_ROOT")}"
CHATTS_HOST_TS_HAYSTACK_PATH="${CHATTS_HOST_TS_HAYSTACK_PATH:-$(map_shared_host_path "$TS_HAYSTACK_ROOT")}"
CHATTS_HOST_TIMESERIESEXAM_PATH="${CHATTS_HOST_TIMESERIESEXAM_PATH:-$(map_shared_host_path "$TIMESERIESEXAM_ROOT")}"

declare -a BENCHMARK_BINDS=()
add_benchmark_bind() {
    local suite="$1" host_path="$2" container_path="$3"
    suite_selected "$suite" || return 0
    [[ -d "$host_path" ]] || {
        echo "$suite host benchmark directory not found: $host_path" >&2
        exit 1
    }
    BENCHMARK_BINDS+=(--bind "$host_path:$container_path:ro")
}
add_benchmark_bind tsrbench "$CHATTS_HOST_TSRBENCH_PATH" "$TSRBENCH_ROOT"
add_benchmark_bind tinybenchmarks "$CHATTS_HOST_TINYBENCH_PATH" "$TINYBENCH_DATASET_ROOT"
add_benchmark_bind ts_haystack "$CHATTS_HOST_TS_HAYSTACK_PATH" "$TS_HAYSTACK_ROOT"
add_benchmark_bind timeseriesexam "$CHATTS_HOST_TIMESERIESEXAM_PATH" "$TIMESERIESEXAM_ROOT"

CONFIG_FILE_SHA256="$("$CHATTS_HOST_PYTHON_BIN" - "$FROZEN_CONFIG" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

CHATTS_HOST_EVAL_OUTPUT_ROOT="$(map_shared_host_path "$EVAL_OUTPUT_ROOT")"
CHATTS_EVAL_LOCK_PATH="${CHATTS_HOST_EVAL_OUTPUT_ROOT}.chatts-evaluation.lock"
if [[ "$PREFLIGHT_ONLY" != "1" ]]; then
    mkdir -p "$(dirname "$CHATTS_EVAL_LOCK_PATH")"
    exec 9>>"$CHATTS_EVAL_LOCK_PATH"
    if ! "$CHATTS_FLOCK_BIN" -n 9; then
        echo "Another evaluation is already writing this model/protocol output." >&2
        echo "Output: $EVAL_OUTPUT_ROOT" >&2
        echo "Lock:   $CHATTS_EVAL_LOCK_PATH" >&2
        exit 75
    fi
fi

echo "============================================================"
echo " ChatTS Dataset Studio standalone Slurm evaluation"
echo " Slurm job:          $SLURM_JOB_ID"
echo " Studio job:         $STUDIO_JOB_ID"
echo " Batch:              ${BATCH_ID:-<none>}"
echo " Frozen config:      $FROZEN_CONFIG"
echo " Config file SHA256: $CONFIG_FILE_SHA256"
echo " Trial config hash:  $TRIAL_CONFIG_HASH"
echo " Model:              $EVAL_MODEL_PATH"
echo " Model name:         $MODEL_NAME"
echo " Evaluation project: $CHATTS_EVALUATION_DIR -> $EVAL_PROJECT_ROOT"
echo " Evaluation output:  $EVAL_OUTPUT_ROOT"
echo " Benchmarks:         $BENCHMARKS"
echo " Protocol hash:      $EVAL_PROTOCOL_HASH"
echo " Evaluation SIF:     $CHATTS_EVAL_SIF_IMAGE"
echo " Output lock:        $CHATTS_EVAL_LOCK_PATH"
echo " Preflight only:     $PREFLIGHT_ONLY"
echo "============================================================"

EVALUATION_OPTS=(
    --nv
    --pwd "$EVAL_PROJECT_ROOT"
    --home "$JOB_HOME"
    --bind "$CHATTS_EVALUATION_DIR:$EVAL_PROJECT_ROOT:ro"
    --bind "$CHATTS_HOST_CHRONOS2_PATH:$EVAL_CHRONOS2_MODEL_PATH:ro"
    --bind "$CHATTS_SHARED_HOST_PATH:$CHATTS_SHARED_CONTAINER_PATH:rw"
    --bind "$JOB_TMP:$JOB_TMP:rw"
    "${BENCHMARK_BINDS[@]}"
    --env LD_PRELOAD=/.singularity.d/libs/libcuda.so.1
    --env CUDA_HOME=/usr/local/cuda
    --env JOB_HOME="$JOB_HOME"
    --env TRITON_CACHE_DIR="$TRITON_CACHE"
    --env XDG_CACHE_HOME="$JOB_TMP/xdg"
    --env HF_HOME="$HF_CACHE"
    --env TRANSFORMERS_CACHE="$HF_CACHE"
    --env HF_MODULES_CACHE="$HF_CACHE/modules"
    --env CHATTS_STUDIO_JOB_ID="$STUDIO_JOB_ID"
    --env CHATTS_STUDIO_CONFIG_SHA256="$CONFIG_FILE_SHA256"
    --env EVAL_PROJECT_ROOT="$EVAL_PROJECT_ROOT"
    --env EVAL_SCRIPT="$EVAL_SCRIPT"
    --env EVAL_MODEL_PATH="$EVAL_MODEL_PATH"
    --env EVAL_OUTPUT_ROOT="$EVAL_OUTPUT_ROOT"
    --env LOG_ROOT="$EVAL_OUTPUT_ROOT/logs"
    --env STATUS_FILE="$EVAL_OUTPUT_ROOT/benchmark_status.tsv"
    --env SUMMARY_FILE="$EVAL_OUTPUT_ROOT/all_benchmarks_summary.md"
    --env MANIFEST_FILE="$EVAL_OUTPUT_ROOT/run_manifest.json"
    --env METRICS_FILE="$EVAL_OUTPUT_ROOT/metrics.json"
    --env EVAL_CHRONOS2_MODEL_PATH="$EVAL_CHRONOS2_MODEL_PATH"
    --env MODEL_NAME="$MODEL_NAME"
    --env TSRBENCH_ROOT="$TSRBENCH_ROOT"
    --env TINYBENCH_DATASET_ROOT="$TINYBENCH_DATASET_ROOT"
    --env TS_HAYSTACK_ROOT="$TS_HAYSTACK_ROOT"
    --env TIMESERIESEXAM_ROOT="$TIMESERIESEXAM_ROOT"
    --env TIMESERIESEXAM_DATA_FILE="$TIMESERIESEXAM_DATA_FILE"
    --env BENCHMARKS="$BENCHMARKS"
    --env RUN_ID="$RUN_ID"
    --env EVAL_PROTOCOL_HASH="$EVAL_PROTOCOL_HASH"
    --env HAYSTACK_SPLIT="$HAYSTACK_SPLIT"
    --env TINY_DATA_PARTITION="$TINY_DATA_PARTITION"
    --env TINY_PARTITION_SEED="$TINY_PARTITION_SEED"
    --env TSR_PROMPT_MODE="$TSR_PROMPT_MODE"
    --env TSR_MAX_MODEL_LEN="$TSR_MAX_MODEL_LEN"
    --env TSR_MAX_NEW_TOKENS="$TSR_MAX_NEW_TOKENS"
    --env TSR_BATCH_SIZE="$TSR_BATCH_SIZE"
    --env TSR_REQUEST_CHUNK_SIZE="$TSR_REQUEST_CHUNK_SIZE"
    --env TINY_MAX_MODEL_LEN="$TINY_MAX_MODEL_LEN"
    --env TINY_REQUEST_CHUNK_SIZE="$TINY_REQUEST_CHUNK_SIZE"
    --env TINY_GPU_MEMORY_UTILIZATION="$TINY_GPU_MEMORY_UTILIZATION"
    --env HAYSTACK_MAX_MODEL_LEN="$HAYSTACK_MAX_MODEL_LEN"
    --env HAYSTACK_MAX_NEW_TOKENS="$HAYSTACK_MAX_NEW_TOKENS"
    --env HAYSTACK_BATCH_SIZE="$HAYSTACK_BATCH_SIZE"
    --env HAYSTACK_REQUEST_CHUNK_SIZE="$HAYSTACK_REQUEST_CHUNK_SIZE"
    --env EXAM_MAX_MODEL_LEN="$EXAM_MAX_MODEL_LEN"
    --env EXAM_MAX_NEW_TOKENS="$EXAM_MAX_NEW_TOKENS"
    --env EXAM_BATCH_SIZE="$EXAM_BATCH_SIZE"
    --env EXAM_REQUEST_CHUNK_SIZE="$EXAM_REQUEST_CHUNK_SIZE"
    --env SEED="$SEED"
    --env DATA_VERSION=
    --env DATASET_SNAPSHOT_HASH=
    --env FORCE_EVAL="$FORCE_EVAL"
    --env SUITE_FORCE_EVAL="$FORCE_EVAL"
    --env PREFLIGHT_ONLY="$PREFLIGHT_ONLY"
    --env MAX_SAMPLES="$MAX_SAMPLES"
    --env OFFLINE="$OFFLINE"
    --env EVAL_GPUS=0,1,2,3,4,5,6,7
    --env EVAL_NUM_GPUS=8
    --env TS_GPUS_PER_PROCESS=2
    --env MODEL_COMPLETION_MARKER=TRAINING_COMPLETE.json
    --env HF_HUB_OFFLINE="$OFFLINE"
    --env TRANSFORMERS_OFFLINE="$OFFLINE"
    --env TOKENIZERS_PARALLELISM=false
    --cleanenv
)

"$CHATTS_SRUN_BIN" "$CHATTS_SINGULARITY_BIN" run \
    "${EVALUATION_OPTS[@]}" \
    "$CHATTS_EVAL_SIF_IMAGE" \
    bash -c '
        set -Eeuo pipefail
        export PROJECT_ROOT="$EVAL_PROJECT_ROOT"
        export MODEL_PATH="$EVAL_MODEL_PATH"
        export OUTPUT_ROOT="$EVAL_OUTPUT_ROOT"
        export CHRONOS2_MODEL_PATH="$EVAL_CHRONOS2_MODEL_PATH"
        export REQUIRE_TRAINING_MARKER=0
        export REQUIRE_MODEL_ON_PREFLIGHT=1
        export PYTHONHASHSEED="$SEED"
        unset AVAILABLE_GPUS_OVERRIDE
        mkdir -p "$JOB_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
            "$HF_HOME" "$HF_MODULES_CACHE"
        bash "$EVAL_SCRIPT"
        if [[ "$PREFLIGHT_ONLY" != "1" ]]; then
            test -f "$OUTPUT_ROOT/benchmark_status.tsv"
            test -f "$OUTPUT_ROOT/all_benchmarks_summary.md"
            test -f "$OUTPUT_ROOT/metrics.json"
        fi
    '

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "Standalone Slurm evaluation preflight passed; no artifacts were created."
else
    echo "Standalone Slurm evaluation completed: $EVAL_OUTPUT_ROOT"
fi

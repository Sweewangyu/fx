#!/usr/bin/env bash
# CHATTS_STUDIO_SBATCH_API=1
#SBATCH --job-name=chatts-studio
#SBATCH --partition=xj-h100-ib-2
#SBATCH --output=log/%j-chatts-studio.out
#SBATCH --error=log/%j-chatts-studio.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --account=nichemodel

set -Eeuo pipefail

# Dataset Studio invocation contract:
#   sbatch ... run_chatts_studio_pipeline.sbatch <immutable-resolved.yaml> <studio-job-id>
#
# The dashboard controls only the two arguments.  Infrastructure paths are
# trusted submission-environment settings and are never interpolated as shell
# commands.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHATTS_HOST_ROOT="${CHATTS_HOST_ROOT:-$HOME}"
CHATTS_TRAINING_DIR="${CHATTS_TRAINING_DIR:-$DEFAULT_TRAINING_DIR}"
CHATTS_SIF_IMAGE="${CHATTS_SIF_IMAGE:-$CHATTS_HOST_ROOT/chatts_v1.sif}"
CHATTS_HOST_CHRONOS2_PATH="${CHATTS_HOST_CHRONOS2_PATH:-$CHATTS_HOST_ROOT/chronos2}"
CHATTS_CONTAINER_TRAINING_DIR="${CHATTS_CONTAINER_TRAINING_DIR:-/workspace/ChatTS-Training}"
CHATTS_SHARED_HOST_PATH="${CHATTS_SHARED_HOST_PATH:-/share}"
CHATTS_SHARED_CONTAINER_PATH="${CHATTS_SHARED_CONTAINER_PATH:-/share}"
CHATTS_HOST_PYTHON_BIN="${CHATTS_HOST_PYTHON_BIN:-python3}"
CHATTS_SRUN_BIN="${CHATTS_SRUN_BIN:-srun}"
CHATTS_SINGULARITY_BIN="${CHATTS_SINGULARITY_BIN:-singularity}"

[[ "$CHATTS_CONTAINER_TRAINING_DIR" == /* ]] || {
    echo "CHATTS_CONTAINER_TRAINING_DIR must be absolute" >&2
    exit 2
}
[[ "$CHATTS_SHARED_CONTAINER_PATH" == /* ]] || {
    echo "CHATTS_SHARED_CONTAINER_PATH must be absolute" >&2
    exit 2
}
for required_dir in \
    "$CHATTS_TRAINING_DIR" \
    "$CHATTS_HOST_CHRONOS2_PATH" \
    "$CHATTS_SHARED_HOST_PATH"; do
    [[ -d "$required_dir" ]] || {
        echo "Required host directory not found: $required_dir" >&2
        exit 1
    }
done
for required_file in \
    "$CHATTS_SIF_IMAGE" \
    "$CHATTS_TRAINING_DIR/scripts/slurm/load_studio_pipeline_config.py" \
    "$CHATTS_TRAINING_DIR/scripts/full/run_chronos2_best_two_stage.sh"; do
    [[ -f "$required_file" ]] || {
        echo "Required host file not found: $required_file" >&2
        exit 1
    }
done
for required_command in \
    "$CHATTS_HOST_PYTHON_BIN" \
    "$CHATTS_SRUN_BIN" \
    "$CHATTS_SINGULARITY_BIN"; do
    command -v "$required_command" >/dev/null 2>&1 || {
        echo "Required command not found: $required_command" >&2
        exit 1
    }
done

if [[ -n "${SINGULARITY_TMPDIR:-}" && -z "${APPTAINER_TMPDIR:-}" ]]; then
    export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
    unset SINGULARITY_TMPDIR
fi

JOB_TMP="${CHATTS_JOB_TMP_ROOT:-/tmp}/chatts_studio_${SLURM_JOB_ID}_${STUDIO_JOB_ID}"
JOB_HOME="$JOB_TMP/home"
TRITON_CACHE="$JOB_TMP/triton"
HF_CACHE="$JOB_TMP/huggingface"
RESOLVED_ENV_FILE="$JOB_TMP/resolved-training.env"
mkdir -p "$JOB_TMP" "$JOB_HOME" "$TRITON_CACHE" "$HF_CACHE"

"$CHATTS_HOST_PYTHON_BIN" \
    "$CHATTS_TRAINING_DIR/scripts/slurm/load_studio_pipeline_config.py" \
    "$FROZEN_CONFIG" \
    --expected-job-id "$STUDIO_JOB_ID" \
    > "$RESOLVED_ENV_FILE"

SEEN_ENV_NAMES=" "
declare -a TRAINING_ENV=()
while IFS= read -r assignment || [[ -n "$assignment" ]]; do
    [[ -n "$assignment" && "$assignment" == *=* ]] || {
        echo "Malformed loader output" >&2
        exit 2
    }
    name="${assignment%%=*}"
    case "$name" in
        CHRONOS2_MODEL_PATH|DATASET_DIR|DATASET_SNAPSHOT_HASH|DATA_VERSION|DEEPSPEED_INCLUDE|FINAL_MODEL_PATH|FORCE_TRAIN|KEEP_STAGE1|MASTER_PORT|MODEL_PATH|OUTPUT_ROOT|PIPELINE_MODE|PREFLIGHT_ONLY|PROJECT_ROOT|S1_LR|S2_LR|SEED|STAGE1_CUTOFF_LEN|STAGE1_DATASETS|STAGE1_EVAL_STEPS|STAGE1_GRADIENT_ACCUMULATION_STEPS|STAGE1_INTERLEAVE_PROBS|STAGE1_LOGGING_STEPS|STAGE1_LR_SCHEDULER_TYPE|STAGE1_MAX_STEPS|STAGE1_MIX_STRATEGY|STAGE1_NUM_TRAIN_EPOCHS|STAGE1_OUT|STAGE1_PER_DEVICE_EVAL_BATCH_SIZE|STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE|STAGE1_PREPROCESSING_NUM_WORKERS|STAGE1_SAVE_STEPS|STAGE1_TIMESERIES_SFT_LR|STAGE1_VAL_SIZE|STAGE1_WARMUP_RATIO|STAGE2_CUTOFF_LEN|STAGE2_DATASETS|STAGE2_EVAL_STEPS|STAGE2_GRADIENT_ACCUMULATION_STEPS|STAGE2_INTERLEAVE_PROBS|STAGE2_LOGGING_STEPS|STAGE2_LR_SCHEDULER_TYPE|STAGE2_MAX_STEPS|STAGE2_MIX_STRATEGY|STAGE2_NUM_TRAIN_EPOCHS|STAGE2_PER_DEVICE_EVAL_BATCH_SIZE|STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE|STAGE2_PREPROCESSING_NUM_WORKERS|STAGE2_SAVE_STEPS|STAGE2_TIMESERIES_SFT_LR|STAGE2_VAL_SIZE|STAGE2_WARMUP_RATIO|TRAINING_RECIPE_HASH|TRIAL_CONFIG_HASH|TRIAL_ID) ;;
        *)
            echo "Unexpected loader environment name: $name" >&2
            exit 2
            ;;
    esac
    [[ "$SEEN_ENV_NAMES" != *" $name "* ]] || {
        echo "Duplicate loader environment name: $name" >&2
        exit 2
    }
    SEEN_ENV_NAMES+="$name "
    export "$assignment"
    TRAINING_ENV+=(--env "$assignment")
done < "$RESOLVED_ENV_FILE"

for required_name in \
    PROJECT_ROOT MODEL_PATH OUTPUT_ROOT FINAL_MODEL_PATH CHRONOS2_MODEL_PATH \
    DATASET_DIR PIPELINE_MODE TRAINING_RECIPE_HASH DATASET_SNAPSHOT_HASH \
    TRIAL_CONFIG_HASH TRIAL_ID; do
    [[ "$SEEN_ENV_NAMES" == *" $required_name "* ]] || {
        echo "Loader did not emit required environment: $required_name" >&2
        exit 2
    }
done
[[ "$PROJECT_ROOT" == "$CHATTS_CONTAINER_TRAINING_DIR" ]] || {
    echo "Resolved training.project_root must match CHATTS_CONTAINER_TRAINING_DIR" >&2
    echo "  resolved: $PROJECT_ROOT" >&2
    echo "  mounted:  $CHATTS_CONTAINER_TRAINING_DIR" >&2
    exit 2
}
CHATTS_CONTAINER_CHRONOS2_PATH="${CHATTS_CONTAINER_CHRONOS2_PATH:-$CHRONOS2_MODEL_PATH}"
[[ "$CHRONOS2_MODEL_PATH" == "$CHATTS_CONTAINER_CHRONOS2_PATH" ]] || {
    echo "Resolved Chronos-2 path must match CHATTS_CONTAINER_CHRONOS2_PATH" >&2
    exit 2
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
require_shared_container_path MODEL_PATH "$MODEL_PATH"
require_shared_container_path DATASET_DIR "$DATASET_DIR"
require_shared_container_path OUTPUT_ROOT "$OUTPUT_ROOT"
require_shared_container_path FINAL_MODEL_PATH "$FINAL_MODEL_PATH"

CONFIG_FILE_SHA256="$("$CHATTS_HOST_PYTHON_BIN" - "$FROZEN_CONFIG" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

echo "============================================================"
echo " ChatTS Dataset Studio Slurm pipeline"
echo " Slurm job:          $SLURM_JOB_ID"
echo " Studio job:         $STUDIO_JOB_ID"
echo " Frozen config:      $FROZEN_CONFIG"
echo " Config file SHA256: $CONFIG_FILE_SHA256"
echo " Trial config hash:  $TRIAL_CONFIG_HASH"
echo " Recipe hash:        $TRAINING_RECIPE_HASH"
echo " Pipeline mode:      $PIPELINE_MODE"
echo " Dataset version:    $DATA_VERSION"
echo " Dataset snapshot:   $DATASET_SNAPSHOT_HASH"
echo " Base model:         $MODEL_PATH"
echo " Dataset directory:  $DATASET_DIR"
echo " Output root:        $OUTPUT_ROOT"
echo " Final model:        $FINAL_MODEL_PATH"
echo " Singularity image:  $CHATTS_SIF_IMAGE"
echo "============================================================"

COMMON_OPTS=(
    --nv
    --pwd "$CHATTS_CONTAINER_TRAINING_DIR"
    --home "$JOB_HOME"
    --bind "$CHATTS_TRAINING_DIR:$CHATTS_CONTAINER_TRAINING_DIR:rw"
    --bind "$CHATTS_HOST_CHRONOS2_PATH:$CHATTS_CONTAINER_CHRONOS2_PATH:ro"
    --bind "$CHATTS_SHARED_HOST_PATH:$CHATTS_SHARED_CONTAINER_PATH:rw"
    --bind "$JOB_TMP:$JOB_TMP:rw"
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
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env TOKENIZERS_PARALLELISM=false
    "${TRAINING_ENV[@]}"
    --cleanenv
)

"$CHATTS_SRUN_BIN" "$CHATTS_SINGULARITY_BIN" run \
    "${COMMON_OPTS[@]}" \
    "$CHATTS_SIF_IMAGE" \
    bash -c '
        set -Eeuo pipefail
        export PYTHONHASHSEED="$SEED"
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        export NCCL_DEBUG=WARN
        export DEEPSPEED_TIMEOUT=120
        mkdir -p "$JOB_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
            "$HF_HOME" "$HF_MODULES_CACHE"
        exec bash "$PROJECT_ROOT/scripts/full/run_chronos2_best_two_stage.sh"
    '

echo "Studio Slurm training completed: $FINAL_MODEL_PATH"

#!/bin/bash
#SBATCH --job-name=chatts-all1
#SBATCH --partition=xj-h100-ib-2
#SBATCH --output=log/%j-chatts-all1.out
#SBATCH --error=log/%j-chatts-all1.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --account=nichemodel

set -Eeuo pipefail

HOST_ROOT=/data/hpc/home/yu.wang17
TRAINING_DIR=$HOST_ROOT/ChatTS-Training
SIF_IMAGE=$HOST_ROOT/chatts_v1.sif

# Chronos-2 权重在宿主机的实际目录。若你的目录名不同，只修改这一行。
HOST_CHRONOS2_PATH="${HOST_CHRONOS2_PATH:-$HOST_ROOT/chronos2}"
CONTAINER_CHRONOS2_PATH=/workspace/chronos2

# 按服务器上的真实路径确认下面几项即可。
MODEL_PATH=/share/airesearch/data/finiverse/model/ChatTS-Qwen3-8B
OUTPUT_DIR=/share/airesearch/data/finiverse/output/ChatTS-all-data-one-stage/seed42-${SLURM_JOB_ID}

# 直接使用六个未经过质量/难度筛选的 annotated source JSONL。
RAW_DATASET_ROOT=/share/airesearch/data/finiverse/traindata/merged_labels/annotated
SELECT_DATASETS=chatts,time-mqa,tsqa
MIN_TOTAL_ROWS=400000

mkdir -p log "$(dirname "$OUTPUT_DIR")"

[[ -f "$MODEL_PATH/config.json" ]] || {
    echo "Base model config not found on host: $MODEL_PATH/config.json" >&2
    exit 1
}
[[ -d "$HOST_CHRONOS2_PATH" ]] || {
    echo "Chronos-2 model directory not found on host: $HOST_CHRONOS2_PATH" >&2
    echo "Edit HOST_CHRONOS2_PATH in this sbatch file to its real host path." >&2
    exit 1
}

# New Apptainer versions prefer APPTAINER_TMPDIR. Preserve the configured
# directory while avoiding the deprecated-variable warning.
if [[ -n "${SINGULARITY_TMPDIR:-}" && -z "${APPTAINER_TMPDIR:-}" ]]; then
    export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
    unset SINGULARITY_TMPDIR
fi

[[ -d "$RAW_DATASET_ROOT" ]] || {
    echo "Full source dataset directory not found: $RAW_DATASET_ROOT" >&2
    exit 1
}

echo "Raw dataset root:  $RAW_DATASET_ROOT"
echo "Dataset selection: $SELECT_DATASETS"
echo "Chronos-2 host:    $HOST_CHRONOS2_PATH"
echo "Model output:      $OUTPUT_DIR"

# Reuse the CUDA/Triton isolation settings from the cluster script that is
# known to work. In particular, HOME and all compiler caches live under a
# job-specific /tmp directory, so DeepSpeed cannot load ~/.triton/*.so built
# against a different NVIDIA driver.
JOB_TMP=/tmp/chatts_all1_${SLURM_JOB_ID}
JOB_HOME=$JOB_TMP/home
TRITON_CACHE=$JOB_TMP/triton
HF_CACHE=$JOB_TMP/huggingface

COMMON_OPTS=(
    --nv
    --pwd /workspace/ChatTS-Training
    --bind "$TRAINING_DIR:/workspace/ChatTS-Training:rw"
    --bind "$HOST_CHRONOS2_PATH:$CONTAINER_CHRONOS2_PATH:ro"
    --bind /share:/share:rw
    --env LD_PRELOAD=/.singularity.d/libs/libcuda.so.1
    --env CUDA_HOME=/usr/local/cuda
    --env JOB_HOME="$JOB_HOME"
    --env TRITON_CACHE_DIR="$TRITON_CACHE"
    --env XDG_CACHE_HOME="$JOB_TMP/xdg"
    --env HF_HOME="$HF_CACHE"
    --env TRANSFORMERS_CACHE="$HF_CACHE"
    --env HF_MODULES_CACHE="$HF_CACHE/modules"
    --env MODEL_PATH="$MODEL_PATH"
    --env CHRONOS2_MODEL_PATH="$CONTAINER_CHRONOS2_PATH"
    --env RAW_DATASET_ROOT="$RAW_DATASET_ROOT"
    --env RUNTIME_DATASET_DIR="$JOB_TMP/dataset_registry"
    --env OUTPUT_DIR="$OUTPUT_DIR"
    --env SELECT_DATASETS="$SELECT_DATASETS"
    --env MIN_TOTAL_ROWS="$MIN_TOTAL_ROWS"
    --env SEED=42
    --env LEARNING_RATE=1e-5
    --env TIMESERIES_SFT_LR=1e-5
    --env CUTOFF_LEN=10000
    --env PER_DEVICE_TRAIN_BATCH_SIZE=1
    --env GRADIENT_ACCUMULATION_STEPS=64
    --env MASTER_PORT=19901
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env TOKENIZERS_PARALLELISM=false
    --cleanenv
)

srun singularity run "${COMMON_OPTS[@]}" "$SIF_IMAGE" \
    bash -c '
        set -Eeuo pipefail
        export HOME="$JOB_HOME"
        rm -rf -- "$JOB_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
        mkdir -p "$JOB_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
            "$HF_HOME" "$HF_MODULES_CACHE"
        exec bash /workspace/ChatTS-Training/scripts/full/run_chronos2_all_data_one_stage.sh
    '

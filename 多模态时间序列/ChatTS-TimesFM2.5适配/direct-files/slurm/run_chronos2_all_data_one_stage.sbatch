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

# 按服务器上的真实路径确认下面三项即可。
MODEL_PATH=/share/airesearch/data/finiverse/model/ChatTS-Qwen3-8B
CHRONOS2_MODEL_PATH=/workspace/chronos2
OUTPUT_DIR=/share/airesearch/data/finiverse/output/ChatTS-all-data-one-stage/seed42-${SLURM_JOB_ID}

# 可选：chatts、time-mqa、tsqa、all。
# 这里表示只训练 ChatTS 原始四个数据集 + Time-MQA + TSQA。
SELECT_DATASETS=chatts,time-mqa,tsqa

# 自动读取 Dataset Studio 当前激活版本（例如 datav3）的真实快照路径。
DATASET_DIR="$(python3 - "$TRAINING_DIR" <<'PY'
import json
import sys
from pathlib import Path

registry = Path(sys.argv[1]) / "data" / "studio_versions"
active = json.loads((registry / "active.json").read_text(encoding="utf-8"))
profile = json.loads((registry / f"{active['version']}.json").read_text(encoding="utf-8"))
print(profile["snapshot_path"])
PY
)"

mkdir -p log "$(dirname "$OUTPUT_DIR")"

# 先在宿主机验证路径。若这里通过而容器内失败，就是 /share 没有挂载。
for required_file in manifest.json dataset_info.json training.env; do
    if [[ ! -f "$DATASET_DIR/$required_file" ]]; then
        echo "Dataset snapshot is incomplete or the active path is stale:" >&2
        echo "  DATASET_DIR=$DATASET_DIR" >&2
        echo "  missing=$DATASET_DIR/$required_file" >&2
        echo "Please activate a complete Dataset Studio version or set DATASET_DIR manually." >&2
        exit 1
    fi
done
[[ -f "$MODEL_PATH/config.json" ]] || {
    echo "Base model config not found on host: $MODEL_PATH/config.json" >&2
    exit 1
}

echo "Dataset snapshot: $DATASET_DIR"
echo "Dataset selection: $SELECT_DATASETS"
echo "Model output:      $OUTPUT_DIR"

srun singularity exec --nv \
    --bind "$TRAINING_DIR:/workspace/ChatTS-Training" \
    --bind /share:/share \
    --pwd /workspace/ChatTS-Training \
    --env MODEL_PATH="$MODEL_PATH" \
    --env CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH" \
    --env DATASET_DIR="$DATASET_DIR" \
    --env OUTPUT_DIR="$OUTPUT_DIR" \
    --env SELECT_DATASETS="$SELECT_DATASETS" \
    --env SEED=42 \
    --env LEARNING_RATE=1e-5 \
    --env TIMESERIES_SFT_LR=1e-5 \
    --env MASTER_PORT=19901 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$SIF_IMAGE" \
    bash /workspace/ChatTS-Training/scripts/full/run_chronos2_all_data_one_stage.sh

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

srun singularity exec --nv \
    --bind "$TRAINING_DIR:/workspace/ChatTS-Training" \
    --pwd /workspace/ChatTS-Training \
    --env MODEL_PATH="$MODEL_PATH" \
    --env CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH" \
    --env DATASET_DIR="$DATASET_DIR" \
    --env OUTPUT_DIR="$OUTPUT_DIR" \
    --env SEED=42 \
    --env LEARNING_RATE=1e-5 \
    --env TIMESERIES_SFT_LR=1e-5 \
    --env MASTER_PORT=19901 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$SIF_IMAGE" \
    bash /workspace/ChatTS-Training/scripts/full/run_chronos2_all_data_one_stage.sh

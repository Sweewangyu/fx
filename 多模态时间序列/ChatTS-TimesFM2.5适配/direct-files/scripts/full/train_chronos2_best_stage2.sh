#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: bash train_chronos2_best_stage2.sh <learning-rate> <stage1-dir> <output-dir>

LR="${1:?Usage: train_chronos2_best_stage2.sh <learning-rate> <stage1-dir> <output-dir>}"
STAGE1_OUT="${2:?Usage: train_chronos2_best_stage2.sh <learning-rate> <stage1-dir> <output-dir>}"
STAGE2_OUT="${3:?Usage: train_chronos2_best_stage2.sh <learning-rate> <stage1-dir> <output-dir>}"

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS-Training}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
FINALIZER="${FINALIZER:-${PROJECT_ROOT}/scripts/finalize_chatts_best_checkpoint.py}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$(dirname "$STAGE2_OUT")/logs/tensorboard/stage2}"
SEED="${SEED:-42}"
DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-19901}"
STAGE2_NUM_TRAIN_EPOCHS="${STAGE2_NUM_TRAIN_EPOCHS:-1}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-0}"

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer." >&2; exit 2; }
[[ "$LR" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || { echo "Invalid learning rate: $LR" >&2; exit 2; }
[[ "$STAGE2_MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "STAGE2_MAX_STEPS must be non-negative." >&2; exit 2; }
[[ -d "$PROJECT_ROOT" ]] || { echo "Training project not found: $PROJECT_ROOT" >&2; exit 1; }
[[ -f "$STAGE1_OUT/config.json" ]] || { echo "Stage1 model config not found: $STAGE1_OUT/config.json" >&2; exit 1; }
[[ -f "$STAGE1_OUT/best_model_manifest.json" ]] || { echo "Stage1 best-model manifest not found." >&2; exit 1; }
[[ -d "$CHRONOS2_MODEL_PATH" ]] || { echo "Chronos-2 model not found: $CHRONOS2_MODEL_PATH" >&2; exit 1; }
[[ -f "$FINALIZER" ]] || { echo "Checkpoint finalizer not found: $FINALIZER" >&2; exit 1; }
[[ ! -e "$STAGE2_OUT" ]] || { echo "Stage2 output already exists: $STAGE2_OUT" >&2; exit 2; }

mkdir -p "$(dirname "$STAGE2_OUT")" "$TENSORBOARD_DIR"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-120}"

TRAIN_LENGTH_ARGS=(--num_train_epochs "$STAGE2_NUM_TRAIN_EPOCHS")
if (( STAGE2_MAX_STEPS > 0 )); then
    TRAIN_LENGTH_ARGS+=(--max_steps "$STAGE2_MAX_STEPS")
fi

echo "[Stage2] seed=$SEED lr=$LR stage1=$STAGE1_OUT output=$STAGE2_OUT"
cd "$PROJECT_ROOT"
deepspeed --include "$DEEPSPEED_INCLUDE" --master_port="$MASTER_PORT" src/train.py \
    --deepspeed ds_config/ds_config_2.json \
    --stage sft \
    --model_name_or_path "$STAGE1_OUT" \
    --ts_encoder_type chronos2 \
    --chronos2_model_name_or_path "$CHRONOS2_MODEL_PATH" \
    --dataset "sft,align_random,finiverse_time_mqa,finiverse_tsaqa" \
    --mix_strategy concat \
    --template chatts \
    --do_train \
    --finetuning_type full \
    --output_dir "$STAGE2_OUT" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate "$LR" \
    --timeseries_sft_lr "$LR" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.02 \
    "${TRAIN_LENGTH_ARGS[@]}" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --plot_loss \
    --bf16 \
    --save_only_model True \
    --save_safetensors False \
    --preprocessing_num_workers 96 \
    --trust_remote_code True \
    --flash_attn fa2 \
    --cutoff_len 2048 \
    --val_size 0.05 \
    --per_device_eval_batch_size 4 \
    --eval_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --seed "$SEED" \
    --data_seed "$SEED" \
    --report_to tensorboard \
    --logging_dir "$TENSORBOARD_DIR"

python3 "$FINALIZER" \
    --checkpoint-dir "$STAGE2_OUT" \
    --stage stage2 \
    --seed "$SEED" \
    --learning-rate "$LR" \
    --chronos2-model-path "$CHRONOS2_MODEL_PATH" \
    --cleanup-checkpoints

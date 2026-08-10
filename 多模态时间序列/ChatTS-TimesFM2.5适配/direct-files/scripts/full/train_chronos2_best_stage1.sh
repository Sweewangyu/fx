#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: bash train_chronos2_best_stage1.sh <learning-rate> <output-dir>

LR="${1:?Usage: train_chronos2_best_stage1.sh <learning-rate> <output-dir>}"
STAGE1_OUT="${2:?Usage: train_chronos2_best_stage1.sh <learning-rate> <output-dir>}"

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS-Training}"
MODEL_PATH="${MODEL_PATH:-/share/airesearch/data/finiverse/model/ChatTS-Qwen3-8B}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
FINALIZER="${FINALIZER:-${PROJECT_ROOT}/scripts/finalize_chatts_best_checkpoint.py}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$(dirname "$STAGE1_OUT")/logs/tensorboard/stage1}"
SEED="${SEED:-42}"
DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-19901}"
STAGE1_NUM_TRAIN_EPOCHS="${STAGE1_NUM_TRAIN_EPOCHS:-3}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-0}"

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer." >&2; exit 2; }
[[ "$LR" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || { echo "Invalid learning rate: $LR" >&2; exit 2; }
[[ "$STAGE1_MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "STAGE1_MAX_STEPS must be non-negative." >&2; exit 2; }
[[ -d "$PROJECT_ROOT" ]] || { echo "Training project not found: $PROJECT_ROOT" >&2; exit 1; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Base model config not found: $MODEL_PATH/config.json" >&2; exit 1; }
[[ -d "$CHRONOS2_MODEL_PATH" ]] || { echo "Chronos-2 model not found: $CHRONOS2_MODEL_PATH" >&2; exit 1; }
[[ -f "$FINALIZER" ]] || { echo "Checkpoint finalizer not found: $FINALIZER" >&2; exit 1; }
[[ ! -e "$STAGE1_OUT" ]] || { echo "Stage1 output already exists: $STAGE1_OUT" >&2; exit 2; }

mkdir -p "$(dirname "$STAGE1_OUT")" "$TENSORBOARD_DIR"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-120}"

TRAIN_LENGTH_ARGS=(--num_train_epochs "$STAGE1_NUM_TRAIN_EPOCHS")
if (( STAGE1_MAX_STEPS > 0 )); then
    TRAIN_LENGTH_ARGS+=(--max_steps "$STAGE1_MAX_STEPS")
fi

echo "[Stage1] seed=$SEED lr=$LR base=$MODEL_PATH output=$STAGE1_OUT"
cd "$PROJECT_ROOT"
deepspeed --include "$DEEPSPEED_INCLUDE" --master_port="$MASTER_PORT" src/train.py \
    --deepspeed ds_config/ds_config_2.json \
    --stage sft \
    --model_name_or_path "$MODEL_PATH" \
    --ts_encoder_type chronos2 \
    --chronos2_model_name_or_path "$CHRONOS2_MODEL_PATH" \
    --dataset "align_256,ift" \
    --interleave_probs "0.9,0.1" \
    --do_train \
    --mix_strategy interleave_over \
    --template chatts \
    --finetuning_type full \
    --output_dir "$STAGE1_OUT" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate "$LR" \
    --timeseries_sft_lr "$LR" \
    --warmup_ratio 0.02 \
    "${TRAIN_LENGTH_ARGS[@]}" \
    --plot_loss \
    --bf16 \
    --save_only_model True \
    --save_safetensors False \
    --preprocessing_num_workers 96 \
    --overwrite_cache \
    --trust_remote_code True \
    --flash_attn fa2 \
    --cutoff_len 2048 \
    --val_size 0.05 \
    --per_device_eval_batch_size 2 \
    --eval_strategy steps \
    --eval_steps 200 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --seed "$SEED" \
    --data_seed "$SEED" \
    --report_to tensorboard \
    --logging_dir "$TENSORBOARD_DIR"

python3 "$FINALIZER" \
    --checkpoint-dir "$STAGE1_OUT" \
    --stage stage1 \
    --seed "$SEED" \
    --learning-rate "$LR" \
    --chronos2-model-path "$CHRONOS2_MODEL_PATH" \
    --cleanup-checkpoints

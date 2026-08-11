#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: bash train_chronos2_best_stage1.sh <learning-rate> <output-dir>

LR="${1:?Usage: train_chronos2_best_stage1.sh <learning-rate> <output-dir>}"
STAGE1_OUT="${2:?Usage: train_chronos2_best_stage1.sh <learning-rate> <output-dir>}"

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS-Training}"
MODEL_PATH="${MODEL_PATH:-/share/airesearch/data/finiverse/model/ChatTS-Qwen3-8B}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data}"
FINALIZER="${FINALIZER:-${PROJECT_ROOT}/scripts/finalize_chatts_best_checkpoint.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$(dirname "$STAGE1_OUT")/logs/tensorboard/stage1}"
SEED="${SEED:-42}"
DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-19901}"
STAGE1_NUM_TRAIN_EPOCHS="${STAGE1_NUM_TRAIN_EPOCHS:-3}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-0}"
STAGE1_TIMESERIES_SFT_LR="${STAGE1_TIMESERIES_SFT_LR:-$LR}"
STAGE1_DATASETS="${STAGE1_DATASETS:-align_256,ift}"
# Preserve an explicitly empty value so callers using concat can disable
# --interleave_probs; keep 0.9/0.1 only when the variable is truly unset.
STAGE1_INTERLEAVE_PROBS="${STAGE1_INTERLEAVE_PROBS-0.9,0.1}"
STAGE1_MIX_STRATEGY="${STAGE1_MIX_STRATEGY:-interleave_over}"
STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE="${STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
STAGE1_GRADIENT_ACCUMULATION_STEPS="${STAGE1_GRADIENT_ACCUMULATION_STEPS:-32}"
STAGE1_LR_SCHEDULER_TYPE="${STAGE1_LR_SCHEDULER_TYPE:-cosine}"
STAGE1_WARMUP_RATIO="${STAGE1_WARMUP_RATIO:-0.02}"
STAGE1_LOGGING_STEPS="${STAGE1_LOGGING_STEPS:-1}"
STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-200}"
STAGE1_EVAL_STEPS="${STAGE1_EVAL_STEPS:-200}"
STAGE1_VAL_SIZE="${STAGE1_VAL_SIZE:-0.05}"
STAGE1_PER_DEVICE_EVAL_BATCH_SIZE="${STAGE1_PER_DEVICE_EVAL_BATCH_SIZE:-2}"
STAGE1_CUTOFF_LEN="${STAGE1_CUTOFF_LEN:-2048}"
STAGE1_PREPROCESSING_NUM_WORKERS="${STAGE1_PREPROCESSING_NUM_WORKERS:-96}"

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer." >&2; exit 2; }
[[ "$LR" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || { echo "Invalid learning rate: $LR" >&2; exit 2; }
[[ "$STAGE1_MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "STAGE1_MAX_STEPS must be non-negative." >&2; exit 2; }
[[ -n "$DATASET_DIR" ]] || { echo "DATASET_DIR must not be empty." >&2; exit 2; }
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
DATASET_ARGS=(--dataset_dir "$DATASET_DIR" --dataset "$STAGE1_DATASETS" --mix_strategy "$STAGE1_MIX_STRATEGY")
if [[ -n "$STAGE1_INTERLEAVE_PROBS" ]]; then
    DATASET_ARGS+=(--interleave_probs "$STAGE1_INTERLEAVE_PROBS")
fi

echo "[Stage1] seed=$SEED lr=$LR ts_lr=$STAGE1_TIMESERIES_SFT_LR base=$MODEL_PATH output=$STAGE1_OUT"
echo "[Stage1] dataset_dir=$DATASET_DIR datasets=$STAGE1_DATASETS mix=$STAGE1_MIX_STRATEGY interleave_probs=${STAGE1_INTERLEAVE_PROBS:-<none>}"
cd "$PROJECT_ROOT"
# DeepSpeed must retain optimizer/scheduler state for Trainer to reload the
# best checkpoint at the end, so save_only_model must remain False.
deepspeed --include "$DEEPSPEED_INCLUDE" --master_port="$MASTER_PORT" src/train.py \
    --deepspeed ds_config/ds_config_2.json \
    --stage sft \
    --model_name_or_path "$MODEL_PATH" \
    --ts_encoder_type chronos2 \
    --chronos2_model_name_or_path "$CHRONOS2_MODEL_PATH" \
    "${DATASET_ARGS[@]}" \
    --do_train \
    --template chatts \
    --finetuning_type full \
    --output_dir "$STAGE1_OUT" \
    --per_device_train_batch_size "$STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$STAGE1_GRADIENT_ACCUMULATION_STEPS" \
    --lr_scheduler_type "$STAGE1_LR_SCHEDULER_TYPE" \
    --logging_steps "$STAGE1_LOGGING_STEPS" \
    --save_strategy steps \
    --save_steps "$STAGE1_SAVE_STEPS" \
    --save_total_limit 1 \
    --learning_rate "$LR" \
    --timeseries_sft_lr "$STAGE1_TIMESERIES_SFT_LR" \
    --warmup_ratio "$STAGE1_WARMUP_RATIO" \
    "${TRAIN_LENGTH_ARGS[@]}" \
    --plot_loss \
    --bf16 \
    --save_only_model False \
    --save_safetensors False \
    --preprocessing_num_workers "$STAGE1_PREPROCESSING_NUM_WORKERS" \
    --overwrite_cache \
    --trust_remote_code True \
    --flash_attn fa2 \
    --cutoff_len "$STAGE1_CUTOFF_LEN" \
    --val_size "$STAGE1_VAL_SIZE" \
    --per_device_eval_batch_size "$STAGE1_PER_DEVICE_EVAL_BATCH_SIZE" \
    --eval_strategy steps \
    --eval_steps "$STAGE1_EVAL_STEPS" \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --seed "$SEED" \
    --data_seed "$SEED" \
    --report_to tensorboard \
    --logging_dir "$TENSORBOARD_DIR"

"$PYTHON_BIN" "$FINALIZER" \
    --checkpoint-dir "$STAGE1_OUT" \
    --stage stage1 \
    --seed "$SEED" \
    --learning-rate "$LR" \
    --chronos2-model-path "$CHRONOS2_MODEL_PATH" \
    --input-model-dir "$MODEL_PATH" \
    --cleanup-checkpoints

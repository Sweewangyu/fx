#!/usr/bin/env bash
# Usage: bash train_timesfm2_5_grid_stage2.sh <LR> <STAGE1_TAG> <STAGE2_TAG>

set -Eeuo pipefail

LR="${1:?Usage: train_timesfm2_5_grid_stage2.sh <LR> <STAGE1_TAG> <STAGE2_TAG>}"
STAGE1_TAG="${2:?Missing STAGE1_TAG}"
STAGE2_TAG="${3:?Missing STAGE2_TAG}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/share/airesearch/data/finiverse/output/ChatTS-Qwen3-1.7B-PR}"
TIMESFM_MODEL_PATH="${TIMESFM_MODEL_PATH:-/workspace/timesfm}"
TRAINING_ROOT="${TRAINING_ROOT:-/workspace/ChatTS-Train/ChatTS-Training}"

MODEL_PATH="$OUTPUT_ROOT/$STAGE1_TAG"
STAGE2_OUT="$OUTPUT_ROOT/$STAGE2_TAG"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Stage 1 checkpoint does not exist: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$TIMESFM_MODEL_PATH/model.safetensors" ]]; then
    echo "Error: TimesFM model.safetensors does not exist under: $TIMESFM_MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"
cd "$TRAINING_ROOT"

echo "[Stage2] LR=$LR  Stage1=$MODEL_PATH  ->  $STAGE2_OUT"
echo "[Stage2] Encoder=timesfm2_5  Backbone=$TIMESFM_MODEL_PATH"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=19901 src/train.py \
    --deepspeed ds_config/ds_config_2.json \
    --stage sft \
    --model_name_or_path "$MODEL_PATH" \
    --ts_encoder_type timesfm2_5 \
    --timesfm_model_name_or_path "$TIMESFM_MODEL_PATH" \
    --dataset "sft,ift,align_random" \
    --interleave_probs "0.6,0.2,0.2" \
    --mix_strategy interleave_over \
    --template chatts \
    --do_train \
    --finetuning_type full \
    --output_dir "$STAGE2_OUT" \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate "$LR" \
    --timeseries_sft_lr "$LR" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.02 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --plot_loss \
    --bf16 \
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
    --report_to tensorboard \
    --logging_dir "$OUTPUT_ROOT/logs/tensorboard/$STAGE2_TAG"

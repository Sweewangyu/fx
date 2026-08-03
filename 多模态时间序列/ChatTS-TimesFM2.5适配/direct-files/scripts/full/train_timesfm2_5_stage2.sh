#!/usr/bin/env bash

set -Eeuo pipefail

# TS-Reasoner-style instruction stage. Explicit encoder selection plus the
# stage-1 config restore the trainable projector while reloading the frozen
# Google checkpoint. The complete ChatTS LLM remains trainable.
# Paper recipe: 30K instructions, global batch 32, lr 2e-5, two epochs.
# With 8 GPUs: 1 sample/GPU * 4 accumulation steps * 8 GPUs = 32.

MODEL_PATH="${MODEL_PATH:-[OUTPUT_PATH_TIMESFM_STAGE_1]}"
TIMESFM_MODEL_PATH="${TIMESFM_MODEL_PATH:-google/timesfm-2.5-200m-pytorch}"
OUTPUT_PATH="${OUTPUT_PATH:-[OUTPUT_PATH_TIMESFM_STAGE_2]}"

NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=120 deepspeed --num_gpus 8 --master_port=19901 src/train.py \
    --deepspeed ds_config/ds_config_3.json \
    --stage sft \
    --model_name_or_path "$MODEL_PATH" \
    --ts_encoder_type timesfm2_5 \
    --timesfm_model_name_or_path "$TIMESFM_MODEL_PATH" \
    --dataset "stage_2_30K" \
    --interleave_probs "1.0" \
    --do_train \
    --mix_strategy "interleave_over" \
    --template "chatts" \
    --finetuning_type full \
    --output_dir "$OUTPUT_PATH" \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.02 \
    --num_train_epochs 2 \
    --plot_loss \
    --bf16 \
    --save_only_model \
    --save_safetensors true \
    --preprocessing_num_workers 96 \
    --trust_remote_code true \
    --cutoff_len 10000

python scripts/full/save_ts_encoder_config.py "$OUTPUT_PATH" \
    --encoder-type timesfm2_5 \
    --backbone-path "$TIMESFM_MODEL_PATH"

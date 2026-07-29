#!/usr/bin/env bash

# Alignment stage: frozen TimesFM 2.5 + trainable TS-to-text projector + ChatTS LLM.
# Install once with: pip install -e ".[timesfm,deepspeed]"

NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=120 deepspeed --num_gpus 8 --master_port=19901 src/train.py \
    --deepspeed ds_config/ds_config_3.json \
    --stage sft \
    --model_name_or_path "[PATH_TO_CHATTS_BASE_MODEL]" \
    --ts_encoder_type timesfm2_5 \
    --timesfm_model_name_or_path "google/timesfm-2.5-200m-pytorch" \
    --dataset "align_256,ift" \
    --interleave_probs "0.9,0.1" \
    --do_train \
    --mix_strategy "interleave_over" \
    --template "chatts" \
    --finetuning_type full \
    --output_dir "[OUTPUT_PATH_TIMESFM_STAGE_1]" \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --timeseries_sft_lr 1e-4 \
    --warmup_ratio 0.02 \
    --max_steps 1000 \
    --plot_loss \
    --bf16 \
    --save_only_model \
    --save_safetensors true \
    --preprocessing_num_workers 96 \
    --trust_remote_code true \
    --cutoff_len 10000

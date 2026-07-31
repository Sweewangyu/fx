#!/usr/bin/env bash

# TS-Reasoner-style alignment stage with a frozen Zeus backbone.
# Trainable: complete ChatTS LLM + TS-to-text projector. Frozen: Zeus.
# Paper recipe: 120K captions, global batch 64, lr 1e-5, one epoch.
# The bundled eager implementation does not require BasicTS or flash-attn.

NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=120 deepspeed --num_gpus 8 --master_port=19901 src/train.py \
    --deepspeed ds_config/ds_config_3.json \
    --stage sft \
    --model_name_or_path "[PATH_TO_CHATTS_BASE_MODEL]" \
    --ts_encoder_type zeus \
    --zeus_model_name_or_path "GestaltCog/zeus" \
    --dataset "stage_1_120K" \
    --interleave_probs "1.0" \
    --do_train \
    --mix_strategy "interleave_over" \
    --template "chatts" \
    --finetuning_type full \
    --output_dir "[OUTPUT_PATH_ZEUS_STAGE_1]" \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.02 \
    --num_train_epochs 1 \
    --plot_loss \
    --bf16 \
    --save_only_model \
    --save_safetensors true \
    --preprocessing_num_workers 96 \
    --trust_remote_code true \
    --cutoff_len 10000

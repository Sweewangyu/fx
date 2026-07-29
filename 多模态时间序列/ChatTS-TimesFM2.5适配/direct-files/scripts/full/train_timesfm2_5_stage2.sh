#!/usr/bin/env bash

# Instruction stage. `auto` reads `timesfm2_5` from the stage-1 config and
# restores the saved projector while reloading the frozen Google checkpoint.

NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=120 deepspeed --num_gpus 8 --master_port=19901 src/train.py \
    --deepspeed ds_config/ds_config_3.json \
    --stage sft \
    --model_name_or_path "[OUTPUT_PATH_TIMESFM_STAGE_1]" \
    --ts_encoder_type auto \
    --dataset "sft,ift,align_random" \
    --interleave_probs "0.6,0.2,0.2" \
    --do_train \
    --mix_strategy "interleave_over" \
    --template "chatts" \
    --finetuning_type full \
    --output_dir "[OUTPUT_PATH_TIMESFM_STAGE_2]" \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --timeseries_sft_lr 3e-5 \
    --warmup_ratio 0.02 \
    --max_steps 400 \
    --plot_loss \
    --bf16 \
    --save_only_model \
    --save_safetensors true \
    --preprocessing_num_workers 96 \
    --trust_remote_code true \
    --cutoff_len 10000

#!/usr/bin/env bash

# One-stage replacement for the TS-Reasoner-style alignment + instruction recipe.
#
# Data exposure (approximately):
#   - stage_1_120K: 120K alignment examples, once
#   - stage_2_30K:   30K instruction examples, twice (about 60K draws)
#
# `interleave_over` keeps sampling until both datasets have been exhausted.
# With probabilities 2/3 and 1/3, the 30K instruction set is repeated about
# twice while the 120K alignment set is consumed once.
#
# Trainable: complete ChatTS LLM + TS-to-text projector.
# Frozen: complete TimesFM 2.5 backbone.
# Effective global batch: 1 sample/GPU * 8 accumulation * 8 GPUs = 64.

NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=120 deepspeed --num_gpus 8 --master_port=19901 src/train.py \
    --deepspeed ds_config/ds_config_3.json \
    --stage sft \
    --model_name_or_path "[PATH_TO_CHATTS_BASE_MODEL]" \
    --ts_encoder_type timesfm2_5 \
    --timesfm_model_name_or_path "google/timesfm-2.5-200m-pytorch" \
    --dataset "stage_1_120K,stage_2_30K" \
    --mix_strategy "interleave_over" \
    --interleave_probs "0.6667,0.3333" \
    --do_train \
    --template "chatts" \
    --finetuning_type full \
    --output_dir "[OUTPUT_PATH_TIMESFM_ONE_STAGE]" \
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

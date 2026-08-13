#!/usr/bin/env bash
set -Eeuo pipefail

# All Dataset Studio datasets, one stage, one epoch, full-parameter SFT.

PROJECT_ROOT="/workspace/ChatTS-Training"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the ChatTS base model}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to a Dataset Studio snapshot}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
# Supported selectors: chatts, time-mqa, tsqa, all. The default is exactly
# ChatTS' four original datasets plus Time-MQA and TSQA.
SELECT_DATASETS="${SELECT_DATASETS:-chatts,time-mqa,tsqa}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
TIMESERIES_SFT_LR="${TIMESERIES_SFT_LR:-$LEARNING_RATE}"
SEED="${SEED:-42}"
MASTER_PORT="${MASTER_PORT:-19901}"

[[ -f "$DATASET_DIR/dataset_info.json" ]] || {
    echo "dataset_info.json not found: $DATASET_DIR/dataset_info.json" >&2
    exit 1
}
[[ -f "$DATASET_DIR/training.env" ]] || {
    echo "training.env not found: $DATASET_DIR/training.env" >&2
    exit 1
}
[[ -f "$MODEL_PATH/config.json" ]] || {
    echo "Base model config not found: $MODEL_PATH/config.json" >&2
    exit 1
}
[[ -d "$CHRONOS2_MODEL_PATH" ]] || {
    echo "Chronos-2 model not found: $CHRONOS2_MODEL_PATH" >&2
    exit 1
}
[[ ! -e "$OUTPUT_DIR" ]] || {
    echo "Output already exists, choose a new OUTPUT_DIR: $OUTPUT_DIR" >&2
    exit 2
}

# Resolve short selectors to the exact versioned keys exported by Dataset
# Studio. The ChatTS recipe intentionally selects each source from its
# canonical stage so the same source is not included twice across stages.
ALL_DATASETS="$(python3 - "$DATASET_DIR/training.env" "$DATASET_DIR/dataset_info.json" "$SELECT_DATASETS" <<'PY'
import json
import shlex
import sys
from pathlib import Path

values = {}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    fields = shlex.split(line)
    if len(fields) == 1 and "=" in fields[0]:
        key, value = fields[0].split("=", 1)
        values[key] = value

info = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
selectors = [item.strip().lower() for item in sys.argv[3].split(",") if item.strip()]
if not selectors:
    raise SystemExit("SELECT_DATASETS must not be empty")

available = []
for variable in ("STAGE1_DATASETS", "STAGE2_DATASETS"):
    for key in values.get(variable, "").split(","):
        if not key or key in available:
            continue
        details = info.get(key)
        if not isinstance(details, dict) or not isinstance(details.get("file_name"), str):
            raise SystemExit(f"Invalid dataset_info entry: {key}")
        relative = Path(details["file_name"])
        available.append(key)

recipes = {
    "chatts": [
        ("stage1", "chatts_align_256"),
        ("stage1", "chatts_ift"),
        ("stage2", "chatts_align_random"),
        ("stage2", "chatts_sft"),
    ],
    "time-mqa": [("stage2", "time_mqa")],
    "time_mqa": [("stage2", "time_mqa")],
    "tsqa": [("stage2", "tsaqa")],
    "tsaqa": [("stage2", "tsaqa")],
}

if "all" in selectors:
    if len(selectors) != 1:
        raise SystemExit("SELECT_DATASETS=all cannot be combined with other selectors")
    print(",".join(available))
    raise SystemExit(0)

requested = []
for selector in selectors:
    try:
        pairs = recipes[selector]
    except KeyError as exc:
        raise SystemExit(
            f"Unknown dataset selector {selector!r}; use chatts,time-mqa,tsqa or all"
        ) from exc
    for pair in pairs:
        if pair not in requested:
            requested.append(pair)

names = []
missing = []
for stage, source in requested:
    matches = []
    for key in available:
        relative = Path(info[key]["file_name"])
        if relative.parent.name == stage and relative.stem == source:
            matches.append(key)
    if len(matches) != 1:
        missing.append(f"{stage}/{source}")
    else:
        names.append(matches[0])
if missing:
    raise SystemExit(
        "Current Dataset Studio snapshot lacks required datasets: " + ", ".join(missing)
    )
print(",".join(names))
PY
)"

mkdir -p "$(dirname "$OUTPUT_DIR")"

echo "============================================================"
echo " ChatTS Chronos-2 all-data one-stage training"
echo " Base model:    $MODEL_PATH"
echo " Dataset dir:   $DATASET_DIR"
echo " Selection:     $SELECT_DATASETS"
echo " Datasets:      $ALL_DATASETS"
echo " Output:        $OUTPUT_DIR"
echo " Epochs:        1"
echo " Mix strategy:  concat"
echo " LLM LR:        $LEARNING_RATE"
echo " Projector LR:  $TIMESERIES_SFT_LR"
echo "============================================================"

cd "$PROJECT_ROOT"
NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=120 \
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port="$MASTER_PORT" src/train.py \
    --deepspeed ds_config/ds_config_2.json \
    --stage sft \
    --model_name_or_path "$MODEL_PATH" \
    --ts_encoder_type chronos2 \
    --chronos2_model_name_or_path "$CHRONOS2_MODEL_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --dataset "$ALL_DATASETS" \
    --mix_strategy concat \
    --do_train \
    --template chatts \
    --finetuning_type full \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate "$LEARNING_RATE" \
    --timeseries_sft_lr "$TIMESERIES_SFT_LR" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.02 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 1 \
    --plot_loss \
    --bf16 \
    --save_only_model False \
    --save_safetensors False \
    --preprocessing_num_workers 32 \
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
    --logging_dir "$OUTPUT_DIR/tensorboard"

python3 scripts/finalize_chatts_best_checkpoint.py \
    --checkpoint-dir "$OUTPUT_DIR" \
    --stage stage1 \
    --seed "$SEED" \
    --learning-rate "$LEARNING_RATE" \
    --chronos2-model-path "$CHRONOS2_MODEL_PATH" \
    --input-model-dir "$MODEL_PATH" \
    --cleanup-checkpoints

echo "Training completed: $OUTPUT_DIR"

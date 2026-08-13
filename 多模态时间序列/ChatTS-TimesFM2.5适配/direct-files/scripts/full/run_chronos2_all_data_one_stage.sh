#!/usr/bin/env bash
set -Eeuo pipefail

# Six unfiltered source datasets, one stage, one epoch, full-parameter SFT.

PROJECT_ROOT="/workspace/ChatTS-Training"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the ChatTS base model}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
RAW_DATASET_ROOT="${RAW_DATASET_ROOT:-/share/airesearch/data/finiverse/traindata/merged_labels/annotated}"
RUNTIME_DATASET_DIR="${RUNTIME_DATASET_DIR:-/tmp/chatts_full_dataset_${SLURM_JOB_ID:-$$}}"
SELECT_DATASETS="${SELECT_DATASETS:-chatts,time-mqa,tsqa}"
MIN_TOTAL_ROWS="${MIN_TOTAL_ROWS:-400000}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
TIMESERIES_SFT_LR="${TIMESERIES_SFT_LR:-$LEARNING_RATE}"
SEED="${SEED:-42}"
MASTER_PORT="${MASTER_PORT:-19901}"

[[ -f "$MODEL_PATH/config.json" ]] || {
    echo "Base model config not found: $MODEL_PATH/config.json" >&2
    exit 1
}
[[ -d "$CHRONOS2_MODEL_PATH" ]] || {
    echo "Chronos-2 model not found: $CHRONOS2_MODEL_PATH" >&2
    exit 1
}
[[ -d "$RAW_DATASET_ROOT" ]] || {
    echo "Raw dataset directory not found: $RAW_DATASET_ROOT" >&2
    exit 1
}
[[ ! -e "$OUTPUT_DIR" ]] || {
    echo "Output already exists, choose a new OUTPUT_DIR: $OUTPUT_DIR" >&2
    exit 2
}
[[ "$MIN_TOTAL_ROWS" =~ ^[0-9]+$ ]] || {
    echo "MIN_TOTAL_ROWS must be a non-negative integer" >&2
    exit 2
}

mkdir -p "$(dirname "$OUTPUT_DIR")" "$RUNTIME_DATASET_DIR"

# Create only a tiny LLaMAFactory registry. The 500K source rows are not
# copied: dataset_info.json points directly to the six unfiltered JSONL files.
PREPARED_JSON="$(python3 - "$RAW_DATASET_ROOT" "$RUNTIME_DATASET_DIR" "$SELECT_DATASETS" "$MIN_TOTAL_ROWS" <<'PY'
import json
import sys
from pathlib import Path

raw_root = Path(sys.argv[1]).resolve()
runtime_root = Path(sys.argv[2]).resolve()
selectors = [item.strip().lower() for item in sys.argv[3].split(",") if item.strip()]
minimum = int(sys.argv[4])

recipes = {
    "chatts": ["chatts_align_256", "chatts_ift", "chatts_align_random", "chatts_sft"],
    "time-mqa": ["time_mqa"],
    "time_mqa": ["time_mqa"],
    "tsqa": ["tsaqa"],
    "tsaqa": ["tsaqa"],
}
if not selectors:
    raise SystemExit("SELECT_DATASETS must not be empty")
if "all" in selectors:
    if len(selectors) != 1:
        raise SystemExit("SELECT_DATASETS=all cannot be combined with other selectors")
    source_names = [
        "chatts_align_256", "chatts_ift", "chatts_align_random",
        "chatts_sft", "time_mqa", "tsaqa",
    ]
else:
    source_names = []
    for selector in selectors:
        try:
            requested = recipes[selector]
        except KeyError as exc:
            raise SystemExit(
                f"Unknown selector {selector!r}; use chatts,time-mqa,tsqa or all"
            ) from exc
        for name in requested:
            if name not in source_names:
                source_names.append(name)

dataset_info = {}
counts = {}
for source_name in source_names:
    path = raw_root / f"{source_name}.jsonl"
    if not path.is_file():
        raise SystemExit(f"Full source dataset not found: {path}")
    with path.open("rb") as stream:
        rows = sum(1 for line in stream if line.strip())
    if rows == 0:
        raise SystemExit(f"Full source dataset is empty: {path}")
    key = f"full_{source_name}"
    counts[source_name] = rows
    dataset_info[key] = {
        "file_name": str(path),
        "columns": {
            "prompt": "input",
            "response": "output",
            "timeseries": "timeseries",
        },
    }

total = sum(counts.values())
if total < minimum:
    detail = ", ".join(f"{name}={rows}" for name, rows in counts.items())
    raise SystemExit(
        f"Selected full datasets have only {total:,} rows (< MIN_TOTAL_ROWS={minimum:,}); {detail}"
    )

(runtime_root / "dataset_info.json").write_text(
    json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
result = {
    "dataset_keys": list(dataset_info),
    "source_counts": counts,
    "total_rows": total,
    "dataset_info": str(runtime_root / "dataset_info.json"),
}
(runtime_root / "full_data_manifest.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
PY
)"

ALL_DATASETS="$(python3 -c 'import json,sys; print(",".join(json.load(sys.stdin)["dataset_keys"]))' <<<"$PREPARED_JSON")"
TOTAL_ROWS="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["total_rows"])' <<<"$PREPARED_JSON")"

echo "============================================================"
echo " ChatTS Chronos-2 FULL-DATA one-stage training"
echo " Base model:       $MODEL_PATH"
echo " Raw dataset root: $RAW_DATASET_ROOT"
echo " Selection:        $SELECT_DATASETS"
echo " Dataset keys:     $ALL_DATASETS"
echo " Raw total rows:   $TOTAL_ROWS"
python3 - "$PREPARED_JSON" <<'PY'
import json
import sys
for name, rows in json.loads(sys.argv[1])["source_counts"].items():
    print(f"   {name:<24} {rows:>10,}")
PY
echo " Output:           $OUTPUT_DIR"
echo " Epochs:           1"
echo " Mix strategy:     concat"
echo " LLM LR:           $LEARNING_RATE"
echo " Projector LR:     $TIMESERIES_SFT_LR"
echo "============================================================"

cd "$PROJECT_ROOT"
NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=120 \
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port="$MASTER_PORT" src/train.py \
    --deepspeed ds_config/ds_config_2.json \
    --stage sft \
    --model_name_or_path "$MODEL_PATH" \
    --ts_encoder_type chronos2 \
    --chronos2_model_name_or_path "$CHRONOS2_MODEL_PATH" \
    --dataset_dir "$RUNTIME_DATASET_DIR" \
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

echo "Full-data training completed: $OUTPUT_DIR"

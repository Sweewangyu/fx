#!/usr/bin/env bash
set -Eeuo pipefail

# Deterministic single-run Chronos-2 Stage1 -> Stage2 training pipeline.
# The final Stage2 directory is the only retained model directory.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS-Training}"
MODEL_PATH="${MODEL_PATH:-/share/airesearch/data/finiverse/model/ChatTS-Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/share/airesearch/data/finiverse/output/ChatTS-msxf-8B-datav1}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
SEED="${SEED:-42}"
S1_LR="${S1_LR:-1e-5}"
S2_LR="${S2_LR:-1e-5}"
STAGE1_DATASETS="${STAGE1_DATASETS:-align_256,ift}"
STAGE2_DATASETS="${STAGE2_DATASETS:-sft,align_random,finiverse_time_mqa,finiverse_tsaqa}"
STAGE1_MIX_STRATEGY="${STAGE1_MIX_STRATEGY:-interleave_over}"
STAGE2_MIX_STRATEGY="${STAGE2_MIX_STRATEGY:-concat}"
STAGE1_TIMESERIES_SFT_LR="${STAGE1_TIMESERIES_SFT_LR:-$S1_LR}"
STAGE2_TIMESERIES_SFT_LR="${STAGE2_TIMESERIES_SFT_LR:-$S2_LR}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

STAGE1_SCRIPT="${STAGE1_SCRIPT:-${PROJECT_ROOT}/scripts/full/train_chronos2_best_stage1.sh}"
STAGE2_SCRIPT="${STAGE2_SCRIPT:-${PROJECT_ROOT}/scripts/full/train_chronos2_best_stage2.sh}"
FINALIZER="${FINALIZER:-${PROJECT_ROOT}/scripts/finalize_chatts_best_checkpoint.py}"
STAGE1_OUT="${STAGE1_OUT:-${OUTPUT_ROOT}/.stage1_seed${SEED}_s1lr_${S1_LR}}"
FINAL_MODEL_PATH="${FINAL_MODEL_PATH:-${OUTPUT_ROOT}/best_seed${SEED}}"
RUN_NAME="${RUN_NAME:-chronos2_seed${SEED}_s1lr_${S1_LR}_s2lr_${S2_LR}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs/${RUN_NAME}}"
READY_MARKER="${FINAL_MODEL_PATH}/TRAINING_COMPLETE.json"

for flag_name in FORCE_TRAIN PREFLIGHT_ONLY; do
    flag_value="${!flag_name}"
    [[ "$flag_value" == "0" || "$flag_value" == "1" ]] || {
        echo "$flag_name must be 0 or 1, got: $flag_value" >&2
        exit 2
    }
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer." >&2; exit 2; }
for lr_name in S1_LR S2_LR; do
    lr_value="${!lr_name}"
    [[ "$lr_value" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || {
        echo "$lr_name must be a positive numeric learning rate, got: $lr_value" >&2
        exit 2
    }
    [[ ! "$lr_value" =~ ^0+([.]0+)?([eE][+-]?[0-9]+)?$ ]] || {
        echo "$lr_name must be greater than zero." >&2
        exit 2
    }
done

require_dir() {
    local label="$1" path="$2"
    [[ -d "$path" ]] || { echo "$label directory not found: $path" >&2; exit 1; }
}

require_file() {
    local label="$1" path="$2"
    [[ -f "$path" ]] || { echo "$label file not found: $path" >&2; exit 1; }
}

safe_remove_model_dir() {
    local target="$1" expected_kind="$2"
    "$PYTHON_BIN" - "$OUTPUT_ROOT" "$target" "$expected_kind" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
target = Path(sys.argv[2]).expanduser().resolve()
kind = sys.argv[3]
patterns = {
    "stage1": re.compile(r"\.stage1_seed[0-9]+_s1lr_[A-Za-z0-9+_.-]+"),
    "final": re.compile(r"best_seed[0-9]+"),
}
if target.parent != root or not patterns[kind].fullmatch(target.name):
    raise SystemExit(f"Refusing to remove path outside the owned output layout: {target}")
print(target)
PY
    if [[ -e "$target" ]]; then
        rm -rf -- "$target"
        echo "Removed previous $expected_kind output: $target"
    fi
}

validate_ready_marker() {
    "$PYTHON_BIN" - "$READY_MARKER" "$FINAL_MODEL_PATH" "$SEED" "$S1_LR" "$S2_LR" <<'PY'
import json
import sys
from pathlib import Path

marker = Path(sys.argv[1])
model_dir = Path(sys.argv[2]).expanduser().resolve()
expected = {"seed": int(sys.argv[3]), "stage1_learning_rate": sys.argv[4], "stage2_learning_rate": sys.argv[5]}
with marker.open(encoding="utf-8") as stream:
    payload = json.load(stream)
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"Existing completion marker mismatch for {key}: {payload.get(key)!r} != {value!r}")
if Path(payload.get("final_model_path", "")).expanduser().resolve() != model_dir:
    raise SystemExit("Existing completion marker points to a different final model path")
if not (model_dir / "config.json").is_file() or not (model_dir / "best_model_manifest.json").is_file():
    raise SystemExit("Completion marker exists but the finalized model metadata is incomplete")
print(f"Validated completed model: {model_dir}")
PY
}

require_dir "Training project" "$PROJECT_ROOT"
require_file "Base model config" "$MODEL_PATH/config.json"
require_dir "Chronos-2" "$CHRONOS2_MODEL_PATH"
require_file "Stage1 runner" "$STAGE1_SCRIPT"
require_file "Stage2 runner" "$STAGE2_SCRIPT"
require_file "Checkpoint finalizer" "$FINALIZER"

AVAILABLE_GPUS="${AVAILABLE_GPUS_OVERRIDE:-$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')}"
if (( AVAILABLE_GPUS < 8 )); then
    echo "Training requires 8 visible GPUs, but PyTorch sees $AVAILABLE_GPUS." >&2
    exit 1
fi

echo "============================================================"
echo " ChatTS Chronos-2 two-stage best-model training"
echo " Base model:       $MODEL_PATH"
echo " Stage1 LR:        $S1_LR"
echo " Stage2 LR:        $S2_LR"
echo " Stage1 datasets:  $STAGE1_DATASETS"
echo " Stage2 datasets:  $STAGE2_DATASETS"
echo " Seed:             $SEED"
echo " Stage1 temporary: $STAGE1_OUT"
echo " Final model:      $FINAL_MODEL_PATH"
echo " Logs:             $LOG_ROOT"
echo "============================================================"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "Training preflight passed. No files were changed."
    exit 0
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$LOG_ROOT/tensorboard"

if [[ -f "$READY_MARKER" && "$FORCE_TRAIN" != "1" ]]; then
    validate_ready_marker
    echo "Training already completed; reusing $FINAL_MODEL_PATH"
    exit 0
fi

if [[ "$FORCE_TRAIN" == "1" ]]; then
    safe_remove_model_dir "$STAGE1_OUT" stage1
    safe_remove_model_dir "$FINAL_MODEL_PATH" final
else
    if [[ -e "$FINAL_MODEL_PATH" ]]; then
        echo "Incomplete final output exists: $FINAL_MODEL_PATH" >&2
        echo "Inspect it or rerun with FORCE_TRAIN=1." >&2
        exit 2
    fi
    if [[ -e "$STAGE1_OUT" ]]; then
        echo "Incomplete Stage1 output exists: $STAGE1_OUT" >&2
        echo "Inspect it or rerun with FORCE_TRAIN=1." >&2
        exit 2
    fi
fi

export PROJECT_ROOT MODEL_PATH OUTPUT_ROOT CHRONOS2_MODEL_PATH SEED FINALIZER
export STAGE1_DATASETS STAGE2_DATASETS STAGE1_MIX_STRATEGY STAGE2_MIX_STRATEGY
export STAGE1_TIMESERIES_SFT_LR STAGE2_TIMESERIES_SFT_LR

echo "$(date '+%Y-%m-%d %H:%M:%S') | Starting Stage1"
TENSORBOARD_DIR="$LOG_ROOT/tensorboard/stage1" \
    bash "$STAGE1_SCRIPT" "$S1_LR" "$STAGE1_OUT" \
    2>&1 | tee "$LOG_ROOT/stage1.log"
cp "$STAGE1_OUT/best_model_manifest.json" "$LOG_ROOT/stage1_best_model_manifest.json"

echo "$(date '+%Y-%m-%d %H:%M:%S') | Starting Stage2"
TENSORBOARD_DIR="$LOG_ROOT/tensorboard/stage2" \
    bash "$STAGE2_SCRIPT" "$S2_LR" "$STAGE1_OUT" "$FINAL_MODEL_PATH" \
    2>&1 | tee "$LOG_ROOT/stage2.log"
cp "$FINAL_MODEL_PATH/best_model_manifest.json" "$LOG_ROOT/stage2_best_model_manifest.json"

# Stage1 is no longer needed after Stage2 has exported and validated its best model.
safe_remove_model_dir "$STAGE1_OUT" stage1

READY_MARKER="$READY_MARKER" \
FINAL_MODEL_PATH="$FINAL_MODEL_PATH" \
MODEL_PATH="$MODEL_PATH" \
LOG_ROOT="$LOG_ROOT" \
SEED="$SEED" \
S1_LR="$S1_LR" \
S2_LR="$S2_LR" \
STAGE1_DATASETS="$STAGE1_DATASETS" \
STAGE2_DATASETS="$STAGE2_DATASETS" \
STAGE1_MIX_STRATEGY="$STAGE1_MIX_STRATEGY" \
STAGE2_MIX_STRATEGY="$STAGE2_MIX_STRATEGY" \
STAGE1_TIMESERIES_SFT_LR="$STAGE1_TIMESERIES_SFT_LR" \
STAGE2_TIMESERIES_SFT_LR="$STAGE2_TIMESERIES_SFT_LR" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

marker = Path(os.environ["READY_MARKER"])
stage1_manifest = Path(os.environ["LOG_ROOT"]) / "stage1_best_model_manifest.json"
stage2_manifest = Path(os.environ["LOG_ROOT"]) / "stage2_best_model_manifest.json"
with stage1_manifest.open(encoding="utf-8") as stream:
    stage1 = json.load(stream)
with stage2_manifest.open(encoding="utf-8") as stream:
    stage2 = json.load(stream)
stage1_export = Path(stage1["exported_model_dir"]).resolve()
stage2_export = Path(stage2["exported_model_dir"]).resolve()
final_model = Path(os.environ["FINAL_MODEL_PATH"]).resolve()
stage2_input = stage2.get("input_best_model")
if not isinstance(stage2_input, dict):
    raise SystemExit("Stage2 manifest does not identify its Stage1 best-model input")
if Path(stage2_input.get("exported_model_dir", "")).resolve() != stage1_export:
    raise SystemExit("Stage2 was not initialized from the finalized Stage1 model")
if stage2_input.get("selected_checkpoint") != stage1.get("selected_checkpoint"):
    raise SystemExit("Stage2 input provenance does not match the Stage1 selected checkpoint")
if stage2_export != final_model:
    raise SystemExit("Stage2 best-model export does not match FINAL_MODEL_PATH")
payload = {
    "status": "complete",
    "seed": int(os.environ["SEED"]),
    "stage1_learning_rate": os.environ["S1_LR"],
    "stage2_learning_rate": os.environ["S2_LR"],
    "stage1_best_eval_loss": stage1["best_metric"],
    "stage2_best_eval_loss": stage2["best_metric"],
    "final_model_path": str(final_model),
    "training_lineage": {
        "stage1": {
            "input_model_path": str(Path(os.environ["MODEL_PATH"]).resolve()),
            "datasets": os.environ["STAGE1_DATASETS"],
            "mix_strategy": os.environ["STAGE1_MIX_STRATEGY"],
            "learning_rate": os.environ["S1_LR"],
            "timeseries_learning_rate": os.environ["STAGE1_TIMESERIES_SFT_LR"],
            "selected_checkpoint": stage1["selected_checkpoint"],
            "best_eval_loss": stage1["best_metric"],
            "exported_best_model_path": stage1["exported_model_dir"],
        },
        "stage2": {
            "input_model_path": stage1["exported_model_dir"],
            "input_stage1_selected_checkpoint": stage1["selected_checkpoint"],
            "datasets": os.environ["STAGE2_DATASETS"],
            "mix_strategy": os.environ["STAGE2_MIX_STRATEGY"],
            "learning_rate": os.environ["S2_LR"],
            "timeseries_learning_rate": os.environ["STAGE2_TIMESERIES_SFT_LR"],
            "selected_checkpoint": stage2["selected_checkpoint"],
            "best_eval_loss": stage2["best_metric"],
            "exported_best_model_path": stage2["exported_model_dir"],
        },
        "evaluation_model_path": stage2["exported_model_dir"],
    },
    "stage1_model_retained": False,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = marker.with_suffix(marker.suffix + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
temporary.replace(marker)
run_manifest = Path(os.environ["LOG_ROOT"]) / "training_run_manifest.json"
with run_manifest.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(f"Training completion marker: {marker}")
PY

validate_ready_marker
echo "Training completed. Only the Stage2 best model is retained: $FINAL_MODEL_PATH"

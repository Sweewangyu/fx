#!/usr/bin/env bash
set -Eeuo pipefail

# Local tinyBenchmarks MCQ evaluation through the ChatTS vLLM implementation.
# This script never runs pip/conda and never downloads benchmark data.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS/ChatTS-main}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/exp/tinybenchmarks_mcq}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASKS_CSV="${TASKS_CSV:-tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande}"
NUM_GPUS="${NUM_GPUS:-auto}"
REQUEST_CHUNK_SIZE="${REQUEST_CHUNK_SIZE:-32}"
MAX_MODEL_LEN="${CHATTS_VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
DTYPE="${DTYPE:-auto}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DATA_PARTITION="${DATA_PARTITION:-all}"
PARTITION_SEED="${PARTITION_SEED:-42}"
ALLOW_SIZE_MISMATCH="${ALLOW_SIZE_MISMATCH:-0}"
FORCE="${FORCE:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
OFFLINE="${OFFLINE:-1}"
FORGETTING_THRESHOLD_PP="${FORGETTING_THRESHOLD_PP:-5.0}"
SUMMARY_BASENAME="${SUMMARY_BASENAME:-tinybenchmarks_mcq_summary}"

# Capture an explicit override once. Normally leave it empty: every ChatTS
# checkpoint is detected independently from its own tensor names/shapes.
ENCODER_TYPE_OVERRIDE="${CHATTS_TS_ENCODER_TYPE:-}"

BASELINE_NAME="${BASELINE_NAME:-}"
MODEL_SPECS=()
TASK_FILE_SPECS=()

usage() {
    cat <<'EOF'
Usage:
  DATASET_ROOT=/workspace/datasets/tinyBenchmarks \
  bash scripts/run_chatts_tinybenchmarks_mcq.sh \
    --model base=/workspace/models/qwen3-base \
    --model chatts=/workspace/checkpoints/chatts-final \
    --baseline base

Options:
  --model NAME=PATH       Repeat for each local model/checkpoint.
  --baseline NAME         Reference row for catastrophic-forgetting deltas.
  --dataset-root PATH     Root containing the five already-downloaded datasets.
  --task-file TASK=PATH   Override discovery for one task; repeat as needed.
  --summary-only          Rebuild tables from existing model result directories.
  --force                 Re-run even when MODEL_OUTPUT/metrics.json exists.
  --allow-size-mismatch   Permit a dataset split with a size other than 100.
  -h, --help              Show this help.

No installation or network access is required. Supported local formats are
JSON, JSONL, Parquet, and (only if `datasets` is already present) save_to_disk.

Useful environment variables:
  CUDA_VISIBLE_DEVICES=0,1        GPUs exposed to vLLM.
  NUM_GPUS=2                      vLLM tensor-parallel size (default: auto).
  REQUEST_CHUNK_SIZE=32           Candidate prompts submitted per call.
  CHATTS_VLLM_MAX_MODEL_LEN=8192  vLLM context allocation.
  MAX_SAMPLES=5                   Smoke-test only the first N examples.
  DATA_PARTITION=search-dev       Locked hash-stratified view (all/search-dev/final-test).
  PARTITION_SEED=42               Stable split seed.
  SEED=42                         vLLM engine/sampling seed.
  OFFLINE=1                       Force local Hugging Face files (default: 1).

For external TS encoders the checkpoint type is auto-detected. Only its local
backbone path may still be needed by model construction:
  CHATTS_TIMESFM_MODEL_PATH=/workspace/timesf
  CHATTS_CHRONOS2_MODEL_PATH=/workspace/chronos-2
  CHATTS_ZEUS_MODEL_PATH=/workspace/zeus
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "--model requires NAME=PATH" >&2; exit 2; }
            MODEL_SPECS+=("$2")
            shift 2
            ;;
        --baseline)
            [[ $# -ge 2 ]] || { echo "--baseline requires NAME" >&2; exit 2; }
            BASELINE_NAME="$2"
            shift 2
            ;;
        --dataset-root)
            [[ $# -ge 2 ]] || { echo "--dataset-root requires PATH" >&2; exit 2; }
            DATASET_ROOT="$2"
            shift 2
            ;;
        --task-file)
            [[ $# -ge 2 ]] || { echo "--task-file requires TASK=PATH" >&2; exit 2; }
            TASK_FILE_SPECS+=("$2")
            shift 2
            ;;
        --summary-only)
            SUMMARY_ONLY=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --allow-size-mismatch)
            ALLOW_SIZE_MISMATCH=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if (( ${#MODEL_SPECS[@]} == 0 )); then
    if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
        MODEL_SPECS+=("${BASE_MODEL_NAME:-base}=${BASE_MODEL_PATH}")
    fi
    if [[ -n "${MODEL_PATH:-}" ]]; then
        MODEL_SPECS+=("${MODEL_NAME:-chatts}=${MODEL_PATH}")
    fi
fi
if (( ${#MODEL_SPECS[@]} == 0 )); then
    echo "No models supplied. Use --model NAME=PATH." >&2
    usage >&2
    exit 2
fi

MODEL_NAMES=()
MODEL_PATHS=()
declare -A SEEN_NAMES=()
for spec in "${MODEL_SPECS[@]}"; do
    [[ "$spec" == *=* ]] || { echo "Invalid --model '$spec'; expected NAME=PATH." >&2; exit 2; }
    name="${spec%%=*}"
    path="${spec#*=}"
    [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid model name: $name" >&2; exit 2; }
    [[ -z "${SEEN_NAMES[$name]:-}" ]] || { echo "Duplicate model name: $name" >&2; exit 2; }
    if [[ -d "$path" ]]; then
        path="$(cd "$path" && pwd)"
    fi
    SEEN_NAMES[$name]=1
    MODEL_NAMES+=("$name")
    MODEL_PATHS+=("$path")
done

if [[ -z "$BASELINE_NAME" ]]; then
    BASELINE_NAME="${MODEL_NAMES[0]}"
fi
[[ -n "${SEEN_NAMES[$BASELINE_NAME]:-}" ]] || {
    echo "Baseline '$BASELINE_NAME' is not in the model list." >&2
    exit 2
}

[[ -d "$PROJECT_ROOT" ]] || { echo "ChatTS project not found: $PROJECT_ROOT" >&2; exit 1; }
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

if [[ "$SUMMARY_ONLY" != "1" ]]; then
    [[ -n "$DATASET_ROOT" ]] || { echo "DATASET_ROOT or --dataset-root is required." >&2; exit 2; }
    [[ -d "$DATASET_ROOT" ]] || { echo "Local dataset root not found: $DATASET_ROOT" >&2; exit 1; }
    DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd)"

    "$PYTHON_BIN" -c 'import torch, transformers, vllm' || {
        echo "The existing ChatTS vLLM environment is incomplete. No installation was attempted." >&2
        exit 1
    }

    if [[ "$NUM_GPUS" == "auto" ]]; then
        NUM_GPUS="$($PYTHON_BIN -c 'import torch; print(max(1, torch.cuda.device_count()))')"
    fi
    [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_GPUS must be a positive integer or auto." >&2; exit 2; }

    if [[ "$OFFLINE" == "1" ]]; then
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
    fi
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

    COMMON_ARGS=(
        --dataset-root "$DATASET_ROOT"
        --tasks "$TASKS_CSV"
        --num-gpus "$NUM_GPUS"
        --request-chunk-size "$REQUEST_CHUNK_SIZE"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --dtype "$DTYPE"
        --seed "$SEED"
        --max-samples "$MAX_SAMPLES"
    )
    case "$DATA_PARTITION" in
        all) ;;
        search-dev|final-test)
            COMMON_ARGS+=(--data-partition "$DATA_PARTITION" --partition-seed "$PARTITION_SEED")
            ;;
        *)
            echo "DATA_PARTITION must be all, search-dev, or final-test: $DATA_PARTITION" >&2
            exit 2
            ;;
    esac
    for task_file in "${TASK_FILE_SPECS[@]}"; do
        COMMON_ARGS+=(--task-file "$task_file")
    done
    if [[ "$ALLOW_SIZE_MISMATCH" == "1" ]]; then
        COMMON_ARGS+=(--allow-size-mismatch)
    fi

    # Validate all paths and schemas before the first GPU allocation.
    (
        cd "$PROJECT_ROOT"
        "$PYTHON_BIN" -m chatts.utils.inference_tinybenchmarks_mcq_vllm \
            --model-path "${MODEL_PATHS[0]}" \
            --model-name "${MODEL_NAMES[0]}" \
            --output-dir "$OUTPUT_ROOT/.data_inspection" \
            "${COMMON_ARGS[@]}" \
            --inspect-data-only
    )

    for index in "${!MODEL_NAMES[@]}"; do
        name="${MODEL_NAMES[$index]}"
        model_path="${MODEL_PATHS[$index]}"
        model_output="$OUTPUT_ROOT/$name"

        [[ -d "$model_path" && -f "$model_path/config.json" ]] || {
            echo "Local model directory/config.json not found: $model_path" >&2
            exit 1
        }
        if [[ -f "$model_output/metrics.json" && "$FORCE" != "1" ]]; then
            echo "[$name] Existing completed result found; skipping: $model_output/metrics.json"
            continue
        fi
        mkdir -p "$model_output"

        is_chatts="$($PYTHON_BIN - "$model_path/config.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
print("1" if any("TSForCausalLM" in str(item) for item in config.get("architectures", [])) else "0")
PY
)"

        encoder_type="none"
        if [[ "$is_chatts" == "1" ]]; then
            if [[ -n "$ENCODER_TYPE_OVERRIDE" ]]; then
                encoder_type="$ENCODER_TYPE_OVERRIDE"
            else
                encoder_type="$($PYTHON_BIN "$PROJECT_ROOT/scripts/inspect_chatts_ts_encoder_checkpoints.py" \
                    "$model_path" --print-detected-only)"
            fi
        fi

        command=(
            "$PYTHON_BIN" -m chatts.utils.inference_tinybenchmarks_mcq_vllm
            --model-path "$model_path"
            --model-name "$name"
            --output-dir "$model_output"
            "${COMMON_ARGS[@]}"
        )

        echo "============================================================"
        echo " tinyBenchmarks MCQ via ChatTS vLLM: $name"
        echo " Model:          $model_path"
        echo " Dataset root:   $DATASET_ROOT"
        echo " Tasks:          $TASKS_CSV"
        echo " ChatTS model:   $is_chatts"
        echo " TS encoder:     $encoder_type"
        echo " Tensor parallel:$NUM_GPUS"
        echo " Seed:           $SEED"
        echo " Output:         $model_output"
        echo "============================================================"

        if [[ "$is_chatts" == "1" ]]; then
            printf 'CHATTS_TS_ENCODER_TYPE=%q ' "$encoder_type" > "$model_output/command.sh"
            printf '%q ' "${command[@]}" >> "$model_output/command.sh"
            printf '\n' >> "$model_output/command.sh"
            (
                cd "$PROJECT_ROOT"
                CHATTS_TS_ENCODER_TYPE="$encoder_type" "${command[@]}"
            ) 2>&1 | tee "$model_output/run.log"
        else
            printf 'env -u CHATTS_TS_ENCODER_TYPE ' > "$model_output/command.sh"
            printf '%q ' "${command[@]}" >> "$model_output/command.sh"
            printf '\n' >> "$model_output/command.sh"
            (
                cd "$PROJECT_ROOT"
                env -u CHATTS_TS_ENCODER_TYPE "${command[@]}"
            ) 2>&1 | tee "$model_output/run.log"
        fi
    done
fi

SUMMARY_ARGS=()
for index in "${!MODEL_NAMES[@]}"; do
    result_root="$OUTPUT_ROOT/${MODEL_NAMES[$index]}"
    if [[ "$SUMMARY_ONLY" == "1" && -f "${MODEL_PATHS[$index]}/metrics.json" ]]; then
        result_root="${MODEL_PATHS[$index]}"
    fi
    SUMMARY_ARGS+=(--model "${MODEL_NAMES[$index]}=$result_root")
done

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/summarize_tinybenchmarks_mcq.py" \
    "${SUMMARY_ARGS[@]}" \
    --baseline "$BASELINE_NAME" \
    --threshold-pp "$FORGETTING_THRESHOLD_PP" \
    --output-dir "$OUTPUT_ROOT" \
    --basename "$SUMMARY_BASENAME"

if (( ${#MODEL_NAMES[@]} < 2 )); then
    echo "Warning: one model gives scores but cannot measure forgetting; add the pre-training base checkpoint." >&2
fi

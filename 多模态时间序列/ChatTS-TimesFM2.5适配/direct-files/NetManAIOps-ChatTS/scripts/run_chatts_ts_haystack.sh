#!/usr/bin/env bash
set -Eeuo pipefail

# ChatTS checkpoint -> official TS-Haystack loaders/prompts/scorers -> vLLM-TS.
# Copy this script and the companion Python files into the same relative paths
# of a NetManAIOps/ChatTS checkout before running it.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS/ChatTS-main}"
TS_HAYSTACK_ROOT="${TS_HAYSTACK_ROOT:-/workspace/TS-Haystack}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TS_HAYSTACK_ROOT}/evaluation/results/chatts}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/ckpt}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"

DATASETS="${DATASETS:-all}"
SPLIT="${SPLIT:-test}"
TASKS="${TASKS:-all}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_GPUS_PER_PROCESS="${NUM_GPUS_PER_PROCESS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
REQUEST_CHUNK_SIZE="${REQUEST_CHUNK_SIZE:-8}"
CHATTS_VLLM_MAX_MODEL_LEN="${CHATTS_VLLM_MAX_MODEL_LEN:-40960}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-500}"
MAX_PROCESSED_INPUT_TOKENS="${MAX_PROCESSED_INPUT_TOKENS:-$((CHATTS_VLLM_MAX_MODEL_LEN - MAX_NEW_TOKENS))}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
FORCE_INFERENCE="${FORCE_INFERENCE:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Optional explicit encoder/path overrides.  When the type is empty, the
# checkpoint tensor inspector distinguishes native MLP and TimesFM directly;
# only a metadata-free 768-d projector needs a Chronos-2/Zeus hint.
TS_ENCODER_TYPE="${TS_ENCODER_TYPE:-${CHATTS_TS_ENCODER_TYPE:-}}"
TIMESFM_MODEL_PATH="${TIMESFM_MODEL_PATH:-${CHATTS_TIMESFM_MODEL_PATH:-}}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-${CHATTS_CHRONOS2_MODEL_PATH:-}}"
ZEUS_MODEL_PATH="${ZEUS_MODEL_PATH:-${CHATTS_ZEUS_MODEL_PATH:-}}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ChatTS project not found: $PROJECT_ROOT" >&2
    exit 1
fi
if [[ ! "$MODEL_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "MODEL_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 1
fi
if [[ "$SCORE_ONLY" != "1" && ! -f "$TS_HAYSTACK_ROOT/src/datasets/registry.py" ]]; then
    echo "Official TS-Haystack source tree not found: $TS_HAYSTACK_ROOT" >&2
    echo "TS_HAYSTACK_ROOT must be an AI-X-Labs/TS-Haystack clone containing src/ and data/." >&2
    exit 1
fi
if [[ "$SCORE_ONLY" != "1" && ! -d "$MODEL_PATH" ]]; then
    echo "Model checkpoint not found: $MODEL_PATH" >&2
    exit 1
fi
if (( MAX_NEW_TOKENS < 1 || CHATTS_VLLM_MAX_MODEL_LEN <= MAX_NEW_TOKENS )); then
    echo "CHATTS_VLLM_MAX_MODEL_LEN must be larger than MAX_NEW_TOKENS." >&2
    exit 1
fi
if (( MAX_PROCESSED_INPUT_TOKENS < 1 || MAX_PROCESSED_INPUT_TOKENS + MAX_NEW_TOKENS > CHATTS_VLLM_MAX_MODEL_LEN )); then
    echo "MAX_PROCESSED_INPUT_TOKENS must reserve MAX_NEW_TOKENS inside max_model_len." >&2
    exit 1
fi

read -r -a DATASET_ARGS <<< "$DATASETS"
read -r -a TASK_ARGS <<< "$TASKS"
read -r -a CONTEXT_ARGS <<< "$CONTEXT_LENGTHS"

if [[ "$SCORE_ONLY" != "1" ]]; then
    if (( NUM_GPUS < 1 || NUM_GPUS_PER_PROCESS < 1 || NUM_GPUS % NUM_GPUS_PER_PROCESS != 0 )); then
        echo "NUM_GPUS must be positive and divisible by NUM_GPUS_PER_PROCESS." >&2
        exit 1
    fi
    AVAILABLE_GPUS="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
    if (( NUM_GPUS > AVAILABLE_GPUS )); then
        echo "Requested NUM_GPUS=$NUM_GPUS, but PyTorch sees $AVAILABLE_GPUS GPU(s)." >&2
        exit 1
    fi

    if [[ -z "$TS_ENCODER_TYPE" ]]; then
        INSPECTOR="$PROJECT_ROOT/scripts/inspect_chatts_ts_encoder_checkpoints.py"
        if [[ ! -f "$INSPECTOR" ]]; then
            echo "Encoder inspector not found: $INSPECTOR" >&2
            exit 1
        fi
        TS_ENCODER_TYPE="$($PYTHON_BIN "$INSPECTOR" "$MODEL_PATH" --print-detected-only)"
        echo "Auto-detected TS encoder from checkpoint weights: $TS_ENCODER_TYPE"
    fi

    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
    export CHATTS_VLLM_MAX_MODEL_LEN
    export CHATTS_TS_ENCODER_TYPE="$TS_ENCODER_TYPE"
    if [[ -n "$TIMESFM_MODEL_PATH" ]]; then
        export CHATTS_TIMESFM_MODEL_PATH="$TIMESFM_MODEL_PATH"
    fi
    if [[ -n "$CHRONOS2_MODEL_PATH" ]]; then
        export CHATTS_CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH"
    fi
    if [[ -n "$ZEUS_MODEL_PATH" ]]; then
        export CHATTS_ZEUS_MODEL_PATH="$ZEUS_MODEL_PATH"
    fi

    INFER_ARGS=(
        --model-path "$MODEL_PATH"
        --model-name "$MODEL_NAME"
        --ts-haystack-root "$TS_HAYSTACK_ROOT"
        --datasets "${DATASET_ARGS[@]}"
        --split "$SPLIT"
        --tasks "${TASK_ARGS[@]}"
        --context-lengths "${CONTEXT_ARGS[@]}"
        --output-root "$OUTPUT_ROOT"
        --num-gpus "$NUM_GPUS"
        --gpus-per-model "$NUM_GPUS_PER_PROCESS"
        --batch-size "$BATCH_SIZE"
        --request-chunk-size "$REQUEST_CHUNK_SIZE"
        --max-new-tokens "$MAX_NEW_TOKENS"
        --temperature "$TEMPERATURE"
        --seed "$SEED"
        --max-processed-input-tokens "$MAX_PROCESSED_INPUT_TOKENS"
        --max-samples "$MAX_SAMPLES"
    )
    if [[ "$ENABLE_THINKING" == "1" ]]; then
        INFER_ARGS+=(--enable-thinking)
    fi
    if [[ "$FORCE_INFERENCE" == "1" ]]; then
        INFER_ARGS+=(--force)
    fi

    echo "============================================================"
    echo " ChatTS x TS-Haystack (official data + official scorer)"
    echo " Model:           $MODEL_PATH"
    echo " TS-Haystack:     $TS_HAYSTACK_ROOT"
    echo " Datasets:        $DATASETS"
    echo " Split/tasks:     $SPLIT / $TASKS"
    echo " Context lengths: $CONTEXT_LENGTHS"
    echo " GPUs:            $NUM_GPUS (${NUM_GPUS_PER_PROCESS} per worker)"
    echo " Encoder:         $TS_ENCODER_TYPE"
    echo " Max model/input: $CHATTS_VLLM_MAX_MODEL_LEN / $MAX_PROCESSED_INPUT_TOKENS"
    echo " Max new tokens:  $MAX_NEW_TOKENS"
    echo " Thinking:        $([[ "$ENABLE_THINKING" == "1" ]] && echo enabled || echo disabled)"
    echo " Output:          $OUTPUT_ROOT"
    echo "============================================================"

    cd "$PROJECT_ROOT"
    "$PYTHON_BIN" -m chatts.utils.inference_ts_haystack_vllm "${INFER_ARGS[@]}"
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/evaluate_ts_haystack.py \
    --results-root "$OUTPUT_ROOT" \
    --model-name "$MODEL_NAME" \
    --datasets "${DATASET_ARGS[@]}"

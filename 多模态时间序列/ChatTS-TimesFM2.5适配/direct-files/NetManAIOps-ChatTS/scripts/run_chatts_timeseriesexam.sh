#!/usr/bin/env bash
set -Eeuo pipefail

# ChatTS checkpoint -> official TimeSeriesExam questions/raw arrays -> vLLM-TS.
# This script does not install packages or download models/datasets at runtime.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS/ChatTS-main}"
TIMESERIESEXAM_ROOT="${TIMESERIESEXAM_ROOT:-/workspace/TimeSeriesExam}"
DATA_FILE_PATH="${DATA_FILE_PATH:-${TIMESERIESEXAM_ROOT}/output/round_3_folder/qa_dataset.json}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/ckpt}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/exp/timeseriesexam}"
PYTHON_BIN="${PYTHON_BIN:-python}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_GPUS_PER_PROCESS="${NUM_GPUS_PER_PROCESS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
REQUEST_CHUNK_SIZE="${REQUEST_CHUNK_SIZE:-64}"
CHATTS_VLLM_MAX_MODEL_LEN="${CHATTS_VLLM_MAX_MODEL_LEN:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_PROCESSED_INPUT_TOKENS="${MAX_PROCESSED_INPUT_TOKENS:-$((CHATTS_VLLM_MAX_MODEL_LEN - MAX_NEW_TOKENS))}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_CONCEPTS="${MAX_CONCEPTS:-3}"

# The official evaluation shell scripts enable all three additions.
ADD_QUESTION_HINT="${ADD_QUESTION_HINT:-1}"
ADD_CONCEPTS="${ADD_CONCEPTS:-1}"
ADD_EXAMPLES="${ADD_EXAMPLES:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
FORCE_INFERENCE="${FORCE_INFERENCE:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
OFFLINE="${OFFLINE:-1}"

# Explicit override is normally unnecessary. The inspector reads checkpoint
# tensor names/shapes; a metadata-free 768-d projector may still need a hint.
TS_ENCODER_TYPE="${TS_ENCODER_TYPE:-${CHATTS_TS_ENCODER_TYPE:-}}"
TIMESFM_MODEL_PATH="${TIMESFM_MODEL_PATH:-${CHATTS_TIMESFM_MODEL_PATH:-}}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-${CHATTS_CHRONOS2_MODEL_PATH:-}}"
ZEUS_MODEL_PATH="${ZEUS_MODEL_PATH:-${CHATTS_ZEUS_MODEL_PATH:-}}"

for value_name in ADD_QUESTION_HINT ADD_CONCEPTS ADD_EXAMPLES ENABLE_THINKING FORCE_INFERENCE SCORE_ONLY OFFLINE; do
    value="${!value_name}"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "$value_name must be 0 or 1, got: $value" >&2
        exit 2
    fi
done
if [[ "$ADD_CONCEPTS" == "0" && "$ADD_EXAMPLES" == "1" ]]; then
    echo "ADD_EXAMPLES=1 requires ADD_CONCEPTS=1." >&2
    exit 2
fi
[[ -d "$PROJECT_ROOT" ]] || { echo "ChatTS project not found: $PROJECT_ROOT" >&2; exit 1; }
[[ "$MODEL_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "MODEL_NAME may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
}
if (( MAX_NEW_TOKENS < 1 || CHATTS_VLLM_MAX_MODEL_LEN <= MAX_NEW_TOKENS )); then
    echo "CHATTS_VLLM_MAX_MODEL_LEN must be larger than MAX_NEW_TOKENS." >&2
    exit 2
fi
if (( MAX_PROCESSED_INPUT_TOKENS < 1 || MAX_PROCESSED_INPUT_TOKENS + MAX_NEW_TOKENS > CHATTS_VLLM_MAX_MODEL_LEN )); then
    echo "MAX_PROCESSED_INPUT_TOKENS must reserve MAX_NEW_TOKENS inside max_model_len." >&2
    exit 2
fi

variant="query"
[[ "$ADD_QUESTION_HINT" == "1" ]] && variant+="_hint"
[[ "$ADD_CONCEPTS" == "1" ]] && variant+="_concepts"
[[ "$ADD_EXAMPLES" == "1" ]] && variant+="_examples"
[[ "$ENABLE_THINKING" == "1" ]] && variant+="_qwen-thinking"
RESULT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${variant}"
RESULT_FILE="${RESULT_DIR}/generated_answer.json"

if [[ "$SCORE_ONLY" != "1" ]]; then
    [[ -d "$TIMESERIESEXAM_ROOT" ]] || {
        echo "Official TimeSeriesExam repository not found: $TIMESERIESEXAM_ROOT" >&2
        exit 1
    }
    [[ -f "$DATA_FILE_PATH" ]] || { echo "Dataset file not found: $DATA_FILE_PATH" >&2; exit 1; }
    [[ -d "$MODEL_PATH" ]] || { echo "Model checkpoint not found: $MODEL_PATH" >&2; exit 1; }
    if [[ "$ADD_CONCEPTS" == "1" && ! -f "$TIMESERIESEXAM_ROOT/evaluate/concepts.py" ]]; then
        echo "Official concepts.py not found under TIMESERIESEXAM_ROOT." >&2
        exit 1
    fi
    if (( NUM_GPUS < 1 || NUM_GPUS_PER_PROCESS < 1 || NUM_GPUS % NUM_GPUS_PER_PROCESS != 0 )); then
        echo "NUM_GPUS must be positive and divisible by NUM_GPUS_PER_PROCESS." >&2
        exit 2
    fi

    AVAILABLE_GPUS="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
    if (( NUM_GPUS > AVAILABLE_GPUS )); then
        echo "Requested NUM_GPUS=$NUM_GPUS, but PyTorch sees $AVAILABLE_GPUS GPU(s)." >&2
        exit 1
    fi
    if [[ -z "$TS_ENCODER_TYPE" ]]; then
        INSPECTOR="$PROJECT_ROOT/scripts/inspect_chatts_ts_encoder_checkpoints.py"
        [[ -f "$INSPECTOR" ]] || { echo "Encoder inspector not found: $INSPECTOR" >&2; exit 1; }
        TS_ENCODER_TYPE="$($PYTHON_BIN "$INSPECTOR" "$MODEL_PATH" --print-detected-only)"
        echo "Auto-detected TS encoder from checkpoint weights: $TS_ENCODER_TYPE"
    fi

    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export CHATTS_VLLM_MAX_MODEL_LEN
    export CHATTS_TS_ENCODER_TYPE="$TS_ENCODER_TYPE"
    if [[ "$OFFLINE" == "1" ]]; then
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
    fi
    [[ -z "$TIMESFM_MODEL_PATH" ]] || export CHATTS_TIMESFM_MODEL_PATH="$TIMESFM_MODEL_PATH"
    [[ -z "$CHRONOS2_MODEL_PATH" ]] || export CHATTS_CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH"
    [[ -z "$ZEUS_MODEL_PATH" ]] || export CHATTS_ZEUS_MODEL_PATH="$ZEUS_MODEL_PATH"

    INFER_ARGS=(
        --model-path "$MODEL_PATH"
        --model-name "$MODEL_NAME"
        --timeseriesexam-root "$TIMESERIESEXAM_ROOT"
        --data-file "$DATA_FILE_PATH"
        --output-path "$RESULT_FILE"
        --num-gpus "$NUM_GPUS"
        --gpus-per-model "$NUM_GPUS_PER_PROCESS"
        --batch-size "$BATCH_SIZE"
        --request-chunk-size "$REQUEST_CHUNK_SIZE"
        --max-new-tokens "$MAX_NEW_TOKENS"
        --temperature "$TEMPERATURE"
        --seed "$SEED"
        --max-processed-input-tokens "$MAX_PROCESSED_INPUT_TOKENS"
        --max-samples "$MAX_SAMPLES"
        --max-concepts "$MAX_CONCEPTS"
    )
    [[ "$ADD_QUESTION_HINT" == "1" ]] || INFER_ARGS+=(--no-question-hint)
    [[ "$ADD_CONCEPTS" == "1" ]] || INFER_ARGS+=(--no-concepts)
    [[ "$ADD_EXAMPLES" == "1" ]] || INFER_ARGS+=(--no-examples)
    [[ "$ENABLE_THINKING" == "0" ]] || INFER_ARGS+=(--enable-thinking)
    [[ "$FORCE_INFERENCE" == "0" ]] || INFER_ARGS+=(--force)

    echo "============================================================"
    echo " ChatTS x TimeSeriesExam (official one-shot protocol)"
    echo " Model:            $MODEL_PATH"
    echo " Dataset:          $DATA_FILE_PATH"
    echo " Variant:          $variant"
    echo " GPUs:             $NUM_GPUS (${NUM_GPUS_PER_PROCESS} per worker)"
    echo " Encoder:          $TS_ENCODER_TYPE"
    echo " Max model/input:  $CHATTS_VLLM_MAX_MODEL_LEN / $MAX_PROCESSED_INPUT_TOKENS"
    echo " Max new tokens:   $MAX_NEW_TOKENS"
    echo " Temperature/seed: $TEMPERATURE / $SEED"
    echo " Qwen thinking:    $([[ "$ENABLE_THINKING" == "1" ]] && echo enabled || echo disabled)"
    echo " Output:           $RESULT_FILE"
    echo "============================================================"

    mkdir -p "$RESULT_DIR"
    cd "$PROJECT_ROOT"
    "$PYTHON_BIN" -m chatts.utils.inference_timeseriesexam_vllm "${INFER_ARGS[@]}"
fi

[[ -f "$RESULT_FILE" ]] || { echo "Result file not found: $RESULT_FILE" >&2; exit 1; }
cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/evaluate_timeseriesexam.py \
    --result-file "$RESULT_FILE" \
    --output-dir "$RESULT_DIR" \
    --model-name "$MODEL_NAME"

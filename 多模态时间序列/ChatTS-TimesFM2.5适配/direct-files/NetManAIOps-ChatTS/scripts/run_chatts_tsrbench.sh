#!/usr/bin/env bash
set -Eeuo pipefail

# One model load -> all TSRBench tasks -> local MCQ evaluation.
# Copy this file to ChatTS/scripts and copy inference_tsrbench_vllm.py to
# ChatTS/chatts/utils before running it.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS/ChatTS-main}"
TSRBENCH_ROOT="${TSRBENCH_ROOT:-/workspace/TSRBench}"
DATASET_ROOT="${DATASET_ROOT:-${TSRBENCH_ROOT}/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TSRBENCH_ROOT}/evaluation/results/embed}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/ckpt}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
DATASETS="${DATASETS:-all}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_GPUS_PER_PROCESS="${NUM_GPUS_PER_PROCESS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
REQUEST_CHUNK_SIZE="${REQUEST_CHUNK_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
FORCE_INFERENCE="${FORCE_INFERENCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Optional encoder override. Leave empty when config/weights are unambiguous.
TS_ENCODER_TYPE="${TS_ENCODER_TYPE:-${CHATTS_TS_ENCODER_TYPE:-}}"
TIMESFM_MODEL_PATH="${TIMESFM_MODEL_PATH:-${CHATTS_TIMESFM_MODEL_PATH:-}}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-${CHATTS_CHRONOS2_MODEL_PATH:-}}"
ZEUS_MODEL_PATH="${ZEUS_MODEL_PATH:-${CHATTS_ZEUS_MODEL_PATH:-}}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ChatTS project not found: $PROJECT_ROOT" >&2
    exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "TSRBench dataset root not found: $DATASET_ROOT" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model checkpoint not found: $MODEL_PATH" >&2
    exit 1
fi
if (( NUM_GPUS < 1 || NUM_GPUS_PER_PROCESS < 1 )); then
    echo "NUM_GPUS and NUM_GPUS_PER_PROCESS must be positive." >&2
    exit 1
fi
if (( NUM_GPUS % NUM_GPUS_PER_PROCESS != 0 )); then
    echo "NUM_GPUS must be divisible by NUM_GPUS_PER_PROCESS." >&2
    exit 1
fi

AVAILABLE_GPUS="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if (( NUM_GPUS > AVAILABLE_GPUS )); then
    echo "Requested NUM_GPUS=$NUM_GPUS, but PyTorch sees only $AVAILABLE_GPUS GPU(s)." >&2
    echo "Reduce NUM_GPUS or correct the job's CUDA device allocation." >&2
    exit 1
fi

# chatts_vllm.py must construct the encoder before vLLM starts loading tensors.
# When config.json has no ts_encoder_type, resolve the architecture from tensor
# names/shapes exactly as the Dataset A/B batch script does.
if [[ -z "$TS_ENCODER_TYPE" ]]; then
    INSPECTOR="$PROJECT_ROOT/scripts/inspect_chatts_ts_encoder_checkpoints.py"
    if [[ ! -f "$INSPECTOR" ]]; then
        echo "Encoder inspector not found: $INSPECTOR" >&2
        echo "Copy scripts/inspect_chatts_ts_encoder_checkpoints.py to the server." >&2
        exit 1
    fi
    TS_ENCODER_TYPE="$($PYTHON_BIN "$INSPECTOR" "$MODEL_PATH" --print-detected-only)"
    echo "Auto-detected TS encoder from checkpoint weights: $TS_ENCODER_TYPE"
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"

if [[ -n "$TS_ENCODER_TYPE" ]]; then
    export CHATTS_TS_ENCODER_TYPE="$TS_ENCODER_TYPE"
fi
if [[ -n "$TIMESFM_MODEL_PATH" ]]; then
    export CHATTS_TIMESFM_MODEL_PATH="$TIMESFM_MODEL_PATH"
fi
if [[ -n "$CHRONOS2_MODEL_PATH" ]]; then
    export CHATTS_CHRONOS2_MODEL_PATH="$CHRONOS2_MODEL_PATH"
fi
if [[ -n "$ZEUS_MODEL_PATH" ]]; then
    export CHATTS_ZEUS_MODEL_PATH="$ZEUS_MODEL_PATH"
fi

read -r -a DATASET_ARGS <<< "$DATASETS"

INFER_ARGS=(
    --model-path "$MODEL_PATH"
    --model-name "$MODEL_NAME"
    --dataset-root "$DATASET_ROOT"
    --datasets "${DATASET_ARGS[@]}"
    --output-root "$OUTPUT_ROOT"
    --num-gpus "$NUM_GPUS"
    --gpus-per-model "$NUM_GPUS_PER_PROCESS"
    --batch-size "$BATCH_SIZE"
    --request-chunk-size "$REQUEST_CHUNK_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --max-samples "$MAX_SAMPLES"
)
if [[ "$FORCE_INFERENCE" == "1" ]]; then
    INFER_ARGS+=(--force)
fi

echo "============================================================"
echo " ChatTS x TSRBench (vLLM time-series embedding modality)"
echo " Model:        $MODEL_PATH"
echo " Dataset root: $DATASET_ROOT"
echo " Tasks:        $DATASETS"
echo " GPUs:         $NUM_GPUS (${NUM_GPUS_PER_PROCESS} per worker)"
echo " vLLM engines: $((NUM_GPUS / NUM_GPUS_PER_PROCESS))"
echo " Output:       $OUTPUT_ROOT"
echo " Encoder:      ${TS_ENCODER_TYPE:-auto}"
echo "============================================================"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m chatts.utils.inference_tsrbench_vllm "${INFER_ARGS[@]}"

"$PYTHON_BIN" scripts/evaluate_tsrbench.py \
    --dataset-root "$DATASET_ROOT" \
    --results-root "$OUTPUT_ROOT" \
    --model-name "$MODEL_NAME" \
    --datasets "${DATASET_ARGS[@]}"

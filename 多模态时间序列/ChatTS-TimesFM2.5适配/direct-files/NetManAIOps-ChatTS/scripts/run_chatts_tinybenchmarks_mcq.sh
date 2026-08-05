#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate one or more base/ChatTS checkpoints on only the multiple-choice
# tinyBenchmarks tasks, then summarize retention relative to a baseline model.
# No time-series input is used and no external judge/RAGAS service is needed.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS/ChatTS-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/exp/tinybenchmarks_mcq}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASKS_CSV="${TASKS_CSV:-tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
DTYPE="${DTYPE:-auto}"
DEVICE="${DEVICE:-cuda:0}"
PARALLELIZE="${PARALLELIZE:-auto}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-0}"
SEED="${SEED:-0,1234,1234,1234}"
FORCE="${FORCE:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
FORGETTING_THRESHOLD_PP="${FORGETTING_THRESHOLD_PP:-5.0}"
SUMMARY_BASENAME="${SUMMARY_BASENAME:-tinybenchmarks_mcq_summary}"
TINYBENCHMARKS_PKL_SHA256="${TINYBENCHMARKS_PKL_SHA256:-c3b6e426dfe7b100fe6d0ee960398e10a8763254bcead3be80cc6bc15abca284}"

BASELINE_NAME="${BASELINE_NAME:-}"
MODEL_SPECS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_chatts_tinybenchmarks_mcq.sh \
    --model base=/path/to/base-model \
    --model chatts=/path/to/chatts-checkpoint \
    --baseline base

Options:
  --model NAME=PATH   Repeat for every model/checkpoint to compare.
  --baseline NAME    Reference model used to calculate score drops.
  --summary-only     Do not run inference; rebuild tables from existing results.
  --force            Run again even if a completed result already exists.
  -h, --help         Show this help.

Environment highlights:
  CUDA_VISIBLE_DEVICES=0,1  Select GPUs visible to the HF backend.
  PARALLELIZE=auto|0|1      Spread one model across visible GPUs (default: auto).
  BATCH_SIZE=1              Safest default; 'auto' is also supported.
  APPLY_CHAT_TEMPLATE=0|1   Default 0 follows the official completion protocol.
  INSTALL_DEPS=1            Run install_tinybenchmarks_mcq.sh first.
  HF_HOME=/path/to/cache    Shared Hugging Face dataset/model cache.
  HF_HUB_OFFLINE=1          Use only an already populated cache.
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
        --summary-only)
            SUMMARY_ONLY=1
            shift
            ;;
        --force)
            FORCE=1
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

# Environment-only single-model compatibility.  A baseline comparison still
# needs at least two --model entries (or BASE_MODEL_PATH + MODEL_PATH).
if (( ${#MODEL_SPECS[@]} == 0 )); then
    if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
        MODEL_SPECS+=("${BASE_MODEL_NAME:-base}=${BASE_MODEL_PATH}")
    fi
    if [[ -n "${MODEL_PATH:-}" ]]; then
        MODEL_SPECS+=("${MODEL_NAME:-chatts}=${MODEL_PATH}")
    fi
fi

if (( ${#MODEL_SPECS[@]} == 0 )); then
    echo "No models supplied. Use --model NAME=PATH at least once." >&2
    usage >&2
    exit 2
fi

MODEL_NAMES=()
MODEL_PATHS=()
declare -A SEEN_NAMES=()
for spec in "${MODEL_SPECS[@]}"; do
    if [[ "$spec" != *=* ]]; then
        echo "Invalid --model '$spec'; expected NAME=PATH." >&2
        exit 2
    fi
    name="${spec%%=*}"
    path="${spec#*=}"
    if [[ ! "$name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "Invalid model name '$name'; use letters, digits, dot, underscore, or hyphen." >&2
        exit 2
    fi
    if [[ -n "${SEEN_NAMES[$name]:-}" ]]; then
        echo "Duplicate model name: $name" >&2
        exit 2
    fi
    if [[ "$path" == *,* ]]; then
        echo "Model paths containing commas are not supported by lm-eval model_args: $path" >&2
        exit 2
    fi
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
if [[ -z "${SEEN_NAMES[$BASELINE_NAME]:-}" ]]; then
    echo "Baseline '$BASELINE_NAME' is not present in the supplied model list." >&2
    exit 2
fi

if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ChatTS project not found: $PROJECT_ROOT" >&2
    exit 1
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

if [[ "$SUMMARY_ONLY" != "1" ]]; then
    if [[ "$INSTALL_DEPS" == "1" ]]; then
        bash "$PROJECT_ROOT/scripts/install_tinybenchmarks_mcq.sh"
    fi

    "$PYTHON_BIN" -c 'import accelerate, lm_eval, tinyBenchmarks, torch, transformers' || {
        echo "Missing tinyBenchmarks dependencies." >&2
        echo "Run: bash scripts/install_tinybenchmarks_mcq.sh" >&2
        exit 1
    }

    # The official tinyBenchmarks package reads tinyBenchmarks.pkl from cwd.
    # Its normal wheel omits that asset, so the installer keeps an editable
    # source tree and we stage the pinned file in an isolated runtime directory.
    CALIBRATION_SOURCE="${TINYBENCHMARKS_PKL:-$($PYTHON_BIN -c 'from pathlib import Path; import tinyBenchmarks; print(Path(tinyBenchmarks.__file__).with_name("tinyBenchmarks.pkl"))')}"
    if [[ ! -f "$CALIBRATION_SOURCE" ]]; then
        echo "tinyBenchmarks calibration asset not found: $CALIBRATION_SOURCE" >&2
        echo "Re-run: bash scripts/install_tinybenchmarks_mcq.sh" >&2
        exit 1
    fi
    actual_calibration_sha="$($PYTHON_BIN -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$CALIBRATION_SOURCE")"
    if [[ "$actual_calibration_sha" != "$TINYBENCHMARKS_PKL_SHA256" ]]; then
        echo "tinyBenchmarks calibration SHA-256 mismatch." >&2
        echo "Expected: $TINYBENCHMARKS_PKL_SHA256" >&2
        echo "Actual:   $actual_calibration_sha" >&2
        exit 1
    fi
    RUNTIME_DIR="$OUTPUT_ROOT/.tinybenchmarks_runtime"
    mkdir -p "$RUNTIME_DIR"
    if [[ ! -f "$RUNTIME_DIR/tinyBenchmarks.pkl" ]] || \
       ! cmp -s "$CALIBRATION_SOURCE" "$RUNTIME_DIR/tinyBenchmarks.pkl"; then
        cp "$CALIBRATION_SOURCE" "$RUNTIME_DIR/tinyBenchmarks.pkl"
    fi

    if [[ "$PARALLELIZE" == "auto" ]]; then
        visible_gpus="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
        if (( visible_gpus > 1 )); then
            PARALLELIZE=1
        else
            PARALLELIZE=0
        fi
    fi
    case "$PARALLELIZE" in
        1|true|True|TRUE) PARALLELIZE_ARG=true ;;
        0|false|False|FALSE) PARALLELIZE_ARG=false ;;
        *) echo "PARALLELIZE must be auto, 0, or 1." >&2; exit 2 ;;
    esac

    if "$PYTHON_BIN" -m lm_eval run --help >/dev/null 2>&1; then
        LMEVAL=("$PYTHON_BIN" -m lm_eval run)
    else
        # Compatibility with the 0.4.x CLI before subcommands were introduced.
        LMEVAL=("$PYTHON_BIN" -m lm_eval)
    fi

    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
fi

if [[ "$SUMMARY_ONLY" != "1" ]]; then
    for index in "${!MODEL_NAMES[@]}"; do
        name="${MODEL_NAMES[$index]}"
        model_path="${MODEL_PATHS[$index]}"
        model_output="$OUTPUT_ROOT/$name"

        if [[ ! -d "$model_path" ]]; then
            echo "Warning: '$model_path' is not a local directory; Hugging Face Hub access will be attempted." >&2
        elif [[ ! -f "$model_path/config.json" ]]; then
            echo "Model directory has no config.json: $model_path" >&2
            exit 1
        fi

        existing_result="$(find "$model_output" -type f -name 'results_*.json' -print 2>/dev/null | sort | tail -n 1 || true)"
        if [[ -n "$existing_result" && "$FORCE" != "1" ]]; then
            echo "[$name] Existing completed result found; skipping: $existing_result"
            continue
        fi

        mkdir -p "$model_output"
        model_args="pretrained=${model_path},tokenizer=${model_path},trust_remote_code=True,backend=causal,dtype=${DTYPE},parallelize=${PARALLELIZE_ARG}"
        command=(
            "${LMEVAL[@]}"
            --model hf
            --model_args "$model_args"
            --tasks "$TASKS_CSV"
            --batch_size "$BATCH_SIZE"
            --device "$DEVICE"
            --seed "$SEED"
            --output_path "$model_output"
            --log_samples
        )
        if [[ "$BATCH_SIZE" == auto* && -n "$MAX_BATCH_SIZE" ]]; then
            command+=(--max_batch_size "$MAX_BATCH_SIZE")
        fi
        if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
            command+=(--apply_chat_template)
        fi

        echo "============================================================"
        echo " tinyBenchmarks MCQ: $name"
        echo " Model:          $model_path"
        echo " Tasks:          $TASKS_CSV"
        echo " Backend:        Hugging Face causal LM (text only)"
        echo " Parallelize:    $PARALLELIZE_ARG"
        echo " Chat template:  $APPLY_CHAT_TEMPLATE"
        echo " Output:         $model_output"
        echo "============================================================"

        printf '%q ' "${command[@]}" > "$model_output/command.sh"
        printf '\n' >> "$model_output/command.sh"
        (
            cd "$RUNTIME_DIR"
            "${command[@]}"
        ) 2>&1 | tee "$model_output/run.log"
    done
fi

SUMMARY_ARGS=()
for index in "${!MODEL_NAMES[@]}"; do
    result_root="$OUTPUT_ROOT/${MODEL_NAMES[$index]}"
    if [[ "$SUMMARY_ONLY" == "1" && -d "${MODEL_PATHS[$index]}" ]]; then
        supplied_result="$(find "${MODEL_PATHS[$index]}" -type f -name 'results_*.json' -print -quit)"
        if [[ -n "$supplied_result" ]]; then
            result_root="${MODEL_PATHS[$index]}"
        fi
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
    echo "Warning: only one model was evaluated. Add the pre-ChatTS/base checkpoint to measure forgetting." >&2
fi

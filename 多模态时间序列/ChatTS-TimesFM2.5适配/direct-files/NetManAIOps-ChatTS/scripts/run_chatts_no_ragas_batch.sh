#!/usr/bin/env bash
set -Eeuo pipefail

# ChatTS Dataset A/B batch inference + categorical/numerical evaluation.
# Automatically iterates over all subdirectories in SEARCH_DIR (excluding 'logs').
# RAGAS is never imported and no judge API is called.
#
# Usage:
#   bash run_chatts_no_ragas_batch.sh                 # infer if needed, then score
#   bash run_chatts_no_ragas_batch.sh --score-only    # reuse existing inference
#   bash run_chatts_no_ragas_batch.sh --infer-only    # inference only
#   bash run_chatts_no_ragas_batch.sh --summary-only  # aggregate existing results
#   TS_ENCODER_TYPE=timesfm2_5 bash run_chatts_no_ragas_batch.sh --infer-only
#
# Every configuration item can also be overridden with an environment variable.

# ==================== User configuration ====================
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS/ChatTS-main}"

# The parent directory containing all model checkpoint subdirectories.
SEARCH_DIR="${SEARCH_DIR:-/share/airesearch/data/finiverse/output/ChatTS-Qwen3-1.7B-PR-grid}"

# Optional: relative path from each subdirectory to the actual model weights.
# Leave empty ("") if the subdirectory itself is the model checkpoint.
#
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-}"

DATASET_A="${DATASET_A:-evaluation/dataset/dataset_a.json}"
DATASET_B="${DATASET_B:-evaluation/dataset/dataset_b.json}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_GPUS_PER_PROCESS="${NUM_GPUS_PER_PROCESS:-2}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Optional fallback for a 768-d external projector that cannot be distinguished
# as Chronos-2 or Zeus from weight shapes alone. Unambiguous native MLP and
# TimesFM checkpoints are always resolved from their own weights per model.
# Supported values: native, timesfm2_5, chronos2, zeus.
# CHATTS_TS_ENCODER_TYPE is accepted too, so either of these one-shot commands works:
#   TS_ENCODER_TYPE=chronos2 bash run_chatts_no_ragas_batch.sh
#   CHATTS_TS_ENCODER_TYPE=chronos2 bash run_chatts_no_ragas_batch.sh
TS_ENCODER_TYPE="${TS_ENCODER_TYPE:-${CHATTS_TS_ENCODER_TYPE:-}}"

# Optional shared local backbone directories/files. Frozen backbone weights are
# intentionally not duplicated inside every ChatTS checkpoint. These aliases
# are mapped to the CHATTS_* variables consumed by chatts_vllm.py.
TIMESFM_MODEL_PATH="${TIMESFM_MODEL_PATH:-${CHATTS_TIMESFM_MODEL_PATH:-}}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-${CHATTS_CHRONOS2_MODEL_PATH:-}}"
ZEUS_MODEL_PATH="${ZEUS_MODEL_PATH:-${CHATTS_ZEUS_MODEL_PATH:-}}"

# Aggregate table written after evaluation. The logs directory is already
# excluded from checkpoint discovery, so summary files are not treated as models.
SUMMARY_DIR="${SUMMARY_DIR:-${SEARCH_DIR%/}/logs}"
SUMMARY_BASENAME="${SUMMARY_BASENAME:-chatts_batch_summary}"

# Set FORCE_INFERENCE=1 to overwrite an already complete generated_answer file.
FORCE_INFERENCE="${FORCE_INFERENCE:-0}"
# ============================================================

case "$TS_ENCODER_TYPE" in
    "" | native | mlp | mlp_patch | mlp-patch | chatts_mlp | \
        timesfm2_5 | timesfm2.5 | timesfm-2.5 | chronos2 | chronos-2 | zeus) ;;
    *)
        echo "Error: unsupported TS_ENCODER_TYPE: $TS_ENCODER_TYPE" >&2
        echo "Use native, timesfm2_5, chronos2, zeus, or leave it empty for auto-detection." >&2
        exit 2
        ;;
esac

MODE="${1:---all}"
case "$MODE" in
    --all | --score-only | --infer-only | --summary-only) ;;
    *)
        echo "Usage: bash $0 [--all|--score-only|--infer-only|--summary-only]" >&2
        exit 2
        ;;
esac

cd "$PROJECT_ROOT"

for dataset_path in "$DATASET_A" "$DATASET_B"; do
    if [[ ! -f "$dataset_path" ]]; then
        echo "Error: dataset does not exist: $PROJECT_ROOT/$dataset_path" >&2
        exit 1
    fi
done

INFERENCE_SCRIPT="chatts/utils/inference_tsmllm_vllm.py"
if [[ ! -f "$INFERENCE_SCRIPT" ]]; then
    echo "Error: inference script does not exist: $PROJECT_ROOT/$INFERENCE_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$SEARCH_DIR" ]]; then
    echo "Error: SEARCH_DIR does not exist: $SEARCH_DIR" >&2
    exit 1
fi

# ---- Collect model subdirectories ----
MODEL_DIRS=()
for entry in "$SEARCH_DIR"/*/; do
    [[ -d "$entry" ]] || continue
    dirname="$(basename "$entry")"
    # Skip logs directory
    [[ "$dirname" == "logs" ]] && continue
    MODEL_DIRS+=("$dirname")
done

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
    echo "Error: no model subdirectories found in $SEARCH_DIR" >&2
    exit 1
fi

echo "=========================================="
echo " Project:   $PROJECT_ROOT"
echo " Search:    $SEARCH_DIR"
echo " Found ${#MODEL_DIRS[@]} model(s): ${MODEL_DIRS[*]}"
if [[ -n "$TS_ENCODER_TYPE" ]]; then
    echo " TS encoder fallback for ambiguous checkpoints: $TS_ENCODER_TYPE"
else
    echo " TS encoder: auto-detect from config and checkpoint weights"
fi
[[ -n "$TIMESFM_MODEL_PATH" ]] && echo " TimesFM backbone: $TIMESFM_MODEL_PATH"
[[ -n "$CHRONOS2_MODEL_PATH" ]] && echo " Chronos-2 backbone: $CHRONOS2_MODEL_PATH"
[[ -n "$ZEUS_MODEL_PATH" ]] && echo " Zeus backbone: $ZEUS_MODEL_PATH"
echo "=========================================="

# ---- Temporary directory for inference script backup ----
TMP_DIR="$(mktemp -d /tmp/chatts-no-ragas.XXXXXX)"
cp "$INFERENCE_SCRIPT" "$TMP_DIR/inference_tsmllm_vllm.py"

restore_source() {
    if [[ -f "$TMP_DIR/inference_tsmllm_vllm.py" ]]; then
        cp "$TMP_DIR/inference_tsmllm_vllm.py" "$INFERENCE_SCRIPT"
    fi
    rm -rf "$TMP_DIR"
}
trap restore_source EXIT INT TERM

# ---- Helper functions ----

detect_encoder_type_from_checkpoint() {
    local model_checkpoint="$1"
    local requested_encoder_type="$2"

    "$PYTHON_BIN" - "$model_checkpoint" "$requested_encoder_type" <<'PY'
from __future__ import annotations

import glob
import json
import os
import sys
from collections.abc import Mapping


checkpoint = os.path.abspath(sys.argv[1])
requested = sys.argv[2].strip().lower()
config_path = os.path.join(checkpoint, "config.json")
config = {}
if os.path.isfile(config_path):
    with open(config_path, "r", encoding="utf-8") as stream:
        config = json.load(stream)

aliases = {
    "mlp": "native",
    "mlp_patch": "native",
    "mlp-patch": "native",
    "chatts_mlp": "native",
    "timesfm2.5": "timesfm2_5",
    "timesfm-2.5": "timesfm2_5",
    "chronos-2": "chronos2",
}
requested = aliases.get(requested, requested)
configured = config.get("ts_encoder_type")
if isinstance(configured, str):
    configured = aliases.get(configured.strip().lower(), configured.strip().lower())
if configured in (None, "", "auto"):
    configured = None

path_fields = {
    "timesfm2_5": "timesfm_model_name_or_path",
    "chronos2": "chronos2_model_name_or_path",
    "zeus": "zeus_model_name_or_path",
}
path_matches = [
    encoder_type
    for encoder_type, field in path_fields.items()
    if config.get(field)
]
def relevant_shape(name: str, shape: tuple[int, ...]):
    """Return (kind, input_dim) for an identifying ChatTS tensor."""
    if (
        "ts_encoder.mlp." in name
        or "ts_encoder.position_embedding." in name
    ):
        return "native", None
    if "ts_encoder.projector.input_norm.weight" in name and shape:
        return "projector", int(shape[0])
    if "ts_encoder.projector.linear_in.weight" in name and len(shape) >= 2:
        return "projector", int(shape[-1])
    return None


native_found = False
projector_dims: set[int] = set()


def record(name: str, shape: tuple[int, ...]) -> None:
    global native_found
    result = relevant_shape(name, shape)
    if result is None:
        return
    kind, input_dim = result
    if kind == "native":
        native_found = True
    elif input_dim is not None:
        projector_dims.add(input_dim)


weight_files: list[str] = []
patterns = (
    "model*.safetensors",
    "pytorch_model*.safetensors",
    "pytorch_model*.bin",
    "model*.bin",
    "model*.pt",
    "pytorch_model*.pt",
)
for pattern in patterns:
    weight_files.extend(glob.glob(os.path.join(checkpoint, pattern)))
weight_files = list(dict.fromkeys(sorted(weight_files)))

for weight_path in weight_files:
    if weight_path.endswith(".safetensors"):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise SystemExit(
                "safetensors is required to inspect this checkpoint."
            ) from exc
        with safe_open(weight_path, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if "ts_encoder." not in name:
                    continue
                record(name, tuple(stream.get_slice(name).get_shape()))
    else:
        import torch

        load_kwargs = {"map_location": "cpu"}
        try:
            state = torch.load(
                weight_path,
                weights_only=True,
                mmap=True,
                **load_kwargs,
            )
        except (TypeError, RuntimeError):
            # Compatibility with older PyTorch releases/checkpoint formats.
            # This may use more RAM than mmap mode, but only runs before vLLM
            # allocates the model.
            try:
                state = torch.load(
                    weight_path,
                    weights_only=True,
                    **load_kwargs,
                )
            except TypeError:
                state = torch.load(weight_path, **load_kwargs)

        def visit(value, prefix: str = "", depth: int = 0) -> None:
            if isinstance(value, torch.Tensor):
                if "ts_encoder." in prefix:
                    record(prefix, tuple(value.shape))
                return
            if isinstance(value, Mapping) and depth < 3:
                for key, child in value.items():
                    child_name = f"{prefix}.{key}" if prefix else str(key)
                    visit(child, child_name, depth + 1)

        visit(state)
        del state

if native_found and projector_dims:
    raise SystemExit(
        "Checkpoint contains both native ts_encoder.mlp.* and external "
        "ts_encoder.projector.* weights; refusing to guess."
    )
if native_found:
    if requested and requested != "native":
        print(
            "Warning: ignoring the requested TS encoder fallback "
            f"`{requested}` because this checkpoint contains native "
            "ChatTS MLP weights.",
            file=sys.stderr,
        )
    print("native")
    raise SystemExit(0)
if projector_dims == {1280}:
    if requested and requested != "timesfm2_5":
        print(
            "Warning: ignoring the requested TS encoder fallback "
            f"`{requested}` because this checkpoint has a 1280-d "
            "TimesFM projector.",
            file=sys.stderr,
        )
    print("timesfm2_5")
    raise SystemExit(0)
if projector_dims == {768}:
    if requested in ("chronos2", "zeus"):
        print(requested)
        raise SystemExit(0)
    if requested:
        raise SystemExit(
            f"The requested TS encoder `{requested}` conflicts with this "
            "checkpoint's 768-d external projector. Use chronos2 or zeus."
        )
    if configured in ("chronos2", "zeus"):
        print(configured)
        raise SystemExit(0)
    if len(path_matches) == 1 and path_matches[0] in ("chronos2", "zeus"):
        print(path_matches[0])
        raise SystemExit(0)
    patch_size = (config.get("ts") or {}).get("patch_size")
    try:
        patch_size = int(patch_size)
    except (TypeError, ValueError):
        patch_size = None
    if patch_size == 16:
        print("chronos2")
        raise SystemExit(0)
    if patch_size == 32:
        print("zeus")
        raise SystemExit(0)
    raise SystemExit(
        "The checkpoint has a 768-d external projector, which can be either "
        "Chronos-2 or Zeus. Their saved projector tensor names and shapes are "
        "identical. Set TS_ENCODER_TYPE=chronos2 or zeus, or retain ts.patch_size "
        "(16 for Chronos-2; 32 for Zeus) in config.json."
    )
if projector_dims:
    raise SystemExit(
        "Unsupported or inconsistent external projector input dimensions: "
        f"{sorted(projector_dims)}."
    )
if requested:
    print(requested)
    raise SystemExit(0)
if configured:
    print(configured)
    raise SystemExit(0)
if len(path_matches) == 1:
    print(path_matches[0])
    raise SystemExit(0)
if len(path_matches) > 1:
    raise SystemExit(
        "Could not identify the TS encoder from checkpoint weights, and "
        "config.json contains multiple backbone path fields "
        f"({', '.join(path_matches)})."
    )
raise SystemExit(
    "Could not find identifying ts_encoder.mlp.*, "
    "ts_encoder.position_embedding.*, or ts_encoder.projector.* tensors "
    "in the checkpoint."
)
PY
}

configure_inference() {
    local model_checkpoint="$1"
    local dataset_path="$2"
    local exp_name="$3"

    "$PYTHON_BIN" - \
        "$INFERENCE_SCRIPT" \
        "$model_checkpoint" \
        "$dataset_path" \
        "$exp_name" \
        "$NUM_GPUS" \
        "$NUM_GPUS_PER_PROCESS" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
model_path = sys.argv[2]
dataset_path = sys.argv[3]
exp_name = sys.argv[4]
num_gpus = int(sys.argv[5])
gpus_per_process = int(sys.argv[6])

text = path.read_text(encoding="utf-8")

replacements = {
    r"^EXP\s*=.*$": f"EXP = {exp_name!r}",
    r"^MODEL_PATH\s*=.*$": f"MODEL_PATH = os.path.abspath({model_path!r})",
    r"^DATASET\s*=.*$": f"DATASET = {dataset_path!r}",
    r"^NUM_GPUS\s*=.*$": f"NUM_GPUS = {num_gpus}",
    r"^NUM_GPUS_PER_PROCESS\s*=.*$": (
        f"NUM_GPUS_PER_PROCESS = {gpus_per_process}"
    ),
}

for pattern, replacement in replacements.items():
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not update config line matching: {pattern}")

path.write_text(text, encoding="utf-8")
PY
}

inference_is_complete() {
    local dataset_path="$1"
    local exp_name="$2"

    "$PYTHON_BIN" - "$dataset_path" "exp/$exp_name" <<'PY'
import glob
import json
import os
import sys

dataset_path, exp_dir = sys.argv[1:3]

with open(dataset_path, "r", encoding="utf-8") as stream:
    dataset_size = len(json.load(stream))

seen = set()
for answer_path in glob.glob(os.path.join(exp_dir, "generated_answer*.json")):
    with open(answer_path, "r", encoding="utf-8") as stream:
        for item in json.load(stream):
            if isinstance(item, dict) and "idx" in item:
                seen.add(int(item["idx"]))

complete = seen == set(range(dataset_size))
print(
    f"Inference status for {exp_dir}: "
    f"{len(seen)}/{dataset_size} responses present"
)
raise SystemExit(0 if complete else 1)
PY
}

run_inference() {
    local model_checkpoint="$1"
    local dataset_path="$2"
    local exp_name="$3"
    local ts_encoder_type="$4"

    if [[ "$FORCE_INFERENCE" != "1" ]] && \
       inference_is_complete "$dataset_path" "$exp_name"; then
        echo "Inference already complete; reusing exp/$exp_name."
        return 0
    fi

    if [[ ! -d "$model_checkpoint" ]]; then
        echo "Error: model checkpoint does not exist: $model_checkpoint" >&2
        return 1
    fi

    echo "Starting inference: $exp_name"
    configure_inference "$model_checkpoint" "$dataset_path" "$exp_name"

    local -a launch_env=(
        "VLLM_WORKER_MULTIPROC_METHOD=spawn"
        "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
    )
    if [[ -n "$ts_encoder_type" ]]; then
        # env passes the selection to the parent Python process and every vLLM
        # spawn worker without requiring a permanent shell export.
        launch_env+=("CHATTS_TS_ENCODER_TYPE=$ts_encoder_type")
    fi
    [[ -n "$TIMESFM_MODEL_PATH" ]] && \
        launch_env+=("CHATTS_TIMESFM_MODEL_PATH=$TIMESFM_MODEL_PATH")
    [[ -n "$CHRONOS2_MODEL_PATH" ]] && \
        launch_env+=("CHATTS_CHRONOS2_MODEL_PATH=$CHRONOS2_MODEL_PATH")
    [[ -n "$ZEUS_MODEL_PATH" ]] && \
        launch_env+=("CHATTS_ZEUS_MODEL_PATH=$ZEUS_MODEL_PATH")

    env "${launch_env[@]}" \
        "$PYTHON_BIN" -m chatts.utils.inference_tsmllm_vllm

    if ! inference_is_complete "$dataset_path" "$exp_name"; then
        echo "Error: inference output is incomplete for $exp_name" >&2
        return 1
    fi
    return 0
}

run_fast_evaluation() {
    local exp_a="$1"
    local exp_b="$2"

    EVAL_WORKERS="$EVAL_WORKERS" \
    EXP_A="$exp_a" \
    EXP_B="$exp_b" \
    DATASET_A="$DATASET_A" \
    DATASET_B="$DATASET_B" \
        "$PYTHON_BIN" <<'PY'
import glob
import json
import multiprocessing as mp
import os
import sys
import types

# evaluate_qa.py imports evaluation.ragas.score at module load time. Register a
# tiny in-memory replacement first, so RAGAS and its configuration are never
# imported and no external judge request can occur.
ragas_stub = types.ModuleType("evaluation.ragas.score")
ragas_stub.calculate_ragas_score = (
    lambda *args, **kwargs: (0.0, {"skipped": True})
)
sys.modules["evaluation.ragas.score"] = ragas_stub

# ChatTS evaluation uses multiprocessing.Pool. Linux fork preserves the stub in
# child workers and keeps evaluation entirely local.
mp.set_start_method("fork", force=True)

import evaluation.evaluate_qa as qa


def load_generated_answers(dataset, exp_name):
    exp_dir = os.path.join("exp", exp_name)
    paths = sorted(glob.glob(os.path.join(exp_dir, "generated_answer*.json")))
    if not paths:
        raise RuntimeError(
            f"No generated answers found under {exp_dir}/generated_answer*.json"
        )

    answer_by_idx = {}
    for path in paths:
        print(f"Loading inference output: {path}")
        with open(path, "r", encoding="utf-8") as stream:
            items = json.load(stream)
        for item in items:
            if isinstance(item, dict) and "idx" in item:
                answer_by_idx[int(item["idx"])] = item

    missing = [
        idx for idx in range(len(dataset))
        if idx not in answer_by_idx
    ]
    if missing:
        raise RuntimeError(
            f"{exp_name} is missing {len(missing)} responses; "
            f"first missing indices: {missing[:20]}"
        )

    return [answer_by_idx[idx] for idx in range(len(dataset))]


def evaluate(dataset_path, exp_name, workers):
    print(f"\nFast local evaluation: {exp_name}")
    with open(dataset_path, "r", encoding="utf-8") as stream:
        dataset = json.load(stream)

    generated_answers = load_generated_answers(dataset, exp_name)
    qa.evaluate_batch_qa(
        dataset,
        generated_answers,
        exp_name,
        num_workers=workers,
    )

    result_path = os.path.join("exp", exp_name, "result.json")
    with open(result_path, "r", encoding="utf-8") as stream:
        raw_result = json.load(stream)

    # Keep a clean result file containing only valid no-RAGAS metrics.
    clean_result = {
        "detail_categorical": raw_result["detail_categorical"],
        "detail_numerical": raw_result["detail_numerical"],
        "overall_categorical": raw_result["overall_categorical"],
        "overall_numerical": raw_result["overall_numerical"],
        "consumed_tokens": raw_result.get("consumed_tokens", 0),
        "reason_skipped": True,
    }

    raw_path = os.path.join("exp", exp_name, "result_raw.json")
    with open(raw_path, "w", encoding="utf-8") as stream:
        json.dump(raw_result, stream, ensure_ascii=False, indent=2)
    with open(result_path, "w", encoding="utf-8") as stream:
        json.dump(clean_result, stream, ensure_ascii=False, indent=2)

    print(json.dumps(clean_result, ensure_ascii=False, indent=2))
    print(f"Saved: {result_path}")


workers = max(1, int(os.environ["EVAL_WORKERS"]))
runs = [
    (os.environ["DATASET_A"], os.environ["EXP_A"]),
    (os.environ["DATASET_B"], os.environ["EXP_B"]),
]

for dataset_path, exp_name in runs:
    evaluate(dataset_path, exp_name, workers)
PY
}

write_batch_summary() {
    local summary_dir="$SUMMARY_DIR"

    mkdir -p "$summary_dir"
    "$PYTHON_BIN" - \
        "$PROJECT_ROOT" \
        "$summary_dir" \
        "$SUMMARY_BASENAME" \
        "${MODEL_DIRS[@]}" \
        --failed-models \
        "${FAILED_MODELS[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from typing import Any


project_root = os.path.abspath(sys.argv[1])
summary_dir = os.path.abspath(sys.argv[2])
summary_basename = sys.argv[3]
remaining_args = sys.argv[4:]
try:
    failed_marker = remaining_args.index("--failed-models")
except ValueError as exc:
    raise SystemExit("Internal error: missing --failed-models marker.") from exc
model_names = remaining_args[:failed_marker]
failed_models = set(remaining_args[failed_marker + 1:])


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_result(model_name: str, dataset_name: str):
    exp_name = f"{model_name}_{dataset_name}"
    path = os.path.join(project_root, "exp", exp_name, "result.json")
    if not os.path.isfile(path):
        return None, path, "missing"
    try:
        with open(path, "r", encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return None, path, f"invalid: {exc}"
    return result, path, "ok"


def mean_or_none(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return statistics.fmean(valid) if valid else None


rows = []
for model_name in model_names:
    result_a, path_a, state_a = load_result(model_name, "dataset_a")
    result_b, path_b, state_b = load_result(model_name, "dataset_b")

    a_categorical = finite_number(
        result_a.get("overall_categorical") if result_a else None
    )
    a_numerical = finite_number(
        result_a.get("overall_numerical") if result_a else None
    )
    b_categorical = finite_number(
        result_b.get("overall_categorical") if result_b else None
    )
    b_numerical = finite_number(
        result_b.get("overall_numerical") if result_b else None
    )
    a_tokens = finite_number(result_a.get("consumed_tokens") if result_a else None)
    b_tokens = finite_number(result_b.get("consumed_tokens") if result_b else None)

    metric_values = [a_categorical, a_numerical, b_categorical, b_numerical]
    if model_name in failed_models:
        status = "current run failed; result files may be stale"
    elif state_a != "ok" or state_b != "ok":
        status = f"dataset_a={state_a}; dataset_b={state_b}"
    elif any(value is None for value in metric_values):
        status = "invalid or missing overall metric"
    else:
        status = "ok"

    rows.append({
        "rank": None,
        "model": model_name,
        "dataset_a_categorical": a_categorical,
        "dataset_a_numerical": a_numerical,
        "dataset_b_categorical": b_categorical,
        "dataset_b_numerical": b_numerical,
        "avg_categorical": mean_or_none([a_categorical, b_categorical]),
        "avg_numerical": mean_or_none([a_numerical, b_numerical]),
        "macro_mean": mean_or_none(metric_values),
        "dataset_a_tokens": int(a_tokens) if a_tokens is not None else None,
        "dataset_b_tokens": int(b_tokens) if b_tokens is not None else None,
        "total_tokens": (
            int(a_tokens + b_tokens)
            if a_tokens is not None and b_tokens is not None
            else None
        ),
        "status": status,
        "dataset_a_result": path_a,
        "dataset_b_result": path_b,
    })


def sort_key(row):
    score = row["macro_mean"]
    return (
        row["status"] != "ok",
        -(score if score is not None else float("-inf")),
        row["model"],
    )


rows.sort(key=sort_key)
rank = 0
for row in rows:
    if row["status"] == "ok" and row["macro_mean"] is not None:
        rank += 1
        row["rank"] = rank

fieldnames = [
    "rank",
    "model",
    "dataset_a_categorical",
    "dataset_a_numerical",
    "dataset_b_categorical",
    "dataset_b_numerical",
    "avg_categorical",
    "avg_numerical",
    "macro_mean",
    "dataset_a_tokens",
    "dataset_b_tokens",
    "total_tokens",
    "status",
    "dataset_a_result",
    "dataset_b_result",
]

csv_path = os.path.join(summary_dir, f"{summary_basename}.csv")
with open(csv_path, "w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def score_text(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def integer_text(value: int | None) -> str:
    return "—" if value is None else str(value)


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


headers = [
    "Rank",
    "Model",
    "A Cat",
    "A Num",
    "B Cat",
    "B Num",
    "Avg Cat",
    "Avg Num",
    "Macro Mean",
    "Tokens",
    "Status",
]
markdown_lines = [
    "# ChatTS Batch Evaluation Summary",
    "",
    (
        "`Macro Mean` is the unweighted mean of Dataset A/B categorical and "
        "numerical scores. Rows are ranked by this value."
    ),
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for row in rows:
    values = [
        integer_text(row["rank"]),
        markdown_cell(row["model"]),
        score_text(row["dataset_a_categorical"]),
        score_text(row["dataset_a_numerical"]),
        score_text(row["dataset_b_categorical"]),
        score_text(row["dataset_b_numerical"]),
        score_text(row["avg_categorical"]),
        score_text(row["avg_numerical"]),
        score_text(row["macro_mean"]),
        integer_text(row["total_tokens"]),
        markdown_cell(row["status"]),
    ]
    markdown_lines.append("| " + " | ".join(values) + " |")

markdown_path = os.path.join(summary_dir, f"{summary_basename}.md")
with open(markdown_path, "w", encoding="utf-8") as stream:
    stream.write("\n".join(markdown_lines) + "\n")

print("\n".join(markdown_lines))
print(f"\nCSV summary:      {csv_path}")
print(f"Markdown summary: {markdown_path}")
PY
}

# ==================== Main batch loop ====================

FAILED_MODELS=()
ALL_RESULTS=()

for model_dir in "${MODEL_DIRS[@]}"; do
    echo ""
    echo "######################################################"
    echo "# Processing model: $model_dir"
    echo "######################################################"

    # Build the full model checkpoint path
    if [[ -n "$CHECKPOINT_SUFFIX" ]]; then
        MODEL_CHECKPOINT="${SEARCH_DIR%/}/${model_dir}/${CHECKPOINT_SUFFIX#/}"
    else
        MODEL_CHECKPOINT="${SEARCH_DIR%/}/${model_dir}"
    fi

    # Experiment names derived from the subdirectory name
    EXP_A="${model_dir}_dataset_a"
    EXP_B="${model_dir}_dataset_b"

    echo "  Checkpoint: $MODEL_CHECKPOINT"
    echo "  EXP_A:      $EXP_A"
    echo "  EXP_B:      $EXP_B"

    model_failed=0
    MODEL_TS_ENCODER_TYPE=""
    if [[ "$MODE" != "--score-only" && "$MODE" != "--summary-only" ]]; then
        if MODEL_TS_ENCODER_TYPE="$(detect_encoder_type_from_checkpoint \
                "$MODEL_CHECKPOINT" "$TS_ENCODER_TYPE")"; then
            echo "  TS encoder: $MODEL_TS_ENCODER_TYPE (resolved per checkpoint)"
        else
            echo "WARNING: Could not detect TS encoder for $model_dir" >&2
            model_failed=1
        fi
    fi

    # ---- Inference ----
    if [[ "$MODE" != "--score-only" && "$MODE" != "--summary-only" && \
          "$model_failed" -eq 0 ]]; then
        echo "==================== Inference ===================="
        if ! run_inference "$MODEL_CHECKPOINT" "$DATASET_A" "$EXP_A" "$MODEL_TS_ENCODER_TYPE"; then
            echo "WARNING: Inference failed for $model_dir (dataset A)" >&2
            model_failed=1
        fi
        if ! run_inference "$MODEL_CHECKPOINT" "$DATASET_B" "$EXP_B" "$MODEL_TS_ENCODER_TYPE"; then
            echo "WARNING: Inference failed for $model_dir (dataset B)" >&2
            model_failed=1
        fi
    fi

    # ---- Evaluation ----
    if [[ "$MODE" != "--infer-only" && "$MODE" != "--summary-only" && \
          "$model_failed" -eq 0 ]]; then
        echo "============= No-RAGAS local evaluation ==========="
        if ! run_fast_evaluation "$EXP_A" "$EXP_B"; then
            echo "WARNING: Evaluation failed for $model_dir" >&2
            model_failed=1
        fi
    fi

    if [[ "$model_failed" -ne 0 ]]; then
        FAILED_MODELS+=("$model_dir")
    else
        ALL_RESULTS+=("$model_dir -> exp/$EXP_A/result.json, exp/$EXP_B/result.json")
    fi
done

# ==================== Summary ====================
echo ""
echo "====================== Batch Summary ========================"

if [[ ${#ALL_RESULTS[@]} -gt 0 ]]; then
    echo "Successful models:"
    for r in "${ALL_RESULTS[@]}"; do
        echo "  ✓ $r"
    done
fi

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed models:"
    for f in "${FAILED_MODELS[@]}"; do
        echo "  ✗ $f"
    done
fi

if [[ "$MODE" != "--infer-only" ]]; then
    echo ""
    echo "================ Aggregated Results Table ==================="
    if ! write_batch_summary; then
        echo "WARNING: Failed to write the aggregated result table." >&2
        FAILED_MODELS+=("<summary generation>")
    fi
else
    echo "Aggregated table skipped in --infer-only mode (no evaluation was run)."
fi

echo "=============================================================="
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    exit 1
fi
echo "All ${#MODEL_DIRS[@]} model(s) processed successfully."

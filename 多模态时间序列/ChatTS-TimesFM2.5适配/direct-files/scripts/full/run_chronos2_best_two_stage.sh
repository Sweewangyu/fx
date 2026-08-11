#!/usr/bin/env bash
set -Eeuo pipefail

# Deterministic Chronos-2 Stage1 -> Stage2 training pipeline.
#
# Backward-compatible default: PIPELINE_MODE=full trains both stages and removes
# the temporary Stage1 model after Stage2.  Autoresearch can train/reuse Stage1
# explicitly with PIPELINE_MODE=stage1|stage2 and STAGE2_FROM.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ChatTS-Training}"
MODEL_PATH="${MODEL_PATH:-/share/airesearch/data/finiverse/model/ChatTS-Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/share/airesearch/data/finiverse/output/ChatTS-msxf-8B-datav1}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-/workspace/chronos2}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"
STAGE2_FROM="${STAGE2_FROM:-}"
KEEP_STAGE1="${KEEP_STAGE1:-0}"
TRIAL_ID="${TRIAL_ID:-}"
TRIAL_CONFIG_HASH="${TRIAL_CONFIG_HASH:-}"
DATASET_SNAPSHOT_HASH="${DATASET_SNAPSHOT_HASH:-}"

SEED="${SEED:-42}"
S1_LR="${S1_LR:-1e-5}"
S2_LR="${S2_LR:-1e-5}"
STAGE1_DATASETS="${STAGE1_DATASETS:-align_256,ift}"
STAGE2_DATASETS="${STAGE2_DATASETS:-sft,align_random,finiverse_time_mqa,finiverse_tsaqa}"
STAGE1_MIX_STRATEGY="${STAGE1_MIX_STRATEGY:-interleave_over}"
STAGE2_MIX_STRATEGY="${STAGE2_MIX_STRATEGY:-concat}"
STAGE1_INTERLEAVE_PROBS="${STAGE1_INTERLEAVE_PROBS:-0.9,0.1}"
STAGE2_INTERLEAVE_PROBS="${STAGE2_INTERLEAVE_PROBS:-}"
STAGE1_TIMESERIES_SFT_LR="${STAGE1_TIMESERIES_SFT_LR:-$S1_LR}"
STAGE2_TIMESERIES_SFT_LR="${STAGE2_TIMESERIES_SFT_LR:-$S2_LR}"
STAGE1_NUM_TRAIN_EPOCHS="${STAGE1_NUM_TRAIN_EPOCHS:-3}"
STAGE2_NUM_TRAIN_EPOCHS="${STAGE2_NUM_TRAIN_EPOCHS:-1}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-0}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-0}"
STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE="${STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE="${STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
STAGE1_GRADIENT_ACCUMULATION_STEPS="${STAGE1_GRADIENT_ACCUMULATION_STEPS:-32}"
STAGE2_GRADIENT_ACCUMULATION_STEPS="${STAGE2_GRADIENT_ACCUMULATION_STEPS:-32}"
STAGE1_LR_SCHEDULER_TYPE="${STAGE1_LR_SCHEDULER_TYPE:-cosine}"
STAGE2_LR_SCHEDULER_TYPE="${STAGE2_LR_SCHEDULER_TYPE:-cosine}"
STAGE1_WARMUP_RATIO="${STAGE1_WARMUP_RATIO:-0.02}"
STAGE2_WARMUP_RATIO="${STAGE2_WARMUP_RATIO:-0.02}"
STAGE1_LOGGING_STEPS="${STAGE1_LOGGING_STEPS:-1}"
STAGE2_LOGGING_STEPS="${STAGE2_LOGGING_STEPS:-1}"
STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-200}"
STAGE2_SAVE_STEPS="${STAGE2_SAVE_STEPS:-100}"
STAGE1_EVAL_STEPS="${STAGE1_EVAL_STEPS:-200}"
STAGE2_EVAL_STEPS="${STAGE2_EVAL_STEPS:-100}"
STAGE1_VAL_SIZE="${STAGE1_VAL_SIZE:-0.05}"
STAGE2_VAL_SIZE="${STAGE2_VAL_SIZE:-0.05}"
STAGE1_PER_DEVICE_EVAL_BATCH_SIZE="${STAGE1_PER_DEVICE_EVAL_BATCH_SIZE:-2}"
STAGE2_PER_DEVICE_EVAL_BATCH_SIZE="${STAGE2_PER_DEVICE_EVAL_BATCH_SIZE:-4}"
STAGE1_CUTOFF_LEN="${STAGE1_CUTOFF_LEN:-2048}"
STAGE2_CUTOFF_LEN="${STAGE2_CUTOFF_LEN:-2048}"
STAGE1_PREPROCESSING_NUM_WORKERS="${STAGE1_PREPROCESSING_NUM_WORKERS:-96}"
STAGE2_PREPROCESSING_NUM_WORKERS="${STAGE2_PREPROCESSING_NUM_WORKERS:-96}"
DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-19901}"

FORCE_TRAIN="${FORCE_TRAIN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

resolve_path() {
    "$PYTHON_BIN" - "$1" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

PROJECT_ROOT="$(resolve_path "$PROJECT_ROOT")"
MODEL_PATH="$(resolve_path "$MODEL_PATH")"
OUTPUT_ROOT="$(resolve_path "$OUTPUT_ROOT")"
CHRONOS2_MODEL_PATH="$(resolve_path "$CHRONOS2_MODEL_PATH")"
DATASET_DIR="$(resolve_path "$DATASET_DIR")"

STAGE1_SCRIPT="${STAGE1_SCRIPT:-${PROJECT_ROOT}/scripts/full/train_chronos2_best_stage1.sh}"
STAGE2_SCRIPT="${STAGE2_SCRIPT:-${PROJECT_ROOT}/scripts/full/train_chronos2_best_stage2.sh}"
FINALIZER="${FINALIZER:-${PROJECT_ROOT}/scripts/finalize_chatts_best_checkpoint.py}"
STAGE1_OUT="${STAGE1_OUT:-${OUTPUT_ROOT}/.stage1_seed${SEED}_s1lr_${S1_LR}}"
FINAL_MODEL_PATH="${FINAL_MODEL_PATH:-${OUTPUT_ROOT}/best_seed${SEED}}"
DEFAULT_RUN_NAME="chronos2_seed${SEED}_s1lr_${S1_LR}_s2lr_${S2_LR}"
if [[ -n "$TRIAL_ID" ]]; then
    DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_${TRIAL_ID}"
fi
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs/${RUN_NAME}}"
STAGE1_SCRIPT="$(resolve_path "$STAGE1_SCRIPT")"
STAGE2_SCRIPT="$(resolve_path "$STAGE2_SCRIPT")"
FINALIZER="$(resolve_path "$FINALIZER")"
STAGE1_OUT="$(resolve_path "$STAGE1_OUT")"
FINAL_MODEL_PATH="$(resolve_path "$FINAL_MODEL_PATH")"
LOG_ROOT="$(resolve_path "$LOG_ROOT")"
if [[ -n "$STAGE2_FROM" ]]; then
    STAGE2_FROM="$(resolve_path "$STAGE2_FROM")"
fi
READY_MARKER="${FINAL_MODEL_PATH}/TRAINING_COMPLETE.json"
STAGE1_READY_MARKER="${STAGE1_OUT}/STAGE1_COMPLETE.json"

case "$PIPELINE_MODE" in
    full|stage1|stage2) ;;
    *) echo "PIPELINE_MODE must be full, stage1, or stage2; got: $PIPELINE_MODE" >&2; exit 2 ;;
esac
for flag_name in FORCE_TRAIN PREFLIGHT_ONLY KEEP_STAGE1; do
    flag_value="${!flag_name}"
    [[ "$flag_value" == "0" || "$flag_value" == "1" ]] || {
        echo "$flag_name must be 0 or 1, got: $flag_value" >&2
        exit 2
    }
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer." >&2; exit 2; }
[[ -n "$DATASET_DIR" ]] || { echo "DATASET_DIR must not be empty." >&2; exit 2; }
[[ -z "$TRIAL_ID" || "$TRIAL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "TRIAL_ID may contain only letters, numbers, dot, underscore, and dash." >&2
    exit 2
}
for lr_name in S1_LR S2_LR STAGE1_TIMESERIES_SFT_LR STAGE2_TIMESERIES_SFT_LR; do
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
for hash_name in TRIAL_CONFIG_HASH DATASET_SNAPSHOT_HASH; do
    hash_value="${!hash_name}"
    [[ -z "$hash_value" || "$hash_value" =~ ^[0-9a-fA-F]{64}$ ]] || {
        echo "$hash_name must be empty or a 64-character SHA256 hex digest." >&2
        exit 2
    }
done
if [[ "$PIPELINE_MODE" == "stage2" && -z "$STAGE2_FROM" ]]; then
    echo "STAGE2_FROM is required when PIPELINE_MODE=stage2." >&2
    exit 2
fi
if [[ "$PIPELINE_MODE" != "stage2" && -n "$STAGE2_FROM" ]]; then
    echo "STAGE2_FROM is only valid when PIPELINE_MODE=stage2." >&2
    exit 2
fi

require_dir() {
    local label="$1" path="$2"
    [[ -d "$path" ]] || { echo "$label directory not found: $path" >&2; exit 1; }
}

require_file() {
    local label="$1" path="$2"
    [[ -f "$path" ]] || { echo "$label file not found: $path" >&2; exit 1; }
}

validate_output_layout() {
    "$PYTHON_BIN" - \
        "$PIPELINE_MODE" "$OUTPUT_ROOT" "$STAGE1_OUT" "$STAGE2_FROM" \
        "$FINAL_MODEL_PATH" "$LOG_ROOT" <<'PY'
import sys
from pathlib import Path

mode = sys.argv[1]
root = Path(sys.argv[2]).resolve()
stage1_out = Path(sys.argv[3]).resolve()
stage2_from = Path(sys.argv[4]).resolve() if sys.argv[4] else None
final = Path(sys.argv[5]).resolve()
log_root = Path(sys.argv[6]).resolve()

if root == Path(root.anchor):
    raise SystemExit(f"OUTPUT_ROOT must not be a filesystem root: {root}")

def require_strict_descendant(label: str, path: Path) -> None:
    if path == root:
        raise SystemExit(f"{label} must not equal OUTPUT_ROOT: {path}")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{label} must be inside OUTPUT_ROOT: {path}") from exc

def overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents

if mode in {"full", "stage1"}:
    require_strict_descendant("STAGE1_OUT", stage1_out)
if mode in {"full", "stage2"}:
    require_strict_descendant("FINAL_MODEL_PATH", final)

model_paths: list[tuple[str, Path]] = []
if mode in {"full", "stage1"}:
    model_paths.append(("STAGE1_OUT", stage1_out))
if mode in {"full", "stage2"}:
    model_paths.append(("FINAL_MODEL_PATH", final))
if mode == "stage2":
    assert stage2_from is not None
    model_paths.append(("STAGE2_FROM", stage2_from))

for index, (left_label, left) in enumerate(model_paths):
    for right_label, right in model_paths[index + 1 :]:
        if overlaps(left, right):
            raise SystemExit(
                f"Unsafe overlapping model paths: {left_label}={left} and "
                f"{right_label}={right}. Model inputs and outputs must be disjoint."
            )
    if overlaps(log_root, left):
        raise SystemExit(
            f"LOG_ROOT must be disjoint from {left_label}: {log_root} vs {left}"
        )
PY
}

write_owned_sentinel() {
    local target="$1" expected_kind="$2"
    "$PYTHON_BIN" - "$OUTPUT_ROOT" "$target" "$expected_kind" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
kind = sys.argv[3]
try:
    target.relative_to(root)
except ValueError as exc:
    raise SystemExit(f"Refusing to own {kind} outside OUTPUT_ROOT: {target}") from exc
if target == root:
    raise SystemExit(f"Refusing to own OUTPUT_ROOT as {kind}: {target}")
sentinel = target.parent / f".{target.name}.chatts-owned-output.json"
payload = {
    "format_version": 1,
    "kind": kind,
    "output_root": str(root),
    "target_path": str(target),
}
target.parent.mkdir(parents=True, exist_ok=True)
if sentinel.is_file():
    with sentinel.open(encoding="utf-8") as stream:
        current = json.load(stream)
    if current != payload:
        raise SystemExit(f"Owned-output sentinel mismatch: {sentinel}")
else:
    temporary = sentinel.with_suffix(sentinel.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(sentinel)
print(f"Owned {kind} output: {target}")
PY
}

safe_remove_model_dir() {
    local target="$1" expected_kind="$2" sentinel
    sentinel="$("$PYTHON_BIN" - "$OUTPUT_ROOT" "$target" "$expected_kind" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
kind = sys.argv[3]
if target == root:
    raise SystemExit(f"Refusing to remove OUTPUT_ROOT as {kind}: {target}")
try:
    target.relative_to(root)
except ValueError as exc:
    raise SystemExit(f"Refusing to remove {kind} outside OUTPUT_ROOT: {target}") from exc

sentinel = target.parent / f".{target.name}.chatts-owned-output.json"
target_exists = os.path.lexists(target)
sentinel_exists = sentinel.is_file()
legacy_patterns = {
    "stage1": re.compile(r"[.]stage1_seed[0-9]+_s1lr_[A-Za-z0-9+_.-]+"),
    "final": re.compile(r"best_seed[0-9]+"),
}
legacy_owned = target.parent == root and bool(legacy_patterns[kind].fullmatch(target.name))
completion_name = "STAGE1_COMPLETE.json" if kind == "stage1" else "TRAINING_COMPLETE.json"
completion_path = target / completion_name
marker_owned = False
if completion_path.is_file():
    with completion_path.open(encoding="utf-8") as stream:
        completion = json.load(stream)
    path_key = "stage1_model_path" if kind == "stage1" else "final_model_path"
    marker_owned = (
        completion.get("status") == "complete"
        and Path(completion.get(path_key, "")).expanduser().resolve() == target
        and (kind != "stage1" or completion.get("kind") == "stage1")
    )
if target_exists or sentinel_exists:
    if sentinel_exists:
        with sentinel.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        expected = {
            "format_version": 1,
            "kind": kind,
            "output_root": str(root),
            "target_path": str(target),
        }
        if payload != expected:
            raise SystemExit(f"Owned-output sentinel mismatch: {sentinel}")
    elif not legacy_owned and not marker_owned:
        raise SystemExit(
            f"Refusing to remove unowned {kind} output without sentinel: {target}"
        )
print(sentinel)
PY
)"
    if [[ -e "$target" || -L "$target" ]]; then
        rm -rf -- "$target"
        echo "Removed previous $expected_kind output: $target"
    fi
    if [[ -f "$sentinel" ]]; then
        rm -f -- "$sentinel"
    fi
}

model_artifact_json() {
    "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1]).resolve()
required = [model_dir / "config.json", model_dir / "best_model_manifest.json"]
for path in required:
    if not path.is_file():
        raise SystemExit(f"Required model artifact file not found: {path}")

weight_patterns = (
    "pytorch_model*.bin",
    "model*.safetensors",
    "adapter_model*.bin",
    "adapter_model*.safetensors",
)
weights: dict[str, Path] = {}
for pattern in weight_patterns:
    for path in model_dir.glob(pattern):
        if path.is_file() and path.stat().st_size > 0:
            weights[path.name] = path
if not weights:
    raise SystemExit(f"No non-empty model weights found in finalized model: {model_dir}")

def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

paths = required + [weights[name] for name in sorted(weights)]
entries = [
    {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest_file(path),
    }
    for path in paths
]
canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
payload = {
    "format_version": 1,
    "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    "files": entries,
}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
PY
}

stage1_source_provenance_json() {
    local source_dir="$1" artifact
    artifact="$(model_artifact_json "$source_dir")"
    MODEL_ARTIFACT_JSON="$artifact" "$PYTHON_BIN" - "$source_dir" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
artifact = json.loads(os.environ["MODEL_ARTIFACT_JSON"])
complete_path = source / "STAGE1_COMPLETE.json"
complete_sha256 = None
if complete_path.is_file():
    complete_sha256 = hashlib.sha256(complete_path.read_bytes()).hexdigest()
identity = {
    "model_artifact_sha256": artifact["sha256"],
    "completion_marker_sha256": complete_sha256,
}
canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
payload = {
    "format_version": 1,
    "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    "model_artifact": artifact,
    "completion_marker_sha256": complete_sha256,
}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
PY
}

validate_stage1_source() {
    local source_dir="$1" source_provenance="$2"
    "$PYTHON_BIN" - "$source_dir" "$SEED" "$source_provenance" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

source = Path(sys.argv[1]).expanduser().resolve()
expected_seed = int(sys.argv[2])
source_provenance = json.loads(sys.argv[3]) if sys.argv[3] else None
current_artifact = source_provenance["model_artifact"] if source_provenance else None
config_path = source / "config.json"
manifest_path = source / "best_model_manifest.json"
if not config_path.is_file():
    raise SystemExit(f"Stage1 model config not found: {config_path}")
if not manifest_path.is_file():
    raise SystemExit(f"Stage1 best-model manifest not found: {manifest_path}")
with manifest_path.open(encoding="utf-8") as stream:
    manifest = json.load(stream)
if manifest.get("stage") != "stage1":
    raise SystemExit("STAGE2_FROM best_model_manifest.json is not a Stage1 manifest")
if Path(manifest.get("exported_model_dir", "")).expanduser().resolve() != source:
    raise SystemExit("Stage1 manifest exported_model_dir does not match STAGE2_FROM")
if manifest.get("seed") != expected_seed:
    raise SystemExit(
        f"Stage1 seed mismatch: {manifest.get('seed')!r} != {expected_seed!r}"
    )
if manifest.get("ts_encoder_type") != "chronos2":
    raise SystemExit("STAGE2_FROM was not finalized as a Chronos-2 Stage1 model")
metric = manifest.get("best_metric")
if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
    raise SystemExit("Stage1 manifest has an invalid best_metric")
if not manifest.get("selected_checkpoint"):
    raise SystemExit("Stage1 manifest has no selected_checkpoint provenance")

complete_path = source / "STAGE1_COMPLETE.json"
if complete_path.is_file():
    with complete_path.open(encoding="utf-8") as stream:
        complete = json.load(stream)
    if complete.get("status") != "complete" or complete.get("kind") != "stage1":
        raise SystemExit("Invalid Stage1 completion marker")
    if Path(complete.get("stage1_model_path", "")).expanduser().resolve() != source:
        raise SystemExit("Stage1 completion marker points to another model")
    expected_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if complete.get("best_model_manifest_sha256") != expected_digest:
        raise SystemExit("Stage1 best-model manifest changed after completion")
    resolved = complete.get("resolved_configuration")
    if not isinstance(resolved, dict):
        raise SystemExit("Stage1 completion marker has no resolved configuration")
    canonical = json.dumps(
        resolved, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    resolved_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if complete.get("resolved_configuration_sha256") != resolved_digest:
        raise SystemExit("Stage1 resolved configuration checksum is invalid")
    recorded_artifact = complete.get("stage1_artifact")
    if recorded_artifact is not None and current_artifact is not None:
        if not isinstance(recorded_artifact, dict):
            raise SystemExit("Stage1 completion marker has an invalid artifact descriptor")
        if recorded_artifact.get("sha256") != current_artifact.get("sha256"):
            raise SystemExit("Stage1 model artifact changed after completion")
print(f"Validated Stage1 input: {source}")
PY
}

STAGE1_CONFIGURATION_KEYS="PROJECT_ROOT MODEL_PATH CHRONOS2_MODEL_PATH DATASET_DIR SEED S1_LR STAGE1_DATASETS STAGE1_MIX_STRATEGY STAGE1_INTERLEAVE_PROBS STAGE1_TIMESERIES_SFT_LR STAGE1_NUM_TRAIN_EPOCHS STAGE1_MAX_STEPS STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE STAGE1_GRADIENT_ACCUMULATION_STEPS STAGE1_LR_SCHEDULER_TYPE STAGE1_WARMUP_RATIO STAGE1_LOGGING_STEPS STAGE1_SAVE_STEPS STAGE1_EVAL_STEPS STAGE1_VAL_SIZE STAGE1_PER_DEVICE_EVAL_BATCH_SIZE STAGE1_CUTOFF_LEN STAGE1_PREPROCESSING_NUM_WORKERS DEEPSPEED_INCLUDE MASTER_PORT STAGE1_SCRIPT FINALIZER STAGE1_OUT"
RESOLVED_CONFIGURATION_KEYS="PIPELINE_MODE PROJECT_ROOT MODEL_PATH OUTPUT_ROOT CHRONOS2_MODEL_PATH DATASET_DIR SEED S1_LR S2_LR STAGE1_DATASETS STAGE2_DATASETS STAGE1_MIX_STRATEGY STAGE2_MIX_STRATEGY STAGE1_INTERLEAVE_PROBS STAGE2_INTERLEAVE_PROBS STAGE1_TIMESERIES_SFT_LR STAGE2_TIMESERIES_SFT_LR STAGE1_NUM_TRAIN_EPOCHS STAGE2_NUM_TRAIN_EPOCHS STAGE1_MAX_STEPS STAGE2_MAX_STEPS STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE STAGE1_GRADIENT_ACCUMULATION_STEPS STAGE2_GRADIENT_ACCUMULATION_STEPS STAGE1_LR_SCHEDULER_TYPE STAGE2_LR_SCHEDULER_TYPE STAGE1_WARMUP_RATIO STAGE2_WARMUP_RATIO STAGE1_LOGGING_STEPS STAGE2_LOGGING_STEPS STAGE1_SAVE_STEPS STAGE2_SAVE_STEPS STAGE1_EVAL_STEPS STAGE2_EVAL_STEPS STAGE1_VAL_SIZE STAGE2_VAL_SIZE STAGE1_PER_DEVICE_EVAL_BATCH_SIZE STAGE2_PER_DEVICE_EVAL_BATCH_SIZE STAGE1_CUTOFF_LEN STAGE2_CUTOFF_LEN STAGE1_PREPROCESSING_NUM_WORKERS STAGE2_PREPROCESSING_NUM_WORKERS DEEPSPEED_INCLUDE MASTER_PORT STAGE1_SCRIPT STAGE2_SCRIPT FINALIZER STAGE1_OUT STAGE2_FROM FINAL_MODEL_PATH RUN_NAME LOG_ROOT KEEP_STAGE1 TRIAL_ID"
export STAGE1_CONFIGURATION_KEYS RESOLVED_CONFIGURATION_KEYS

export PROJECT_ROOT MODEL_PATH OUTPUT_ROOT CHRONOS2_MODEL_PATH DATASET_DIR PIPELINE_MODE STAGE2_FROM KEEP_STAGE1
export TRIAL_ID TRIAL_CONFIG_HASH DATASET_SNAPSHOT_HASH SEED S1_LR S2_LR PYTHON_BIN
export STAGE1_DATASETS STAGE2_DATASETS STAGE1_MIX_STRATEGY STAGE2_MIX_STRATEGY
export STAGE1_INTERLEAVE_PROBS STAGE2_INTERLEAVE_PROBS
export STAGE1_TIMESERIES_SFT_LR STAGE2_TIMESERIES_SFT_LR
export STAGE1_NUM_TRAIN_EPOCHS STAGE2_NUM_TRAIN_EPOCHS STAGE1_MAX_STEPS STAGE2_MAX_STEPS
export STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE
export STAGE1_GRADIENT_ACCUMULATION_STEPS STAGE2_GRADIENT_ACCUMULATION_STEPS
export STAGE1_LR_SCHEDULER_TYPE STAGE2_LR_SCHEDULER_TYPE STAGE1_WARMUP_RATIO STAGE2_WARMUP_RATIO
export STAGE1_LOGGING_STEPS STAGE2_LOGGING_STEPS STAGE1_SAVE_STEPS STAGE2_SAVE_STEPS
export STAGE1_EVAL_STEPS STAGE2_EVAL_STEPS STAGE1_VAL_SIZE STAGE2_VAL_SIZE
export STAGE1_PER_DEVICE_EVAL_BATCH_SIZE STAGE2_PER_DEVICE_EVAL_BATCH_SIZE
export STAGE1_CUTOFF_LEN STAGE2_CUTOFF_LEN
export STAGE1_PREPROCESSING_NUM_WORKERS STAGE2_PREPROCESSING_NUM_WORKERS
export DEEPSPEED_INCLUDE MASTER_PORT STAGE1_SCRIPT STAGE2_SCRIPT FINALIZER STAGE1_OUT FINAL_MODEL_PATH
export RUN_NAME LOG_ROOT

configuration_sha256() {
    local key_variable="$1"
    "$PYTHON_BIN" - "$key_variable" <<'PY'
import hashlib
import json
import os
import sys

keys = os.environ[sys.argv[1]].split()
payload = {key: os.environ[key] for key in keys}
canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
PY
}

STAGE1_RESOLVED_CONFIGURATION_SHA256="$(configuration_sha256 STAGE1_CONFIGURATION_KEYS)"
RESOLVED_CONFIGURATION_SHA256="$(configuration_sha256 RESOLVED_CONFIGURATION_KEYS)"
export STAGE1_RESOLVED_CONFIGURATION_SHA256 RESOLVED_CONFIGURATION_SHA256

validate_ready_marker() {
    local final_artifact current_stage1_provenance
    final_artifact="$(model_artifact_json "$FINAL_MODEL_PATH")"
    current_stage1_provenance="${STAGE2_INPUT_PROVENANCE_JSON:-}"
    "$PYTHON_BIN" - \
        "$READY_MARKER" "$FINAL_MODEL_PATH" "$final_artifact" \
        "$current_stage1_provenance" <<'PY'
import json
import os
import sys
from pathlib import Path

marker = Path(sys.argv[1])
model_dir = Path(sys.argv[2]).expanduser().resolve()
current_artifact = json.loads(sys.argv[3])
current_stage1 = json.loads(sys.argv[4]) if sys.argv[4] else None
with marker.open(encoding="utf-8") as stream:
    payload = json.load(stream)

if Path(payload.get("final_model_path", "")).expanduser().resolve() != model_dir:
    raise SystemExit("Existing completion marker points to a different final model path")

legacy_allowed = (
    os.environ["PIPELINE_MODE"] == "full"
    and not os.environ["TRIAL_ID"]
    and not os.environ["TRIAL_CONFIG_HASH"]
    and not os.environ["DATASET_SNAPSHOT_HASH"]
    and "final_artifact" not in payload
)
if legacy_allowed:
    legacy_expected = {
        "status": "complete",
        "seed": int(os.environ["SEED"]),
        "stage1_learning_rate": os.environ["S1_LR"],
        "stage2_learning_rate": os.environ["S2_LR"],
    }
    for key, value in legacy_expected.items():
        if payload.get(key) != value:
            raise SystemExit(
                f"Existing legacy completion marker mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    print(f"Validated legacy completed model: {model_dir}")
    raise SystemExit(0)

expected = {
    "status": "complete",
    "trial_id": os.environ["TRIAL_ID"],
    "trial_config_hash": os.environ["TRIAL_CONFIG_HASH"],
    "dataset_snapshot_hash": os.environ["DATASET_SNAPSHOT_HASH"],
    "resolved_configuration_sha256": os.environ["RESOLVED_CONFIGURATION_SHA256"],
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"Existing completion marker mismatch for {key}: "
            f"{payload.get(key)!r} != {value!r}"
        )
recorded_artifact = payload.get("final_artifact")
if not isinstance(recorded_artifact, dict):
    raise SystemExit("Completion marker has no final artifact digest")
if recorded_artifact != current_artifact:
    raise SystemExit("Final model artifact digest mismatch; refusing cached reuse")
if os.environ["PIPELINE_MODE"] == "stage2":
    recorded_stage1 = payload.get("stage1_input_provenance")
    if not isinstance(recorded_stage1, dict) or current_stage1 is None:
        raise SystemExit("Completion marker has no Stage1 input provenance digest")
    if recorded_stage1.get("sha256") != current_stage1.get("sha256"):
        raise SystemExit("Stage1 input provenance digest mismatch; refusing cached reuse")
print(f"Validated completed model: {model_dir}")
PY
}

validate_stage1_marker_for_reuse() {
    local stage1_artifact
    stage1_artifact="$(model_artifact_json "$STAGE1_OUT")"
    "$PYTHON_BIN" - "$STAGE1_READY_MARKER" "$STAGE1_OUT" "$stage1_artifact" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

marker = Path(sys.argv[1])
model_dir = Path(sys.argv[2]).expanduser().resolve()
current_artifact = json.loads(sys.argv[3])
with marker.open(encoding="utf-8") as stream:
    payload = json.load(stream)
expected = {
    "status": "complete",
    "kind": "stage1",
    "trial_config_hash": os.environ["TRIAL_CONFIG_HASH"],
    "dataset_snapshot_hash": os.environ["DATASET_SNAPSHOT_HASH"],
    "resolved_configuration_sha256": os.environ["STAGE1_RESOLVED_CONFIGURATION_SHA256"],
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"Existing Stage1 marker mismatch for {key}: {payload.get(key)!r} != {value!r}"
        )
if Path(payload.get("stage1_model_path", "")).expanduser().resolve() != model_dir:
    raise SystemExit("Existing Stage1 marker points to a different model path")
resolved = payload.get("resolved_configuration")
if not isinstance(resolved, dict):
    raise SystemExit("Stage1 marker has no resolved configuration")
canonical = json.dumps(resolved, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != payload.get(
    "resolved_configuration_sha256"
):
    raise SystemExit("Stage1 resolved configuration checksum is invalid")
recorded_artifact = payload.get("stage1_artifact")
if not isinstance(recorded_artifact, dict):
    raise SystemExit("Stage1 marker has no model artifact digest; FORCE_TRAIN=1 is required")
if recorded_artifact != current_artifact:
    raise SystemExit("Stage1 model artifact digest mismatch; refusing cached reuse")
print(f"Validated completed Stage1 model: {model_dir}")
PY
}

write_stage1_marker() {
    local stage1_artifact
    stage1_artifact="$(model_artifact_json "$STAGE1_OUT")"
    READY_PATH="$STAGE1_READY_MARKER" STAGE1_ARTIFACT_JSON="$stage1_artifact" \
        "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

marker = Path(os.environ["READY_PATH"])
model_dir = Path(os.environ["STAGE1_OUT"]).expanduser().resolve()
manifest_path = model_dir / "best_model_manifest.json"
with manifest_path.open(encoding="utf-8") as stream:
    manifest = json.load(stream)
configuration = {
    key: os.environ[key] for key in os.environ["STAGE1_CONFIGURATION_KEYS"].split()
}
command = ["bash", os.environ["STAGE1_SCRIPT"], os.environ["S1_LR"], str(model_dir)]
command_json = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
payload = {
    "format_version": 3,
    "status": "complete",
    "kind": "stage1",
    "trial_id": os.environ["TRIAL_ID"],
    "trial_config_hash": os.environ["TRIAL_CONFIG_HASH"],
    "dataset_snapshot_hash": os.environ["DATASET_SNAPSHOT_HASH"],
    "resolved_configuration": configuration,
    "resolved_configuration_sha256": os.environ["STAGE1_RESOLVED_CONFIGURATION_SHA256"],
    "command": command,
    "command_sha256": hashlib.sha256(command_json.encode("utf-8")).hexdigest(),
    "stage1_model_path": str(model_dir),
    "best_model_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "stage1_artifact": json.loads(os.environ["STAGE1_ARTIFACT_JSON"]),
    "best_eval_loss": manifest["best_metric"],
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = marker.with_suffix(marker.suffix + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
temporary.replace(marker)
PY
}

write_final_marker() {
    local final_artifact
    final_artifact="$(model_artifact_json "$FINAL_MODEL_PATH")"
    STAGE1_INPUT="$1" STAGE1_RETAINED="$2" READY_MARKER="$READY_MARKER" LOG_ROOT="$LOG_ROOT" \
        FINAL_ARTIFACT_JSON="$final_artifact" \
        STAGE1_INPUT_PROVENANCE_JSON="$STAGE2_INPUT_PROVENANCE_JSON" \
        "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

marker = Path(os.environ["READY_MARKER"])
stage1_manifest_path = Path(os.environ["LOG_ROOT"]) / "stage1_best_model_manifest.json"
stage2_manifest_path = Path(os.environ["LOG_ROOT"]) / "stage2_best_model_manifest.json"
with stage1_manifest_path.open(encoding="utf-8") as stream:
    stage1 = json.load(stream)
with stage2_manifest_path.open(encoding="utf-8") as stream:
    stage2 = json.load(stream)

stage1_export = Path(stage1["exported_model_dir"]).expanduser().resolve()
stage2_export = Path(stage2["exported_model_dir"]).expanduser().resolve()
stage1_input = Path(os.environ["STAGE1_INPUT"]).expanduser().resolve()
final_model = Path(os.environ["FINAL_MODEL_PATH"]).expanduser().resolve()
stage1_input_provenance = json.loads(os.environ["STAGE1_INPUT_PROVENANCE_JSON"])
final_artifact = json.loads(os.environ["FINAL_ARTIFACT_JSON"])
if stage1_export != stage1_input:
    raise SystemExit("Stage1 log manifest does not describe the selected Stage1 input")
stage2_input = stage2.get("input_best_model")
if not isinstance(stage2_input, dict):
    raise SystemExit("Stage2 manifest does not identify its Stage1 best-model input")
if Path(stage2_input.get("exported_model_dir", "")).expanduser().resolve() != stage1_export:
    raise SystemExit("Stage2 was not initialized from the finalized Stage1 model")
if stage2_input.get("selected_checkpoint") != stage1.get("selected_checkpoint"):
    raise SystemExit("Stage2 input provenance does not match the Stage1 selected checkpoint")
if stage2_export != final_model:
    raise SystemExit("Stage2 best-model export does not match FINAL_MODEL_PATH")

configuration = {
    key: os.environ[key] for key in os.environ["RESOLVED_CONFIGURATION_KEYS"].split()
}
stage1_command = None
if os.environ["PIPELINE_MODE"] == "full":
    stage1_command = [
        "bash", os.environ["STAGE1_SCRIPT"], os.environ["S1_LR"], str(stage1_input)
    ]
stage2_command = [
    "bash", os.environ["STAGE2_SCRIPT"], os.environ["S2_LR"],
    str(stage1_input), str(final_model)
]
commands_json = json.dumps(
    {"stage1": stage1_command, "stage2": stage2_command},
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
source_complete_path = Path(os.environ["LOG_ROOT"]) / "stage1_complete_input.json"
source_complete = None
if source_complete_path.is_file():
    with source_complete_path.open(encoding="utf-8") as stream:
        source_complete = json.load(stream)
stage1_configuration = {}
if isinstance(source_complete, dict) and isinstance(
    source_complete.get("resolved_configuration"), dict
):
    stage1_configuration = source_complete["resolved_configuration"]

payload = {
    "format_version": 3,
    "status": "complete",
    "pipeline_mode": os.environ["PIPELINE_MODE"],
    "trial_id": os.environ["TRIAL_ID"],
    "trial_config_hash": os.environ["TRIAL_CONFIG_HASH"],
    "dataset_snapshot_hash": os.environ["DATASET_SNAPSHOT_HASH"],
    "resolved_configuration": configuration,
    "resolved_configuration_sha256": os.environ["RESOLVED_CONFIGURATION_SHA256"],
    "commands": {"stage1": stage1_command, "stage2": stage2_command},
    "commands_sha256": hashlib.sha256(commands_json.encode("utf-8")).hexdigest(),
    "seed": int(os.environ["SEED"]),
    "stage1_learning_rate": stage1.get("learning_rate", os.environ["S1_LR"]),
    "stage2_learning_rate": os.environ["S2_LR"],
    "stage1_best_eval_loss": stage1["best_metric"],
    "stage2_best_eval_loss": stage2["best_metric"],
    "final_model_path": str(final_model),
    "stage1_input_provenance": stage1_input_provenance,
    "final_artifact": final_artifact,
    "training_lineage": {
        "stage1": {
            "input_model_path": stage1.get("input_model_dir"),
            "dataset_dir": stage1_configuration.get(
                "DATASET_DIR", os.environ["DATASET_DIR"]
            ),
            "datasets": stage1_configuration.get(
                "STAGE1_DATASETS", os.environ["STAGE1_DATASETS"]
            ),
            "mix_strategy": stage1_configuration.get(
                "STAGE1_MIX_STRATEGY", os.environ["STAGE1_MIX_STRATEGY"]
            ),
            "learning_rate": stage1.get("learning_rate", os.environ["S1_LR"]),
            "timeseries_learning_rate": stage1_configuration.get(
                "STAGE1_TIMESERIES_SFT_LR", os.environ["STAGE1_TIMESERIES_SFT_LR"]
            ),
            "selected_checkpoint": stage1["selected_checkpoint"],
            "best_eval_loss": stage1["best_metric"],
            "exported_best_model_path": stage1["exported_model_dir"],
            "completion_marker": source_complete,
        },
        "stage2": {
            "input_model_path": stage1["exported_model_dir"],
            "input_stage1_selected_checkpoint": stage1["selected_checkpoint"],
            "dataset_dir": os.environ["DATASET_DIR"],
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
    "stage1_model_retained": os.environ["STAGE1_RETAINED"] == "1",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = marker.with_suffix(marker.suffix + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
temporary.replace(marker)
run_manifest = Path(os.environ["LOG_ROOT"]) / "training_run_manifest.json"
run_temporary = run_manifest.with_suffix(run_manifest.suffix + ".tmp")
with run_temporary.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
run_temporary.replace(run_manifest)
PY
}

ensure_training_run_manifest() {
    READY_MARKER="$READY_MARKER" LOG_ROOT="$LOG_ROOT" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

marker = Path(os.environ["READY_MARKER"])
run_manifest = Path(os.environ["LOG_ROOT"]) / "training_run_manifest.json"
with marker.open(encoding="utf-8") as stream:
    payload = json.load(stream)
temporary = run_manifest.with_suffix(run_manifest.suffix + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
temporary.replace(run_manifest)
PY
}

validate_output_layout
require_dir "Training project" "$PROJECT_ROOT"
require_dir "Chronos-2" "$CHRONOS2_MODEL_PATH"
require_dir "Dataset" "$DATASET_DIR"
require_file "Checkpoint finalizer" "$FINALIZER"
if [[ "$PIPELINE_MODE" == "full" || "$PIPELINE_MODE" == "stage1" ]]; then
    require_file "Base model config" "$MODEL_PATH/config.json"
    require_file "Stage1 runner" "$STAGE1_SCRIPT"
fi
if [[ "$PIPELINE_MODE" == "full" || "$PIPELINE_MODE" == "stage2" ]]; then
    require_file "Stage2 runner" "$STAGE2_SCRIPT"
fi
if [[ "$PIPELINE_MODE" == "stage2" ]]; then
    validate_stage1_source "$STAGE2_FROM" ""
    STAGE2_INPUT_PROVENANCE_JSON="$(stage1_source_provenance_json "$STAGE2_FROM")"
    validate_stage1_source "$STAGE2_FROM" "$STAGE2_INPUT_PROVENANCE_JSON"
    export STAGE2_INPUT_PROVENANCE_JSON
else
    STAGE2_INPUT_PROVENANCE_JSON=""
    export STAGE2_INPUT_PROVENANCE_JSON
fi

AVAILABLE_GPUS="${AVAILABLE_GPUS_OVERRIDE:-$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')}"
[[ "$AVAILABLE_GPUS" =~ ^[0-9]+$ ]] || { echo "Unable to determine visible GPU count: $AVAILABLE_GPUS" >&2; exit 1; }
if (( AVAILABLE_GPUS < 8 )); then
    echo "Training requires 8 visible GPUs, but PyTorch sees $AVAILABLE_GPUS." >&2
    exit 1
fi

echo "============================================================"
echo " ChatTS Chronos-2 best-model training"
echo " Pipeline mode:     $PIPELINE_MODE"
echo " Trial ID:          ${TRIAL_ID:-<none>}"
echo " Base model:        $MODEL_PATH"
echo " Dataset directory: $DATASET_DIR"
echo " Stage1 LR:         $S1_LR"
echo " Stage2 LR:         $S2_LR"
echo " Stage1 datasets:   $STAGE1_DATASETS"
echo " Stage2 datasets:   $STAGE2_DATASETS"
echo " Seed:              $SEED"
echo " Stage1 output:     $STAGE1_OUT"
if [[ "$PIPELINE_MODE" == "stage2" ]]; then
    echo " Stage2 input:      $STAGE2_FROM"
fi
echo " Final model:       $FINAL_MODEL_PATH"
echo " Logs:              $LOG_ROOT"
echo "============================================================"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "Training preflight passed. No files were changed."
    exit 0
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$LOG_ROOT/tensorboard"

case "$PIPELINE_MODE" in
    full)
        if [[ -f "$READY_MARKER" && "$FORCE_TRAIN" != "1" ]]; then
            validate_ready_marker
            ensure_training_run_manifest
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
        ;;
    stage1)
        if [[ -f "$STAGE1_READY_MARKER" && "$FORCE_TRAIN" != "1" ]]; then
            validate_stage1_marker_for_reuse
            echo "Stage1 already completed; reusing $STAGE1_OUT"
            exit 0
        fi
        if [[ "$FORCE_TRAIN" == "1" ]]; then
            safe_remove_model_dir "$STAGE1_OUT" stage1
        elif [[ -e "$STAGE1_OUT" ]]; then
            echo "Incomplete Stage1 output exists: $STAGE1_OUT" >&2
            echo "Inspect it or rerun with FORCE_TRAIN=1." >&2
            exit 2
        fi
        ;;
    stage2)
        if [[ -f "$READY_MARKER" && "$FORCE_TRAIN" != "1" ]]; then
            validate_ready_marker
            ensure_training_run_manifest
            echo "Stage2 already completed; reusing $FINAL_MODEL_PATH"
            exit 0
        fi
        if [[ "$FORCE_TRAIN" == "1" ]]; then
            safe_remove_model_dir "$FINAL_MODEL_PATH" final
        elif [[ -e "$FINAL_MODEL_PATH" ]]; then
            echo "Incomplete Stage2 output exists: $FINAL_MODEL_PATH" >&2
            echo "Inspect it or rerun with FORCE_TRAIN=1." >&2
            exit 2
        fi
        ;;
esac

if [[ "$PIPELINE_MODE" == "full" || "$PIPELINE_MODE" == "stage1" ]]; then
    write_owned_sentinel "$STAGE1_OUT" stage1
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Starting Stage1"
    TENSORBOARD_DIR="$LOG_ROOT/tensorboard/stage1" \
        bash "$STAGE1_SCRIPT" "$S1_LR" "$STAGE1_OUT" \
        2>&1 | tee "$LOG_ROOT/stage1.log"
    cp "$STAGE1_OUT/best_model_manifest.json" "$LOG_ROOT/stage1_best_model_manifest.json"
    write_stage1_marker
    cp "$STAGE1_READY_MARKER" "$LOG_ROOT/stage1_complete_input.json"
    if [[ "$PIPELINE_MODE" == "stage1" ]]; then
        echo "Stage1 training completed and retained: $STAGE1_OUT"
        exit 0
    fi
fi

if [[ "$PIPELINE_MODE" == "stage2" ]]; then
    STAGE2_INPUT="$STAGE2_FROM"
    cp "$STAGE2_INPUT/best_model_manifest.json" "$LOG_ROOT/stage1_best_model_manifest.json"
    if [[ -f "$STAGE2_INPUT/STAGE1_COMPLETE.json" ]]; then
        cp "$STAGE2_INPUT/STAGE1_COMPLETE.json" "$LOG_ROOT/stage1_complete_input.json"
    else
        rm -f -- "$LOG_ROOT/stage1_complete_input.json"
    fi
else
    STAGE2_INPUT="$STAGE1_OUT"
fi

if [[ "$PIPELINE_MODE" == "full" ]]; then
    STAGE2_INPUT_PROVENANCE_JSON="$(stage1_source_provenance_json "$STAGE2_INPUT")"
    export STAGE2_INPUT_PROVENANCE_JSON
fi

write_owned_sentinel "$FINAL_MODEL_PATH" final
echo "$(date '+%Y-%m-%d %H:%M:%S') | Starting Stage2"
TENSORBOARD_DIR="$LOG_ROOT/tensorboard/stage2" \
    bash "$STAGE2_SCRIPT" "$S2_LR" "$STAGE2_INPUT" "$FINAL_MODEL_PATH" \
    2>&1 | tee "$LOG_ROOT/stage2.log"
cp "$FINAL_MODEL_PATH/best_model_manifest.json" "$LOG_ROOT/stage2_best_model_manifest.json"

STAGE1_RETAINED=1
if [[ "$PIPELINE_MODE" == "full" && "$KEEP_STAGE1" == "0" ]]; then
    safe_remove_model_dir "$STAGE1_OUT" stage1
    STAGE1_RETAINED=0
fi
write_final_marker "$STAGE2_INPUT" "$STAGE1_RETAINED"

echo "Training completed: $FINAL_MODEL_PATH"

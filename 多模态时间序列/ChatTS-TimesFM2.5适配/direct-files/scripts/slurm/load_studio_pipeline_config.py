#!/usr/bin/env python3
"""Validate a frozen Dataset Studio YAML and emit safe training env assignments.

The output format is deliberately simple: one ``NAME=value`` assignment per
line.  The Slurm launcher reads each line as data and never evaluates it as
shell source.  Only the fixed variables in ``KEY_TO_ENV`` can be emitted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


STAGE_KEYS = frozenset(
    {
        "learning_rate",
        "timeseries_learning_rate",
        "datasets",
        "mix_strategy",
        "interleave_probs",
        "num_train_epochs",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "lr_scheduler_type",
        "warmup_ratio",
        "logging_steps",
        "save_steps",
        "eval_steps",
        "val_size",
        "per_device_eval_batch_size",
        "cutoff_len",
        "preprocessing_num_workers",
    }
)

PIPELINE_KEYS = frozenset(
    {
        "seed",
        "data_version",
        "dataset_snapshot_hash",
        "training_recipe_hash",
        "force_train",
        "force_eval",
        "preflight_only",
        "max_samples",
        "offline",
        "trial_id",
        "trial_config_hash",
        # Accept the naming variants used during the Stage1-only transition.
        "training_mode",
        "pipeline_mode",
        "mode",
    }
)

TRAINING_KEYS = frozenset(
    {
        "project_root",
        "script",
        "base_model_path",
        "output_root",
        "final_model_path",
        "chronos2_model_path",
        "dataset_dir",
        "keep_stage1",
        "deepspeed_include",
        "master_port",
        "stage1",
        "stage2",
        "pipeline_mode",
        "mode",
        "stage1_output_path",
        "stage1_model_path",
    }
)

KEY_TO_ENV = {
    "pipeline.seed": "SEED",
    "pipeline.data_version": "DATA_VERSION",
    "pipeline.dataset_snapshot_hash": "DATASET_SNAPSHOT_HASH",
    "pipeline.training_recipe_hash": "TRAINING_RECIPE_HASH",
    "pipeline.force_train": "FORCE_TRAIN",
    "pipeline.preflight_only": "PREFLIGHT_ONLY",
    "pipeline.trial_id": "TRIAL_ID",
    "pipeline.trial_config_hash": "TRIAL_CONFIG_HASH",
    "training.project_root": "PROJECT_ROOT",
    "training.base_model_path": "MODEL_PATH",
    "training.output_root": "OUTPUT_ROOT",
    "training.final_model_path": "FINAL_MODEL_PATH",
    "training.chronos2_model_path": "CHRONOS2_MODEL_PATH",
    "training.dataset_dir": "DATASET_DIR",
    "training.keep_stage1": "KEEP_STAGE1",
    "training.deepspeed_include": "DEEPSPEED_INCLUDE",
    "training.master_port": "MASTER_PORT",
    "training.stage1.learning_rate": "S1_LR",
    "training.stage1.timeseries_learning_rate": "STAGE1_TIMESERIES_SFT_LR",
    "training.stage1.datasets": "STAGE1_DATASETS",
    "training.stage1.mix_strategy": "STAGE1_MIX_STRATEGY",
    "training.stage1.interleave_probs": "STAGE1_INTERLEAVE_PROBS",
    "training.stage1.num_train_epochs": "STAGE1_NUM_TRAIN_EPOCHS",
    "training.stage1.max_steps": "STAGE1_MAX_STEPS",
    "training.stage1.per_device_train_batch_size": "STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE",
    "training.stage1.gradient_accumulation_steps": "STAGE1_GRADIENT_ACCUMULATION_STEPS",
    "training.stage1.lr_scheduler_type": "STAGE1_LR_SCHEDULER_TYPE",
    "training.stage1.warmup_ratio": "STAGE1_WARMUP_RATIO",
    "training.stage1.logging_steps": "STAGE1_LOGGING_STEPS",
    "training.stage1.save_steps": "STAGE1_SAVE_STEPS",
    "training.stage1.eval_steps": "STAGE1_EVAL_STEPS",
    "training.stage1.val_size": "STAGE1_VAL_SIZE",
    "training.stage1.per_device_eval_batch_size": "STAGE1_PER_DEVICE_EVAL_BATCH_SIZE",
    "training.stage1.cutoff_len": "STAGE1_CUTOFF_LEN",
    "training.stage1.preprocessing_num_workers": "STAGE1_PREPROCESSING_NUM_WORKERS",
    "training.stage2.learning_rate": "S2_LR",
    "training.stage2.timeseries_learning_rate": "STAGE2_TIMESERIES_SFT_LR",
    "training.stage2.datasets": "STAGE2_DATASETS",
    "training.stage2.mix_strategy": "STAGE2_MIX_STRATEGY",
    "training.stage2.interleave_probs": "STAGE2_INTERLEAVE_PROBS",
    "training.stage2.num_train_epochs": "STAGE2_NUM_TRAIN_EPOCHS",
    "training.stage2.max_steps": "STAGE2_MAX_STEPS",
    "training.stage2.per_device_train_batch_size": "STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE",
    "training.stage2.gradient_accumulation_steps": "STAGE2_GRADIENT_ACCUMULATION_STEPS",
    "training.stage2.lr_scheduler_type": "STAGE2_LR_SCHEDULER_TYPE",
    "training.stage2.warmup_ratio": "STAGE2_WARMUP_RATIO",
    "training.stage2.logging_steps": "STAGE2_LOGGING_STEPS",
    "training.stage2.save_steps": "STAGE2_SAVE_STEPS",
    "training.stage2.eval_steps": "STAGE2_EVAL_STEPS",
    "training.stage2.val_size": "STAGE2_VAL_SIZE",
    "training.stage2.per_device_eval_batch_size": "STAGE2_PER_DEVICE_EVAL_BATCH_SIZE",
    "training.stage2.cutoff_len": "STAGE2_CUTOFF_LEN",
    "training.stage2.preprocessing_num_workers": "STAGE2_PREPROCESSING_NUM_WORKERS",
}

SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
DATASET_LIST_RE = re.compile(r"[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*")


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if value[0:1] in {"'", '"'}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid quoted YAML scalar: {value}") from exc
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        return [] if not body else [parse_scalar(item) for item in body.split(",")]
    if re.fullmatch(r"[-+]?[0-9]+", value):
        return int(value)
    if re.fullmatch(
        r"[-+]?(?:[0-9]+[.][0-9]*|[0-9]*[.][0-9]+)(?:[eE][-+]?[0-9]+)?",
        value,
    ):
        return float(value)
    return value


def fallback_yaml_load(text: str) -> dict[str, Any]:
    """Parse the mapping-only YAML subset emitted by Dataset Studio."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value

    root: dict[str, Any] = {}
    parents: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for number, original in enumerate(text.splitlines(), 1):
        if not original.strip() or original.lstrip().startswith("#"):
            continue
        content = original.split(" #", 1)[0].rstrip()
        indent = len(content) - len(content.lstrip(" "))
        if "\t" in original[:indent] or indent % 2:
            raise ValueError(f"line {number}: indentation must use multiples of two spaces")
        stripped = content.strip()
        if stripped.startswith("-") or ":" not in stripped:
            raise ValueError(f"line {number}: only nested mappings and inline lists are supported")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
            raise ValueError(f"line {number}: invalid key {key!r}")
        while indent <= parents[-1][0]:
            parents.pop()
        parent = parents[-1][1]
        if key in parent:
            raise ValueError(f"line {number}: duplicate key {key!r}")
        if raw_value.strip():
            parent[key] = parse_scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            parents.append((indent, child))
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        payload = fallback_yaml_load(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise TypeError("top-level YAML value must be a mapping")
    return payload


def mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def reject_unknown(value: dict[str, Any], allowed: frozenset[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {field} fields: {', '.join(unknown)}")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def shell_value(value: Any, field: str) -> str:
    if isinstance(value, bool):
        result = "1" if value else "0"
    elif isinstance(value, list):
        if any(isinstance(item, (dict, list)) for item in value):
            raise ValueError(f"{field} must not contain nested values")
        result = ",".join(shell_value(item, field) for item in value)
    elif isinstance(value, (str, int, float)):
        result = str(value)
    else:
        raise TypeError(f"{field} has unsupported value {value!r}")
    if "\n" in result or "\r" in result or "\x00" in result:
        raise ValueError(f"{field} must not contain NUL or newline characters")
    return result


def require_absolute(value: Any, field: str) -> str:
    result = shell_value(value, field)
    if not result or not PurePosixPath(result).is_absolute():
        raise ValueError(f"{field} must be an absolute POSIX path")
    return result


def normalise_mode(pipeline: dict[str, Any], training: dict[str, Any]) -> str:
    candidates = [
        pipeline.get("training_mode"),
        pipeline.get("pipeline_mode"),
        pipeline.get("mode"),
        training.get("pipeline_mode"),
        training.get("mode"),
    ]
    supplied = [str(value) for value in candidates if value not in (None, "")]
    aliases = {
        "full": "full",
        "two_stage": "full",
        "two-stage": "full",
        "stage1": "stage1",
        "stage1_only": "stage1",
        "stage1-only": "stage1",
    }
    normalised = []
    for value in supplied or ["full"]:
        try:
            normalised.append(aliases[value])
        except KeyError as exc:
            raise ValueError("training mode must be full or stage1") from exc
    if len(set(normalised)) != 1:
        raise ValueError(f"conflicting training mode fields: {supplied}")
    return normalised[0]


def validate_stage(stage: dict[str, Any], field: str) -> None:
    reject_unknown(stage, STAGE_KEYS, field)
    missing = sorted(STAGE_KEYS - set(stage))
    if missing:
        raise ValueError(f"{field} is missing fields: {', '.join(missing)}")
    datasets = shell_value(stage["datasets"], f"{field}.datasets")
    if not DATASET_LIST_RE.fullmatch(datasets):
        raise ValueError(f"{field}.datasets must be a non-empty comma-separated dataset list")
    strategy = stage["mix_strategy"]
    if strategy not in {"concat", "interleave_under", "interleave_over"}:
        raise ValueError(f"{field}.mix_strategy is invalid")
    probabilities = shell_value(stage["interleave_probs"], f"{field}.interleave_probs")
    if strategy == "concat" and probabilities:
        raise ValueError(f"{field}.interleave_probs must be empty for concat")


def validate_and_resolve(
    payload: dict[str, Any], expected_job_id: str
) -> dict[str, str]:
    unknown_top = sorted(set(payload) - {"pipeline", "containers", "training", "evaluation"})
    if unknown_top:
        raise ValueError(f"unknown top-level fields: {', '.join(unknown_top)}")
    pipeline = mapping(payload, "pipeline")
    training = mapping(payload, "training")
    reject_unknown(pipeline, PIPELINE_KEYS, "pipeline")
    reject_unknown(training, TRAINING_KEYS, "training")
    if "containers" in payload and not isinstance(payload["containers"], dict):
        raise ValueError("containers must be a mapping")
    if "evaluation" in payload and not isinstance(payload["evaluation"], dict):
        raise ValueError("evaluation must be a mapping")

    stage1 = mapping(training, "stage1")
    stage2 = mapping(training, "stage2")
    validate_stage(stage1, "training.stage1")
    validate_stage(stage2, "training.stage2")

    trial_id = shell_value(pipeline.get("trial_id"), "pipeline.trial_id")
    if trial_id != expected_job_id:
        raise ValueError(
            f"pipeline.trial_id does not match submitted Studio job: {trial_id!r} != {expected_job_id!r}"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", trial_id):
        raise ValueError("pipeline.trial_id is not a safe Studio job id")
    validated_hashes: dict[str, str] = {}
    for key in ("trial_config_hash", "dataset_snapshot_hash", "training_recipe_hash"):
        raw_value = pipeline.get(key)
        if not isinstance(raw_value, str):
            raise TypeError(f"pipeline.{key} must be a string")
        value = shell_value(raw_value, f"pipeline.{key}")
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"pipeline.{key} must be a 64-character SHA256")
        validated_hashes[key] = value.lower()
    version = shell_value(pipeline.get("data_version"), "pipeline.data_version")
    if not re.fullmatch(r"datav[1-9][0-9]*", version):
        raise ValueError("pipeline.data_version must use canonical datavN form")

    expected_hash = validated_hashes["trial_config_hash"]
    hash_payload = json.loads(json.dumps(payload))
    del hash_payload["pipeline"]["trial_config_hash"]
    actual_hash = canonical_hash(hash_payload)
    if actual_hash != expected_hash:
        raise ValueError(
            "pipeline.trial_config_hash does not match the frozen resolved YAML: "
            f"{actual_hash} != {expected_hash}"
        )

    mode = normalise_mode(pipeline, training)
    project_root = require_absolute(training.get("project_root"), "training.project_root")
    configured_script = require_absolute(training.get("script"), "training.script")
    if PurePosixPath(configured_script).name != "run_chronos2_best_two_stage.sh":
        raise ValueError("training.script must name run_chronos2_best_two_stage.sh")
    base_model = require_absolute(training.get("base_model_path"), "training.base_model_path")
    output_root = require_absolute(training.get("output_root"), "training.output_root")
    final_model = require_absolute(training.get("final_model_path"), "training.final_model_path")
    chronos = require_absolute(training.get("chronos2_model_path"), "training.chronos2_model_path")
    dataset_dir = require_absolute(training.get("dataset_dir"), "training.dataset_dir")

    root_path = PurePosixPath(output_root)
    final_path = PurePosixPath(final_model)
    if root_path == PurePosixPath(root_path.root):
        raise ValueError("training.output_root must not be a filesystem root")
    if final_path == root_path or root_path not in final_path.parents:
        raise ValueError("training.final_model_path must be strictly inside training.output_root")
    recipe_id = f"recipe-{validated_hashes['training_recipe_hash'][:16]}"
    if root_path.name != recipe_id:
        raise ValueError(
            f"training.output_root must end in the frozen recipe id {recipe_id}"
        )
    if any(root_path == PurePosixPath(value) for value in (base_model, chronos, dataset_dir)):
        raise ValueError("training.output_root must not equal an input path")

    seed = pipeline.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("pipeline.seed must be a non-negative integer")
    for flag in ("force_train", "preflight_only"):
        if not isinstance(pipeline.get(flag), bool):
            raise TypeError(f"pipeline.{flag} must be a boolean")
    if not isinstance(training.get("keep_stage1"), bool):
        raise TypeError("training.keep_stage1 must be a boolean")

    flattened: dict[str, Any] = {}
    for source_key, env_name in KEY_TO_ENV.items():
        cursor: Any = payload
        for part in source_key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(f"frozen YAML is missing {source_key}")
            cursor = cursor[part]
        flattened[env_name] = cursor
    flattened["PIPELINE_MODE"] = mode
    # Stage1-only is a final product, not the disposable hidden Stage1 used by
    # the full recipe.  Save it directly to the recipe's final model path.
    if mode == "stage1":
        flattened["STAGE1_OUT"] = final_model
        flattened["KEEP_STAGE1"] = True
    elif training.get("stage1_output_path") not in (None, "") or training.get(
        "stage1_model_path"
    ) not in (None, ""):
        stage1_output = require_absolute(
            training.get("stage1_output_path") or training["stage1_model_path"],
            "training.stage1_output_path",
        )
        flattened["STAGE1_OUT"] = stage1_output

    # Preserve the already validated paths verbatim in the emitted contract.
    flattened.update(
        {
            "PROJECT_ROOT": project_root,
            "MODEL_PATH": base_model,
            "OUTPUT_ROOT": output_root,
            "FINAL_MODEL_PATH": final_model,
            "CHRONOS2_MODEL_PATH": chronos,
            "DATASET_DIR": dataset_dir,
        }
    )
    return {
        name: shell_value(value, name)
        for name, value in sorted(flattened.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file")
    parser.add_argument("--expected-job-id", required=True)
    args = parser.parse_args()
    path = Path(args.config_file).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Frozen Studio configuration not found: {path}")
    try:
        payload = load_yaml(path)
        environment = validate_and_resolve(payload, args.expected_job_id)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid frozen Studio configuration {path}: {exc}") from exc
    for name, value in environment.items():
        print(f"{name}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

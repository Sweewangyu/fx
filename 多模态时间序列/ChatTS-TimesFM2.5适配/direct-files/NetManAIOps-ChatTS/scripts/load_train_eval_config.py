#!/usr/bin/env python3
"""Load the ChatTS train/evaluate YAML file as whitelisted shell variables.

PyYAML is used when available.  A small dependency-free fallback supports the
nested mapping/scalar syntax used by the bundled example, so the host does not
need an extra package merely to launch the Docker pipeline.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shlex
from pathlib import Path
from typing import Any

KEY_TO_ENV = {
    "pipeline.seed": "SEED",
    "pipeline.force_train": "FORCE_TRAIN",
    "pipeline.force_eval": "FORCE_EVAL",
    "pipeline.preflight_only": "PREFLIGHT_ONLY",
    "pipeline.max_samples": "MAX_SAMPLES",
    "pipeline.offline": "OFFLINE",
    "containers.training": "TRAIN_CONTAINER",
    "containers.evaluation": "EVAL_CONTAINER",
    "training.project_root": "TRAIN_PROJECT_ROOT",
    "training.script": "TRAIN_SCRIPT",
    "training.base_model_path": "BASE_MODEL_PATH",
    "training.output_root": "TRAIN_OUTPUT_ROOT",
    "training.final_model_path": "FINAL_MODEL_PATH",
    "training.chronos2_model_path": "TRAIN_CHRONOS2_MODEL_PATH",
    "training.dataset_dir": "DATASET_DIR",
    "training.keep_stage1": "KEEP_STAGE1",
    "training.deepspeed_include": "DEEPSPEED_INCLUDE",
    "training.master_port": "MASTER_PORT",
    "training.stage1.learning_rate": "S1_LR",
    "training.stage1.timeseries_learning_rate": "STAGE1_TIMESERIES_SFT_LR",
    "training.stage1.datasets": "STAGE1_DATASETS",
    "training.stage1.interleave_probs": "STAGE1_INTERLEAVE_PROBS",
    "training.stage1.mix_strategy": "STAGE1_MIX_STRATEGY",
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
    "training.stage2.interleave_probs": "STAGE2_INTERLEAVE_PROBS",
    "training.stage2.mix_strategy": "STAGE2_MIX_STRATEGY",
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
    "evaluation.project_root": "EVAL_PROJECT_ROOT",
    "evaluation.script": "EVAL_SCRIPT",
    "evaluation.model_name": "MODEL_NAME",
    "evaluation.output_root": "EVAL_OUTPUT_ROOT",
    "evaluation.chronos2_model_path": "EVAL_CHRONOS2_MODEL_PATH",
    "evaluation.tsrbench_root": "TSRBENCH_ROOT",
    "evaluation.tinybench_dataset_root": "TINYBENCH_DATASET_ROOT",
    "evaluation.ts_haystack_root": "TS_HAYSTACK_ROOT",
    "evaluation.timeseriesexam_root": "TIMESERIESEXAM_ROOT",
    "evaluation.timeseriesexam_data_file": "TIMESERIESEXAM_DATA_FILE",
    "evaluation.benchmarks": "BENCHMARKS",
    "evaluation.run_id": "RUN_ID",
    "evaluation.protocol_hash": "EVAL_PROTOCOL_HASH",
    "evaluation.haystack_split": "HAYSTACK_SPLIT",
    "evaluation.tiny_data_partition": "TINY_DATA_PARTITION",
    "evaluation.tiny_partition_seed": "TINY_PARTITION_SEED",
}


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
    if re.fullmatch(r"[-+]?(?:[0-9]+[.][0-9]*|[0-9]*[.][0-9]+)(?:[eE][-+]?[0-9]+)?", value):
        return float(value)
    return value


def fallback_yaml_load(text: str) -> dict[str, Any]:
    """Parse the mapping-only subset used by the example configuration."""
    root: dict[str, Any] = {}
    parents: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for number, original in enumerate(text.splitlines(), 1):
        if not original.strip() or original.lstrip().startswith("#"):
            continue
        content = original.split(" #", 1)[0].rstrip()
        indent = len(content) - len(content.lstrip(" "))
        if "\t" in original[:indent] or indent % 2:
            raise ValueError(f"line {number}: use multiples of two spaces for indentation")
        stripped = content.strip()
        if stripped.startswith("-") or ":" not in stripped:
            raise ValueError(
                f"line {number}: fallback parser accepts nested mappings and inline lists only"
            )
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
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("top-level YAML value must be a mapping")
    return payload


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise TypeError(f"configuration key under {prefix or '<root>'} is not a string")
        child_prefix = f"{prefix}.{key}" if prefix else key
        result.update(flatten(child, child_prefix))
    return result


def shell_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        if any(isinstance(item, (dict, list)) for item in value):
            raise ValueError("nested lists/mappings are not valid parameter values")
        return ",".join(shell_value(item) for item in value)
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ValueError(f"unsupported YAML value: {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file")
    args = parser.parse_args()
    path = Path(args.config_file).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Configuration file not found: {path}")

    try:
        flattened = flatten(load_yaml(path))
        unknown = sorted(set(flattened) - set(KEY_TO_ENV))
        if unknown:
            raise ValueError("unknown configuration keys: " + ", ".join(unknown))
        for key in sorted(flattened):
            value = flattened[key]
            if value is None:
                continue
            env_name = KEY_TO_ENV[key]
            # Inline environment variables have the highest precedence.
            if env_name in os.environ:
                continue
            print(f"{env_name}={shlex.quote(shell_value(value))}")
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid pipeline YAML {path}: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .hashing import hash_object

DEFAULTS: dict[str, Any] = {
    "schema_version": "chatts-autoresearch-v1",
    "runtime": {
        "python_bin": "python3",
        "seed": 42,
        "gpu_ids": "0,1,2,3,4,5,6,7",
        "master_port": 19901,
        "dry_run": False,
    },
    "deepseek": {
        "enabled": True,
        "base_url": "http://localhost:30000/v1",
        "model": "/models",
        "api_key_env": "DEEPSEEK_API_KEY",
        "concurrency": 8,
        "timeout_seconds": 120,
        "max_retries": 2,
        "temperature": 0.0,
        "response_format": "json_schema",
        "prompt_version": "chatts-label-v2",
    },
    "labeling": {"max_samples": 0, "sources": [], "input_char_limit": 6000, "output_char_limit": 6000},
    "data": {
        "snapshot_name": "filtered-v1",
        "baseline_snapshot": "raw",
        "minimum_quality": 0.0,
        "missing_label_policy": "keep",
        "drop_exact_duplicates": True,
        "drop_cross_source_duplicates": True,
        "drop_near_duplicates": False,
        "near_duplicate_hamming": 3,
        "source_weights": {},
        "difficulty_weights": {"easy": 1.0, "medium": 1.0, "hard": 1.0},
        "aliases": {},
    },
    "training": {
        "stage1_learning_rate": 1e-5,
        "stage2_learning_rate": 1e-5,
        "stage1_timeseries_learning_rate": 1e-5,
        "stage2_timeseries_learning_rate": 1e-5,
        "stage1_datasets": "align_256,ift",
        "stage2_datasets": "sft,align_random,finiverse_time_mqa,finiverse_tsaqa",
        "stage1_mix_strategy": "interleave_over",
        "stage2_mix_strategy": "concat",
        "stage1_interleave_probs": "0.9,0.1",
        "stage2_interleave_probs": "",
        "stage1_epochs": 3,
        "stage2_epochs": 1,
        "stage2_warmup_ratio": 0.02,
        "stage2_scheduler": "cosine",
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "cutoff_len": 2048,
        "val_size": 0.05,
    },
    "search": {
        "proxy_trials": 6,
        "proxy_max_steps": 300,
        "full_finalists": 2,
        "proposal_mode": "deepseek",
        "learning_rates": [5e-6, 1e-5, 2e-5],
        "projector_lr_ratios": [0.5, 1.0, 2.0],
        "warmup_ratios": [0.01, 0.02, 0.05],
        "schedulers": ["cosine", "linear"],
        "epochs": [1, 2],
        "minimum_qualities": [0.0, 0.5, 0.7],
        "source_weight_range": [0.5, 2.0],
    },
    "evaluation": {
        "search_benchmarks": "tsrbench,timeseriesexam",
        "guard_benchmarks": "ts_haystack,tinybenchmarks",
        "final_benchmarks": "tsrbench,timeseriesexam,ts_haystack,tinybenchmarks",
        "search_split": "search-dev",
        "final_split": "final-test",
        "haystack_search_split": "validation",
        "haystack_final_split": "test",
        "tiny_partition_seed": 42,
        "search_max_samples": 0,
        "final_max_samples": 0,
        "split_sources": [],
    },
    "gates": {
        "tiny_average_max_drop": 0.01,
        "tiny_task_max_drop": 0.02,
        "haystack_iou_max_drop": 0.02,
        "coverage_max_drop": 0.01,
        "tiny_expected_tasks": [
            "tinyArc",
            "tinyHellaswag",
            "tinyMMLU",
            "tinyTruthfulQA",
            "tinyWinogrande",
        ],
    },
}


class ConfigError(ValueError):
    pass


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _resolve_relative_paths(data: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    path_keys = {
        "train_project",
        "train_script",
        "eval_project",
        "eval_script",
        "datav2_root",
        "chronos2_model",
        "base_model",
        "tsrbench_root",
        "timeseriesexam_root",
        "timeseriesexam_data_file",
        "ts_haystack_root",
        "tinybench_root",
    }
    for key in path_keys:
        raw = data.get("paths", {}).get(key)
        if not raw:
            continue
        path = Path(str(raw))
        data["paths"][key] = str(path if path.is_absolute() else (config_dir / path).resolve())
    for source in data.get("evaluation", {}).get("split_sources", []):
        if source.get("path"):
            path = Path(str(source["path"]))
            source["path"] = str(path if path.is_absolute() else (config_dir / path).resolve())
    return data


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    source_path: Path

    def get(self, dotted: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in dotted.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value is None or value == "":
            raise ConfigError(f"Missing required configuration: {dotted}")
        return value

    @property
    def output_root(self) -> Path:
        raw = Path(str(self.require("runtime.output_root")))
        return raw if raw.is_absolute() else (self.source_path.parent / raw).resolve()

    @property
    def sanitized(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.data)
        # Only the environment variable name is stored; the secret value is never read here.
        return payload

    @property
    def fingerprint(self) -> str:
        return hash_object(self.sanitized)


def validate_config(data: Mapping[str, Any]) -> None:
    if data.get("schema_version") != "chatts-autoresearch-v1":
        raise ConfigError("schema_version must be chatts-autoresearch-v1")
    required = (
        "train_project",
        "train_script",
        "eval_project",
        "eval_script",
        "datav2_root",
        "datav2_registry",
        "datav2_manifest",
        "chronos2_model",
        "base_model",
    )
    paths = data.get("paths", {})
    evaluation = data.get("evaluation", {})
    supported_benchmarks = {
        "tsrbench",
        "timeseriesexam",
        "ts_haystack",
        "tinybenchmarks",
    }
    selected_benchmarks: set[str] = set()
    for key in ("search_benchmarks", "guard_benchmarks", "final_benchmarks"):
        values = {
            item.strip()
            for item in str(evaluation.get(key, "")).split(",")
            if item.strip()
        }
        unknown = values - supported_benchmarks
        if unknown:
            raise ConfigError(f"evaluation.{key} has unsupported benchmarks: {sorted(unknown)}")
        selected_benchmarks.update(values)
    benchmark_paths = {
        "tsrbench": ("tsrbench_root",),
        "timeseriesexam": ("timeseriesexam_root", "timeseriesexam_data_file"),
        "ts_haystack": ("ts_haystack_root",),
        "tinybenchmarks": ("tinybench_root",),
    }
    required = (*required, *(key for suite in selected_benchmarks for key in benchmark_paths[suite]))
    missing = [f"paths.{key}" for key in required if not paths.get(key)]
    if missing:
        raise ConfigError(f"Missing required configuration: {', '.join(missing)}")
    if int(data["runtime"].get("seed", -1)) != 42:
        raise ConfigError("V1 fixes runtime.seed to 42")
    try:
        gpu_ids = [
            int(item.strip())
            for item in str(data["runtime"].get("gpu_ids", "")).split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise ConfigError("runtime.gpu_ids must be comma-separated integers") from exc
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8 or any(item < 0 for item in gpu_ids):
        raise ConfigError("V1 requires exactly eight distinct non-negative runtime.gpu_ids")
    try:
        master_port = int(data["runtime"].get("master_port", 0))
    except (TypeError, ValueError) as exc:
        raise ConfigError("runtime.master_port must be an integer") from exc
    if not 1 <= master_port <= 65535:
        raise ConfigError("runtime.master_port must be in [1, 65535]")
    if evaluation.get("search_split") != "search-dev" or evaluation.get("final_split") != "final-test":
        raise ConfigError("V1 fixes evaluation splits to search-dev and final-test")
    if evaluation.get("haystack_search_split") not in {"train", "validation"}:
        raise ConfigError("evaluation.haystack_search_split must be train or validation")
    if evaluation.get("haystack_final_split") != "test":
        raise ConfigError("evaluation.haystack_final_split must be test")
    if int(evaluation.get("tiny_partition_seed", -1)) != 42:
        raise ConfigError("V1 fixes evaluation.tiny_partition_seed to 42")
    if data.get("deepseek", {}).get("response_format") not in {
        "json_schema",
        "json_object",
    }:
        raise ConfigError("deepseek.response_format must be json_schema or json_object")
    search = data.get("search", {})
    proxy_trials = int(search.get("proxy_trials", 0))
    full_finalists = int(search.get("full_finalists", 0))
    if proxy_trials < 1:
        raise ConfigError("search.proxy_trials must be positive")
    if int(search.get("proxy_max_steps", 0)) <= 0:
        raise ConfigError("search.proxy_max_steps must be positive")
    if not 1 <= full_finalists <= proxy_trials:
        raise ConfigError(
            "search.full_finalists must be between 1 and search.proxy_trials"
        )
    if search.get("proposal_mode") not in {"deepseek", "deterministic"}:
        raise ConfigError("search.proposal_mode must be deepseek or deterministic")
    if data["data"].get("missing_label_policy") not in {"keep", "drop", "error"}:
        raise ConfigError("data.missing_label_policy must be keep, drop, or error")
    baseline_snapshot = data["data"].get("baseline_snapshot")
    snapshot_name = data["data"].get("snapshot_name")
    if baseline_snapshot not in {"raw", "prepared", snapshot_name}:
        raise ConfigError(
            "data.baseline_snapshot must be raw, prepared, or equal data.snapshot_name"
        )
    for key in (
        "learning_rates",
        "projector_lr_ratios",
        "warmup_ratios",
        "schedulers",
        "epochs",
        "minimum_qualities",
    ):
        values = search.get(key)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"search.{key} must be a non-empty list")
    try:
        learning_rates = [float(value) for value in search["learning_rates"]]
        projector_ratios = [float(value) for value in search["projector_lr_ratios"]]
        warmup_ratios = [float(value) for value in search["warmup_ratios"]]
        minimum_qualities = [float(value) for value in search["minimum_qualities"]]
    except (TypeError, ValueError) as exc:
        raise ConfigError("numeric search grids must contain only numbers") from exc
    if any(value <= 0 for value in learning_rates + projector_ratios):
        raise ConfigError("learning-rate grids and projector ratios must be positive")
    if any(not 0 <= value < 1 for value in warmup_ratios):
        raise ConfigError("search.warmup_ratios values must be in [0, 1)")
    if any(not 0 <= value <= 1 for value in minimum_qualities):
        raise ConfigError("search.minimum_qualities values must be in [0, 1]")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in search["epochs"]
    ):
        raise ConfigError("search.epochs values must be positive integers")
    if any(value not in {"cosine", "linear"} for value in search["schedulers"]):
        raise ConfigError("search.schedulers values must be cosine or linear")
    source_range = search.get("source_weight_range")
    if not isinstance(source_range, list) or len(source_range) != 2:
        raise ConfigError("search.source_weight_range must contain exactly two values")
    try:
        low, high = [float(item) for item in source_range]
    except (TypeError, ValueError) as exc:
        raise ConfigError("search.source_weight_range must be numeric") from exc
    if not (0 < low <= high):
        raise ConfigError("search.source_weight_range must be positive and ordered")
    for key in ("search_max_samples", "final_max_samples"):
        try:
            max_samples = int(evaluation.get(key, 0))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"evaluation.{key} must be an integer") from exc
        if max_samples < 0:
            raise ConfigError(f"evaluation.{key} must be non-negative")
    if int(evaluation.get("final_max_samples", 0)) != 0:
        raise ConfigError("V1 requires evaluation.final_max_samples=0 for full final-test")
    training = data.get("training", {})
    positive = (
        "stage1_learning_rate",
        "stage2_learning_rate",
        "stage1_timeseries_learning_rate",
        "stage2_timeseries_learning_rate",
        "stage1_epochs",
        "stage2_epochs",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "cutoff_len",
    )
    for key in positive:
        try:
            value = float(training[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"training.{key} must be numeric") from exc
        if value <= 0:
            raise ConfigError(f"training.{key} must be positive")
    for key in ("stage1_datasets", "stage2_datasets", "stage1_mix_strategy", "stage2_mix_strategy"):
        if not str(training.get(key, "")).strip():
            raise ConfigError(f"training.{key} must not be empty")
    warmup = float(training.get("stage2_warmup_ratio", -1))
    val_size = float(training.get("val_size", -1))
    if not 0 <= warmup < 1 or not 0 <= val_size < 1:
        raise ConfigError("training warmup_ratio and val_size must be in [0, 1)")
    expected_tiny = data.get("gates", {}).get("tiny_expected_tasks")
    if (
        not isinstance(expected_tiny, list)
        or not expected_tiny
        or any(not isinstance(item, str) or not item.strip() for item in expected_tiny)
        or len(set(expected_tiny)) != len(expected_tiny)
    ):
        raise ConfigError("gates.tiny_expected_tasks must be a non-empty list of unique names")


def load_config(path: str | Path) -> Config:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"Configuration file not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise ConfigError("Top-level YAML value must be a mapping")
    data = _resolve_relative_paths(_expand(_merge(DEFAULTS, loaded)), source.parent)
    validate_config(data)
    return Config(data=data, source_path=source)


def dump_resolved(config: Config, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(config.sanitized, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(destination)

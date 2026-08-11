from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .models import StudioError

BENCHMARKS = ("tsrbench", "tinybenchmarks", "ts_haystack", "timeseriesexam")
SCHEDULERS = ("cosine", "linear", "constant", "constant_with_warmup")
MIX_STRATEGIES = ("concat", "interleave_under", "interleave_over")

STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "stage1": {
        "learning_rate": "1e-5",
        "timeseries_learning_rate": "1e-5",
        "mix_strategy": "concat",
        "interleave_probs": "",
        "num_train_epochs": 3,
        "max_steps": 0,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.02,
        "logging_steps": 1,
        "save_steps": 200,
        "eval_steps": 200,
        "val_size": 0.05,
        "per_device_eval_batch_size": 2,
        "cutoff_len": 2048,
        "preprocessing_num_workers": 96,
    },
    "stage2": {
        "learning_rate": "1e-5",
        "timeseries_learning_rate": "1e-5",
        "mix_strategy": "concat",
        "interleave_probs": "",
        "num_train_epochs": 1,
        "max_steps": 0,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.02,
        "logging_steps": 1,
        "save_steps": 100,
        "eval_steps": 100,
        "val_size": 0.05,
        "per_device_eval_batch_size": 4,
        "cutoff_len": 2048,
        "preprocessing_num_workers": 96,
    },
}

EVALUATION_DEFAULTS: dict[str, Any] = {
    "benchmarks": list(BENCHMARKS),
    "max_samples": 0,
    "offline": True,
    "force_eval": False,
    "haystack_split": "test",
    "tiny_data_partition": "all",
    "tiny_partition_seed": 42,
    "tsr_prompt_mode": "answer_only",
    "tsr_max_model_len": 12288,
    "tsr_max_new_tokens": 8,
    "tsr_batch_size": 16,
    "tsr_request_chunk_size": 128,
    "tiny_max_model_len": 6000,
    "tiny_request_chunk_size": 16,
    "tiny_gpu_memory_utilization": 0.70,
    "haystack_max_model_len": 40960,
    "haystack_max_new_tokens": 500,
    "haystack_batch_size": 1,
    "haystack_request_chunk_size": 8,
    "exam_max_model_len": 8192,
    "exam_max_new_tokens": 1024,
    "exam_batch_size": 8,
    "exam_request_chunk_size": 64,
}

_STAGE_KEYS = frozenset(STAGE_DEFAULTS["stage1"])
_TRAINING_KEYS = frozenset(
    {
        "profile",
        "seed",
        "force_train",
        "keep_stage1",
        "deepspeed_include",
        "master_port",
        "stage1",
        "stage2",
        # Display-only values sent by older/newer frontends. They may not
        # override the server's fixed integration paths.
        "project_root",
        "base_model_path",
        "output_root",
    }
)
_EVALUATION_KEYS = frozenset(
    set(EVALUATION_DEFAULTS)
    | {
        "protocol_hash",
        "project_root",
        "model_path",
        "output_root",
        "tsrbench_root",
        "tinybench_dataset_root",
        "ts_haystack_root",
        "timeseriesexam_root",
        "timeseriesexam_data_file",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise StudioError(f"{field} must be an object")
    return value


def _reject_unknown(value: dict[str, Any], allowed: frozenset[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise StudioError(f"Unknown {field} fields: {unknown}")


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1, "0", "1"):
        return bool(int(value))
    raise StudioError(f"{field} must be true/false or 0/1")


def _as_int(value: Any, field: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise StudioError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StudioError(f"{field} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise StudioError(f"{field} must be an integer")
    if parsed < minimum or (maximum is not None and parsed > maximum):
        bound = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise StudioError(f"{field} must be {bound}")
    return parsed


def _as_float(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive_minimum: bool = True,
    inclusive_maximum: bool = True,
) -> float:
    if isinstance(value, bool):
        raise StudioError(f"{field} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise StudioError(f"{field} must be a number") from exc
    if not (parsed == parsed and abs(parsed) != float("inf")):
        raise StudioError(f"{field} must be finite")
    if minimum is not None and (
        parsed < minimum or (parsed == minimum and not inclusive_minimum)
    ):
        operator = ">=" if inclusive_minimum else ">"
        raise StudioError(f"{field} must be {operator} {minimum}")
    if maximum is not None and (
        parsed > maximum or (parsed == maximum and not inclusive_maximum)
    ):
        operator = "<=" if inclusive_maximum else "<"
        raise StudioError(f"{field} must be {operator} {maximum}")
    return parsed


def _as_lr(value: Any, field: str) -> str:
    parsed = _as_float(value, field, minimum=0, inclusive_minimum=False)
    # Retain a compact scientific representation in the resolved YAML.
    return f"{parsed:.12g}"


def _fixed_value(
    request: dict[str, Any], request_key: str, integration: dict[str, Any], integration_key: str
) -> Any:
    expected = integration.get(integration_key)
    supplied = request.get(request_key)
    if (
        supplied not in (None, "")
        and expected not in (None, "")
        and str(supplied) != str(expected)
    ):
        raise StudioError(
            f"{request_key} is fixed by the server configuration and cannot be overridden"
        )
    if expected in (None, ""):
        raise StudioError(f"Missing server integration setting: {integration_key}")
    return expected


def _version_record_fields(record: dict[str, Any]) -> tuple[str, str, dict[str, list[str]]]:
    snapshot_dir = (
        record.get("snapshot_path")
        or record.get("snapshot_dir")
        or record.get("output_dir")
        or record.get("path")
    )
    snapshot_hash = record.get("dataset_snapshot_hash") or record.get("content_hash")
    dataset_names = record.get("dataset_names")
    if not isinstance(snapshot_dir, str) or not snapshot_dir:
        raise StudioError("Version record has no snapshot_dir")
    if not isinstance(snapshot_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot_hash):
        raise StudioError("Version record has no valid dataset_snapshot_hash")
    if not isinstance(dataset_names, dict):
        composition = record.get("composition")
        if isinstance(composition, dict):
            dataset_names = {
                stage: details.get("dataset_names") if isinstance(details, dict) else None
                for stage, details in composition.items()
            }
    if not isinstance(dataset_names, dict):
        raise StudioError("Version record has no dataset_names")
    parsed_names: dict[str, list[str]] = {}
    for stage in ("stage1", "stage2"):
        names = dataset_names.get(stage)
        if not isinstance(names, list) or not names or any(not isinstance(item, str) for item in names):
            raise StudioError(f"Version record has no non-empty {stage} dataset list")
        parsed_names[stage] = names
    return snapshot_dir, snapshot_hash, parsed_names


def _normalise_probs(value: Any, *, count: int, field: str, strategy: str) -> str:
    if strategy == "concat":
        if value in (None, "", []):
            return ""
        raise StudioError(f"{field} must be empty when mix_strategy=concat")
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        raw_items = value
    else:
        raise StudioError(f"{field} must be a comma-separated string or list")
    if len(raw_items) != count:
        raise StudioError(f"{field} must contain {count} probabilities")
    probabilities = [
        _as_float(item, field, minimum=0, maximum=1) for item in raw_items
    ]
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise StudioError(f"{field} probabilities must sum to 1")
    return ",".join(f"{item:.12g}" for item in probabilities)


def _resolve_stage(
    stage: str, request: dict[str, Any], dataset_names: list[str]
) -> dict[str, Any]:
    _reject_unknown(request, _STAGE_KEYS, f"training.{stage}")
    values = {**STAGE_DEFAULTS[stage], **request}
    strategy = values["mix_strategy"]
    if strategy not in MIX_STRATEGIES:
        raise StudioError(f"training.{stage}.mix_strategy must be one of {MIX_STRATEGIES}")
    resolved = {
        "learning_rate": _as_lr(values["learning_rate"], f"training.{stage}.learning_rate"),
        "timeseries_learning_rate": _as_lr(
            values["timeseries_learning_rate"],
            f"training.{stage}.timeseries_learning_rate",
        ),
        "datasets": ",".join(dataset_names),
        "mix_strategy": strategy,
        "interleave_probs": _normalise_probs(
            values["interleave_probs"],
            count=len(dataset_names),
            field=f"training.{stage}.interleave_probs",
            strategy=strategy,
        ),
        "num_train_epochs": _as_float(
            values["num_train_epochs"],
            f"training.{stage}.num_train_epochs",
            minimum=0,
            inclusive_minimum=False,
        ),
        "max_steps": _as_int(values["max_steps"], f"training.{stage}.max_steps"),
        "per_device_train_batch_size": _as_int(
            values["per_device_train_batch_size"],
            f"training.{stage}.per_device_train_batch_size",
            minimum=1,
        ),
        "gradient_accumulation_steps": _as_int(
            values["gradient_accumulation_steps"],
            f"training.{stage}.gradient_accumulation_steps",
            minimum=1,
        ),
        "lr_scheduler_type": values["lr_scheduler_type"],
        "warmup_ratio": _as_float(
            values["warmup_ratio"],
            f"training.{stage}.warmup_ratio",
            minimum=0,
            maximum=1,
            inclusive_maximum=False,
        ),
        "logging_steps": _as_int(
            values["logging_steps"], f"training.{stage}.logging_steps", minimum=1
        ),
        "save_steps": _as_int(
            values["save_steps"], f"training.{stage}.save_steps", minimum=1
        ),
        "eval_steps": _as_int(
            values["eval_steps"], f"training.{stage}.eval_steps", minimum=1
        ),
        "val_size": _as_float(
            values["val_size"],
            f"training.{stage}.val_size",
            minimum=0,
            maximum=1,
            inclusive_minimum=False,
            inclusive_maximum=False,
        ),
        "per_device_eval_batch_size": _as_int(
            values["per_device_eval_batch_size"],
            f"training.{stage}.per_device_eval_batch_size",
            minimum=1,
        ),
        "cutoff_len": _as_int(
            values["cutoff_len"], f"training.{stage}.cutoff_len", minimum=1
        ),
        "preprocessing_num_workers": _as_int(
            values["preprocessing_num_workers"],
            f"training.{stage}.preprocessing_num_workers",
            minimum=1,
        ),
    }
    if resolved["lr_scheduler_type"] not in SCHEDULERS:
        raise StudioError(
            f"training.{stage}.lr_scheduler_type must be one of {SCHEDULERS}"
        )
    if resolved["save_steps"] % resolved["eval_steps"]:
        raise StudioError(
            f"training.{stage}.save_steps must be a multiple of eval_steps"
        )
    return resolved


def _benchmarks(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = value
    else:
        raise StudioError("evaluation.benchmarks must be a list or comma-separated string")
    if not values or len(values) != len(set(values)):
        raise StudioError("evaluation.benchmarks must be non-empty and contain no duplicates")
    unknown = sorted(set(values) - set(BENCHMARKS))
    if unknown:
        raise StudioError(f"Unknown benchmarks: {unknown}")
    return values


def _versioned_name(base: str, version: str) -> str:
    clean = re.sub(r"(?:[-_]?data-?v[0-9]+)$", "", base.rstrip("/"), flags=re.IGNORECASE)
    return f"{clean}-{version}"


def _resolve_evaluation(request: dict[str, Any], integration: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown(request, _EVALUATION_KEYS, "evaluation")
    values = {**EVALUATION_DEFAULTS, **request}
    roots = {
        "project_root": _fixed_value(request, "project_root", integration, "eval_project_root"),
        "script": integration.get("evaluation_script"),
        "chronos2_model_path": integration.get("eval_chronos2_model_path"),
        "tsrbench_root": _fixed_value(request, "tsrbench_root", integration, "tsrbench_root"),
        "tinybench_dataset_root": _fixed_value(
            request, "tinybench_dataset_root", integration, "tinybench_dataset_root"
        ),
        "ts_haystack_root": _fixed_value(
            request, "ts_haystack_root", integration, "ts_haystack_root"
        ),
        "timeseriesexam_root": _fixed_value(
            request, "timeseriesexam_root", integration, "timeseriesexam_root"
        ),
        "timeseriesexam_data_file": _fixed_value(
            request,
            "timeseriesexam_data_file",
            integration,
            "timeseriesexam_data_file",
        ),
    }
    for key in ("script", "chronos2_model_path"):
        if not isinstance(roots[key], str) or not roots[key]:
            raise StudioError(f"Missing server integration setting: evaluation_{key}")

    benchmarks = _benchmarks(values["benchmarks"])
    tiny_gpu_memory_utilization = _as_float(
        values["tiny_gpu_memory_utilization"],
        "evaluation.tiny_gpu_memory_utilization",
        minimum=0,
        maximum=1,
        inclusive_minimum=False,
        inclusive_maximum=False,
    )
    result = {
        **roots,
        "benchmarks": ",".join(benchmarks),
        "haystack_split": values["haystack_split"],
        "tiny_data_partition": values["tiny_data_partition"],
        "tiny_partition_seed": _as_int(
            values["tiny_partition_seed"], "evaluation.tiny_partition_seed"
        ),
        "tsr_prompt_mode": values["tsr_prompt_mode"],
        "tsr_max_model_len": _as_int(
            values["tsr_max_model_len"], "evaluation.tsr_max_model_len", minimum=1
        ),
        "tsr_max_new_tokens": _as_int(
            values["tsr_max_new_tokens"], "evaluation.tsr_max_new_tokens", minimum=1
        ),
        "tsr_batch_size": _as_int(
            values["tsr_batch_size"], "evaluation.tsr_batch_size", minimum=1
        ),
        "tsr_request_chunk_size": _as_int(
            values["tsr_request_chunk_size"],
            "evaluation.tsr_request_chunk_size",
            minimum=1,
        ),
        "tiny_max_model_len": _as_int(
            values["tiny_max_model_len"], "evaluation.tiny_max_model_len", minimum=1
        ),
        "tiny_request_chunk_size": _as_int(
            values["tiny_request_chunk_size"],
            "evaluation.tiny_request_chunk_size",
            minimum=1,
        ),
        "tiny_gpu_memory_utilization": f"{tiny_gpu_memory_utilization:.2f}",
        "haystack_max_model_len": _as_int(
            values["haystack_max_model_len"],
            "evaluation.haystack_max_model_len",
            minimum=1,
        ),
        "haystack_max_new_tokens": _as_int(
            values["haystack_max_new_tokens"],
            "evaluation.haystack_max_new_tokens",
            minimum=1,
        ),
        "haystack_batch_size": _as_int(
            values["haystack_batch_size"], "evaluation.haystack_batch_size", minimum=1
        ),
        "haystack_request_chunk_size": _as_int(
            values["haystack_request_chunk_size"],
            "evaluation.haystack_request_chunk_size",
            minimum=1,
        ),
        "exam_max_model_len": _as_int(
            values["exam_max_model_len"], "evaluation.exam_max_model_len", minimum=1
        ),
        "exam_max_new_tokens": _as_int(
            values["exam_max_new_tokens"], "evaluation.exam_max_new_tokens", minimum=1
        ),
        "exam_batch_size": _as_int(
            values["exam_batch_size"], "evaluation.exam_batch_size", minimum=1
        ),
        "exam_request_chunk_size": _as_int(
            values["exam_request_chunk_size"],
            "evaluation.exam_request_chunk_size",
            minimum=1,
        ),
    }
    if result["haystack_split"] not in ("train", "validation", "test"):
        raise StudioError("evaluation.haystack_split must be train, validation, or test")
    if result["tiny_data_partition"] not in ("all", "search-dev", "final-test"):
        raise StudioError(
            "evaluation.tiny_data_partition must be all, search-dev, or final-test"
        )
    if result["tsr_prompt_mode"] not in ("answer_only", "official"):
        raise StudioError("evaluation.tsr_prompt_mode must be answer_only or official")
    for prefix in ("tsr", "haystack", "exam"):
        if result[f"{prefix}_max_model_len"] <= result[f"{prefix}_max_new_tokens"]:
            raise StudioError(
                f"evaluation.{prefix}_max_model_len must be greater than max_new_tokens"
            )
    protocol_hash = values.get("protocol_hash")
    if protocol_hash not in (None, ""):
        if not isinstance(protocol_hash, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", protocol_hash
        ):
            raise StudioError("evaluation.protocol_hash must be a SHA256 or empty")
        result["protocol_hash"] = protocol_hash.lower()
    return result


def public_pipeline_defaults(integration: dict[str, Any]) -> dict[str, Any]:
    """Return safe UI defaults without exposing host-side executable paths."""
    script = integration.get("pipeline_script")
    return {
        "training": {
            "profile": "chronos2-full",
            "base_model_path": integration.get("base_model_path"),
            "output_root": integration.get("model_output_base"),
            "seed": 42,
            "force_train": False,
            "keep_stage1": False,
            "deepspeed_include": integration.get(
                "deepspeed_include", "localhost:0,1,2,3,4,5,6,7"
            ),
            "master_port": integration.get("master_port", 19901),
            "stage1": dict(STAGE_DEFAULTS["stage1"]),
            "stage2": dict(STAGE_DEFAULTS["stage2"]),
        },
        "evaluation": {
            **EVALUATION_DEFAULTS,
            "output_root": integration.get("evaluation_output_base"),
            "tsrbench_root": integration.get("tsrbench_root"),
            "tinybench_dataset_root": integration.get("tinybench_dataset_root"),
            "ts_haystack_root": integration.get("ts_haystack_root"),
            "timeseriesexam_root": integration.get("timeseriesexam_root"),
            "timeseriesexam_data_file": integration.get("timeseriesexam_data_file"),
        },
        "integration": {
            "enabled": bool(script and Path(str(script)).expanduser().is_file()),
            "training_root": integration.get("training_root"),
            "evaluation_root": integration.get("evaluation_root"),
            "training_container": integration.get("training_container", "chatts"),
            "evaluation_container": integration.get("evaluation_container", "ragas"),
        },
    }


def resolve_pipeline_request(
    payload: dict[str, Any], version_record: dict[str, Any], integration: dict[str, Any]
) -> dict[str, Any]:
    allowed_root = {"mode", "version", "training", "evaluation"}
    unknown_root = sorted(set(payload) - allowed_root)
    if unknown_root:
        raise StudioError(f"Unknown pipeline fields: {unknown_root}")
    if payload.get("mode", "train_eval") != "train_eval":
        raise StudioError("The safe dashboard launcher currently supports mode=train_eval only")

    version = version_record.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"datav[1-9][0-9]*", version):
        raise StudioError("Version record has no canonical datavN version")
    if payload.get("version") not in (None, version):
        raise StudioError("Requested version does not match the resolved version record")
    snapshot_dir, snapshot_hash, dataset_names = _version_record_fields(version_record)

    training_request = _mapping(payload.get("training"), "training")
    evaluation_request = _mapping(payload.get("evaluation"), "evaluation")
    _reject_unknown(training_request, _TRAINING_KEYS, "training")
    if training_request.get("profile", "chronos2-full") != "chronos2-full":
        raise StudioError("Only the fixed chronos2-full training profile is supported")

    seed = _as_int(training_request.get("seed", 42), "training.seed")
    master_port = _as_int(
        training_request.get("master_port", integration.get("master_port", 19901)),
        "training.master_port",
        minimum=1,
        maximum=65535,
    )
    deepspeed_include = training_request.get(
        "deepspeed_include",
        integration.get("deepspeed_include", "localhost:0,1,2,3,4,5,6,7"),
    )
    if not isinstance(deepspeed_include, str) or not re.fullmatch(
        r"localhost:[0-9]+(?:,[0-9]+)*", deepspeed_include
    ):
        raise StudioError("training.deepspeed_include must look like localhost:0,1,...")

    train_project_root = _fixed_value(
        training_request, "project_root", integration, "train_project_root"
    )
    base_model_path = _fixed_value(
        training_request, "base_model_path", integration, "base_model_path"
    )
    model_output_base = integration.get("model_output_base")
    if not isinstance(model_output_base, str) or not model_output_base:
        raise StudioError("Missing server integration setting: model_output_base")
    train_output_root = _versioned_name(model_output_base, version)
    supplied_output = training_request.get("output_root")
    if supplied_output not in (None, "", train_output_root):
        raise StudioError("training.output_root is derived from data version and cannot be changed")
    final_model_path = f"{train_output_root.rstrip('/')}/best_seed{seed}"

    train_script = integration.get("training_script")
    train_chronos = integration.get("train_chronos2_model_path")
    if not isinstance(train_script, str) or not train_script:
        raise StudioError("Missing server integration setting: training_script")
    if not isinstance(train_chronos, str) or not train_chronos:
        raise StudioError("Missing server integration setting: train_chronos2_model_path")

    stage1 = _resolve_stage(
        "stage1", _mapping(training_request.get("stage1"), "training.stage1"), dataset_names["stage1"]
    )
    stage2 = _resolve_stage(
        "stage2", _mapping(training_request.get("stage2"), "training.stage2"), dataset_names["stage2"]
    )
    evaluation = _resolve_evaluation(evaluation_request, integration)

    model_name_base = integration.get("model_name_base", "chatts-msxf-8B")
    if not isinstance(model_name_base, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", model_name_base):
        raise StudioError("model_name_base must be a safe slug")
    model_name = f"{_versioned_name(model_name_base, version)}-seed{seed}"
    eval_output_base = integration.get("evaluation_output_base")
    if not isinstance(eval_output_base, str) or not eval_output_base:
        raise StudioError("Missing server integration setting: evaluation_output_base")
    eval_output_root = f"{eval_output_base.rstrip('/')}/{model_name}"
    supplied_eval_output = evaluation_request.get("output_root")
    if supplied_eval_output not in (None, "", eval_output_root):
        raise StudioError("evaluation.output_root is derived from data version and cannot be changed")
    supplied_model = evaluation_request.get("model_path")
    if supplied_model not in (None, "", final_model_path):
        raise StudioError("evaluation.model_path is derived from training output and cannot be changed")

    run_id = f"chronos2-{version}-seed{seed}-full"
    config = {
        "pipeline": {
            "seed": seed,
            "data_version": version,
            "dataset_snapshot_hash": snapshot_hash,
            "force_train": _as_bool(
                training_request.get("force_train", False), "training.force_train"
            ),
            "force_eval": _as_bool(
                evaluation_request.get("force_eval", False), "evaluation.force_eval"
            ),
            "preflight_only": False,
            "max_samples": _as_int(
                evaluation_request.get("max_samples", 0), "evaluation.max_samples"
            ),
            "offline": _as_bool(
                evaluation_request.get("offline", True), "evaluation.offline"
            ),
        },
        "containers": {
            "training": integration.get("training_container", "chatts"),
            "evaluation": integration.get("evaluation_container", "ragas"),
        },
        "training": {
            "project_root": train_project_root,
            "script": train_script,
            "base_model_path": base_model_path,
            "output_root": train_output_root,
            "final_model_path": final_model_path,
            "chronos2_model_path": train_chronos,
            "dataset_dir": snapshot_dir,
            "keep_stage1": _as_bool(
                training_request.get("keep_stage1", False), "training.keep_stage1"
            ),
            "deepspeed_include": deepspeed_include,
            "master_port": master_port,
            "stage1": stage1,
            "stage2": stage2,
        },
        "evaluation": {
            **evaluation,
            "model_name": model_name,
            "output_root": eval_output_root,
            "run_id": run_id,
        },
    }
    result = {
        "schema_version": "chatts-dataset-studio-pipeline-v1",
        "version": version,
        "dataset_snapshot_hash": snapshot_hash,
        "config": config,
        "derived": {
            "train_output_root": train_output_root,
            "final_model_path": final_model_path,
            "model_name": model_name,
            "evaluation_output_root": eval_output_root,
            "run_id": run_id,
        },
    }
    result["config_hash"] = _hash(config)
    return result


class PipelineJobs:
    """Persistent, local-only subprocess launcher for the fixed host pipeline."""

    def __init__(self, state_root: str | Path, pipeline_script: str | Path | None):
        self.state_root = Path(state_root).expanduser().resolve()
        self.jobs_root = self.state_root / "jobs"
        self.config_root = self.state_root / "configs"
        self.logs_root = self.state_root / "logs"
        self.status_root = self.state_root / "worker-status"
        self.run_lock_path = self.state_root / ".pipeline-run.lock"
        self.pipeline_script = (
            Path(pipeline_script).expanduser().resolve() if pipeline_script else None
        )
        self.lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self._recover()

    def _path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise StudioError("Invalid job id")
        return self.jobs_root / f"{job_id}.json"

    def _write(self, job: dict[str, Any]) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        path = self._path(job["job_id"])
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        temporary.write_text(
            json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StudioError(f"Cannot read pipeline job {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise StudioError(f"Pipeline job is not an object: {path}")
        return value

    def list(self) -> list[dict[str, Any]]:
        if not self.jobs_root.is_dir():
            return []
        jobs = [self._read(path) for path in self.jobs_root.glob("*.json")]
        return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)

    def get(self, job_id: str, *, include_log: bool = True) -> dict[str, Any]:
        path = self._path(job_id)
        if not path.is_file():
            raise StudioError(f"Unknown pipeline job: {job_id}")
        job = self._read(path)
        if include_log:
            log_path = Path(job["log_path"])
            if log_path.is_file():
                with log_path.open("rb") as stream:
                    stream.seek(max(0, log_path.stat().st_size - 100_000))
                    job["log_tail"] = stream.read().decode("utf-8", errors="replace")
            else:
                job["log_tail"] = ""
        return job

    def _status_path(self, job_id: str) -> Path:
        return self.status_root / f"{job_id}.json"

    @staticmethod
    def _pid_alive(pid: Any) -> bool:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _read_worker_status(self, job_id: str) -> dict[str, Any] | None:
        path = self._status_path(job_id)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or value.get("job_id") != job_id:
            return None
        return value

    def _apply_worker_status(self, job_id: str, status: dict[str, Any]) -> None:
        with self.lock:
            job = self.get(job_id, include_log=False)
            for key in (
                "status",
                "pid",
                "started_at",
                "finished_at",
                "duration_seconds",
                "exit_code",
                "error",
            ):
                if key in status:
                    job[key] = status[key]
            self._write(job)
            self.processes.pop(job_id, None)

    def _mark_interrupted(self, job_id: str) -> None:
        with self.lock:
            job = self.get(job_id, include_log=False)
            if job.get("status") not in {"queued", "running"}:
                return
            job.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "exit_code": 125,
                    "error": "Pipeline worker exited without a final status record",
                }
            )
            self._write(job)
            self.processes.pop(job_id, None)

    def _watch(self, job_id: str, process: subprocess.Popen[bytes] | None = None) -> None:
        if process is not None:
            process.wait()
        else:
            while True:
                status = self._read_worker_status(job_id)
                if status and status.get("status") in {"completed", "failed"}:
                    break
                job = self.get(job_id, include_log=False)
                pid = (status or {}).get("pid", job.get("pid"))
                if not self._pid_alive(pid):
                    break
                time.sleep(0.25)

        # The worker atomically writes its status immediately before exiting. A
        # tiny retry also covers slow network filesystems used by the HPC setup.
        for _ in range(20):
            status = self._read_worker_status(job_id)
            if status and status.get("status") in {"completed", "failed"}:
                self._apply_worker_status(job_id, status)
                return
            time.sleep(0.05)
        self._mark_interrupted(job_id)

    def _recover(self) -> None:
        for job in self.list():
            if job.get("status") not in {"queued", "running"}:
                continue
            job_id = job.get("job_id")
            if not isinstance(job_id, str):
                continue
            status = self._read_worker_status(job_id)
            if status and status.get("status") in {"completed", "failed"}:
                self._apply_worker_status(job_id, status)
                continue
            pid = (status or {}).get("pid", job.get("pid"))
            if self._pid_alive(pid):
                threading.Thread(
                    target=self._watch,
                    args=(job_id,),
                    name=f"pipeline-recover-{job_id[:8]}",
                    daemon=True,
                ).start()
            else:
                self._mark_interrupted(job_id)

    def start(self, resolved: dict[str, Any], *, preflight: bool = False) -> dict[str, Any]:
        if self.pipeline_script is None or not self.pipeline_script.is_file():
            raise StudioError(
                "One-click pipeline is disabled or pipeline_script does not exist on the host"
            )
        with self.lock:
            active = [
                item
                for item in self.list()
                if item.get("status") in {"queued", "running"}
            ]
            if active:
                raise StudioError(f"A pipeline job is already active: {active[0]['job_id']}")
            job_id = uuid.uuid4().hex
            self.config_root.mkdir(parents=True, exist_ok=True)
            self.logs_root.mkdir(parents=True, exist_ok=True)
            config_path = self.config_root / f"{job_id}.yaml"
            log_path = self.logs_root / f"{job_id}.log"
            config = json.loads(json.dumps(resolved["config"]))
            config["pipeline"]["preflight_only"] = preflight
            config["pipeline"]["trial_id"] = job_id
            effective_config_hash = _hash(config)
            config["pipeline"]["trial_config_hash"] = effective_config_hash
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            job = {
                "schema_version": "chatts-dataset-studio-job-v1",
                "job_id": job_id,
                "kind": "preflight" if preflight else "train_eval",
                "status": "queued",
                "created_at": _utc_now(),
                "version": resolved["version"],
                "dataset_snapshot_hash": resolved["dataset_snapshot_hash"],
                "config_hash": effective_config_hash,
                "config_path": str(config_path),
                "log_path": str(log_path),
                "derived": resolved["derived"],
            }
            self._write(job)
            status_path = self._status_path(job_id)
            worker_command = [
                sys.executable,
                str(Path(__file__).with_name("pipeline_worker.py")),
                "--job-id",
                job_id,
                "--lock-path",
                str(self.run_lock_path),
                "--status-path",
                str(status_path),
                "--script",
                str(self.pipeline_script),
                "--config",
                str(config_path),
                "--cwd",
                str(self.pipeline_script.parent.parent),
            ]
            try:
                with log_path.open("wb") as log_stream:
                    process = subprocess.Popen(  # noqa: S603 - configured executable only.
                        worker_command,
                        cwd=self.pipeline_script.parent.parent,
                        env=dict(os.environ),
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            except OSError as exc:
                job.update(
                    {
                        "status": "failed",
                        "finished_at": _utc_now(),
                        "error": f"Could not start fixed pipeline script: {exc}",
                    }
                )
                self._write(job)
                raise StudioError(job["error"]) from exc
            job.update({"status": "running", "started_at": _utc_now(), "pid": process.pid})
            self._write(job)
            self.processes[job_id] = process
            threading.Thread(
                target=self._watch,
                args=(job_id, process),
                name=f"pipeline-{job_id[:8]}",
                daemon=True,
            ).start()
            return job

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .models import StudioError
from .slurm import resolve_slurm_launcher, slurm_readiness

BENCHMARKS = ("tsrbench", "tinybenchmarks", "ts_haystack", "timeseriesexam")
MAX_EVALUATION_MODELS = 64
EVALUATION_SBATCH_MARKER = "# CHATTS_STUDIO_EVALUATION_SBATCH_API=1"
DEFAULT_EVALUATION_SBATCH_NAME = "run_chatts_studio_evaluation.sbatch"
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
_EXECUTION_KEYS = frozenset({"backend", "sbatch_path"})
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


def _user_posix_absolute_path(value: Any, field: str) -> str:
    """Validate a container/shared-filesystem path supplied by the dashboard."""

    if not isinstance(value, str) or not value:
        raise StudioError(f"{field} must be a non-empty absolute POSIX path")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise StudioError(f"{field} must not contain NUL or newline characters")
    if not PurePosixPath(value).is_absolute():
        raise StudioError(f"{field} must be a non-empty absolute POSIX path")
    return value


def _standalone_model_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise StudioError("model_paths must be a list of absolute POSIX paths")
    if not value:
        raise StudioError("model_paths must contain at least one model path")
    if len(value) > MAX_EVALUATION_MODELS:
        raise StudioError(
            f"model_paths accepts at most {MAX_EVALUATION_MODELS} paths per batch"
        )
    result: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        field = f"model_paths[{index}]"
        path = _user_posix_absolute_path(raw, field)
        if path == "/" or path.endswith("/"):
            raise StudioError(f"{field} must name a model directory, without a trailing slash")
        raw_parts = path.split("/")
        if any(part in {".", ".."} for part in raw_parts):
            raise StudioError(f"{field} must not contain . or .. path components")
        canonical = posixpath.normpath(path)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def _standalone_model_name(model_path: str) -> str:
    basename = PurePosixPath(model_path).name
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", basename).strip("._-")
    slug = slug[:80].rstrip("._-") or "model"
    path_hash = hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{path_hash}"


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


_MODEL_SCALE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*([BM])(?=$|[^A-Za-z0-9])"
)


def _model_scale(base_model_path: str) -> str | None:
    """Return the final 8B/4B/1.7B-style scale in a model directory name."""

    matches = list(_MODEL_SCALE_RE.finditer(PurePosixPath(base_model_path).name))
    if not matches:
        return None
    match = matches[-1]
    return f"{match.group(1)}{match.group(2).upper()}"


def _with_model_scale(template: str, scale: str | None) -> str:
    """Replace the final model-scale token in a configured output template."""

    if scale is None:
        return template
    matches = list(_MODEL_SCALE_RE.finditer(template))
    if not matches:
        return template
    match = matches[-1]
    return f"{template[:match.start()]}{scale}{template[match.end():]}"


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
    else:
        # A resolved Studio job must always carry an immutable evaluation
        # identity.  The benchmark runner can additionally fingerprint code
        # and data files, but this hash guarantees that no page-selected
        # protocol value is silently dropped between Studio and the runner.
        result["protocol_hash"] = _hash(
            {
                "schema_version": "chatts-evaluation-protocol-v1",
                "evaluation": result,
            }
        )
    return result


def _evaluation_slurm_integration(integration: dict[str, Any]) -> dict[str, Any]:
    result = dict(integration)
    result["slurm_sbatch"] = integration.get(
        "slurm_evaluation_sbatch", DEFAULT_EVALUATION_SBATCH_NAME
    )
    return result


def _evaluation_pipeline_script(integration: dict[str, Any]) -> str | None:
    value = integration.get("evaluation_pipeline_script")
    if isinstance(value, str) and value:
        return value
    evaluation_root = integration.get("evaluation_root")
    if isinstance(evaluation_root, str) and evaluation_root:
        return str(Path(evaluation_root) / "scripts" / "run_eval_only.sh")
    return None


def _resolve_evaluation_slurm_launcher(
    integration: dict[str, Any], requested: Any = None
) -> dict[str, Any]:
    launcher = resolve_slurm_launcher(
        _evaluation_slurm_integration(integration), requested
    )
    try:
        header = Path(launcher["path"]).read_bytes()[:8192].decode(
            "utf-8", errors="replace"
        )
    except OSError as exc:
        raise StudioError(f"Cannot read trusted evaluation Slurm launcher: {exc}") from exc
    if EVALUATION_SBATCH_MARKER not in header.splitlines():
        raise StudioError(
            "Slurm launcher does not implement the standalone evaluation contract: "
            f"{launcher['path']}"
        )
    return launcher


def resolve_evaluation_requests(
    payload: dict[str, Any], integration: dict[str, Any]
) -> list[dict[str, Any]]:
    """Resolve one immutable standalone-evaluation job per unique model path."""

    allowed_root = {"model_paths", "evaluation", "execution"}
    unknown_root = sorted(set(payload) - allowed_root)
    if unknown_root:
        raise StudioError(f"Unknown standalone evaluation fields: {unknown_root}")
    model_paths = _standalone_model_paths(payload.get("model_paths"))
    evaluation_request = _mapping(payload.get("evaluation"), "evaluation")
    execution_request = _mapping(payload.get("execution"), "execution")
    _reject_unknown(evaluation_request, _EVALUATION_KEYS, "evaluation")
    _reject_unknown(execution_request, _EXECUTION_KEYS, "execution")

    backend = execution_request.get(
        "backend", integration.get("execution_mode", "docker_host")
    )
    if backend not in {"docker_host", "slurm"}:
        raise StudioError("execution.backend must be docker_host or slurm")
    execution: dict[str, Any] = {"backend": backend}
    if backend == "docker_host":
        script_value = _evaluation_pipeline_script(integration)
        if not isinstance(script_value, str) or not script_value:
            raise StudioError(
                "Missing server integration setting: evaluation_pipeline_script"
            )
        script_path = Path(script_value).expanduser().resolve()
        execution.update(
            {
                "pipeline_script": str(script_path),
                "execution_root": str(script_path.parent.parent),
            }
        )
    else:
        launcher = _resolve_evaluation_slurm_launcher(
            integration, execution_request.get("sbatch_path")
        )
        execution.update(
            {
                "sbatch_path": launcher["path"],
                "sbatch_relative_path": launcher["relative_path"],
                "sbatch_sha256": launcher["sha256"],
                "execution_root": str(Path(launcher["root"]).parent),
            }
        )

    evaluation = _resolve_evaluation(evaluation_request, integration)
    pipeline_config = {
        "task_type": "standalone_evaluation",
        "seed": 42,
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
    }
    evaluation_protocol = dict(evaluation)
    evaluation_protocol.pop("protocol_hash", None)
    computed_protocol_hash = _hash(
        {
            "schema_version": "chatts-evaluation-protocol-v1",
            "evaluation": evaluation_protocol,
            "max_samples": pipeline_config["max_samples"],
            "offline": pipeline_config["offline"],
        }
    )
    supplied_protocol_hash = evaluation_request.get("protocol_hash")
    if (
        supplied_protocol_hash not in (None, "")
        and str(supplied_protocol_hash).lower() != computed_protocol_hash
    ):
        raise StudioError(
            "evaluation.protocol_hash is an expected hash and does not match "
            "the server-resolved evaluation protocol"
        )
    evaluation["protocol_hash"] = computed_protocol_hash
    protocol_id = f"protocol-{computed_protocol_hash[:16]}"

    eval_output_base = integration.get("evaluation_output_base")
    if not isinstance(eval_output_base, str) or not eval_output_base:
        raise StudioError("Missing server integration setting: evaluation_output_base")
    supplied_output = evaluation_request.get("output_root")
    supplied_model = evaluation_request.get("model_path")
    resolved: list[dict[str, Any]] = []
    for model_path in model_paths:
        model_name = _standalone_model_name(model_path)
        output_root = (
            f"{eval_output_base.rstrip('/')}/{model_name}/{protocol_id}"
        )
        if supplied_output not in (None, "", output_root):
            raise StudioError(
                "evaluation.output_root is derived from model path and protocol "
                "and cannot be changed"
            )
        if supplied_model not in (None, "", model_path):
            raise StudioError(
                "evaluation.model_path must be supplied through model_paths"
            )
        run_id = f"{model_name}-{protocol_id}-eval"
        config: dict[str, Any] = {
            "pipeline": dict(pipeline_config),
            "containers": {
                "evaluation": integration.get("evaluation_container", "ragas")
            },
            "evaluation": {
                **evaluation,
                "model_path": model_path,
                "model_name": model_name,
                "output_root": output_root,
                "run_id": run_id,
            },
            "slurm": {},
        }
        if backend == "slurm":
            slurm_paths = {
                "evaluation_host_root": integration.get("slurm_evaluation_root")
                or integration.get("evaluation_root"),
                "evaluation_sif_image": integration.get("slurm_evaluation_sif_image"),
                "chronos2_host_root": integration.get("slurm_chronos2_host_root"),
                "tsrbench_host_root": integration.get("slurm_tsrbench_host_root"),
                "tinybench_host_root": integration.get("slurm_tinybench_host_root"),
                "ts_haystack_host_root": integration.get(
                    "slurm_ts_haystack_host_root"
                ),
                "timeseriesexam_host_root": integration.get(
                    "slurm_timeseriesexam_host_root"
                ),
            }
            config["slurm"] = {
                key: str(Path(str(value)).expanduser().resolve())
                for key, value in slurm_paths.items()
                if value not in (None, "")
            }
        result = {
            "schema_version": "chatts-dataset-studio-evaluation-v1",
            "mode": "evaluate",
            "task_type": "standalone_evaluation",
            "execution": dict(execution),
            "config": config,
            "derived": {
                "model_path": model_path,
                "model_name": model_name,
                "evaluation_output_root": output_root,
                "evaluation_protocol_id": protocol_id,
                "run_id": run_id,
            },
        }
        result["config_hash"] = _hash(config)
        resolved.append(result)
    return resolved


def _docker_readiness(integration: dict[str, Any], *, include_evaluation: bool) -> list[str]:
    reasons: list[str] = []
    if shutil.which("docker") is None:
        reasons.append(
            "Docker CLI is unavailable to Dataset Studio; run the Studio control plane "
            "on the Docker host, not inside the training/evaluation container"
        )
    path_keys = ["pipeline_script", "training_root"]
    if include_evaluation:
        path_keys.append("evaluation_root")
    for key in path_keys:
        value = integration.get(key)
        if not isinstance(value, str) or not value:
            reasons.append(f"integration.{key} is not configured")
            continue
        path = Path(value).expanduser()
        if key == "pipeline_script":
            if not path.is_file():
                reasons.append(
                    f"integration.pipeline_script does not exist or is not a file: {path}"
                )
        elif not path.is_dir():
            reasons.append(f"integration.{key} does not exist or is not a directory: {path}")
    required = (
        [
            "train_project_root",
            "eval_project_root",
            "training_script",
            "evaluation_script",
            "model_output_base",
            "evaluation_output_base",
            "train_chronos2_model_path",
            "eval_chronos2_model_path",
            "tsrbench_root",
            "tinybench_dataset_root",
            "ts_haystack_root",
            "timeseriesexam_root",
            "timeseriesexam_data_file",
        ]
        if include_evaluation
        else [
            "train_project_root",
            "training_script",
            "model_output_base",
            "train_chronos2_model_path",
        ]
    )
    for key in required:
        value = integration.get(key)
        if not isinstance(value, str) or not value:
            reasons.append(f"integration.{key} is not configured")
    return reasons


def _slurm_profile_readiness(integration: dict[str, Any]) -> dict[str, Any]:
    status = slurm_readiness(integration)
    reasons = list(status["disabled_reasons"])
    for key in (
        "training_root",
        "evaluation_root",
        "train_project_root",
        "training_script",
        "model_output_base",
        "train_chronos2_model_path",
        "eval_project_root",
        "evaluation_script",
        "evaluation_output_base",
        "eval_chronos2_model_path",
        "tsrbench_root",
        "tinybench_dataset_root",
        "ts_haystack_root",
        "timeseriesexam_root",
        "timeseriesexam_data_file",
    ):
        value = integration.get(key)
        if not isinstance(value, str) or not value:
            reasons.append(f"integration.{key} is not configured")
    return {**status, "enabled": not reasons, "disabled_reasons": reasons}


def _standalone_evaluation_required_settings(
    integration: dict[str, Any], reasons: list[str]
) -> None:
    for key in (
        "evaluation_root",
        "eval_project_root",
        "evaluation_script",
        "evaluation_output_base",
        "eval_chronos2_model_path",
        "tsrbench_root",
        "tinybench_dataset_root",
        "ts_haystack_root",
        "timeseriesexam_root",
        "timeseriesexam_data_file",
    ):
        value = integration.get(key)
        if not isinstance(value, str) or not value:
            reasons.append(f"integration.{key} is not configured")


def _docker_evaluation_readiness(integration: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if shutil.which("docker") is None:
        reasons.append(
            "Docker CLI is unavailable to Dataset Studio; run the Studio control plane "
            "on the Docker host, not inside the evaluation container"
        )
    for key, kind, value in (
        (
            "evaluation_pipeline_script",
            "file",
            _evaluation_pipeline_script(integration),
        ),
        ("evaluation_root", "directory", integration.get("evaluation_root")),
    ):
        if not isinstance(value, str) or not value:
            reasons.append(f"integration.{key} is not configured")
            continue
        path = Path(value).expanduser()
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not exists:
            reasons.append(
                f"integration.{key} does not exist or is not a {kind}: {path}"
            )
    _standalone_evaluation_required_settings(integration, reasons)
    return {"enabled": not reasons, "disabled_reasons": reasons}


def _slurm_evaluation_readiness(integration: dict[str, Any]) -> dict[str, Any]:
    evaluation_integration = _evaluation_slurm_integration(integration)
    status = slurm_readiness(evaluation_integration)
    reasons = list(status["disabled_reasons"])
    launchers: list[dict[str, Any]] = []
    for item in status.get("launchers", []):
        relative = item.get("relative_path")
        try:
            launcher = _resolve_evaluation_slurm_launcher(integration, relative)
        except StudioError:
            continue
        launchers.append(
            {
                "relative_path": launcher["relative_path"],
                "sha256": launcher["sha256"],
            }
        )
    default_sbatch: str | None = None
    try:
        default_sbatch = _resolve_evaluation_slurm_launcher(integration)[
            "relative_path"
        ]
    except StudioError as exc:
        message = str(exc)
        if message not in reasons:
            reasons.append(message)
    _standalone_evaluation_required_settings(integration, reasons)
    return {
        **status,
        "enabled": not reasons,
        "disabled_reasons": reasons,
        "launchers": launchers,
        "default_sbatch": default_sbatch,
    }


def public_pipeline_defaults(integration: dict[str, Any]) -> dict[str, Any]:
    """Return safe UI defaults and per-backend readiness information."""
    execution_mode = integration.get("execution_mode", "docker_host")
    docker_full_reasons = _docker_readiness(integration, include_evaluation=True)
    docker_stage1_reasons = _docker_readiness(integration, include_evaluation=True)
    slurm_status = _slurm_profile_readiness(integration)
    evaluation_docker_status = _docker_evaluation_readiness(integration)
    evaluation_slurm_status = _slurm_evaluation_readiness(integration)
    backend_status = {
        "docker_host": {
            "enabled": not docker_full_reasons,
            "disabled_reasons": docker_full_reasons,
            "profiles": {
                "chronos2-full": {
                    "enabled": not docker_full_reasons,
                    "disabled_reasons": docker_full_reasons,
                },
                "chronos2-stage1": {
                    "enabled": not docker_stage1_reasons,
                    "disabled_reasons": docker_stage1_reasons,
                },
            },
        },
        "slurm": {
            **slurm_status,
            "profiles": {
                profile: {
                    "enabled": slurm_status["enabled"],
                    "disabled_reasons": slurm_status["disabled_reasons"],
                }
                for profile in ("chronos2-full", "chronos2-stage1")
            },
        },
    }
    if execution_mode not in backend_status:
        disabled_reasons = [
            f"integration.execution_mode must be docker_host or slurm, got: {execution_mode}"
        ]
        enabled = False
    else:
        disabled_reasons = list(backend_status[execution_mode]["disabled_reasons"])
        enabled = not disabled_reasons
    default_base_model = integration.get("base_model_path")
    default_scale = (
        _model_scale(default_base_model) if isinstance(default_base_model, str) else None
    )
    configured_model_output = integration.get("model_output_base")
    default_model_output = (
        _with_model_scale(configured_model_output, default_scale)
        if isinstance(configured_model_output, str)
        else configured_model_output
    )
    evaluation_backend_status = {
        "docker_host": evaluation_docker_status,
        "slurm": evaluation_slurm_status,
    }
    evaluation_default_status = evaluation_backend_status.get(execution_mode)
    if evaluation_default_status is None:
        evaluation_disabled_reasons = [
            f"integration.execution_mode must be docker_host or slurm, got: {execution_mode}"
        ]
    else:
        evaluation_disabled_reasons = list(
            evaluation_default_status["disabled_reasons"]
        )
    return {
        "training": {
            "profile": "chronos2-full",
            "base_model_path": default_base_model,
            "output_root": default_model_output,
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
            "enabled": enabled,
            "disabled_reasons": disabled_reasons,
            "execution_mode": execution_mode,
            "backends": backend_status,
            "training_root": integration.get("training_root"),
            "evaluation_root": integration.get("evaluation_root"),
            "training_container": integration.get("training_container", "chatts"),
            "evaluation_container": integration.get("evaluation_container", "ragas"),
            "evaluation_only": {
                "enabled": not evaluation_disabled_reasons,
                "disabled_reasons": evaluation_disabled_reasons,
                "execution_mode": execution_mode,
                "backends": evaluation_backend_status,
                "evaluation_container": integration.get(
                    "evaluation_container", "ragas"
                ),
                "evaluation_root": integration.get("evaluation_root"),
                "max_batch_models": MAX_EVALUATION_MODELS,
            },
        },
    }


def resolve_pipeline_request(
    payload: dict[str, Any], version_record: dict[str, Any], integration: dict[str, Any]
) -> dict[str, Any]:
    allowed_root = {"mode", "version", "training", "evaluation", "execution"}
    unknown_root = sorted(set(payload) - allowed_root)
    if unknown_root:
        raise StudioError(f"Unknown pipeline fields: {unknown_root}")

    version = version_record.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"datav[1-9][0-9]*", version):
        raise StudioError("Version record has no canonical datavN version")
    if payload.get("version") not in (None, version):
        raise StudioError("Requested version does not match the resolved version record")
    snapshot_dir, snapshot_hash, dataset_names = _version_record_fields(version_record)

    training_request = _mapping(payload.get("training"), "training")
    evaluation_request = _mapping(payload.get("evaluation"), "evaluation")
    execution_request = _mapping(payload.get("execution"), "execution")
    _reject_unknown(training_request, _TRAINING_KEYS, "training")
    _reject_unknown(evaluation_request, _EVALUATION_KEYS, "evaluation")
    _reject_unknown(execution_request, _EXECUTION_KEYS, "execution")

    profile = training_request.get("profile", "chronos2-full")
    if profile not in {"chronos2-full", "chronos2-stage1"}:
        raise StudioError(
            "training.profile must be chronos2-full or chronos2-stage1"
        )
    pipeline_mode = "stage1" if profile == "chronos2-stage1" else "full"
    backend = execution_request.get(
        "backend", integration.get("execution_mode", "docker_host")
    )
    if backend not in {"docker_host", "slurm"}:
        raise StudioError("execution.backend must be docker_host or slurm")
    # Every supported training profile/backend is an end-to-end
    # train-then-evaluate pipeline.  Legacy frontends used ``train`` for
    # Stage1/Slurm; accept those spellings during rollout but canonicalize the
    # frozen job to train_eval so downstream launchers cannot skip evaluation.
    expected_mode = "train_eval"
    requested_mode = payload.get("mode", expected_mode)
    accepted_modes = {expected_mode, "train"}
    if pipeline_mode == "stage1":
        accepted_modes.add("train_stage1")
    else:
        accepted_modes.add("train_full")
    if requested_mode not in accepted_modes:
        raise StudioError(
            f"mode={requested_mode!r} is incompatible with profile={profile!r} "
            f"and execution backend={backend!r}; use mode={expected_mode}"
        )

    execution: dict[str, Any] = {"backend": backend}
    if backend == "slurm":
        launcher = resolve_slurm_launcher(
            integration, execution_request.get("sbatch_path")
        )
        execution.update(
            {
                "sbatch_path": launcher["path"],
                "sbatch_relative_path": launcher["relative_path"],
                "sbatch_sha256": launcher["sha256"],
                "training_root": str(
                    Path(str(integration.get("training_root", "")))
                    .expanduser()
                    .resolve()
                ),
            }
        )

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
    base_model_path = _user_posix_absolute_path(
        training_request.get("base_model_path", integration.get("base_model_path")),
        "training.base_model_path",
    )
    model_scale = _model_scale(base_model_path)
    model_output_base = integration.get("model_output_base")
    if not isinstance(model_output_base, str) or not model_output_base:
        raise StudioError("Missing server integration setting: model_output_base")
    model_output_base = _with_model_scale(model_output_base, model_scale)
    version_output_root = _versioned_name(model_output_base, version)

    train_script = integration.get("training_script")
    train_chronos = integration.get("train_chronos2_model_path")
    if not isinstance(train_script, str) or not train_script:
        raise StudioError("Missing server integration setting: training_script")
    if not isinstance(train_chronos, str) or not train_chronos:
        raise StudioError("Missing server integration setting: train_chronos2_model_path")

    stage1 = _resolve_stage(
        "stage1", _mapping(training_request.get("stage1"), "training.stage1"), dataset_names["stage1"]
    )
    stage2: dict[str, Any]
    if pipeline_mode == "full":
        stage2 = _resolve_stage(
            "stage2",
            _mapping(training_request.get("stage2"), "training.stage2"),
            dataset_names["stage2"],
        )
    else:
        # Stage2 is deliberately not part of a Stage1-only experiment identity.
        # Validate only its shape/keys so stale hidden UI fields cannot alter the
        # output path or accidentally become executable configuration.
        stage2_request = _mapping(training_request.get("stage2"), "training.stage2")
        _reject_unknown(stage2_request, _STAGE_KEYS, "training.stage2")
        stage2 = _resolve_stage("stage2", {}, dataset_names["stage2"])

    evaluation = _resolve_evaluation(evaluation_request, integration)

    # A data version is not an experiment identity. Two runs over the same
    # snapshot can still produce different weights when any training setting
    # changes. Keep a stable, training-only hash (excluding force/retry and all
    # evaluation controls) in every model path so one recipe cannot overwrite
    # another recipe's checkpoints.
    training_recipe: dict[str, Any] = {
        "schema_version": "chatts-training-recipe-v1",
        "profile": profile,
        "data_version": version,
        "dataset_snapshot_hash": snapshot_hash,
        "base_model_path": base_model_path,
        "time_series_encoder": {
            "type": "chronos2",
            "model_path": train_chronos,
        },
        "seed": seed,
        "deepspeed_include": deepspeed_include,
        "stage1": stage1,
    }
    if pipeline_mode == "full":
        training_recipe["stage2"] = stage2
    training_recipe_hash = _hash(training_recipe)
    training_recipe_id = f"recipe-{training_recipe_hash[:16]}"
    train_output_root = f"{version_output_root.rstrip('/')}/experiments/{training_recipe_id}"
    supplied_output = training_request.get("output_root")
    if supplied_output not in (None, "", train_output_root, version_output_root):
        raise StudioError(
            "training.output_root is derived from data version and training recipe"
        )
    final_model_path = (
        f"{train_output_root}/best_stage1_seed{seed}"
        if pipeline_mode == "stage1"
        else f"{train_output_root}/best_seed{seed}"
    )

    pipeline_config: dict[str, Any] = {
        "pipeline_mode": pipeline_mode,
        "seed": seed,
        "data_version": version,
        "dataset_snapshot_hash": snapshot_hash,
        "training_recipe_hash": training_recipe_hash,
        "force_train": _as_bool(
            training_request.get("force_train", False), "training.force_train"
        ),
        "preflight_only": False,
    }
    pipeline_config.update(
        {
            "force_eval": _as_bool(
                evaluation_request.get("force_eval", False),
                "evaluation.force_eval",
            ),
            "max_samples": _as_int(
                evaluation_request.get("max_samples", 0),
                "evaluation.max_samples",
            ),
            "offline": _as_bool(
                evaluation_request.get("offline", True), "evaluation.offline"
            ),
        }
    )
    evaluation_protocol = dict(evaluation)
    evaluation_protocol.pop("protocol_hash", None)
    computed_protocol_hash = _hash(
        {
            "schema_version": "chatts-evaluation-protocol-v1",
            "evaluation": evaluation_protocol,
            "max_samples": pipeline_config["max_samples"],
            "offline": pipeline_config["offline"],
        }
    )
    supplied_protocol_hash = evaluation_request.get("protocol_hash")
    if (
        supplied_protocol_hash not in (None, "")
        and str(supplied_protocol_hash).lower() != computed_protocol_hash
    ):
        raise StudioError(
            "evaluation.protocol_hash is an expected hash and does not match "
            "the server-resolved evaluation protocol"
        )
    evaluation["protocol_hash"] = computed_protocol_hash
    training_config: dict[str, Any] = {
        "project_root": train_project_root,
        "script": train_script,
        "base_model_path": base_model_path,
        "output_root": train_output_root,
        "final_model_path": final_model_path,
        "chronos2_model_path": train_chronos,
        "dataset_dir": snapshot_dir,
        "keep_stage1": (
            True
            if pipeline_mode == "stage1"
            else _as_bool(
                training_request.get("keep_stage1", False), "training.keep_stage1"
            )
        ),
        "deepspeed_include": deepspeed_include,
        "master_port": master_port,
        "stage1": stage1,
    }
    if pipeline_mode == "stage1":
        training_config["stage1_model_path"] = final_model_path
    training_config["stage2"] = stage2

    config: dict[str, Any] = {
        "pipeline": pipeline_config,
        "containers": {
            "training": integration.get("training_container", "chatts"),
            "evaluation": integration.get("evaluation_container", "ragas"),
        },
        "training": training_config,
    }

    model_name: str | None = None
    eval_output_root: str | None = None
    run_id: str | None = None
    model_name_base = integration.get("model_name_base", "chatts-msxf-8B")
    if not isinstance(model_name_base, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", model_name_base
    ):
        raise StudioError("model_name_base must be a safe slug")
    model_name_base = _with_model_scale(model_name_base, model_scale)
    model_name = (
        f"{_versioned_name(model_name_base, version)}-seed{seed}-{training_recipe_id}"
    )
    eval_output_base = integration.get("evaluation_output_base")
    if not isinstance(eval_output_base, str) or not eval_output_base:
        raise StudioError("Missing server integration setting: evaluation_output_base")
    # Evaluation artifacts are protocol-scoped.  Runs that share one training
    # recipe/model but change a benchmark, partition, prompt, token budget, or
    # offline policy must never write into the same cache/output directory.
    eval_protocol_id = f"protocol-{evaluation['protocol_hash'][:16]}"
    eval_output_root = (
        f"{eval_output_base.rstrip('/')}/{model_name}/{eval_protocol_id}"
    )
    supplied_eval_output = evaluation_request.get("output_root")
    if supplied_eval_output not in (None, "", eval_output_root):
        raise StudioError(
            "evaluation.output_root is derived from data version and cannot be changed"
        )
    supplied_model = evaluation_request.get("model_path")
    if supplied_model not in (None, "", final_model_path):
        raise StudioError(
            "evaluation.model_path is derived from training output and cannot be changed"
        )
    run_id = (
        f"chronos2-{version}-seed{seed}-{training_recipe_id}-"
        f"{eval_protocol_id}-{pipeline_mode}"
    )
    completion_marker = (
        "STAGE1_COMPLETE.json"
        if pipeline_mode == "stage1"
        else "TRAINING_COMPLETE.json"
    )
    config["evaluation"] = {
        **evaluation,
        "model_path": final_model_path,
        "model_name": model_name,
        "output_root": eval_output_root,
        "run_id": run_id,
        "model_completion_marker": completion_marker,
    }
    if backend == "slurm":
        slurm_paths = {
            "evaluation_host_root": integration.get("slurm_evaluation_root")
            or integration.get("evaluation_root"),
            "evaluation_sif_image": integration.get("slurm_evaluation_sif_image"),
            "chronos2_host_root": integration.get("slurm_chronos2_host_root"),
            "tsrbench_host_root": integration.get("slurm_tsrbench_host_root"),
            "tinybench_host_root": integration.get("slurm_tinybench_host_root"),
            "ts_haystack_host_root": integration.get(
                "slurm_ts_haystack_host_root"
            ),
            "timeseriesexam_host_root": integration.get(
                "slurm_timeseriesexam_host_root"
            ),
        }
        config["slurm"] = {
            key: str(Path(str(value)).expanduser().resolve())
            for key, value in slurm_paths.items()
            if value not in (None, "")
        }
    result = {
        "schema_version": "chatts-dataset-studio-pipeline-v1",
        "version": version,
        "dataset_snapshot_hash": snapshot_hash,
        "dataset_names": dataset_names,
        "mode": expected_mode,
        "profile": profile,
        "pipeline_mode": pipeline_mode,
        "execution": execution,
        "config": config,
        "derived": {
            "version_output_root": version_output_root,
            "train_output_root": train_output_root,
            "final_model_path": final_model_path,
            "training_recipe_id": training_recipe_id,
            "training_recipe_hash": training_recipe_hash,
            "training_recipe": training_recipe,
            "model_name": model_name,
            "evaluation_output_root": eval_output_root,
            "evaluation_protocol_id": eval_protocol_id,
            "run_id": run_id,
            "model_scale": model_scale,
        },
    }
    result["config_hash"] = _hash(config)
    return result


class PipelineJobs:
    """Persist jobs while separating local Docker FIFO from Slurm scheduling."""

    def __init__(
        self,
        state_root: str | Path,
        pipeline_script: str | Path | None,
        integration: dict[str, Any] | None = None,
    ):
        self.state_root = Path(state_root).expanduser().resolve()
        self.jobs_root = self.state_root / "jobs"
        self.config_root = self.state_root / "configs"
        self.logs_root = self.state_root / "logs"
        self.run_records_root = self.state_root / "run-records"
        self.status_root = self.state_root / "worker-status"
        self.run_lock_path = self.state_root / ".pipeline-run.lock"
        self.pipeline_script = (
            Path(pipeline_script).expanduser().resolve() if pipeline_script else None
        )
        self.integration = dict(integration or {})
        evaluation_pipeline_script = self.integration.get(
            "evaluation_pipeline_script"
        )
        if not isinstance(evaluation_pipeline_script, str) or not evaluation_pipeline_script:
            evaluation_pipeline_script = _evaluation_pipeline_script(self.integration)
        self.evaluation_pipeline_script = (
            Path(evaluation_pipeline_script).expanduser().resolve()
            if isinstance(evaluation_pipeline_script, str)
            and evaluation_pipeline_script
            else None
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

    def _list_raw(self) -> list[dict[str, Any]]:
        if not self.jobs_root.is_dir():
            return []
        return [self._read(path) for path in self.jobs_root.glob("*.json")]

    @staticmethod
    def _fifo_key(job: dict[str, Any]) -> tuple[int, str, str]:
        sequence = job.get("queue_sequence")
        return (
            sequence if isinstance(sequence, int) and not isinstance(sequence, bool) else 0,
            str(job.get("created_at", "")),
            str(job.get("job_id", "")),
        )

    def _next_queue_sequence_locked(self) -> int:
        sequences = [
            value
            for job in self._list_raw()
            if isinstance((value := job.get("queue_sequence")), int)
            and not isinstance(value, bool)
            and value > 0
        ]
        return max(sequences, default=0) + 1

    def _decorate_queue_positions(
        self, jobs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # Only Docker jobs wait in Dataset Studio. Slurm jobs are submitted
        # immediately and any PENDING order belongs to the cluster scheduler.
        positions = {
            str(job.get("job_id")): position
            for position, job in enumerate(
                sorted(
                    (
                        item
                        for item in jobs
                        if item.get("status") == "queued"
                        and item.get("execution_backend", "docker_host")
                        == "docker_host"
                    ),
                    key=self._fifo_key,
                ),
                1,
            )
        }
        decorated = []
        for job in jobs:
            current = dict(job)
            job_id = str(current.get("job_id", ""))
            if job_id in positions:
                current["queue_position"] = positions[job_id]
            else:
                current.pop("queue_position", None)
            decorated.append(current)
        return decorated

    def list(self) -> list[dict[str, Any]]:
        jobs = self._list_raw()
        for job in jobs:
            job_id = job.get("job_id")
            if not isinstance(job_id, str):
                continue
            worker_status = self._read_worker_status(job_id)
            if not worker_status:
                continue
            # A worker publishes its terminal status immediately before it
            # exits.  Keep the durable job record authoritative for terminal
            # transitions so callers never observe "completed" before the
            # watcher has committed that state to jobs/<id>.json.
            if worker_status.get("status") in {"completed", "failed", "canceled"} and job.get(
                "status"
            ) not in {"completed", "failed", "canceled"}:
                continue
            for key in (
                "status",
                "pid",
                "started_at",
                "finished_at",
                "duration_seconds",
                "exit_code",
                "error",
                "execution_backend",
                "scheduler_job_id",
                "scheduler_state",
            ):
                if key in worker_status:
                    job[key] = worker_status[key]
        jobs = self._decorate_queue_positions(jobs)
        return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)

    def get(self, job_id: str, *, include_log: bool = True) -> dict[str, Any]:
        path = self._path(job_id)
        if not path.is_file():
            raise StudioError(f"Unknown pipeline job: {job_id}")
        job = self._read(path)
        if job.get("status") == "queued":
            positions = {
                str(item.get("job_id")): item.get("queue_position")
                for item in self._decorate_queue_positions(self._list_raw())
            }
            position = positions.get(job_id)
            if position is None:
                # The scheduler may have promoted this item between the first
                # read and the queue scan. Return the authoritative latest
                # record instead of surfacing a transient KeyError to the API.
                job = self._read(path)
                job.pop("queue_position", None)
            else:
                job["queue_position"] = position
        else:
            job.pop("queue_position", None)
        worker_status = self._read_worker_status(job_id)
        if worker_status:
            terminal_before_commit = worker_status.get("status") in {
                "completed",
                "failed",
                "canceled",
            } and job.get("status") not in {"completed", "failed", "canceled"}
            if not terminal_before_commit:
                for key in (
                    "status",
                    "pid",
                    "started_at",
                    "finished_at",
                    "duration_seconds",
                    "exit_code",
                    "error",
                    "execution_backend",
                    "scheduler_job_id",
                    "scheduler_state",
                ):
                    if key in worker_status:
                        job[key] = worker_status[key]
        if include_log:
            log_path = Path(job["log_path"])
            if log_path.is_file():
                with log_path.open("rb") as stream:
                    stream.seek(max(0, log_path.stat().st_size - 100_000))
                    job["log_tail"] = stream.read().decode("utf-8", errors="replace")
            else:
                job["log_tail"] = ""
            for label, key in (
                ("Slurm stdout", "scheduler_stdout_path"),
                ("Slurm stderr", "scheduler_stderr_path"),
            ):
                value = job.get(key)
                if not isinstance(value, str):
                    continue
                scheduler_id = job.get("scheduler_job_id")
                if isinstance(scheduler_id, str):
                    value = value.replace("%j", scheduler_id)
                scheduler_log = Path(value)
                if not scheduler_log.is_file():
                    continue
                with scheduler_log.open("rb") as stream:
                    stream.seek(max(0, scheduler_log.stat().st_size - 100_000))
                    tail = stream.read().decode("utf-8", errors="replace")
                if tail:
                    job["log_tail"] += f"\n===== {label} =====\n{tail}"
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

    def _apply_worker_status_locked(
        self, job_id: str, status: dict[str, Any]
    ) -> None:
        job = self._read(self._path(job_id))
        for key in (
            "status",
            "pid",
            "started_at",
            "finished_at",
            "duration_seconds",
            "exit_code",
            "error",
            "execution_backend",
            "scheduler_job_id",
            "scheduler_state",
        ):
            if key in status:
                job[key] = status[key]
        self._write(job)
        self.processes.pop(job_id, None)

    def _mark_interrupted_locked(self, job_id: str) -> None:
        job = self._read(self._path(job_id))
        if job.get("status") != "running":
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

    def _worker_command(self, job: dict[str, Any]) -> list[str]:
        backend = job.get("execution_backend", "docker_host")
        execution_cwd = job.get("execution_cwd")
        if not isinstance(execution_cwd, str) or not execution_cwd:
            job_script = self._docker_script_for_job(job)
            if backend == "docker_host" and job_script is not None:
                execution_cwd = str(job_script.parent.parent)
            else:
                raise StudioError("Queued job has no execution working directory")
        command = [
            sys.executable,
            str(Path(__file__).with_name("pipeline_worker.py")),
            "--job-id",
            str(job["job_id"]),
            "--status-path",
            str(self._status_path(str(job["job_id"]))),
            "--backend",
            str(backend),
            "--config",
            str(job["config_path"]),
            "--cwd",
            execution_cwd,
        ]
        if backend == "docker_host":
            job_script = self._docker_script_for_job(job)
            if job_script is None:
                raise StudioError("One-click Docker pipeline is disabled")
            command.extend(
                [
                    "--lock-path",
                    str(self.run_lock_path),
                    "--script",
                    str(job_script),
                ]
            )
        elif backend == "slurm":
            command.extend(
                [
                    "--sbatch-path",
                    str(job["sbatch_path"]),
                    "--sbatch-sha256",
                    str(job["sbatch_sha256"]),
                    "--scheduler-stdout",
                    str(job["scheduler_stdout_path"]),
                    "--scheduler-stderr",
                    str(job["scheduler_stderr_path"]),
                    "--poll-seconds",
                    str(self.integration.get("slurm_poll_seconds", 5.0)),
                    "--accounting-grace-seconds",
                    str(self.integration.get("slurm_accounting_grace_seconds", 120.0)),
                ]
            )
            scheduler_job_id = job.get("scheduler_job_id")
            if isinstance(scheduler_job_id, str):
                command.extend(["--scheduler-job-id", scheduler_job_id])
            if job.get("kind") in {"preflight", "evaluation_preflight"}:
                command.append("--slurm-preflight")
        else:
            raise StudioError(f"Unknown execution backend in queued job: {backend}")
        return command

    def _docker_script_for_job(self, job: dict[str, Any]) -> Path | None:
        value = job.get("pipeline_script_path")
        if isinstance(value, str) and value:
            return Path(value).expanduser().resolve()
        return self.pipeline_script

    def _launch_locked(self, job: dict[str, Any]) -> bool:
        job_id = str(job["job_id"])
        backend = job.get("execution_backend", "docker_host")
        docker_script = self._docker_script_for_job(job)
        if backend == "docker_host" and (
            docker_script is None or not docker_script.is_file()
        ):
            job.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "exit_code": 126,
                    "error": (
                        "Could not start fixed pipeline script: pipeline_script "
                        "is missing on the host"
                    ),
                }
            )
            self._write(job)
            return False
        if backend == "slurm":
            sbatch_path = Path(str(job.get("sbatch_path", "")))
            expected_hash = job.get("sbatch_sha256")
            try:
                actual_hash = hashlib.sha256(sbatch_path.read_bytes()).hexdigest()
            except OSError as exc:
                actual_hash = None
                job.update(
                    {
                        "status": "failed",
                        "finished_at": _utc_now(),
                        "exit_code": 126,
                        "error": f"Could not read frozen Slurm launcher: {exc}",
                    }
                )
            if actual_hash is None or actual_hash != expected_hash:
                if actual_hash is not None:
                    job.update(
                        {
                            "status": "failed",
                            "finished_at": _utc_now(),
                            "exit_code": 126,
                            "error": "Slurm launcher changed after this job was frozen",
                        }
                    )
                self._write(job)
                return False
        elif backend != "docker_host":
            job.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "exit_code": 126,
                    "error": f"Unknown execution backend: {backend}",
                }
            )
            self._write(job)
            return False
        config_path = Path(str(job.get("config_path", "")))
        if not config_path.is_file():
            job.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "exit_code": 126,
                    "error": f"Could not start queued pipeline: config is missing: {config_path}",
                }
            )
            self._write(job)
            return False

        status_path = self._status_path(job_id)
        try:
            status_path.unlink(missing_ok=True)
            self.logs_root.mkdir(parents=True, exist_ok=True)
            execution_cwd = job.get("execution_cwd")
            if not isinstance(execution_cwd, str) or not execution_cwd:
                execution_cwd = (
                    str(docker_script.parent.parent)
                    if docker_script is not None
                    else str(self.state_root)
                )
            with Path(str(job["log_path"])).open("wb") as log_stream:
                process = subprocess.Popen(  # noqa: S603 - configured executable only.
                    self._worker_command(job),
                    cwd=Path(execution_cwd),
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
                    "exit_code": 126,
                    "error": f"Could not start fixed pipeline script: {exc}",
                }
            )
            self._write(job)
            return False

        job.update(
            {
                "status": "running",
                "started_at": _utc_now(),
                "pid": process.pid,
            }
        )
        job.pop("queue_position", None)
        self._write(job)
        self.processes[job_id] = process
        threading.Thread(
            target=self._watch,
            args=(job_id, process),
            name=f"pipeline-{job_id[:8]}",
            daemon=True,
        ).start()
        return True

    def _dispatch_next_locked(self) -> None:
        jobs = self._list_raw()
        queued_docker = sorted(
            (
                job
                for job in jobs
                if job.get("status") == "queued"
                and job.get("execution_backend", "docker_host") == "docker_host"
            ),
            key=self._fifo_key,
        )
        docker_running = any(
            job.get("status") == "running"
            and job.get("execution_backend", "docker_host") == "docker_host"
            for job in jobs
        )
        if not docker_running:
            # Docker uses one shared pair of training/evaluation containers, so
            # it retains the durable local FIFO and a process-level run lock.
            for job in queued_docker:
                job_script = self._docker_script_for_job(job)
                if job_script is None or not job_script.is_file():
                    # Preserve a durable queued job while the host checkout is
                    # temporarily unavailable (for example during a restart).
                    break
                if self._launch_locked(job):
                    break

        # Slurm owns GPU admission and scheduling. Submit every frozen Slurm
        # job immediately; do not serialize them behind Docker or each other.
        queued_slurm = sorted(
            (
                job
                for job in jobs
                if job.get("status") == "queued"
                and job.get("execution_backend") == "slurm"
            ),
            key=self._fifo_key,
        )
        for job in queued_slurm:
            self._launch_locked(job)

    def _watch(self, job_id: str, process: subprocess.Popen[bytes] | None = None) -> None:
        if process is not None:
            process.wait()
        else:
            while True:
                status = self._read_worker_status(job_id)
                if status and status.get("status") in {"completed", "failed", "canceled"}:
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
            if status and status.get("status") in {"completed", "failed", "canceled"}:
                with self.lock:
                    self._apply_worker_status_locked(job_id, status)
                    self._dispatch_next_locked()
                return
            time.sleep(0.05)
        with self.lock:
            self._mark_interrupted_locked(job_id)
            self._dispatch_next_locked()

    def _recover(self) -> None:
        recovered_running: list[str] = []
        with self.lock:
            for job in sorted(self._list_raw(), key=self._fifo_key):
                if job.get("status") not in {"queued", "running"}:
                    continue
                job_id = job.get("job_id")
                if not isinstance(job_id, str):
                    continue
                status = self._read_worker_status(job_id)
                if status and status.get("status") in {"completed", "failed", "canceled"}:
                    self._apply_worker_status_locked(job_id, status)
                    continue
                if (
                    job.get("execution_backend") == "slurm"
                    and status
                    and isinstance(status.get("scheduler_job_id"), str)
                ):
                    job["scheduler_job_id"] = status["scheduler_job_id"]
                    if "scheduler_state" in status:
                        job["scheduler_state"] = status["scheduler_state"]
                    self._write(job)
                pid = (status or {}).get("pid", job.get("pid"))
                if self._pid_alive(pid):
                    # Covers a service crash after Popen but before the queued
                    # record was promoted to running.
                    if job.get("status") == "queued":
                        job["status"] = "running"
                        for key in ("pid", "started_at"):
                            if status and key in status:
                                job[key] = status[key]
                        job.setdefault("pid", pid)
                        job.setdefault("started_at", _utc_now())
                        self._write(job)
                    recovered_running.append(job_id)
                elif job.get("status") == "running":
                    # Recheck once because the worker may have atomically
                    # published its terminal status between the first read and
                    # the PID check.
                    final_status = self._read_worker_status(job_id)
                    if final_status and final_status.get("status") in {
                        "completed",
                        "failed",
                        "canceled",
                    }:
                        self._apply_worker_status_locked(job_id, final_status)
                    elif (
                        job.get("execution_backend") == "slurm"
                        and status
                        and isinstance(status.get("scheduler_job_id"), str)
                    ):
                        # The local tracker may die while the scheduler job keeps
                        # running (for example after a control-plane reboot).
                        # Requeue a tracker for the existing immutable Slurm id;
                        # never submit a second training job.
                        job["status"] = "queued"
                        job["scheduler_job_id"] = status["scheduler_job_id"]
                        if "scheduler_state" in status:
                            job["scheduler_state"] = status["scheduler_state"]
                        job.pop("pid", None)
                        self._write(job)
                    else:
                        self._mark_interrupted_locked(job_id)
                # A genuine queued item has no worker yet and remains queued.
            self._dispatch_next_locked()

        for job_id in recovered_running:
            threading.Thread(
                target=self._watch,
                args=(job_id,),
                name=f"pipeline-recover-{job_id[:8]}",
                daemon=True,
            ).start()

    @staticmethod
    def _diff_values(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
        if isinstance(before, dict) and isinstance(after, dict):
            changes: list[dict[str, Any]] = []
            for key in sorted(set(before) | set(after)):
                child = f"{path}.{key}" if path else key
                if key not in before:
                    changes.append({"path": child, "before": None, "after": after[key]})
                elif key not in after:
                    changes.append({"path": child, "before": before[key], "after": None})
                else:
                    changes.extend(PipelineJobs._diff_values(before[key], after[key], child))
            return changes
        if before != after:
            return [{"path": path, "before": before, "after": after}]
        return []

    @staticmethod
    def _comparison_config(config: dict[str, Any]) -> dict[str, Any]:
        comparable = json.loads(json.dumps(config))
        pipeline = comparable.get("pipeline")
        if isinstance(pipeline, dict):
            for key in ("trial_id", "trial_config_hash", "preflight_only"):
                pipeline.pop(key, None)
            if pipeline.get("pipeline_mode") == "stage1":
                training = comparable.get("training")
                if isinstance(training, dict):
                    # The Slurm verifier receives a canonical Stage2 block for
                    # snapshot compatibility, but Stage1-only never executes it.
                    # Do not report that inert block as an experiment change.
                    training.pop("stage2", None)
        return comparable

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _previous_training_comparison(self) -> tuple[str | None, dict[str, Any] | None]:
        for job in self.list():
            if job.get("kind") not in {
                "train_eval",
                "train_eval_stage1",
                "train_eval_full",
                # Read older run records written before all modes evaluated.
                "train_stage1",
                "train_full",
            }:
                continue
            comparison_path = job.get("comparison_path")
            if isinstance(comparison_path, str):
                comparison = self._read_mapping(Path(comparison_path))
                if comparison is not None:
                    return str(job.get("job_id")), comparison
            config_path = job.get("config_path")
            if not isinstance(config_path, str) or not Path(config_path).is_file():
                continue
            try:
                previous_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(previous_config, dict):
                continue
            return str(job.get("job_id")), {
                "data": {
                    "version": job.get("version"),
                    "dataset_snapshot_hash": job.get("dataset_snapshot_hash"),
                    "snapshot_dir": previous_config.get("training", {}).get("dataset_dir"),
                },
                "config": self._comparison_config(previous_config),
            }
        return None, None

    def _write_run_record(
        self,
        job_id: str,
        resolved: dict[str, Any],
        config: dict[str, Any],
        config_path: Path,
    ) -> dict[str, Any]:
        record_root = self.run_records_root / job_id
        record_root.mkdir(parents=True, exist_ok=False)
        exported_config = record_root / "pipeline_config.resolved.yaml"
        exported_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

        snapshot_dir = Path(str(config["training"]["dataset_dir"]))
        snapshot_manifest_path = snapshot_dir / "manifest.json"
        snapshot_manifest = self._read_mapping(snapshot_manifest_path)
        data_export = {
            "schema_version": "chatts-dataset-studio-training-data-v1",
            "exported_at": _utc_now(),
            "version": resolved["version"],
            "dataset_snapshot_hash": resolved["dataset_snapshot_hash"],
            "snapshot_dir": str(snapshot_dir),
            "snapshot_manifest_path": str(snapshot_manifest_path),
            "dataset_names": resolved.get("dataset_names", {}),
            "snapshot_manifest": snapshot_manifest,
        }
        data_path = record_root / "training_data.json"
        data_path.write_text(
            json.dumps(data_export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        resolved_execution = resolved.get("execution", {"backend": "docker_host"})
        comparison = {
            "data": {key: value for key, value in data_export.items() if key != "exported_at"},
            "config": self._comparison_config(config),
            "execution": {
                key: resolved_execution.get(key)
                for key in ("backend", "sbatch_relative_path", "sbatch_sha256")
                if resolved_execution.get(key) is not None
            },
        }
        comparison_path = record_root / "comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        previous_job_id, previous = self._previous_training_comparison()
        changes = self._diff_values(previous, comparison) if previous is not None else []
        diff_export = {
            "schema_version": "chatts-dataset-studio-run-diff-v1",
            "current_job_id": job_id,
            "previous_job_id": previous_job_id,
            "has_previous_run": previous is not None,
            "change_count": len(changes),
            "changes": changes,
        }
        diff_json_path = record_root / "diff_from_previous.json"
        diff_json_path.write_text(
            json.dumps(diff_export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        diff_md_path = record_root / "diff_from_previous.md"
        lines = ["# 与上一次训练的差异", ""]
        if previous is None:
            lines.append("这是第一条训练记录，没有可比较的上一次训练。")
        elif not changes:
            lines.append(f"与上一次训练 `{previous_job_id}` 的数据和参数完全一致。")
        else:
            lines.extend(
                [
                    f"上一次训练：`{previous_job_id}`",
                    "",
                    f"共 {len(changes)} 项变化。",
                    "",
                    "| 配置项 | 上一次 | 本次 |",
                    "| --- | --- | --- |",
                ]
            )
            for change in changes:
                before = json.dumps(change["before"], ensure_ascii=False, sort_keys=True)
                after = json.dumps(change["after"], ensure_ascii=False, sort_keys=True)
                lines.append(
                    f"| `{change['path']}` | `{before.replace('|', '&#124;')}` | "
                    f"`{after.replace('|', '&#124;')}` |"
                )
        diff_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        record_artifacts = {
            "训练参数配置": str(exported_config),
            "训练数据清单": str(data_path),
            "与上次差异 JSON": str(diff_json_path),
            "与上次差异报告": str(diff_md_path),
            "不可变数据快照": str(snapshot_dir),
        }
        if resolved_execution.get("backend") == "slurm":
            record_artifacts["Slurm 提交脚本"] = str(
                resolved_execution.get("sbatch_path")
            )
        evaluation_output = resolved.get("derived", {}).get(
            "evaluation_output_root"
        )
        if isinstance(evaluation_output, str) and evaluation_output:
            record_artifacts["评测输出目录"] = evaluation_output
            record_artifacts["评测状态表"] = (
                f"{evaluation_output.rstrip('/')}/benchmark_status.tsv"
            )
            record_artifacts["评测汇总"] = (
                f"{evaluation_output.rstrip('/')}/all_benchmarks_summary.md"
            )
            record_artifacts["评测指标"] = f"{evaluation_output.rstrip('/')}/metrics.json"
        record = {
            "schema_version": "chatts-dataset-studio-run-record-v1",
            "job_id": job_id,
            "created_at": _utc_now(),
            "version": resolved["version"],
            "dataset_snapshot_hash": resolved["dataset_snapshot_hash"],
            "config_hash": config["pipeline"]["trial_config_hash"],
            "execution": resolved_execution,
            "previous_job_id": previous_job_id,
            "artifacts": record_artifacts,
        }
        record_path = record_root / "run_record.json"
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return {
            **record["artifacts"],
            "运行档案": str(record_path),
            "comparison": str(comparison_path),
            "diff_summary": {
                **{key: value for key, value in diff_export.items() if key != "changes"},
                "changes": changes[:100],
                "displayed_change_count": min(len(changes), 100),
                "truncated": len(changes) > 100,
            },
        }

    def _write_evaluation_record(
        self,
        job_id: str,
        batch_id: str,
        resolved: dict[str, Any],
        config: dict[str, Any],
        config_path: Path,
    ) -> dict[str, Any]:
        record_root = self.run_records_root / job_id
        record_root.mkdir(parents=True, exist_ok=False)
        exported_config = record_root / "evaluation_config.resolved.yaml"
        exported_config.write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        derived = resolved.get("derived", {})
        output_root = derived.get("evaluation_output_root")
        record = {
            "schema_version": "chatts-dataset-studio-evaluation-record-v1",
            "job_id": job_id,
            "batch_id": batch_id,
            "created_at": _utc_now(),
            "model_path": derived.get("model_path"),
            "model_name": derived.get("model_name"),
            "run_id": derived.get("run_id"),
            "evaluation_protocol_id": derived.get("evaluation_protocol_id"),
            "config_hash": config["pipeline"]["trial_config_hash"],
            "execution": resolved.get("execution", {"backend": "docker_host"}),
            "output_root": output_root,
        }
        record_path = record_root / "evaluation_record.json"
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        artifacts = {
            "评测参数配置": str(exported_config),
            "被评测模型": str(derived.get("model_path", "")),
            "评测运行档案": str(record_path),
        }
        if isinstance(output_root, str) and output_root:
            artifacts.update(
                {
                    "评测输出目录": output_root,
                    "评测状态表": f"{output_root.rstrip('/')}/benchmark_status.tsv",
                    "评测汇总": f"{output_root.rstrip('/')}/all_benchmarks_summary.md",
                    "评测指标": f"{output_root.rstrip('/')}/metrics.json",
                }
            )
        return artifacts

    def _execution_spec(
        self, resolved: dict[str, Any]
    ) -> tuple[str, Path, Path | None]:
        execution = _mapping(resolved.get("execution"), "resolved.execution")
        backend = execution.get("backend", "docker_host")
        if backend == "docker_host":
            configured = execution.get("pipeline_script")
            script_path = (
                Path(configured).expanduser().resolve()
                if isinstance(configured, str) and configured
                else self.pipeline_script
            )
            allowed_scripts = {
                path
                for path in (self.pipeline_script, self.evaluation_pipeline_script)
                if path is not None
            }
            if script_path is None or script_path not in allowed_scripts:
                raise StudioError("Resolved Docker pipeline script is not server-trusted")
            if not script_path.is_file():
                raise StudioError(
                    "One-click pipeline is disabled or its fixed script does not exist on the host"
                )
            execution_root = execution.get("execution_root")
            execution_cwd = (
                Path(execution_root).expanduser().resolve()
                if isinstance(execution_root, str) and execution_root
                else script_path.parent.parent
            )
            if not execution_cwd.is_dir():
                raise StudioError(
                    f"Docker pipeline working directory is unavailable: {execution_cwd}"
                )
            return backend, execution_cwd, script_path
        elif backend == "slurm":
            if shutil.which("sbatch") is None:
                raise StudioError("Slurm backend is disabled because sbatch is unavailable")
            sbatch_path = Path(str(execution.get("sbatch_path", "")))
            sbatch_sha256 = execution.get("sbatch_sha256")
            try:
                current_sha256 = hashlib.sha256(sbatch_path.read_bytes()).hexdigest()
            except OSError as exc:
                raise StudioError(f"Cannot read trusted Slurm launcher: {exc}") from exc
            if current_sha256 != sbatch_sha256:
                raise StudioError("Trusted Slurm launcher changed after request resolution")
            execution_cwd = Path(
                str(
                    execution.get("execution_root")
                    or execution.get("training_root", "")
                )
            )
            if not execution_cwd.is_dir():
                raise StudioError(
                    f"Slurm execution root is unavailable: {execution_cwd}"
                )
            return backend, execution_cwd, None
        else:
            raise StudioError(f"Unknown execution backend: {backend}")

    def _create_job_locked(
        self,
        resolved: dict[str, Any],
        *,
        preflight: bool,
        batch_id: str | None,
        batch_index: int | None,
        batch_size: int | None,
        execution_spec: tuple[str, Path, Path | None],
        queue_sequence: int,
    ) -> str:
        backend, execution_cwd, docker_script = execution_spec
        execution = _mapping(resolved.get("execution"), "resolved.execution")
        standalone = resolved.get("task_type") == "standalone_evaluation"
        job_id = uuid.uuid4().hex
        self.config_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        config_path = self.config_root / f"{job_id}.yaml"
        log_path = self.logs_root / f"{job_id}.log"
        config = json.loads(json.dumps(resolved["config"]))
        config["pipeline"]["preflight_only"] = preflight
        config["pipeline"]["trial_id"] = job_id
        if standalone and batch_id is not None:
            config["pipeline"]["batch_id"] = batch_id
        effective_config_hash = _hash(config)
        config["pipeline"]["trial_config_hash"] = effective_config_hash
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        artifacts = {
            "评测参数配置" if standalone else "训练参数配置": str(config_path)
        }
        comparison_path = None
        diff_summary = None
        if not preflight:
            if standalone:
                if batch_id is None:
                    raise StudioError("Standalone evaluation job is missing its batch id")
                artifacts = self._write_evaluation_record(
                    job_id, batch_id, resolved, config, config_path
                )
            else:
                artifacts = self._write_run_record(
                    job_id, resolved, config, config_path
                )
                comparison_path = artifacts.pop("comparison")
                diff_summary = artifacts.pop("diff_summary")
        scheduler_stdout_path = (
            self.logs_root / f"{job_id}-slurm-%j.out"
            if backend == "slurm"
            else None
        )
        scheduler_stderr_path = (
            self.logs_root / f"{job_id}-slurm-%j.err"
            if backend == "slurm"
            else None
        )
        if backend == "slurm":
            artifacts["Slurm 提交脚本"] = str(execution["sbatch_path"])
            artifacts["Slurm 标准输出"] = str(scheduler_stdout_path)
            artifacts["Slurm 标准错误"] = str(scheduler_stderr_path)
        if standalone:
            kind = "evaluation_preflight" if preflight else "evaluation"
        elif preflight:
            kind = "preflight"
        elif resolved.get("pipeline_mode") == "stage1":
            kind = "train_eval_stage1"
        elif backend == "slurm":
            kind = "train_eval_full"
        else:
            kind = "train_eval"
        derived = resolved.get("derived", {})
        job = {
            "schema_version": "chatts-dataset-studio-job-v1",
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": _utc_now(),
            "queue_sequence": queue_sequence,
            "version": resolved.get("version"),
            "dataset_snapshot_hash": resolved.get("dataset_snapshot_hash"),
            "config_hash": effective_config_hash,
            "config_path": str(config_path),
            "comparison_path": comparison_path,
            "diff_from_previous": diff_summary,
            "log_path": str(log_path),
            "execution_backend": backend,
            "execution_cwd": str(execution_cwd),
            "derived": derived,
            "artifacts": artifacts,
        }
        if standalone:
            job.update(
                {
                    "batch_id": batch_id,
                    "batch_index": batch_index,
                    "batch_size": batch_size,
                    "model_path": derived.get("model_path"),
                    "model_name": derived.get("model_name"),
                    "run_id": derived.get("run_id"),
                }
            )
        if backend == "docker_host" and docker_script is not None:
            job["pipeline_script_path"] = str(docker_script)
        if backend == "slurm":
            job.update(
                {
                    "sbatch_path": str(execution["sbatch_path"]),
                    "sbatch_relative_path": execution.get("sbatch_relative_path"),
                    "sbatch_sha256": execution["sbatch_sha256"],
                    "scheduler_stdout_path": str(scheduler_stdout_path),
                    "scheduler_stderr_path": str(scheduler_stderr_path),
                }
            )
        self._write(job)
        return job_id

    def start(self, resolved: dict[str, Any], *, preflight: bool = False) -> dict[str, Any]:
        if resolved.get("task_type") == "standalone_evaluation":
            return self.start_many([resolved], preflight=preflight)["jobs"][0]
        execution_spec = self._execution_spec(resolved)
        with self.lock:
            queue_sequence = self._next_queue_sequence_locked()
            job_id = self._create_job_locked(
                resolved,
                preflight=preflight,
                batch_id=None,
                batch_index=None,
                batch_size=None,
                execution_spec=execution_spec,
                queue_sequence=queue_sequence,
            )
            self._dispatch_next_locked()
            return self.get(job_id, include_log=False)

    def start_many(
        self, resolved_items: list[dict[str, Any]], *, preflight: bool = False
    ) -> dict[str, Any]:
        if not resolved_items:
            raise StudioError("Standalone evaluation batch is empty")
        # Validate every executable, working directory and immutable launcher
        # hash before writing the first durable job.
        execution_specs = [self._execution_spec(item) for item in resolved_items]
        batch_id = uuid.uuid4().hex
        with self.lock:
            first_sequence = self._next_queue_sequence_locked()
            job_ids = [
                self._create_job_locked(
                    item,
                    preflight=preflight,
                    batch_id=batch_id,
                    batch_index=index,
                    batch_size=len(resolved_items),
                    execution_spec=execution_specs[index - 1],
                    queue_sequence=first_sequence + index - 1,
                )
                for index, item in enumerate(resolved_items, 1)
            ]
            self._dispatch_next_locked()
            return {
                "batch_id": batch_id,
                "job_count": len(job_ids),
                "jobs": [self.get(job_id, include_log=False) for job_id in job_ids],
            }

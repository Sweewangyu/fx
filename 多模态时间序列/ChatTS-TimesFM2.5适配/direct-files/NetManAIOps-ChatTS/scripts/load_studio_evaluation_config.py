#!/usr/bin/env python3
"""Validate a frozen standalone-evaluation YAML and emit safe env assignments.

The same validator is used by the Docker-host and Slurm entrypoints.  Output is
one literal ``NAME=value`` record per line; callers treat each line as data and
never evaluate it as shell source.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

PIPELINE_KEYS = frozenset(
    {
        "task_type",
        "seed",
        "force_eval",
        "preflight_only",
        "max_samples",
        "offline",
        "trial_id",
        "trial_config_hash",
        "batch_id",
    }
)
REQUIRED_PIPELINE_KEYS = PIPELINE_KEYS - {"batch_id"}
CONTAINER_KEYS = frozenset({"evaluation"})
EVALUATION_KEYS = frozenset(
    {
        "project_root",
        "script",
        "model_path",
        "model_name",
        "output_root",
        "chronos2_model_path",
        "tsrbench_root",
        "tinybench_dataset_root",
        "ts_haystack_root",
        "timeseriesexam_root",
        "timeseriesexam_data_file",
        "benchmarks",
        "run_id",
        "protocol_hash",
        "haystack_split",
        "tiny_data_partition",
        "tiny_partition_seed",
        "tsr_prompt_mode",
        "tsr_max_model_len",
        "tsr_max_new_tokens",
        "tsr_batch_size",
        "tsr_request_chunk_size",
        "tiny_max_model_len",
        "tiny_request_chunk_size",
        "tiny_gpu_memory_utilization",
        "haystack_max_model_len",
        "haystack_max_new_tokens",
        "haystack_batch_size",
        "haystack_request_chunk_size",
        "exam_max_model_len",
        "exam_max_new_tokens",
        "exam_batch_size",
        "exam_request_chunk_size",
    }
)
SLURM_KEYS = frozenset(
    {
        "evaluation_host_root",
        "evaluation_sif_image",
        "chronos2_host_root",
        "tsrbench_host_root",
        "tinybench_host_root",
        "ts_haystack_host_root",
        "timeseriesexam_host_root",
    }
)

KEY_TO_ENV = {
    "pipeline.task_type": "TASK_TYPE",
    "pipeline.seed": "SEED",
    "pipeline.force_eval": "FORCE_EVAL",
    "pipeline.preflight_only": "PREFLIGHT_ONLY",
    "pipeline.max_samples": "MAX_SAMPLES",
    "pipeline.offline": "OFFLINE",
    "pipeline.trial_id": "TRIAL_ID",
    "pipeline.trial_config_hash": "TRIAL_CONFIG_HASH",
    "pipeline.batch_id": "BATCH_ID",
    "containers.evaluation": "EVAL_CONTAINER",
    "evaluation.project_root": "EVAL_PROJECT_ROOT",
    "evaluation.script": "EVAL_SCRIPT",
    "evaluation.model_path": "EVAL_MODEL_PATH",
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
    "evaluation.tsr_prompt_mode": "TSR_PROMPT_MODE",
    "evaluation.tsr_max_model_len": "TSR_MAX_MODEL_LEN",
    "evaluation.tsr_max_new_tokens": "TSR_MAX_NEW_TOKENS",
    "evaluation.tsr_batch_size": "TSR_BATCH_SIZE",
    "evaluation.tsr_request_chunk_size": "TSR_REQUEST_CHUNK_SIZE",
    "evaluation.tiny_max_model_len": "TINY_MAX_MODEL_LEN",
    "evaluation.tiny_request_chunk_size": "TINY_REQUEST_CHUNK_SIZE",
    "evaluation.tiny_gpu_memory_utilization": "TINY_GPU_MEMORY_UTILIZATION",
    "evaluation.haystack_max_model_len": "HAYSTACK_MAX_MODEL_LEN",
    "evaluation.haystack_max_new_tokens": "HAYSTACK_MAX_NEW_TOKENS",
    "evaluation.haystack_batch_size": "HAYSTACK_BATCH_SIZE",
    "evaluation.haystack_request_chunk_size": "HAYSTACK_REQUEST_CHUNK_SIZE",
    "evaluation.exam_max_model_len": "EXAM_MAX_MODEL_LEN",
    "evaluation.exam_max_new_tokens": "EXAM_MAX_NEW_TOKENS",
    "evaluation.exam_batch_size": "EXAM_BATCH_SIZE",
    "evaluation.exam_request_chunk_size": "EXAM_REQUEST_CHUNK_SIZE",
    "slurm.evaluation_host_root": "CHATTS_EVALUATION_DIR",
    "slurm.evaluation_sif_image": "CHATTS_EVAL_SIF_IMAGE",
    "slurm.chronos2_host_root": "CHATTS_HOST_CHRONOS2_PATH",
    "slurm.tsrbench_host_root": "CHATTS_HOST_TSRBENCH_PATH",
    "slurm.tinybench_host_root": "CHATTS_HOST_TINYBENCH_PATH",
    "slurm.ts_haystack_host_root": "CHATTS_HOST_TS_HAYSTACK_PATH",
    "slurm.timeseriesexam_host_root": "CHATTS_HOST_TIMESERIESEXAM_PATH",
}

SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
SAFE_SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}")


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
    if value == "{}":
        return {}
    if value == "[]":
        return []
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
    path = PurePosixPath(result)
    if not result or not path.is_absolute() or path == PurePosixPath(path.root):
        raise ValueError(f"{field} must be an absolute non-root POSIX path")
    return result


def require_int(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def validate_and_resolve(
    payload: dict[str, Any], expected_job_id: str | None = None
) -> dict[str, str]:
    unknown_top = sorted(set(payload) - {"pipeline", "containers", "evaluation", "slurm"})
    missing_top = sorted({"pipeline", "containers", "evaluation"} - set(payload))
    if unknown_top or missing_top:
        raise ValueError(
            f"top-level fields mismatch; missing={missing_top}, unknown={unknown_top}"
        )
    pipeline = mapping(payload, "pipeline")
    containers = mapping(payload, "containers")
    evaluation = mapping(payload, "evaluation")
    slurm_value = payload.get("slurm", {})
    if not isinstance(slurm_value, dict):
        raise TypeError("slurm must be a mapping")
    slurm = slurm_value
    reject_unknown(pipeline, PIPELINE_KEYS, "pipeline")
    reject_unknown(containers, CONTAINER_KEYS, "containers")
    reject_unknown(evaluation, EVALUATION_KEYS, "evaluation")
    reject_unknown(slurm, SLURM_KEYS, "slurm")

    missing_pipeline = sorted(REQUIRED_PIPELINE_KEYS - set(pipeline))
    missing_evaluation = sorted(EVALUATION_KEYS - set(evaluation))
    if missing_pipeline:
        raise ValueError(f"pipeline is missing fields: {', '.join(missing_pipeline)}")
    if set(containers) != CONTAINER_KEYS:
        raise ValueError("containers must contain exactly evaluation")
    if missing_evaluation:
        raise ValueError(f"evaluation is missing fields: {', '.join(missing_evaluation)}")

    if pipeline["task_type"] != "standalone_evaluation":
        raise ValueError("pipeline.task_type must be standalone_evaluation")
    for key in ("force_eval", "preflight_only", "offline"):
        if not isinstance(pipeline[key], bool):
            raise TypeError(f"pipeline.{key} must be a boolean")
    require_int(pipeline["seed"], "pipeline.seed", minimum=0)
    require_int(pipeline["max_samples"], "pipeline.max_samples", minimum=0)

    trial_id = shell_value(pipeline["trial_id"], "pipeline.trial_id")
    if not SAFE_SLUG_RE.fullmatch(trial_id):
        raise ValueError("pipeline.trial_id must be a safe slug")
    if expected_job_id is not None and trial_id != expected_job_id:
        raise ValueError(
            "pipeline.trial_id does not match submitted Studio job: "
            f"{trial_id!r} != {expected_job_id!r}"
        )
    batch_id = pipeline.get("batch_id")
    if batch_id is not None and not SAFE_SLUG_RE.fullmatch(
        shell_value(batch_id, "pipeline.batch_id")
    ):
        raise ValueError("pipeline.batch_id must be a safe slug")

    expected_hash = shell_value(
        pipeline["trial_config_hash"], "pipeline.trial_config_hash"
    ).lower()
    if not SHA256_RE.fullmatch(expected_hash):
        raise ValueError("pipeline.trial_config_hash must be a 64-character SHA256")
    hash_payload = json.loads(json.dumps(payload))
    del hash_payload["pipeline"]["trial_config_hash"]
    actual_hash = canonical_hash(hash_payload)
    if actual_hash != expected_hash:
        raise ValueError(
            "pipeline.trial_config_hash does not match the frozen resolved YAML: "
            f"{actual_hash} != {expected_hash}"
        )

    container = shell_value(containers["evaluation"], "containers.evaluation")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", container):
        raise ValueError("containers.evaluation must be a safe container name")

    absolute_fields = (
        "project_root",
        "script",
        "model_path",
        "output_root",
        "chronos2_model_path",
        "tsrbench_root",
        "tinybench_dataset_root",
        "ts_haystack_root",
        "timeseriesexam_root",
        "timeseriesexam_data_file",
    )
    paths = {
        key: require_absolute(evaluation[key], f"evaluation.{key}")
        for key in absolute_fields
    }
    if PurePosixPath(paths["script"]).name != "run_all_chatts_benchmarks.sh":
        raise ValueError("evaluation.script must name run_all_chatts_benchmarks.sh")
    if PurePosixPath(paths["project_root"]) not in PurePosixPath(paths["script"]).parents:
        raise ValueError("evaluation.script must be inside evaluation.project_root")
    if PurePosixPath(paths["timeseriesexam_root"]) not in PurePosixPath(
        paths["timeseriesexam_data_file"]
    ).parents:
        raise ValueError(
            "evaluation.timeseriesexam_data_file must be inside timeseriesexam_root"
        )

    for key in ("model_name", "run_id"):
        value = shell_value(evaluation[key], f"evaluation.{key}")
        if not SAFE_SLUG_RE.fullmatch(value):
            raise ValueError(f"evaluation.{key} must be a safe slug")
    protocol_hash = shell_value(
        evaluation["protocol_hash"], "evaluation.protocol_hash"
    ).lower()
    if not SHA256_RE.fullmatch(protocol_hash):
        raise ValueError("evaluation.protocol_hash must be a 64-character SHA256")
    protocol_id = f"protocol-{protocol_hash[:16]}"
    run_id = shell_value(evaluation["run_id"], "evaluation.run_id")
    if not run_id.endswith(f"-{protocol_id}-eval"):
        raise ValueError(f"evaluation.run_id must end in -{protocol_id}-eval")
    if PurePosixPath(paths["output_root"]).name != protocol_id:
        raise ValueError(f"evaluation.output_root must end in {protocol_id}")

    benchmarks = shell_value(evaluation["benchmarks"], "evaluation.benchmarks")
    requested = benchmarks.split(",")
    allowed = {"tsrbench", "tinybenchmarks", "ts_haystack", "timeseriesexam"}
    if not requested or any(item not in allowed for item in requested):
        raise ValueError("evaluation.benchmarks contains an unsupported benchmark")
    if len(requested) != len(set(requested)):
        raise ValueError("evaluation.benchmarks contains duplicates")
    if evaluation["haystack_split"] not in {"train", "validation", "test"}:
        raise ValueError("evaluation.haystack_split is invalid")
    if evaluation["tiny_data_partition"] not in {"all", "search-dev", "final-test"}:
        raise ValueError("evaluation.tiny_data_partition is invalid")
    if evaluation["tsr_prompt_mode"] not in {
        "answer_only",
        "official",
        "json_reasoning",
    }:
        raise ValueError("evaluation.tsr_prompt_mode is invalid")

    integer_fields = (
        "tiny_partition_seed",
        "tsr_max_model_len",
        "tsr_max_new_tokens",
        "tsr_batch_size",
        "tsr_request_chunk_size",
        "tiny_max_model_len",
        "tiny_request_chunk_size",
        "haystack_max_model_len",
        "haystack_max_new_tokens",
        "haystack_batch_size",
        "haystack_request_chunk_size",
        "exam_max_model_len",
        "exam_max_new_tokens",
        "exam_batch_size",
        "exam_request_chunk_size",
    )
    for key in integer_fields:
        minimum = 0 if key == "tiny_partition_seed" else 1
        require_int(evaluation[key], f"evaluation.{key}", minimum=minimum)
    for prefix in ("tsr", "haystack", "exam"):
        if evaluation[f"{prefix}_max_model_len"] <= evaluation[f"{prefix}_max_new_tokens"]:
            raise ValueError(
                f"evaluation.{prefix}_max_model_len must exceed max_new_tokens"
            )
    try:
        tiny_memory = float(evaluation["tiny_gpu_memory_utilization"])
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation.tiny_gpu_memory_utilization must be numeric") from exc
    if not 0 < tiny_memory <= 1:
        raise ValueError("evaluation.tiny_gpu_memory_utilization must be in (0, 1]")

    for key, value in slurm.items():
        require_absolute(value, f"slurm.{key}")

    flattened: dict[str, Any] = {}
    for section_name, section in (
        ("pipeline", pipeline),
        ("containers", containers),
        ("evaluation", evaluation),
        ("slurm", slurm),
    ):
        for key, value in section.items():
            flattened[f"{section_name}.{key}"] = value
    return {
        KEY_TO_ENV[key]: shell_value(value, key)
        for key, value in flattened.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--expected-job-id")
    args = parser.parse_args()
    path = args.config_file.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Configuration file not found: {path}")
    try:
        resolved = validate_and_resolve(load_yaml(path), args.expected_job_id)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid standalone evaluation YAML {path}: {exc}") from exc
    for name in sorted(resolved):
        print(f"{name}={resolved[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

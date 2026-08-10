"""Evaluate a ChatTS checkpoint on the official TS-Haystack datasets.

This adapter intentionally imports TS-Haystack's own QADataset classes.  The
official code is responsible for hydrating/reconstructing signals, formatting
the benchmark prompt, extracting typed answers, and scoring IoU/timestamps.
ChatTS only replaces the model inference path and receives the unmodified
one-dimensional channels through vLLM's ``timeseries`` modality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Keep this import at module scope.  vLLM uses spawn, and every child must run
# ChatTS model/processor registration before constructing an engine.
import chatts.vllm.chatts_vllm as _chatts_vllm  # noqa: F401,E402


@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    registry_name: str
    label_class: str | None = None
    channel_count: int = 1
    source_hz: float | None = None


DATASET_SPECS = {
    "capture24": DatasetSpec(
        "capture24", "capture24_haystack_cot", channel_count=3, source_hz=100.0
    ),
    "sleep_stages": DatasetSpec(
        "sleep_stages",
        "sleep_psg_haystack",
        label_class="sleep_stages",
        channel_count=13,
        source_hz=100.0,
    ),
    "sleep_arousals": DatasetSpec(
        "sleep_arousals",
        "sleep_psg_haystack",
        label_class="arousals",
        channel_count=13,
        source_hz=100.0,
    ),
    "ltaf": DatasetSpec(
        "ltaf", "ltaf_haystack", channel_count=2, source_hz=128.0
    ),
    "uk_dale": DatasetSpec("uk_dale", "uk_dale_haystack", channel_count=1),
}

DATASET_ALIASES = {
    "capture24_haystack": "capture24",
    "capture24_haystack_cot": "capture24",
    "sleep_psg_stages": "sleep_stages",
    "sleep_psg_arousals": "sleep_arousals",
    "ltaf_haystack": "ltaf",
    "uk-dale": "uk_dale",
    "uk_dale_haystack": "uk_dale",
}

RESUMABLE_STATUSES = {"ok", "skipped_input_length"}
VLLM_REJECTED_PREFIX = "VLLM_INPUT_REJECTED:"


def resolve_dataset_specs(requested: Iterable[str]) -> list[DatasetSpec]:
    names = [str(value).strip().lower() for value in requested if str(value).strip()]
    if not names or "all" in names:
        names = list(DATASET_SPECS)

    resolved: list[DatasetSpec] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for name in names:
        canonical = DATASET_ALIASES.get(name, name)
        if canonical not in DATASET_SPECS:
            unknown.append(name)
            continue
        if canonical not in seen:
            resolved.append(DATASET_SPECS[canonical])
            seen.add(canonical)
    if unknown:
        raise ValueError(
            f"Unknown TS-Haystack dataset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(DATASET_SPECS)}"
        )
    return resolved


def normalize_context_lengths(values: Iterable[str]) -> list[str | int | float]:
    normalized: list[str | int | float] = []
    for value in values:
        raw = str(value).strip().lower()
        if not raw:
            continue
        if raw in {"all", "full"}:
            normalized.append(raw)
            continue
        if raw.endswith("s"):
            raw = raw[:-1]
        raw = raw.replace("_", ".")
        try:
            number = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid context length: {value!r}") from exc
        normalized.append(int(number) if number.is_integer() else number)
    return normalized or ["all"]


def _validate_ts_haystack_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    required = root / "src" / "datasets" / "registry.py"
    if not required.is_file():
        raise FileNotFoundError(
            f"TS-Haystack source tree not found at {root}. Expected {required}. "
            "TS_HAYSTACK_ROOT must point to a clone of AI-X-Labs/TS-Haystack, "
            "not just to one downloaded parquet directory."
        )
    return root


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _clear_official_dataset_cache(dataset_cls: type[Any]) -> None:
    """Avoid cross-contamination between official dataset configurations.

    In particular, the current SleepPSG class cache key does not contain
    ``label_class``.  Clearing the class-owned caches before construction lets
    stages and arousals safely run sequentially in one model process.
    """
    for attr in ("_raw_split_cache", "_formatted_split_cache"):
        if attr in dataset_cls.__dict__:
            setattr(dataset_cls, attr, {})
    if "_cached_config" in dataset_cls.__dict__:
        setattr(dataset_cls, "_cached_config", None)


def load_official_dataset(
    get_dataset_class: Any,
    spec: DatasetSpec,
    *,
    split: str,
    tasks: list[str],
    context_lengths: list[str | int | float],
) -> Any:
    dataset_cls = get_dataset_class(spec.registry_name)
    _clear_official_dataset_cache(dataset_cls)
    kwargs: dict[str, Any] = {
        "split": split,
        "EOS_TOKEN": "",
        "tasks": tasks,
        "context_lengths_seconds": context_lengths,
    }
    if spec.label_class:
        kwargs["label_class"] = spec.label_class
    dataset = dataset_cls(**kwargs)
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"Official TS-Haystack loader returned zero {split} samples for "
            f"{spec.alias}. Check tasks/context filters and the data tree under "
            "TS_HAYSTACK_ROOT/data/."
        )
    return dataset


def _as_series(value: Any, *, label: str) -> np.ndarray:
    series = np.asarray(value, dtype=np.float64).squeeze()
    if series.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got {series.shape}")
    if series.size == 0:
        raise ValueError(f"{label} is empty")
    if not np.isfinite(series).all():
        raise ValueError(f"{label} contains NaN or infinity")
    return series


def build_official_prompt(sample: dict[str, Any]) -> tuple[str, list[np.ndarray]]:
    """Insert ChatTS markers between official pre/channel/post prompt chunks."""
    labels = list(sample.get("time_series_text") or [])
    raw_series = list(sample.get("time_series") or [])
    if not labels or len(labels) != len(raw_series):
        raise ValueError(
            "Official sample has mismatched time_series_text/time_series fields: "
            f"{len(labels)} labels vs {len(raw_series)} channels"
        )

    prompt = str(sample.get("pre_prompt", "")).strip()
    series: list[np.ndarray] = []
    for index, (label, values) in enumerate(zip(labels, raw_series)):
        series.append(_as_series(values, label=f"channel {index + 1}"))
        prompt += f"\n{str(label).strip()} <ts><ts/>"
    post_prompt = str(sample.get("post_prompt", "")).strip()
    if post_prompt:
        prompt += f"\n{post_prompt}"
    if prompt.count("<ts><ts/>") != len(series):
        raise ValueError("Internal placeholder/channel count mismatch")
    return prompt, series


def apply_chat_template(
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str,
    enable_thinking: bool,
) -> str:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(conversation, **kwargs)
    except TypeError:
        # Transformers/tokenizer versions predating Qwen3 do not expose the
        # enable_thinking keyword.  Their template has no Qwen3 thinking knob.
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(conversation, **kwargs)


def estimate_vllm_prompt_tokens(
    tokenizer: Any,
    templated_prompt: str,
    series: list[np.ndarray],
    patch_size: int,
) -> int:
    """Mirror ChatTS processor expansion of every ``<ts><ts/>`` marker."""
    if patch_size < 1:
        raise ValueError(f"Invalid time-series patch size: {patch_size}")
    text_tokens = len(tokenizer.encode(templated_prompt, add_special_tokens=False))
    total_tokens = text_tokens - 2 * len(series)
    for values in series:
        mean = float(np.mean(values))
        centered = values - mean
        scale = 1.0
        if np.any(np.abs(centered) >= 3.0):
            scale = float(np.max(np.abs(centered)) / 3.0)
        metadata = (
            f"[offset={-mean:.4f}|scaling={scale:.4f}|length={len(values)}|"
            f"max={float(np.max(values)):.4f}|min={float(np.min(values)):.4f}|"
            f"left={float(values[0]):.4f}|right={float(values[-1]):.4f}]<ts>"
        )
        metadata_tokens = len(tokenizer.encode(metadata, add_special_tokens=False))
        patch_tokens = math.ceil(len(values) / patch_size)
        total_tokens += metadata_tokens + max(0, patch_tokens - 1)
    return total_tokens


def _raw_row(dataset: Any, index: int) -> dict[str, Any] | None:
    raw_dataset = getattr(dataset, "_raw_dataset", None)
    if raw_dataset is None:
        return None
    row = raw_dataset[index]
    return dict(row) if row is not None else None


def infer_series_shape(
    raw: dict[str, Any] | None, spec: DatasetSpec
) -> tuple[int, int] | None:
    """Infer (channels, points-per-channel) before hydrating a huge signal."""
    if not raw:
        return None
    points: int | None = None
    if spec.alias == "capture24":
        value = raw.get("context_length_samples")
        if value is not None:
            points = int(value)
    elif spec.alias.startswith("sleep_"):
        start_ms = raw.get("window_start_ms")
        end_ms = raw.get("window_end_ms")
        if start_ms is not None and end_ms is not None:
            points = int(round((float(end_ms) - float(start_ms)) / 1000.0 * 100.0))
    elif spec.alias == "ltaf":
        start_ms = raw.get("window_start_ms")
        end_ms = raw.get("window_end_ms")
        hz = float(raw.get("source_hz") or spec.source_hz or 128.0)
        if start_ms is not None and end_ms is not None:
            points = int(round((float(end_ms) - float(start_ms)) / 1000.0 * hz))
    elif spec.alias == "uk_dale":
        context_s = raw.get("context_length_s")
        dt_s = raw.get("dt_s", 6.0)
        if context_s is not None and float(dt_s) > 0:
            points = int(round(float(context_s) / float(dt_s)))
    if points is None or points < 1:
        return None
    return spec.channel_count, points


def _context_seconds(sample: dict[str, Any], spec: DatasetSpec) -> float | str | None:
    if sample.get("context_length_s") is not None:
        value = sample["context_length_s"]
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)
    if sample.get("context_length_samples") is not None and spec.source_hz:
        return float(sample["context_length_samples"]) / spec.source_hz
    if sample.get("window_start_ms") is not None and sample.get("window_end_ms") is not None:
        return (float(sample["window_end_ms"]) - float(sample["window_start_ms"])) / 1000.0
    return None


def _base_result(
    *,
    index: int,
    spec: DatasetSpec,
    split: str,
    source: dict[str, Any] | None,
) -> dict[str, Any]:
    source = source or {}
    return {
        "idx": index,
        "dataset": spec.alias,
        "official_dataset": spec.registry_name,
        "split": split,
        "task_type": str(source.get("task_type", "unknown")),
        "answer_type": str(source.get("answer_type", "unknown")),
        "context_length_s": _context_seconds(source, spec),
        "question": str(source.get("question", "")),
        "ground_truth": str(source.get("direct_answer") or source.get("answer", "")),
    }


def _safe_model_name(model_path: str, requested: str | None) -> str:
    if requested and not re.fullmatch(r"[A-Za-z0-9_.-]+", requested):
        raise ValueError(
            "--model-name may contain only letters, digits, dot, underscore, and dash"
        )
    value = requested or Path(model_path.rstrip("/")).name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "chatts"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_dump(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(rows), stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _load_existing(path: Path, protocol: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    completed: dict[int, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict) or "idx" not in item:
            continue
        if item.get("protocol") != protocol:
            continue
        if item.get("status") in RESUMABLE_STATUSES:
            completed[int(item["idx"])] = item
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChatTS vLLM inference on TS-Haystack")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--ts-haystack-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--context-lengths", nargs="+", default=["all"])
    parser.add_argument("--output-root", default="evaluation/results/chatts")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpus-per-model", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--request-chunk-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-processed-input-tokens", type=int, default=0)
    parser.add_argument("--max-mm-per-prompt", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_gpus < 1 or args.gpus_per_model < 1:
        raise ValueError("GPU counts must be positive")
    if args.num_gpus % args.gpus_per_model:
        raise ValueError("--num-gpus must be divisible by --gpus-per-model")
    if args.batch_size < 1 or args.request_chunk_size < 1:
        raise ValueError("batch and chunk sizes must be positive")
    if args.max_new_tokens < 1 or args.max_processed_input_tokens < 0:
        raise ValueError("token limits are invalid")

    ts_root = _validate_ts_haystack_root(Path(args.ts_haystack_root))
    ts_haystack_revision = _git_revision(ts_root)
    model_path = str(Path(args.model_path).expanduser().resolve())
    model_name = _safe_model_name(model_path, args.model_name)
    specs = resolve_dataset_specs(args.datasets)
    context_lengths = normalize_context_lengths(args.context_lengths)
    tasks = [str(value).strip() for value in args.tasks if str(value).strip()] or ["all"]
    output_root = Path(args.output_root).expanduser().resolve()

    # Official loaders use relative data paths.  Keep their repository as CWD
    # for the full run, and place it first on sys.path before importing `src`.
    os.chdir(ts_root)
    sys.path.insert(0, str(ts_root))
    try:
        from src.datasets.registry import get_dataset_class
    except Exception as exc:
        raise RuntimeError(
            "Could not import TS-Haystack's official dataset registry. Install "
            "its dataset runtime dependencies into the same Python environment "
            "that runs ChatTS (datasets, pyarrow, pandas, scipy, wfdb, and "
            "matplotlib). The adapter imports the source checkout directly."
        ) from exc

    datasets: list[tuple[DatasetSpec, Any, int]] = []
    for spec in specs:
        dataset = load_official_dataset(
            get_dataset_class,
            spec,
            split=args.split,
            tasks=tasks,
            context_lengths=context_lengths,
        )
        limit = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
        datasets.append((spec, dataset, limit))
        print(f"[TS-Haystack] {spec.alias}: loaded {len(dataset)}; selected {limit}")

    from chatts.utils.llm_utils import LLMClient
    from transformers import AutoConfig
    from vllm import SamplingParams

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    patch_size = _chatts_vllm.get_time_series_patch_size(model_config)
    print(f"[TS-Haystack] time-series encoder patch_size={patch_size}")
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        seed=args.seed,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    client = LLMClient(
        model_path=model_path,
        engine="vllm-ts",
        num_gpus=args.num_gpus,
        gpus_per_model=args.gpus_per_model,
        batch_size=args.batch_size,
        system_prompt=args.system_prompt,
    )

    try:
        client.wait_for_ready()
        for spec, dataset, selected_size in datasets:
            protocol = {
                "version": 1,
                "model_path": model_path,
                "official_dataset": spec.registry_name,
                "official_code_revision": ts_haystack_revision,
                "label_class": spec.label_class,
                "split": args.split,
                "tasks": tasks,
                "context_lengths": context_lengths,
                "enable_thinking": bool(args.enable_thinking),
                "system_prompt": args.system_prompt,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "patch_size": patch_size,
                "max_processed_input_tokens": args.max_processed_input_tokens,
                "official_dataset_size": len(dataset),
                "selected_dataset_size": selected_size,
            }
            result_path = output_root / f"{spec.alias}_{model_name}" / "generated_answer.json"
            completed = {} if args.force else _load_existing(result_path, protocol)
            pending = [index for index in range(selected_size) if index not in completed]
            print(
                f"[TS-Haystack] {spec.alias}: total={selected_size}, "
                f"completed={len(completed)}, pending={len(pending)}"
            )

            for start in range(0, len(pending), args.request_chunk_size):
                indices = pending[start : start + args.request_chunk_size]
                prompts: list[str] = []
                templated_prompts: list[str] = []
                series_batch: list[list[np.ndarray]] = []
                samples: list[dict[str, Any]] = []
                valid_indices: list[int] = []
                token_counts: list[tuple[int, int]] = []

                for index in indices:
                    raw = _raw_row(dataset, index)
                    shape = infer_series_shape(raw, spec)
                    if shape is not None and args.max_processed_input_tokens > 0:
                        channels, points = shape
                        patch_lower_bound = channels * math.ceil(points / patch_size)
                        if patch_lower_bound > args.max_processed_input_tokens:
                            reason = (
                                f"time-series patch tokens >= {patch_lower_bound} exceed "
                                f"processed input limit {args.max_processed_input_tokens}"
                            )
                            result = _base_result(
                                index=index, spec=spec, split=args.split, source=raw
                            )
                            result.update(
                                {
                                    "status": "skipped_input_length",
                                    "error": reason,
                                    "channel_count": channels,
                                    "points_per_channel": points,
                                    "patch_token_lower_bound": patch_lower_bound,
                                    "protocol": protocol,
                                }
                            )
                            completed[index] = result
                            print(f"[TS-Haystack] {spec.alias}[{index}] skipped: {reason}")
                            continue

                    try:
                        sample = dataset[index]
                        prompt, series = build_official_prompt(sample)
                        if len(series) > args.max_mm_per_prompt:
                            raise ValueError(
                                f"sample has {len(series)} channels; "
                                f"limit is {args.max_mm_per_prompt}"
                            )
                        templated = apply_chat_template(
                            client.tokenizer,
                            prompt,
                            system_prompt=args.system_prompt,
                            enable_thinking=args.enable_thinking,
                        )
                        text_tokens = len(
                            client.tokenizer.encode(templated, add_special_tokens=False)
                        )
                        processed_tokens = estimate_vllm_prompt_tokens(
                            client.tokenizer, templated, series, patch_size
                        )
                        if (
                            args.max_processed_input_tokens > 0
                            and processed_tokens > args.max_processed_input_tokens
                        ):
                            reason = (
                                f"processed input tokens {processed_tokens} exceed "
                                f"limit {args.max_processed_input_tokens}"
                            )
                            result = _base_result(
                                index=index, spec=spec, split=args.split, source=sample
                            )
                            result.update(
                                {
                                    "status": "skipped_input_length",
                                    "error": reason,
                                    "channel_count": len(series),
                                    "series_lengths": [len(values) for values in series],
                                    "text_input_tokens": text_tokens,
                                    "processed_input_tokens": processed_tokens,
                                    "protocol": protocol,
                                }
                            )
                            completed[index] = result
                            print(f"[TS-Haystack] {spec.alias}[{index}] skipped: {reason}")
                            continue
                    except Exception as exc:
                        result = _base_result(
                            index=index, spec=spec, split=args.split, source=raw
                        )
                        result.update(
                            {
                                "status": "preparation_error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "protocol": protocol,
                            }
                        )
                        completed[index] = result
                        print(f"[TS-Haystack] {spec.alias}[{index}] error: {exc}")
                        continue

                    prompts.append(prompt)
                    templated_prompts.append(templated)
                    series_batch.append(series)
                    samples.append(sample)
                    valid_indices.append(index)
                    token_counts.append((text_tokens, processed_tokens))

                if templated_prompts:
                    responses = client.llm_batch_generate(
                        templated_prompts,
                        series_batch,
                        use_chat_template=False,
                        sampling_params=sampling_params,
                    )
                    for index, prompt, sample, series, counts, response in zip(
                        valid_indices,
                        prompts,
                        samples,
                        series_batch,
                        token_counts,
                        responses,
                    ):
                        text_tokens, processed_tokens = counts
                        result = _base_result(
                            index=index, spec=spec, split=args.split, source=sample
                        )
                        result.update(
                            {
                                "prompt": prompt,
                                "raw_response": response or "",
                                "channel_count": len(series),
                                "series_lengths": [len(values) for values in series],
                                "text_input_tokens": text_tokens,
                                "processed_input_tokens": processed_tokens,
                                "protocol": protocol,
                            }
                        )
                        if not response or str(response).startswith(VLLM_REJECTED_PREFIX):
                            result.update(
                                {
                                    "status": "generation_error",
                                    "predicted_answer": "",
                                    "correct": False,
                                    "error": response or "empty model response",
                                }
                            )
                        else:
                            try:
                                predicted = dataset.extract_answer(str(response), sample)
                                evaluation = dataset.evaluate_answer(predicted, sample)
                                result.update(
                                    {
                                        "status": "ok",
                                        "ground_truth": dataset.get_ground_truth(sample),
                                        "predicted_answer": predicted,
                                        "correct": bool(evaluation.get("correct", False)),
                                        "evaluation": _json_safe(evaluation),
                                    }
                                )
                            except Exception as exc:
                                result.update(
                                    {
                                        "status": "evaluation_error",
                                        "predicted_answer": "",
                                        "correct": False,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )
                        completed[index] = result

                ordered = [completed[index] for index in sorted(completed)]
                _atomic_dump(ordered, result_path)
                print(
                    f"[TS-Haystack] {spec.alias}: saved {len(ordered)}/{selected_size} "
                    f"to {result_path}"
                )
    finally:
        client.kill()


if __name__ == "__main__":
    main()

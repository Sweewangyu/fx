"""Run TimeSeriesExam through ChatTS' native vLLM time-series modality.

The benchmark question, choices, hints, concepts, fixed one-shot example, and
generation settings follow the official TimeSeriesExam implementation.  The
only modality-specific change is replacing comma-separated values/images with
one ``<ts><ts/>`` marker per raw numeric series.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# vLLM uses spawn; every worker must register ChatTS before engine creation.
import chatts.vllm.chatts_vllm as _chatts_vllm  # noqa: F401,E402


VLLM_REJECTED_PREFIX = "VLLM_INPUT_REJECTED:"
RESUMABLE_STATUSES = {"ok", "skipped_input_length"}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read the official GitHub JSON or Hugging Face JSONL/Parquet export."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TimeSeriesExam data file not found: {path}")
    if path.suffix.lower() in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as stream:
            if path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in stream if line.strip()]
            else:
                payload = json.load(stream)
                rows = payload if isinstance(payload, list) else payload.get("data")
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return [_plain(row) for row in rows]
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Reading the Hugging Face Parquet export requires pandas and a "
                "Parquet engine. Alternatively use the official qa_dataset.json."
            ) from exc
        frame = pd.read_parquet(path)
        return [_plain(row) for row in frame.to_dict(orient="records")]
    raise ValueError(f"Unsupported TimeSeriesExam format: {path.suffix}")


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("TimeSeriesExam data file is empty")
    required = {"question", "options", "answer", "category", "subcategory"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Not a TimeSeriesExam data file; missing fields: {missing}")
    for index, row in enumerate(rows):
        options = row.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= 26:
            raise ValueError(f"row {index}: options must contain 2-26 entries")
        if row.get("answer") not in options:
            raise ValueError(f"row {index}: answer is absent from options")
        _main_series(row)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_concepts(root: Path) -> dict[str, Any]:
    """Load CONCEPTS without importing the official evaluator's API packages."""
    source = root.expanduser().resolve() / "evaluate" / "concepts.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"Official concept definitions not found: {source}. Set "
            "TIMESERIESEXAM_ROOT to the official repository clone, or disable "
            "ADD_CONCEPTS and ADD_EXAMPLES."
        )
    name = "_chatts_timeseriesexam_official_concepts"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official concept definitions: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    concepts = getattr(module, "CONCEPTS", None)
    if not isinstance(concepts, dict):
        raise ValueError(f"Official {source} does not define a CONCEPTS mapping")
    return concepts


def _as_series(value: Any, *, label: str) -> np.ndarray:
    series = np.asarray(value, dtype=np.float64).squeeze()
    if series.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got {series.shape}")
    if series.size == 0:
        raise ValueError(f"{label} is empty")
    if not np.isfinite(series).all():
        raise ValueError(f"{label} contains NaN or infinity")
    return series


def _present_series(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
        return np.asarray(value).size > 0
    except Exception:
        return False


def _main_series(sample: dict[str, Any]) -> list[np.ndarray]:
    if _present_series(sample.get("ts")):
        return [_as_series(sample["ts"], label="time series")]
    if _present_series(sample.get("ts1")) and _present_series(sample.get("ts2")):
        return [
            _as_series(sample["ts1"], label="time series 1"),
            _as_series(sample["ts2"], label="time series 2"),
        ]
    raise ValueError("sample has neither ts nor a complete ts1/ts2 pair")


def _official_query(
    sample: dict[str, Any],
    concepts: dict[str, Any] | None,
    *,
    add_question_hint: bool,
    add_concepts: bool,
    add_examples: bool,
    max_concepts: int,
) -> tuple[str, list[list[np.ndarray]]]:
    """Reproduce TimeSeriesQADataset.__getitem__ and return concept examples."""
    options_string = "\n".join(
        f"{chr(65 + index)}) {option}"
        for index, option in enumerate(sample["options"])
    )
    prompt = (
        f"{sample['question']}\n        \n"
        "        Choose From Following Options: \n        \n"
        f"        {options_string}\n"
    )
    example_groups: list[list[np.ndarray]] = []
    if add_concepts:
        if concepts is None:
            raise ValueError("concept definitions were not loaded")
        concept_strings = []
        for concept_index, concept_key in enumerate(
            list(sample.get("relevant_concepts") or [])[:max_concepts]
        ):
            if concept_key not in concepts:
                raise KeyError(f"Unknown official concept: {concept_key!r}")
            concept = concepts[concept_key]
            if add_examples:
                concept_strings.append(
                    f"{concept.concept_name}: {concept.concept_description}. "
                    f"{concept.concept_example_string}, marked as example "
                    f"{concept_index + 1}."
                )
                raw_group = concept.concept_example
                if not isinstance(raw_group, (list, tuple)):
                    raw_group = [raw_group]
                example_groups.append(
                    [
                        _as_series(values, label=f"concept example {concept_index + 1}")
                        for values in raw_group
                    ]
                )
            else:
                concept_strings.append(
                    f"{concept.concept_name}: {concept.concept_description}."
                )
        prompt += "Here are some relevant concepts: \n            "
        prompt += "\n".join(concept_strings) + "\n"

    if add_question_hint:
        prompt += f"Here is a hint that might help you: {sample.get('question_hint', '')}."

    prompt += f"""{sample.get('format_hint', '')}.

        Here is an example question answer pair to help you understand the format better:

        EXAMPLE QUESTION:

        What is the most likely autocorrelation at lag 1 for the given time series?

        Choose From Following Options:

        A) High positive autocorrelation
B) No autocorrelation
C) Negative autocorrelation
Now, answer the question.

        EXAMPLE RESPONSE:

        Based on the given time series, the data points appear to fluctuate randomly around the mean with no clear pattern of persistence or trend. This suggests that the time series does not exhibit a strong relationship between consecutive data points.

Given the options:

A) High positive autocorrelation
B) No autocorrelation
C) Negative autocorrelation

The most likely autocorrelation at lag 1 for the given time series is:

B) No autocorrelation

        Now, answer the given question and also explain your thought process: """
    return prompt, example_groups


def prepare_sample(
    sample: dict[str, Any],
    concepts: dict[str, Any] | None = None,
    *,
    add_question_hint: bool = True,
    add_concepts: bool = True,
    add_examples: bool = True,
    max_concepts: int = 3,
) -> tuple[str, list[np.ndarray]]:
    """Build the official one-shot prompt with ChatTS time-series markers."""
    main_series = _main_series(sample)
    query, example_groups = _official_query(
        sample,
        concepts,
        add_question_hint=add_question_hint,
        add_concepts=add_concepts,
        add_examples=add_examples,
        max_concepts=max_concepts,
    )

    series: list[np.ndarray] = []
    if len(main_series) == 1:
        if example_groups:
            base = (
                "You are given one time series. Some examples of the relevant "
                "concepts mentioned below are also provided.\n"
                "Concept example time series:\n"
            )
        else:
            base = "You are given one time series.\n"
    else:
        if example_groups:
            base = (
                "You are given two time series. Some examples of the relevant "
                "concepts mentioned below are also provided.\n"
                "Concept example time series:\n"
            )
        else:
            base = "You are given two time series.\n"

    for example_index, group in enumerate(example_groups, 1):
        if len(group) == 1:
            base += f"Example {example_index} time series: <ts><ts/>\n"
        else:
            for channel_index, _ in enumerate(group, 1):
                base += (
                    f"Example {example_index} time series {channel_index}: "
                    "<ts><ts/>\n"
                )
        series.extend(group)

    if len(main_series) == 1:
        base += "Time series: <ts><ts/>\n\n"
    else:
        base += "Time series 1: <ts><ts/>\nTime series 2: <ts><ts/>\n\n"
    series.extend(main_series)
    base += (
        "Now, answer the following question based on the time series. In your "
        "analysis, try not to repeat large chunk of values in the time series "
        "to save space. Question: \n"
    )
    prompt = base + query
    if prompt.count("<ts><ts/>") != len(series):
        raise RuntimeError("placeholder/time-series count mismatch")
    return prompt, series


def apply_chat_template(
    tokenizer: Any, prompt: str, *, system_prompt: str, enable_thinking: bool
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
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(conversation, **kwargs)


def estimate_vllm_prompt_tokens(
    tokenizer: Any,
    templated_prompt: str,
    series: list[np.ndarray],
    patch_size: int,
) -> int:
    """Mirror ChatTS processor expansion of each multimodal marker."""
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
        total_tokens += len(tokenizer.encode(metadata, add_special_tokens=False))
        total_tokens += max(0, math.ceil(len(values) / patch_size) - 1)
    return total_tokens


def gold_letter(sample: dict[str, Any]) -> str:
    return chr(65 + sample["options"].index(sample["answer"]))


def extract_answer(response: str | None, options: list[Any] | None = None) -> str | None:
    if not response:
        return None
    text = response.strip()
    # Prefer explicit final-answer syntax. Use the last match because the
    # official prompt encourages models to discuss choices before concluding.
    final_patterns = (
        r"<answer>\s*([A-Z])\s*</answer>",
        r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*([A-Z])(?:[.)]|\b)",
    )
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1).upper()
            if options is None or ord(answer) - 65 < len(options):
                return answer

    # Next choose the last letter+matching-option occurrence, not the first
    # option repeated in a "Given the options" discussion.
    if options:
        option_matches: list[tuple[int, str]] = []
        for index, option in enumerate(options):
            letter = chr(65 + index)
            for match in re.finditer(
                rf"(?:^|\n|\s){letter}\)\s*{re.escape(str(option).strip())}",
                text,
                re.IGNORECASE,
            ):
                option_matches.append((match.start(), letter))
        if option_matches:
            return max(option_matches)[1]

    patterns = (
        r"(?:^|\n)\s*([A-Z])\)\s*",
        r"(?:^|\n)\s*([A-Z])\s*(?:\r?\n|$)",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1).upper()
            if options is None or ord(answer) - 65 < len(options):
                return answer
    return None


def score_response(sample: dict[str, Any], response: str | None) -> dict[str, Any]:
    raw = response or ""
    letter = gold_letter(sample)
    answer = str(sample["answer"])
    parsed = extract_answer(raw, sample.get("options"))
    return {
        "answer": parsed,
        "official_flexible_correct": f"{letter}) {answer}".lower() in raw.lower(),
        "official_strict_correct": answer.lower() in raw.split("\n")[-1].lower(),
        "letter_correct": parsed == letter,
    }


def _atomic_dump(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _load_existing(path: Path, protocol_key: str) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    return {
        int(item["idx"]): item
        for item in rows
        if isinstance(item, dict)
        and "idx" in item
        and item.get("protocol_key") == protocol_key
        and item.get("status") in RESUMABLE_STATUSES
    }


def _safe_model_name(model_path: str, requested: str | None) -> str:
    value = requested or Path(model_path.rstrip("/")).name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "chatts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChatTS vLLM on TimeSeriesExam")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--timeseriesexam-root", required=True)
    parser.add_argument("--data-file")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpus-per-model", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--request-chunk-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-processed-input-tokens", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-concepts", type=int, default=3)
    parser.add_argument("--no-question-hint", action="store_true")
    parser.add_argument("--no-concepts", action="store_true")
    parser.add_argument("--no-examples", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--inspect-data-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_gpus < 1 or args.gpus_per_model < 1:
        raise ValueError("GPU counts must be positive")
    if args.num_gpus % args.gpus_per_model:
        raise ValueError("--num-gpus must be divisible by --gpus-per-model")
    if args.request_chunk_size < 1 or args.max_new_tokens < 1:
        raise ValueError("request chunk size and max new tokens must be positive")
    if args.max_concepts < 1 or args.max_processed_input_tokens < 0:
        raise ValueError("invalid concept/token limit")

    root = Path(args.timeseriesexam_root).expanduser().resolve()
    data_path = (
        Path(args.data_file).expanduser().resolve()
        if args.data_file
        else root / "output" / "round_3_folder" / "qa_dataset.json"
    )
    rows = read_rows(data_path)
    validate_rows(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    add_question_hint = not args.no_question_hint
    add_concepts = not args.no_concepts
    add_examples = add_concepts and not args.no_examples
    concepts = load_official_concepts(root) if add_concepts else None
    source_sha256 = _file_sha256(data_path)
    protocol_key = (
        f"hint={int(add_question_hint)};concepts={int(add_concepts)};"
        f"examples={int(add_examples)};max_concepts={args.max_concepts};"
        f"thinking={int(args.enable_thinking)};max_new={args.max_new_tokens};"
        f"temperature={args.temperature};seed={args.seed};rows={len(rows)};"
        f"data_sha256={source_sha256}"
    )
    protocol = {
        "benchmark": "TimeSeriesExam",
        "source_file": str(data_path),
        "source_sha256": source_sha256,
        "selected_dataset_size": len(rows),
        "add_question_hint": add_question_hint,
        "add_concepts": add_concepts,
        "add_examples": add_examples,
        "max_concepts": args.max_concepts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "enable_qwen_thinking": args.enable_thinking,
        "time_series_modality": "raw ChatTS <ts><ts/>",
    }

    # Validate every row and every selected concept before reserving GPU memory.
    modality_counts: dict[int, int] = {}
    for sample in rows:
        _, series = prepare_sample(
            sample,
            concepts,
            add_question_hint=add_question_hint,
            add_concepts=add_concepts,
            add_examples=add_examples,
            max_concepts=args.max_concepts,
        )
        modality_counts[len(series)] = modality_counts.get(len(series), 0) + 1
    print(
        f"[TimeSeriesExam] data={data_path}, rows={len(rows)}, "
        f"time-series-per-prompt={dict(sorted(modality_counts.items()))}"
    )
    if args.inspect_data_only:
        return

    from chatts.utils.llm_utils import LLMClient
    from transformers import AutoConfig
    from vllm import SamplingParams

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    patch_size = _chatts_vllm.get_time_series_patch_size(config)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        seed=args.seed,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    model_name = _safe_model_name(args.model_path, args.model_name)
    output_path = Path(args.output_path).expanduser().resolve()
    completed = {} if args.force else _load_existing(output_path, protocol_key)
    client = LLMClient(
        model_path=os.path.abspath(args.model_path),
        engine="vllm-ts",
        num_gpus=args.num_gpus,
        gpus_per_model=args.gpus_per_model,
        batch_size=args.batch_size,
    )
    try:
        client.wait_for_ready()
        pending = [index for index in range(len(rows)) if index not in completed]
        print(
            f"[TimeSeriesExam] model={model_name}, patch_size={patch_size}, "
            f"completed={len(completed)}, pending={len(pending)}"
        )
        for start in range(0, len(pending), args.request_chunk_size):
            batch_indices = pending[start : start + args.request_chunk_size]
            prompts: list[str] = []
            templated_prompts: list[str] = []
            series_batch: list[list[np.ndarray]] = []
            valid_indices: list[int] = []
            input_sizes: list[int] = []

            for index in batch_indices:
                sample = rows[index]
                prompt, series = prepare_sample(
                    sample,
                    concepts,
                    add_question_hint=add_question_hint,
                    add_concepts=add_concepts,
                    add_examples=add_examples,
                    max_concepts=args.max_concepts,
                )
                templated = apply_chat_template(
                    client.tokenizer,
                    prompt,
                    system_prompt=client.system_prompt,
                    enable_thinking=args.enable_thinking,
                )
                processed_tokens = estimate_vllm_prompt_tokens(
                    client.tokenizer, templated, series, patch_size
                )
                if (
                    args.max_processed_input_tokens > 0
                    and processed_tokens > args.max_processed_input_tokens
                ):
                    reason = (
                        f"processed input tokens {processed_tokens} > "
                        f"{args.max_processed_input_tokens}"
                    )
                    completed[index] = {
                        "idx": index,
                        "id": sample.get("id"),
                        "tid": sample.get("tid"),
                        "category": sample.get("category"),
                        "subcategory": sample.get("subcategory"),
                        "difficulty": sample.get("difficulty"),
                        "question": sample.get("question"),
                        "prompt": prompt,
                        "options": sample.get("options"),
                        "gold_answer": sample.get("answer"),
                        "gold_letter": gold_letter(sample),
                        "answer": None,
                        "response": "",
                        "status": "skipped_input_length",
                        "error": reason,
                        "processed_input_tokens": processed_tokens,
                        "protocol_key": protocol_key,
                        "protocol": protocol,
                    }
                    continue
                prompts.append(prompt)
                templated_prompts.append(templated)
                series_batch.append(series)
                valid_indices.append(index)
                input_sizes.append(processed_tokens)

            if templated_prompts:
                responses = client.llm_batch_generate(
                    templated_prompts,
                    series_batch,
                    use_chat_template=False,
                    sampling_params=sampling_params,
                )
                for index, prompt, response, processed_tokens, submitted_series in zip(
                    valid_indices, prompts, responses, input_sizes, series_batch
                ):
                    sample = rows[index]
                    raw = response or ""
                    rejected = raw.startswith(VLLM_REJECTED_PREFIX)
                    scores = score_response(sample, None if rejected else raw)
                    completed[index] = {
                        "idx": index,
                        "id": sample.get("id"),
                        "tid": sample.get("tid"),
                        "category": sample.get("category"),
                        "subcategory": sample.get("subcategory"),
                        "difficulty": sample.get("difficulty"),
                        "question_type": sample.get("question_type"),
                        "question": sample.get("question"),
                        "prompt": prompt,
                        "options": sample.get("options"),
                        "gold_answer": sample.get("answer"),
                        "gold_letter": gold_letter(sample),
                        "response": raw,
                        **scores,
                        "status": "vllm_input_rejected" if rejected else "ok",
                        "error": raw[len(VLLM_REJECTED_PREFIX) :].strip() if rejected else None,
                        "processed_input_tokens": processed_tokens,
                        "num_time_series": len(submitted_series),
                        "protocol_key": protocol_key,
                        "protocol": protocol,
                    }

            ordered = [completed[index] for index in sorted(completed)]
            _atomic_dump(ordered, output_path)
            print(f"[TimeSeriesExam] saved {len(ordered)}/{len(rows)} -> {output_path}")
    finally:
        client.kill()


if __name__ == "__main__":
    main()

"""Run a ChatTS checkpoint on TSRBench with the existing ChatTS vLLM backend.

The model is loaded once, then every selected TSRBench JSONL file is evaluated.
Raw time-series arrays are passed to ChatTS.  The checkpoint processor performs
the same SP normalization used by the existing Dataset A/B vLLM script.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# This import MUST stay at module scope and MUST happen before any vLLM engine
# is created.  vLLM uses the ``spawn`` start method, so each child imports this
# module again but does not call main().  Importing inside main() would leave
# the child registry unaware of Qwen2/3TSForCausalLM and its multimodal
# processor, causing a fallback to TransformersForCausalLM and a registry
# KeyError during WorkerProc initialization.
import chatts.vllm.chatts_vllm as _chatts_vllm  # noqa: F401,E402


TASK_PATHS = {
    "perception": "perception/perception.jsonl",
    "causal_reasoning": "reasoning/causal_reasoning.jsonl",
    "inductive_reasoning": "reasoning/inductive_reasoning.jsonl",
    "numerical_reasoning": "reasoning/numerical_reasoning.jsonl",
    "temporal_relation_reasoning": "reasoning/temporal_relation_reasoning.jsonl",
    "etiological_reasoning": "reasoning/etiological_reasoning.jsonl",
    "abductive_reasoning": "reasoning/abductive_reasoning.jsonl",
    "deductive_reasoning": "reasoning/deductive_reasoning.jsonl",
    "time_series_forecasting": "prediction/time_series_forecasting.jsonl",
    "event_prediction": "prediction/event_prediction.jsonl",
    "qualitative_decision": "decision/qualitative_decision.jsonl",
    "quantitative_decision": "decision/quantitative_decision.jsonl",
}

TASK_ALIASES = {
    "math_reasoning": "numerical_reasoning",
    "event_forecast": "event_prediction",
    "pattern_decision": "qualitative_decision",
}

PROMPT_MODES = ("answer_only", "official")

ANSWER_ONLY_INSTRUCTION = (
    "\n\nReturn exactly one uppercase option letter (A-G) and no other text."
)


def _official_answer_instruction(choice_text: str) -> str:
    """The answer instruction used by TSRBench's official ChatTS runner."""
    return (
        " Select from the options below:\n" + choice_text
        + "\nOutput your reasoning process in <think> tags and final answer "
        "as a single letter in <answer> tags. Format:\n"
        "<think>Your reasoning here (less than 2048 tokens)</think>\n"
        "<answer>A</answer>"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        first = stream.read(1)
        stream.seek(0)
        if first == "[":
            value = json.load(stream)
            if not isinstance(value, list):
                raise ValueError(f"Expected a JSON list in {path}")
            return value

        rows = []
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        return rows


def discover_dataset_files(
    dataset_root: Path, requested: Iterable[str]
) -> list[tuple[str, Path]]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"TSRBench dataset root does not exist: {dataset_root}")

    names = [TASK_ALIASES.get(item.strip(), item.strip()) for item in requested if item.strip()]
    if not names or "all" in names:
        names = list(TASK_PATHS)

    by_stem: dict[str, Path] = {}
    for path in dataset_root.rglob("*.jsonl"):
        by_stem.setdefault(path.stem, path)

    resolved = []
    missing = []
    for name in names:
        canonical = TASK_ALIASES.get(name, name)
        expected = dataset_root / TASK_PATHS.get(canonical, f"{canonical}.jsonl")
        path = expected if expected.is_file() else by_stem.get(canonical)
        if path is None or not path.is_file():
            missing.append(canonical)
        else:
            resolved.append((canonical, path.resolve()))

    if missing:
        available = ", ".join(sorted(by_stem)) or "none"
        raise FileNotFoundError(
            f"Missing TSRBench dataset(s): {', '.join(missing)}. "
            f"Discovered JSONL stems: {available}"
        )
    return resolved


def _format_choices(choices: Any) -> str:
    labels = "ABCDEFG"
    if isinstance(choices, dict):
        return "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))
    if isinstance(choices, list):
        formatted = []
        for index, value in enumerate(choices):
            text = str(value).strip()
            if re.match(r"^[A-G][.)]\s*", text, flags=re.IGNORECASE):
                formatted.append(text)
            else:
                formatted.append(f"{labels[index]}. {text}")
        return "\n".join(formatted)
    return ""


def _format_choices_official(choices: Any) -> str:
    """Match TSRBench's format_choices function byte-for-byte in behavior."""
    labels = "ABCDEFG"
    if isinstance(choices, dict):
        return "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))
    if isinstance(choices, list):
        return "\n".join(
            f"{labels[index]}. {value}" for index, value in enumerate(choices)
        )
    return str(choices)


def _question_already_has_choices(question: str, choices: Any) -> bool:
    if not isinstance(choices, (list, dict)) or not choices:
        return False
    count = len(choices)
    markers = sum(
        bool(re.search(rf"(?:^|\n)\s*{letter}[.)]\s", question, re.IGNORECASE))
        for letter in "ABCDEFG"[:count]
    )
    return markers >= min(2, count)


def _as_1d_series(value: Any, *, label: str) -> np.ndarray:
    series = np.asarray(value, dtype=np.float64).squeeze()
    if series.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {series.shape}")
    if series.size == 0:
        raise ValueError(f"{label} is empty")
    if not np.isfinite(series).all():
        raise ValueError(f"{label} contains NaN or infinity")
    return series


def _append_choices(question: str, choices: Any) -> str:
    choice_text = _format_choices(choices)
    if choice_text and not _question_already_has_choices(question, choices):
        question += "\n\nOptions:\n" + choice_text
    return question


def _standard_prompt(
    sample: dict[str, Any], dataset_name: str, prompt_mode: str
) -> tuple[str, list[np.ndarray]]:
    if "question" not in sample or "timeseries" not in sample:
        raise KeyError("A standard TSRBench sample requires question and timeseries")

    raw_series = sample["timeseries"]
    if not isinstance(raw_series, list) or not raw_series:
        raise ValueError("timeseries must be a non-empty list")

    raw_names = sample.get("name_of_series")
    if not isinstance(raw_names, list):
        raw_names = []

    series: list[np.ndarray] = []
    names: list[str] = []
    for index, values in enumerate(raw_series):
        default_name = (
            f"Series {index + 1}"
            if prompt_mode == "official"
            else f"Time series {index + 1}"
        )
        name = str(raw_names[index]) if index < len(raw_names) else default_name
        if dataset_name == "temporal_relation_reasoning" and name.strip().lower() == "time stamps":
            continue
        series.append(_as_1d_series(values, label=name))
        names.append(name)

    if not series:
        raise ValueError("No numeric time series remained after preprocessing")

    question = str(sample["question"])
    if prompt_mode == "official":
        # Keep this deliberately equivalent to TSRBench's build_prompt_standard
        # / build_prompt_temporal implementation.
        question += " Here are the time series"
        for name in names:
            question += f" '{name}': <ts><ts/>. "
        question += _official_answer_instruction(
            _format_choices_official(sample.get("choices"))
        )
        return question, series

    question = question.strip()
    question = _append_choices(question, sample.get("choices"))
    placeholder_count = question.count("<ts><ts/>")
    if placeholder_count == 0:
        question += "\n\nNumeric time-series inputs:\n"
        question += "\n".join(f"- {name}: <ts><ts/>" for name in names)
    elif placeholder_count != len(series):
        raise ValueError(
            f"Prompt has {placeholder_count} <ts><ts/> placeholders but sample has {len(series)} series"
        )

    return question + ANSWER_ONLY_INSTRUCTION, series


def _abductive_prompt(
    sample: dict[str, Any], prompt_mode: str
) -> tuple[str, list[np.ndarray]]:
    context = sample["context"]
    mcq = sample["multiple_choice_question"]
    numerical = sample["numerical_time_series"]

    history_events = list(context.get("history_events", []))
    history_times = list(context.get("history_times", []))
    future_events = list(context.get("future_events", []))
    future_times = list(context.get("future_times", []))

    def values_for(preferred_key: str, fallback_index: int) -> list[float]:
        key = preferred_key
        if key not in numerical:
            keys = sorted(numerical)
            if fallback_index >= len(keys):
                raise KeyError(f"Cannot find {preferred_key} in numerical_time_series")
            key = keys[fallback_index]
        item = numerical[key]
        return list(item.get("history", [])) + list(item.get("future", []))

    team_a = values_for("wp_Team A", 0)
    team_b = values_for("wp_Team B", 1)
    all_times = history_times + future_times
    min_length = min(len(all_times), len(team_a), len(team_b))
    all_times = all_times[:min_length]
    team_a = team_a[:min_length]
    team_b = team_b[:min_length]
    critical = len(history_times)
    lower = max(0, critical - 10)
    upper = min(min_length, critical + 10)
    all_times = all_times[lower:upper]
    team_a = team_a[lower:upper]
    team_b = team_b[lower:upper]

    past = "\n".join(
        f"- {time}: {event}"
        for time, event in zip(history_times[-10:], history_events[-10:])
    )
    future = "\n".join(
        f"- {time}: {event}"
        for time, event in zip(future_times[:10], future_events[:10])
    )

    if prompt_mode == "official":
        time_series_table = "Time | Team A Win Prob | Team B Win Prob\n"
        time_series_table += "-" * 60 + "\n"
        for time_value, value_a, value_b in zip(all_times, team_a, team_b):
            time_series_table += f"{time_value} | {value_a:.3f} | {value_b:.3f}\n"
        question = (
            "You are an expert in basketball game analysis. Your task is to perform abductive reasoning.\n"
            "Given a sequence of past events, future events, and corresponding time series data from a game, "
            "determine the most plausible event that occurred in between to link them.\n\n"
            "--- CONTEXT ---\n"
            f"Past Events (History):\n{past}\n"
            "\n... [A CRITICAL EVENT HAPPENED HERE] ...\n\n"
            f"Future Events:\n{future}\n\n"
            f"Time Series Data (Win Probability):\n{time_series_table.strip()}\n\n"
            "--- TASK ---\n"
            f"{mcq['question']}"
            " Here are the time series"
            " 'Team A Win Probability': <ts><ts/>. "
            " 'Team B Win Probability': <ts><ts/>. "
        )
        question += _official_answer_instruction(
            _format_choices_official(mcq.get("choices"))
        )
    else:
        question = (
            "Infer the most plausible missing basketball event between the observed past and future.\n\n"
            f"Past events:\n{past}\n\n"
            "[A CRITICAL EVENT IS MISSING HERE]\n\n"
            f"Future events:\n{future}\n\n"
            f"Question: {mcq['question']}"
        )
        question = _append_choices(question, mcq.get("choices"))
        question += (
            "\n\nNumeric time-series inputs around the missing event:\n"
            "- Team A win probability: <ts><ts/>\n"
            "- Team B win probability: <ts><ts/>"
            + ANSWER_ONLY_INSTRUCTION
        )
    return question, [
        _as_1d_series(team_a, label="Team A win probability"),
        _as_1d_series(team_b, label="Team B win probability"),
    ]


def prepare_sample(
    sample: dict[str, Any], dataset_name: str, prompt_mode: str = "answer_only"
) -> tuple[str, list[np.ndarray]]:
    if dataset_name == "abductive_reasoning" or (
        "multiple_choice_question" in sample and "numerical_time_series" in sample
    ):
        prompt, series = _abductive_prompt(sample, prompt_mode)
    else:
        prompt, series = _standard_prompt(sample, dataset_name, prompt_mode)

    if prompt.count("<ts><ts/>") != len(series):
        raise ValueError("Internal error: placeholder count does not match time-series count")
    return prompt, series


def extract_answer(response: str | None) -> str | None:
    if not response:
        return None
    text = response.strip()
    text = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", text, flags=re.IGNORECASE)

    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:
            continue
        if isinstance(value, dict) and "answer" in value:
            match = re.search(r"[A-G]", str(value["answer"]), re.IGNORECASE)
            if match:
                return match.group(0).upper()

    patterns = (
        r"<answer>\s*([A-G])\s*</answer>",
        r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([A-G])",
        r"(?:^|\n)\s*(?:final\s+answer\s*[:：]?\s*)?([A-G])[.)]",
        r"^\s*([A-G])\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def canonicalize_response(
    response: str | None, prompt_mode: str
) -> tuple[str, str | None, str | None]:
    """Return saved response, parsed answer, and optional reasoning path."""
    raw = response or ""
    answer = extract_answer(raw)
    if answer is None:
        return raw, None, None

    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    reasoning_path = think_match.group(1).strip() if think_match else None
    if prompt_mode == "official":
        return raw, answer, reasoning_path

    canonical = json.dumps({"answer": answer}, ensure_ascii=False)
    return canonical, answer, None


def _is_valid_official_response(response: str | None) -> bool:
    if not response:
        return False
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    answer_match = re.search(r"<answer>\s*([A-G])\s*</answer>", response, re.DOTALL)
    return bool(think_match and think_match.group(1).strip() and answer_match)


def apply_chat_template(
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str,
    enable_thinking: bool,
    prompt_mode: str,
) -> str:
    """Build the exact official ChatTS chat wrapper or the checkpoint template."""
    if prompt_mode == "official":
        # TSRBench's official ChatTS script builds this string manually.  Its
        # thinking request lives in the user prompt, not in Qwen3's template.
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>"
            f"<|im_start|>user\n{prompt}<|im_end|>"
            f"<|im_start|>assistant\n"
        )

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _safe_model_name(model_path: str, requested_name: str | None) -> str:
    value = requested_name or Path(model_path.rstrip("/")).name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "chatts"


def _load_existing(path: Path, prompt_mode: str) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    return {
        int(item["idx"]): item
        for item in rows
        if isinstance(item, dict)
        and "idx" in item
        and item.get("response")
        and item.get("prompt_mode") == prompt_mode
    }


def _atomic_dump(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChatTS vLLM inference on TSRBench")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--output-root", default="evaluation/results/embed")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpus-per-model", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--request-chunk-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--max-mm-per-prompt", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default="answer_only",
        help="answer_only disables reasoning; official reproduces TSRBench's ChatTS XML prompt.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode (disabled by default).",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def generate_with_retries(
    client: Any,
    prompts: list[str],
    series_batch: list[list[np.ndarray]],
    sampling_params: Any,
    *,
    prompt_mode: str,
    max_retries: int,
) -> list[str | None]:
    responses: list[str | None] = [None] * len(prompts)
    remaining = list(range(len(prompts)))
    total_attempts = max(1, max_retries)

    for attempt in range(total_attempts):
        generated = client.llm_batch_generate(
            [prompts[index] for index in remaining],
            [series_batch[index] for index in remaining],
            use_chat_template=False,
            sampling_params=sampling_params,
        )
        for index, response in zip(remaining, generated):
            responses[index] = response

        if prompt_mode == "official":
            remaining = [
                index for index in remaining
                if not _is_valid_official_response(responses[index])
            ]
        else:
            remaining = [
                index for index in remaining
                if extract_answer(responses[index]) is None
            ]
        if not remaining:
            break
        print(
            f"[TSRBench] invalid output for {len(remaining)} sample(s); "
            f"attempt {attempt + 1}/{total_attempts}"
        )

    return responses


def main() -> None:
    args = parse_args()
    if args.num_gpus < 1 or args.gpus_per_model < 1:
        raise ValueError("GPU counts must be positive")
    if args.num_gpus % args.gpus_per_model:
        raise ValueError("--num-gpus must be divisible by --gpus-per-model")
    if args.request_chunk_size < 1:
        raise ValueError("--request-chunk-size must be positive")
    if args.max_retries < 0 or args.max_input_tokens < 0:
        raise ValueError("token limits and retry count cannot be negative")

    datasets = discover_dataset_files(Path(args.dataset_root), args.datasets)
    model_name = _safe_model_name(args.model_path, args.model_name)
    output_root = Path(args.output_root).resolve()

    from chatts.utils.llm_utils import LLMClient
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    client = LLMClient(
        model_path=os.path.abspath(args.model_path),
        engine="vllm-ts",
        num_gpus=args.num_gpus,
        gpus_per_model=args.gpus_per_model,
        batch_size=args.batch_size,
    )

    try:
        client.wait_for_ready()
        for dataset_name, dataset_path in datasets:
            rows = read_jsonl(dataset_path)
            if args.max_samples > 0:
                rows = rows[: args.max_samples]

            result_dir = output_root / f"{dataset_name}_{model_name}"
            result_path = result_dir / "generated_answer.json"
            completed = (
                {} if args.force else _load_existing(result_path, args.prompt_mode)
            )
            pending = [index for index in range(len(rows)) if index not in completed]
            print(
                f"[TSRBench] {dataset_name}: total={len(rows)}, "
                f"completed={len(completed)}, pending={len(pending)}"
            )

            for start in range(0, len(pending), args.request_chunk_size):
                indices = pending[start : start + args.request_chunk_size]
                prompts: list[str] = []
                series_batch: list[list[np.ndarray]] = []
                valid_indices: list[int] = []

                for index in indices:
                    try:
                        prompt, series = prepare_sample(
                            rows[index], dataset_name, args.prompt_mode
                        )
                        if len(series) > args.max_mm_per_prompt:
                            raise ValueError(
                                f"sample has {len(series)} time series; limit is {args.max_mm_per_prompt}"
                            )
                    except Exception as exc:
                        print(f"[TSRBench] {dataset_name}[{index}] skipped: {exc}")
                        continue
                    prompts.append(prompt)
                    series_batch.append(series)
                    valid_indices.append(index)

                if not prompts:
                    continue
                templated_prompts = [
                    apply_chat_template(
                        client.tokenizer,
                        prompt,
                        system_prompt=client.system_prompt,
                        enable_thinking=args.enable_thinking,
                        prompt_mode=args.prompt_mode,
                    )
                    for prompt in prompts
                ]
                if args.max_input_tokens > 0:
                    kept = []
                    for position, templated_prompt in enumerate(templated_prompts):
                        input_tokens = len(
                            client.tokenizer(
                                templated_prompt, add_special_tokens=False
                            )["input_ids"]
                        )
                        if input_tokens > args.max_input_tokens:
                            print(
                                f"[TSRBench] {dataset_name}[{valid_indices[position]}] "
                                f"skipped: input tokens {input_tokens} > {args.max_input_tokens}"
                            )
                        else:
                            kept.append(position)
                    prompts = [prompts[position] for position in kept]
                    templated_prompts = [templated_prompts[position] for position in kept]
                    series_batch = [series_batch[position] for position in kept]
                    valid_indices = [valid_indices[position] for position in kept]
                    if not prompts:
                        continue

                responses = generate_with_retries(
                    client,
                    templated_prompts,
                    series_batch,
                    sampling_params,
                    prompt_mode=args.prompt_mode,
                    max_retries=args.max_retries,
                )
                for index, prompt, templated_prompt, response in zip(
                    valid_indices, prompts, templated_prompts, responses
                ):
                    canonical_response, parsed_answer, reasoning_path = canonicalize_response(
                        response, args.prompt_mode
                    )
                    result = {
                        "idx": index,
                        "question_text": (
                            templated_prompt if args.prompt_mode == "official" else prompt
                        ),
                        "response": canonical_response,
                        "raw_response": response or "",
                        "answer": parsed_answer,
                        "prompt_mode": args.prompt_mode,
                    }
                    if reasoning_path:
                        result["reasoning_path"] = reasoning_path
                    completed[index] = result

                ordered = [completed[index] for index in sorted(completed)]
                _atomic_dump(ordered, result_path)
                print(
                    f"[TSRBench] {dataset_name}: saved {len(ordered)}/{len(rows)} "
                    f"to {result_path}"
                )
    finally:
        client.kill()


if __name__ == "__main__":
    main()

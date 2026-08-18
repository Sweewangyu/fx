"""Small, dependency-light helpers for TSRBench inference tracing."""

from __future__ import annotations

import re
from typing import Any, Iterable


def analyze_official_response(response: str | None) -> dict[str, Any]:
    """Explain whether a response satisfies TSRBench's official XML contract.

    The validity rule intentionally matches the production inference runner:
    tags must be lower-case, the reasoning block must be non-empty, and the
    answer must be one upper-case letter from A through G.
    """
    raw = response or ""
    reasons: list[str] = []
    if not raw.strip():
        reasons.append("empty_response")

    think_open = "<think>" in raw
    think_close = "</think>" in raw
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    reasoning_path = think_match.group(1).strip() if think_match else None
    if not think_open:
        reasons.append("missing_<think>")
    if not think_close:
        reasons.append("missing_</think>")
    if think_match and not reasoning_path:
        reasons.append("empty_think")

    answer_open = "<answer>" in raw
    answer_close = "</answer>" in raw
    answer_match = re.search(r"<answer>\s*([A-G])\s*</answer>", raw, re.DOTALL)
    answer = answer_match.group(1) if answer_match else None
    if not answer_open:
        reasons.append("missing_<answer>")
    if not answer_close:
        reasons.append("missing_</answer>")
    if answer_open and answer_close and answer is None:
        tagged = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        value = tagged.group(1).strip() if tagged else ""
        reasons.append(f"invalid_answer_value:{value[:80]}")

    valid = bool(think_match and reasoning_path and answer_match)
    return {
        "official_valid": valid,
        "parsed_answer": answer,
        "reasoning_path": reasoning_path,
        "invalid_reasons": [] if valid else reasons,
        "has_think_open": think_open,
        "has_think_close": think_close,
        "has_answer_open": answer_open,
        "has_answer_close": answer_close,
        "response_chars": len(raw),
    }


def summarize_series(values: Iterable[float], *, preview: int = 5) -> dict[str, Any]:
    """Return compact JSON-safe statistics without copying the full series."""
    numbers = [float(value) for value in values]
    if not numbers:
        return {"length": 0, "head": [], "tail": []}
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    return {
        "length": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": mean,
        "std": variance**0.5,
        "left": numbers[0],
        "right": numbers[-1],
        "head": numbers[:preview],
        "tail": numbers[-preview:],
    }


def ground_truth(sample: dict[str, Any], dataset_name: str) -> str | None:
    """Read the MCQ label using the same fields as evaluate_tsrbench.py."""
    if dataset_name == "abductive_reasoning":
        value = sample.get("multiple_choice_question", {}).get("answer")
    else:
        value = sample.get("answer")
    if value is None:
        return None
    match = re.search(r"[A-G]", str(value), re.IGNORECASE)
    return match.group(0).upper() if match else None


def sample_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep question metadata in a trace while omitting large numeric arrays."""
    omitted = {"timeseries", "numerical_time_series"}
    return {key: value for key, value in sample.items() if key not in omitted}

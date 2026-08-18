from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from chatts.utils.tsrbench_trace import (
    analyze_json_reasoning_response,
    analyze_official_response,
    summarize_series,
)


def _load_renderer():
    path = Path(__file__).parents[1] / "scripts" / "render_tsrbench_trace.py"
    spec = importlib.util.spec_from_file_location("render_tsrbench_trace", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_response_diagnostics_preserve_strict_contract() -> None:
    valid = analyze_official_response(
        "<think>The peak occurs last.</think>\n<answer>B</answer>"
    )
    assert valid["official_valid"] is True
    assert valid["parsed_answer"] == "B"
    assert valid["reasoning_path"] == "The peak occurs last."
    assert valid["invalid_reasons"] == []

    truncated = analyze_official_response("<think>I am still reasoning")
    assert truncated["official_valid"] is False
    assert "missing_</think>" in truncated["invalid_reasons"]
    assert "missing_<answer>" in truncated["invalid_reasons"]

    lower_case = analyze_official_response(
        "<think>Reasoning.</think><answer>b</answer>"
    )
    assert lower_case["official_valid"] is False
    assert "invalid_answer_value:b" in lower_case["invalid_reasons"]


def test_series_summary_is_compact() -> None:
    summary = summarize_series(range(10), preview=2)
    assert summary["length"] == 10
    assert summary["head"] == [0.0, 1.0]
    assert summary["tail"] == [8.0, 9.0]
    assert summary["mean"] == 4.5


def test_json_reasoning_requires_strict_valid_json() -> None:
    valid = analyze_json_reasoning_response(
        '{"reason":"Humidity is rising.","answer":"B"}'
    )
    assert valid["format_valid"] is True
    assert valid["parsed_answer"] == "B"
    assert valid["reasoning_path"] == "Humidity is rising."

    bare = analyze_json_reasoning_response("B")
    assert bare["format_valid"] is False
    assert any(reason.startswith("invalid_json:") for reason in bare["invalid_reasons"])

    pseudo_json = analyze_json_reasoning_response("{reason: x, answer: B}")
    assert pseudo_json["format_valid"] is False


def test_renderer_shows_every_attempt(tmp_path: Path) -> None:
    renderer = _load_renderer()
    trace = tmp_path / "trace.jsonl"
    row = {
        "dataset": "event_prediction",
        "idx": 7,
        "model": "demo",
        "prompt_mode": "official",
        "input_tokens": 100,
        "processed_input_tokens": 140,
        "source_sample": {"question": "What happens next?"},
        "time_series": [{"length": 3, "head": [1, 2, 3]}],
        "templated_prompt": "official prompt",
        "ground_truth": "B",
        "final_answer": "B",
        "final_valid": True,
        "correct": True,
        "attempts": [
            {
                "attempt": 1,
                "official_valid": False,
                "parsed_answer": None,
                "invalid_reasons": ["missing_</think>"],
                "raw_response": "<think>truncated",
                "response_chars": 16,
                "output_tokens": 5,
            },
            {
                "attempt": 2,
                "official_valid": True,
                "parsed_answer": "B",
                "invalid_reasons": [],
                "raw_response": "<think>done</think><answer>B</answer>",
                "response_chars": 38,
                "output_tokens": 12,
            },
        ],
    }
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = renderer.render(renderer.read_jsonl(trace), trace)
    assert "Attempt 1 — invalid" in report
    assert "Attempt 2 — valid" in report
    assert "missing_</think>" in report
    assert "Correct | yes" in report

#!/usr/bin/env python3
"""Render ChatTS TSRBench retry traces as a readable Markdown report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _fenced(value: Any, language: str = "text") -> str:
    text = "" if value is None else str(value)
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def render(rows: list[dict[str, Any]], source: Path) -> str:
    total_attempts = sum(len(row.get("attempts", [])) for row in rows)
    valid = sum(row.get("final_valid") is True for row in rows)
    invalid = sum(row.get("final_valid") is False for row in rows)
    scored = [row for row in rows if row.get("correct") is not None]
    correct = sum(row.get("correct") is True for row in scored)

    lines = [
        "# ChatTS × TSRBench structured-reasoning inference trace",
        "",
        f"Trace source: `{source}`",
        "",
        "This is a diagnostic subset report. Accuracy below uses only traced, "
        "scorable cases; it is not the full TSRBench score.",
        "",
        "## Summary",
        "",
        "| Cases | Attempts | Final valid | Final invalid | Correct / scorable |",
        "|---:|---:|---:|---:|---:|",
        f"| {len(rows)} | {total_attempts} | {valid} | {invalid} | "
        f"{correct} / {len(scored)} |",
        "",
    ]

    for row in rows:
        dataset = row.get("dataset", "unknown")
        index = row.get("idx", "?")
        lines.extend(
            [
                f"## {dataset}[{index}]",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| Model | `{row.get('model', '-')}` |",
                f"| Prompt mode | `{row.get('prompt_mode', '-')}` |",
                f"| Text / processed input tokens | {row.get('input_tokens', '-')} / "
                f"{row.get('processed_input_tokens', '-')} |",
                f"| Attempts | {len(row.get('attempts', []))} |",
                f"| Final valid | {_fmt_bool(row.get('final_valid'))} |",
                f"| Predicted / truth | `{row.get('final_answer')}` / "
                f"`{row.get('ground_truth')}` |",
                f"| Correct | {_fmt_bool(row.get('correct'))} |",
                "",
            ]
        )
        if row.get("error"):
            lines.extend([f"Error: `{row['error']}`", ""])

        source_sample = row.get("source_sample", {})
        lines.extend(
            [
                "### Source item (large numeric arrays omitted)",
                "",
                _fenced(json.dumps(source_sample, ensure_ascii=False, indent=2), "json"),
                "",
                "### Time-series summaries",
                "",
                _fenced(
                    json.dumps(row.get("time_series", []), ensure_ascii=False, indent=2),
                    "json",
                ),
                "",
                "### Sampling protocol",
                "",
                _fenced(
                    json.dumps(row.get("sampling", {}), ensure_ascii=False, indent=2),
                    "json",
                ),
                "",
                "### Exact rendered prompt sent to vLLM",
                "",
                _fenced(row.get("templated_prompt", row.get("raw_prompt", ""))),
                "",
            ]
        )

        attempts = row.get("attempts", [])
        if not attempts:
            lines.extend(["### Attempts", "", "No generation attempt was made.", ""])
        for attempt in attempts:
            format_valid = attempt.get(
                "format_valid", attempt.get("official_valid")
            )
            status = "valid" if format_valid else "invalid"
            reasons = attempt.get("invalid_reasons") or []
            reason_text = ", ".join(str(reason) for reason in reasons) or "none"
            lines.extend(
                [
                    f"### Attempt {attempt.get('attempt', '?')} — {status}",
                    "",
                    f"Parsed answer: `{attempt.get('parsed_answer')}`  ",
                    f"Output tokens / characters: {attempt.get('output_tokens', '-')} / "
                    f"{attempt.get('response_chars', '-')}  ",
                    f"Invalid reasons: `{reason_text}`",
                    "",
                    _fenced(attempt.get("raw_response", "")),
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_jsonl")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.trace_jsonl).resolve()
    output = Path(args.output).resolve()
    rows = read_jsonl(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(rows, source), encoding="utf-8")
    print(f"Trace report: {output}")


if __name__ == "__main__":
    main()

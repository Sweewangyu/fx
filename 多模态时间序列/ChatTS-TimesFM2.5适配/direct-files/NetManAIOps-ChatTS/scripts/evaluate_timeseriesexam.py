#!/usr/bin/env python3
"""Aggregate ChatTS x TimeSeriesExam predictions without re-running inference."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Result file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [row for row in rows if isinstance(row, dict)]


def summarize(
    records: list[dict[str, Any]],
    *,
    scope: str,
    key: str,
    expected_total: int | None = None,
) -> dict[str, Any]:
    observed = len(records)
    total = max(observed, int(expected_total or 0))
    generated = sum(item.get("status") == "ok" for item in records)
    parsed = sum(item.get("answer") not in (None, "") for item in records)
    official_flexible = sum(
        bool(item.get("official_flexible_correct")) for item in records
    )
    official_strict = sum(
        bool(item.get("official_strict_correct")) for item in records
    )
    letter_correct = sum(bool(item.get("letter_correct")) for item in records)
    statuses: dict[str, int] = defaultdict(int)
    for item in records:
        statuses[str(item.get("status", "unknown"))] += 1
    return {
        "scope": scope,
        "key": key,
        "total": total,
        "observed": observed,
        "generated": generated,
        "parsed": parsed,
        "official_flexible_correct": official_flexible,
        "official_strict_correct": official_strict,
        "letter_correct": letter_correct,
        "coverage": generated / total if total else None,
        "parse_rate": parsed / generated if generated else None,
        "official_flexible_accuracy": official_flexible / total if total else None,
        "official_strict_accuracy": official_strict / total if total else None,
        "letter_accuracy": letter_correct / total if total else None,
        "letter_accuracy_parsed": letter_correct / parsed if parsed else None,
        "statuses": dict(sorted(statuses.items())),
    }


def grouped_rows(
    records: list[dict[str, Any]],
    field: str,
    scope: str,
    normalizer: Callable[[Any], str] = lambda value: str(value or "unknown"),
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[normalizer(item.get(field))].append(item)
    return [
        summarize(values, scope=scope, key=key)
        for key, values in sorted(grouped.items())
    ]


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TimeSeriesExam results")
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--model-name", default="chatts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_file = Path(args.result_file).expanduser().resolve()
    records = load_records(result_file)
    if not records:
        raise ValueError(f"No result records: {result_file}")

    protocol = records[0].get("protocol") or {}
    expected = int(protocol.get("selected_dataset_size") or len(records))
    overall = summarize(records, scope="overall", key="all", expected_total=expected)
    categories = grouped_rows(records, "category", "category")
    subcategories = grouped_rows(records, "subcategory", "subcategory")
    difficulties = grouped_rows(
        records,
        "difficulty",
        "difficulty",
        lambda value: str(value or "unknown").lower(),
    )

    print(
        f"{'Category':24} {'Done':>11} {'Parsed':>11} "
        f"{'Official':>10} {'Strict':>10} {'LetterAcc':>10}"
    )
    print("-" * 84)
    for row in categories:
        print(
            f"{row['key'][:24]:24} "
            f"{row['generated']:4}/{row['total']:<6} "
            f"{row['parsed']:4}/{row['generated']:<6} "
            f"{_fmt(row['official_flexible_accuracy']):>10} "
            f"{_fmt(row['official_strict_accuracy']):>10} "
            f"{_fmt(row['letter_accuracy']):>10}"
        )
    print("-" * 84)
    print(
        f"{'Overall':24} "
        f"{overall['generated']:4}/{overall['total']:<6} "
        f"{overall['parsed']:4}/{overall['generated']:<6} "
        f"{_fmt(overall['official_flexible_accuracy']):>10} "
        f"{_fmt(overall['official_strict_accuracy']):>10} "
        f"{_fmt(overall['letter_accuracy']):>10}"
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_file.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model_name,
        "primary_metric": "official_flexible_accuracy",
        "metric_notes": {
            "official_flexible_accuracy": (
                "Official TimeSeriesExam flexible rule: the response contains "
                "'<letter>) <gold option text>'."
            ),
            "official_strict_accuracy": (
                "Official strict rule: the final response line contains the gold option text."
            ),
            "letter_accuracy": "Robustly parsed option letter equals the gold letter.",
        },
        "protocol": protocol,
        "overall": overall,
        "per_category": categories,
        "per_subcategory": subcategories,
        "per_difficulty": difficulties,
    }
    json_path = output_dir / f"timeseriesexam_summary_{args.model_name}.json"
    csv_path = output_dir / f"timeseriesexam_summary_{args.model_name}.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    fields = [
        "scope",
        "key",
        "total",
        "observed",
        "generated",
        "parsed",
        "official_flexible_correct",
        "official_strict_correct",
        "letter_correct",
        "coverage",
        "parse_rate",
        "official_flexible_accuracy",
        "official_strict_accuracy",
        "letter_accuracy",
        "letter_accuracy_parsed",
        "statuses",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in [overall, *categories, *subcategories, *difficulties]:
            value = dict(row)
            value["statuses"] = json.dumps(value["statuses"], ensure_ascii=False)
            writer.writerow(value)

    print(f"\nSummary JSON: {json_path}")
    print(f"Summary CSV:  {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate locally scored ChatTS x TS-Haystack result files.

Per-sample correctness is produced by TS-Haystack's official dataset methods
during inference.  This script only aggregates those stored decisions; it does
not introduce a second parser or an LLM judge.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("capture24", "sleep_stages", "sleep_arousals", "ltaf", "uk_dale")
ALIASES = {
    "capture24_haystack": "capture24",
    "capture24_haystack_cot": "capture24",
    "sleep_psg_stages": "sleep_stages",
    "sleep_psg_arousals": "sleep_arousals",
    "ltaf_haystack": "ltaf",
    "uk-dale": "uk_dale",
    "uk_dale_haystack": "uk_dale",
}


def resolve_datasets(values: Iterable[str]) -> list[str]:
    names = [str(value).strip().lower() for value in values if str(value).strip()]
    if not names or "all" in names:
        return list(DATASETS)
    resolved: list[str] = []
    for name in names:
        canonical = ALIASES.get(name, name)
        if canonical not in DATASETS:
            raise ValueError(f"Unknown TS-Haystack dataset: {name}")
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [item for item in value if isinstance(item, dict)]


def context_label(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)):
        number = float(value)
        return f"{int(number) if number.is_integer() else number:g}s"
    text = str(value).strip()
    return text if text.endswith("s") or text == "full" else f"{text}s"


def summarize(
    records: list[dict[str, Any]],
    *,
    scope: str,
    dataset: str,
    key: str,
    expected_total: int | None = None,
) -> dict[str, Any]:
    statuses: dict[str, int] = defaultdict(int)
    generated = parsed = correct = 0
    ious: list[float] = []
    timestamp_errors: list[float] = []
    for item in records:
        status = str(item.get("status", "unknown"))
        statuses[status] += 1
        if status in {"ok", "evaluation_error"}:
            generated += 1
        if item.get("predicted_answer") not in (None, ""):
            parsed += 1
        if bool(item.get("correct", False)):
            correct += 1
        evaluation = item.get("evaluation") or {}
        if evaluation.get("iou") is not None:
            ious.append(float(evaluation["iou"]))
        if evaluation.get("timestamp_error_s") is not None:
            timestamp_errors.append(float(evaluation["timestamp_error_s"]))

    observed = len(records)
    total = max(observed, int(expected_total or 0))
    missing = max(0, total - observed)
    skipped = statuses.get("skipped_input_length", 0)
    errors = sum(
        count
        for status, count in statuses.items()
        if status not in {"ok", "skipped_input_length"}
    ) + missing
    return {
        "scope": scope,
        "dataset": dataset,
        "key": key,
        "total": total,
        "observed": observed,
        "generated": generated,
        "parsed": parsed,
        "correct": correct,
        "skipped_input_length": skipped,
        "errors_or_missing": errors,
        "coverage": generated / total if total else None,
        "parse_rate": parsed / generated if generated else None,
        "accuracy_strict": correct / total if total else None,
        "accuracy_generated": correct / generated if generated else None,
        "mean_iou": sum(ious) / len(ious) if ious else None,
        "mean_timestamp_error_s": (
            sum(timestamp_errors) / len(timestamp_errors)
            if timestamp_errors
            else None
        ),
        "statuses": dict(sorted(statuses.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ChatTS TS-Haystack scores")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    return parser.parse_args()


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    if not args.model_name or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
        for character in args.model_name
    ):
        raise ValueError(
            "--model-name may contain only letters, digits, dot, underscore, and dash"
        )
    results_root = Path(args.results_root).expanduser().resolve()
    datasets = resolve_datasets(args.datasets)

    all_records: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    expected_overall = 0

    for dataset in datasets:
        path = results_root / f"{dataset}_{args.model_name}" / "generated_answer.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing result file for {dataset}: {path}. "
                "Select only completed DATASETS or run inference first."
            )
        records = load_results(path)
        if not records:
            raise ValueError(f"Result file contains no sample records: {path}")
        expected = 0
        protocol = records[0].get("protocol") or {}
        expected = int(protocol.get("selected_dataset_size") or len(records))
        expected_overall += expected
        all_records.extend(records)
        dataset_rows.append(
            summarize(
                records,
                scope="dataset",
                dataset=dataset,
                key=dataset,
                expected_total=expected,
            )
        )

        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in records:
            by_task[str(item.get("task_type", "unknown"))].append(item)
            by_context[context_label(item.get("context_length_s"))].append(item)
        for task, values in sorted(by_task.items()):
            task_rows.append(
                summarize(
                    values,
                    scope="task",
                    dataset=dataset,
                    key=task,
                )
            )
        for context, values in sorted(by_context.items()):
            context_rows.append(
                summarize(
                    values,
                    scope="context",
                    dataset=dataset,
                    key=context,
                )
            )

    overall = summarize(
        all_records,
        scope="overall",
        dataset="all",
        key="all",
        expected_total=expected_overall,
    )

    print(
        f"{'Dataset':20} {'Generated':>13} {'Parsed':>13} {'Skipped':>8} "
        f"{'Strict':>9} {'GenAcc':>9}"
    )
    print("-" * 82)
    for row in dataset_rows:
        print(
            f"{row['dataset']:20} "
            f"{row['generated']:5}/{row['total']:<7} "
            f"{row['parsed']:5}/{row['generated']:<7} "
            f"{row['skipped_input_length']:8} "
            f"{_fmt(row['accuracy_strict']):>9} "
            f"{_fmt(row['accuracy_generated']):>9}"
        )
    print("-" * 82)
    print(
        f"{'Overall':20} "
        f"{overall['generated']:5}/{overall['total']:<7} "
        f"{overall['parsed']:5}/{overall['generated']:<7} "
        f"{overall['skipped_input_length']:8} "
        f"{_fmt(overall['accuracy_strict']):>9} "
        f"{_fmt(overall['accuracy_generated']):>9}"
    )

    summary = {
        "model": args.model_name,
        "protocol": (
            "Per-sample parsing and scoring use the official TS-Haystack "
            "QADataset.extract_answer/evaluate_answer methods."
        ),
        "overall": overall,
        "per_dataset": dataset_rows,
        "per_task": task_rows,
        "per_context": context_rows,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    json_path = results_root / f"ts_haystack_summary_{args.model_name}.json"
    csv_path = results_root / f"ts_haystack_summary_{args.model_name}.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    flat_rows = [overall, *dataset_rows, *task_rows, *context_rows]
    csv_fields = [
        "scope",
        "dataset",
        "key",
        "total",
        "observed",
        "generated",
        "parsed",
        "correct",
        "skipped_input_length",
        "errors_or_missing",
        "coverage",
        "parse_rate",
        "accuracy_strict",
        "accuracy_generated",
        "mean_iou",
        "mean_timestamp_error_s",
        "statuses",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in flat_rows:
            value = dict(row)
            value["statuses"] = json.dumps(value["statuses"], ensure_ascii=False)
            writer.writerow(value)

    print(f"\nSummary JSON: {json_path}")
    print(f"Summary CSV:  {csv_path}")
    if overall["skipped_input_length"]:
        print(
            "Note: Strict includes input-length skips in the denominator; "
            "GenAcc scores only samples actually generated."
        )


if __name__ == "__main__":
    main()

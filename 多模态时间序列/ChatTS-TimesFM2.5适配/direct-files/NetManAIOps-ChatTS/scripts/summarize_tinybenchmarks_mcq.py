#!/usr/bin/env python3
"""Summarize tinyBenchmarks MCQ results and screen for general-capability loss.

The official GPIRT/IRT++ estimates are the primary scores.  We also average the
logged binary per-item metrics to expose raw accuracy on each 100-item anchor
set.  A screening warning is not a statistical proof of catastrophic
forgetting; it is a configurable diagnostic requiring follow-up on full tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any


TASK_METRICS = OrderedDict(
    [
        ("tinyArc", "acc_norm"),
        ("tinyHellaswag", "acc_norm"),
        ("tinyMMLU", "acc_norm"),
        ("tinyTruthfulQA", "acc"),
        ("tinyWinogrande", "acc_norm"),
    ]
)


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --model {spec!r}; expected NAME=RESULT_DIRECTORY")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise ValueError(f"Invalid --model {spec!r}; expected NAME=RESULT_DIRECTORY")
    return name, Path(path).expanduser().resolve()


def latest_result_file(root: Path) -> Path:
    candidates = list(root.rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No results_*.json found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def extract_metric(task_result: dict[str, Any], metric: str) -> float | None:
    for key in (f"{metric},none", metric):
        value = finite_number(task_result.get(key))
        if value is not None:
            return value
    for key, raw_value in task_result.items():
        if key.split(",", 1)[0] == metric and "stderr" not in key:
            value = finite_number(raw_value)
            if value is not None:
                return value
    return None


def load_anchor_accuracy(
    result_path: Path, task: str, metric: str
) -> tuple[float | None, int, str | None]:
    timestamp = result_path.stem.removeprefix("results_")
    expected = result_path.parent / f"samples_{task}_{timestamp}.jsonl"
    if expected.is_file():
        sample_path = expected
    else:
        candidates = list(result_path.parent.glob(f"samples_{task}_*.jsonl"))
        if not candidates:
            return None, 0, None
        sample_path = max(
            candidates, key=lambda path: (path.stat().st_mtime_ns, path.name)
        )

    values: list[float] = []
    with sample_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {sample_path}:{line_number}") from exc
            value = finite_number(item.get(metric))
            if value is not None:
                values.append(value)
    return (
        statistics.fmean(values) if values else None,
        len(values),
        str(sample_path),
    )


def load_model_result(name: str, root: Path) -> dict[str, Any]:
    result_path = latest_result_file(root)
    with result_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    task_results = payload.get("results")
    if not isinstance(task_results, dict):
        raise ValueError(f"Missing results object in {result_path}")

    tasks: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for task, metric in TASK_METRICS.items():
        raw = task_results.get(task)
        if not isinstance(raw, dict):
            missing.append(task)
            tasks[task] = {
                "estimate": None,
                "anchor_accuracy": None,
                "anchor_n": 0,
                "sample_file": None,
            }
            continue
        estimate = extract_metric(raw, metric)
        anchor_accuracy, anchor_n, sample_file = load_anchor_accuracy(
            result_path, task, metric
        )
        if estimate is None:
            missing.append(task)
        tasks[task] = {
            "metric": metric,
            "estimate": estimate,
            "anchor_accuracy": anchor_accuracy,
            "anchor_n": anchor_n,
            "sample_file": sample_file,
        }

    estimates = [tasks[task]["estimate"] for task in TASK_METRICS]
    anchors = [tasks[task]["anchor_accuracy"] for task in TASK_METRICS]
    complete_estimates = all(value is not None for value in estimates)
    complete_anchors = all(value is not None for value in anchors)
    task_hashes = payload.get("task_hashes", {})
    if not isinstance(task_hashes, dict):
        task_hashes = {}

    return {
        "name": name,
        "root": str(root),
        "result_file": str(result_path),
        "tasks": tasks,
        "task_hashes": {task: task_hashes.get(task) for task in TASK_METRICS},
        "macro_estimate": statistics.fmean(estimates) if complete_estimates else None,
        "macro_anchor_accuracy": statistics.fmean(anchors) if complete_anchors else None,
        "missing_tasks": sorted(set(missing)),
        "config": payload.get("config", {}),
        "date": payload.get("date"),
    }


def protocol_match(reference: dict[str, Any], current: dict[str, Any]) -> str:
    reference_hashes = reference["task_hashes"]
    current_hashes = current["task_hashes"]
    known = 0
    for task in TASK_METRICS:
        left = reference_hashes.get(task)
        right = current_hashes.get(task)
        if left and right:
            known += 1
            if left != right:
                return "mismatch"
    return "match" if known == len(TASK_METRICS) else "unknown"


def percentage_points(current: float | None, reference: float | None) -> float | None:
    if current is None or reference is None:
        return None
    return 100.0 * (current - reference)


def enrich_rows(
    rows: list[dict[str, Any]], baseline_name: str, threshold_pp: float
) -> list[dict[str, Any]]:
    by_name = {row["name"]: row for row in rows}
    if baseline_name not in by_name:
        raise ValueError(f"Baseline {baseline_name!r} is not in model results")
    reference = by_name[baseline_name]

    for row in rows:
        match = protocol_match(reference, row)
        row["baseline"] = baseline_name
        row["protocol_match"] = match
        row["macro_delta_pp"] = percentage_points(
            row["macro_estimate"], reference["macro_estimate"]
        )
        row["macro_anchor_delta_pp"] = percentage_points(
            row["macro_anchor_accuracy"], reference["macro_anchor_accuracy"]
        )
        if row["macro_estimate"] is not None and reference["macro_estimate"]:
            row["retention_percent"] = (
                100.0 * row["macro_estimate"] / reference["macro_estimate"]
            )
        else:
            row["retention_percent"] = None

        dropped_tasks = 0
        large_drop_tasks = 0
        for task in TASK_METRICS:
            current_task = row["tasks"][task]
            reference_task = reference["tasks"][task]
            estimate_delta = percentage_points(
                current_task["estimate"], reference_task["estimate"]
            )
            anchor_delta = percentage_points(
                current_task["anchor_accuracy"],
                reference_task["anchor_accuracy"],
            )
            current_task["delta_pp"] = estimate_delta
            current_task["anchor_delta_pp"] = anchor_delta
            if estimate_delta is not None and estimate_delta < 0:
                dropped_tasks += 1
            if estimate_delta is not None and estimate_delta <= -threshold_pp:
                large_drop_tasks += 1

        row["dropped_tasks"] = dropped_tasks
        row["large_drop_tasks"] = large_drop_tasks
        if row["name"] == baseline_name:
            row["screening_flag"] = "reference"
        elif row["missing_tasks"]:
            row["screening_flag"] = "incomplete"
        elif match == "mismatch":
            row["screening_flag"] = "invalid_protocol_mismatch"
        elif (
            row["macro_delta_pp"] is not None
            and row["macro_delta_pp"] <= -threshold_pp
            and dropped_tasks >= 3
        ):
            row["screening_flag"] = "forgetting_warning"
        else:
            row["screening_flag"] = "no_large_drop_detected"
    return rows


def display_percent(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.2f}"


def display_number(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def flat_row(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "model": row["name"],
        "baseline": row["baseline"],
        "protocol_match": row["protocol_match"],
        "macro_estimate_percent": display_percent(row["macro_estimate"]),
        "macro_delta_pp": display_number(row["macro_delta_pp"]),
        "macro_anchor_accuracy_percent": display_percent(
            row["macro_anchor_accuracy"]
        ),
        "macro_anchor_delta_pp": display_number(row["macro_anchor_delta_pp"]),
        "retention_percent": display_number(row["retention_percent"]),
        "dropped_tasks": row["dropped_tasks"],
        "large_drop_tasks": row["large_drop_tasks"],
        "screening_flag": row["screening_flag"],
        "result_file": row["result_file"],
    }
    for task in TASK_METRICS:
        task_row = row["tasks"][task]
        output[f"{task}_estimate_percent"] = display_percent(task_row["estimate"])
        output[f"{task}_delta_pp"] = display_number(task_row["delta_pp"])
        output[f"{task}_anchor_accuracy_percent"] = display_percent(
            task_row["anchor_accuracy"]
        )
        output[f"{task}_anchor_n"] = task_row["anchor_n"]
    return output


def markdown_report(rows: list[dict[str, Any]], threshold_pp: float) -> str:
    lines = [
        "# tinyBenchmarks MCQ catastrophic-forgetting screen",
        "",
        "Primary values are official GPIRT/IRT++ full-benchmark estimates. "
        "Anchor accuracy is the unadjusted accuracy on each 100-item tiny set.",
        "",
        "| Model | GPIRT macro (%) | Δ vs base (pp) | Anchor macro (%) | "
        "Retention (%) | Protocol | Screen |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {macro} | {delta} | {anchor} | {retention} | {protocol} | {flag} |".format(
                name=row["name"],
                macro=display_percent(row["macro_estimate"]) or "N/A",
                delta=display_number(row["macro_delta_pp"]) or "N/A",
                anchor=display_percent(row["macro_anchor_accuracy"]) or "N/A",
                retention=display_number(row["retention_percent"]) or "N/A",
                protocol=row["protocol_match"],
                flag=row["screening_flag"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-task GPIRT/IRT++ estimates",
            "",
            "| Model | " + " | ".join(TASK_METRICS) + " |",
            "| --- | " + " | ".join(["---:"] * len(TASK_METRICS)) + " |",
        ]
    )
    for row in rows:
        values = [display_percent(row["tasks"][task]["estimate"]) or "N/A" for task in TASK_METRICS]
        lines.append(f"| {row['name']} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"`forgetting_warning` requires a macro drop of at least {threshold_pp:.2f} percentage "
            "points and declines on at least three of five tasks.",
            "This is a low-cost screening rule, not a statistical diagnosis. Confirm a warning "
            "with the corresponding full benchmarks and, ideally, multiple evaluation seeds.",
            "`protocol_match=mismatch` means prompts/documents differed, so the comparison is invalid.",
            "",
            "No experimental result is invented by this report; every score is read from lm-eval outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    rows: list[dict[str, Any]], output_dir: Path, basename: str, threshold_pp: float
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}.csv"
    json_path = output_dir / f"{basename}.json"
    markdown_path = output_dir / f"{basename}.md"

    flat_rows = [flat_row(row) for row in rows]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    payload = {
        "benchmark": "tinyBenchmarks multiple-choice subset",
        "tasks": list(TASK_METRICS),
        "primary_metric": "GPIRT/IRT++ estimate",
        "threshold_pp": threshold_pp,
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    markdown_path.write_text(markdown_report(rows, threshold_pp), encoding="utf-8")
    return csv_path, json_path, markdown_path


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        specs = []
        for name, offset in (("base", 0.0), ("chatts", -0.08)):
            model_root = root / name / "model"
            model_root.mkdir(parents=True)
            stamp = "2026-08-05T00-00-00"
            results = {}
            hashes = {}
            for task, metric in TASK_METRICS.items():
                results[task] = {f"{metric},none": 0.70 + offset}
                hashes[task] = f"hash-{task}"
                sample_path = model_root / f"samples_{task}_{stamp}.jsonl"
                with sample_path.open("w", encoding="utf-8") as stream:
                    for index in range(100):
                        value = 1.0 if index < (70 + int(offset * 100)) else 0.0
                        stream.write(json.dumps({"doc_id": index, metric: value}) + "\n")
            result_path = model_root / f"results_{stamp}.json"
            result_path.write_text(
                json.dumps({"results": results, "task_hashes": hashes}),
                encoding="utf-8",
            )
            specs.append((name, root / name))

        rows = enrich_rows(
            [load_model_result(name, path) for name, path in specs],
            baseline_name="base",
            threshold_pp=5.0,
        )
        assert rows[0]["screening_flag"] == "reference"
        assert rows[1]["screening_flag"] == "forgetting_warning"
        assert abs(rows[1]["macro_delta_pp"] + 8.0) < 1e-9
        output_paths = write_outputs(rows, root / "summary", "test", 5.0)
        assert all(path.is_file() for path in output_paths)
    print("Self-test passed: result discovery, GPIRT/raw aggregation, protocol hashes, and warning rule.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--baseline")
    parser.add_argument("--threshold-pp", type=float, default=5.0)
    parser.add_argument("--output-dir", default="exp/tinybenchmarks_mcq")
    parser.add_argument("--basename", default="tinybenchmarks_mcq_summary")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.model:
        raise SystemExit("At least one --model NAME=DIR is required.")
    if args.threshold_pp < 0:
        raise SystemExit("--threshold-pp must be non-negative.")

    specs = [parse_model_spec(spec) for spec in args.model]
    names = [name for name, _ in specs]
    if len(names) != len(set(names)):
        raise SystemExit("Duplicate model names are not allowed.")
    baseline = args.baseline or names[0]
    rows = [load_model_result(name, path) for name, path in specs]
    enrich_rows(rows, baseline, args.threshold_pp)
    paths = write_outputs(
        rows,
        Path(args.output_dir).expanduser().resolve(),
        args.basename,
        args.threshold_pp,
    )
    report = Path(paths[2]).read_text(encoding="utf-8")
    print(report)
    print("Saved:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

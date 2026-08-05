#!/usr/bin/env python3
"""Summarize raw ChatTS-vLLM tinyBenchmarks MCQ results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any


TASK_METRICS = OrderedDict(
    [
        ("tinyArc", "accuracy_norm"),
        ("tinyHellaswag", "accuracy_norm"),
        ("tinyMMLU", "accuracy_norm"),
        ("tinyTruthfulQA", "mc2_probability_mass"),
        ("tinyWinogrande", "accuracy_norm"),
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


def find_metrics(root: Path) -> Path:
    direct = root / "metrics.json"
    if direct.is_file():
        return direct
    candidates = list(root.rglob("metrics.json")) if root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No metrics.json found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def load_model_result(name: str, root: Path) -> dict[str, Any]:
    path = find_metrics(root)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, dict):
        raise ValueError(f"Missing tasks object in {path}")

    tasks: dict[str, dict[str, Any]] = {}
    missing = []
    for task, expected_metric in TASK_METRICS.items():
        raw = raw_tasks.get(task)
        if not isinstance(raw, dict):
            missing.append(task)
            tasks[task] = {
                "metric": expected_metric,
                "score": None,
                "num_samples": 0,
                "protocol_hash": None,
                "dataset_source": None,
            }
            continue
        score = finite_number(raw.get("score"))
        metric = str(raw.get("metric", ""))
        if score is None or metric != expected_metric:
            missing.append(task)
        tasks[task] = {
            "metric": metric,
            "score": score,
            "num_samples": int(raw.get("num_samples", 0)),
            "protocol_hash": raw.get("protocol_hash"),
            "dataset_source": raw.get("dataset_source"),
        }

    scores = [tasks[task]["score"] for task in TASK_METRICS]
    complete = all(score is not None for score in scores)
    return {
        "name": name,
        "root": str(root),
        "result_file": str(path),
        "model_path": payload.get("model_path"),
        "tasks": tasks,
        "macro_score": math.fsum(scores) / len(scores) if complete else None,
        "missing_tasks": sorted(set(missing)),
        "evaluator": payload.get("evaluator"),
        "evaluator_version": payload.get("evaluator_version"),
    }


def protocol_match(reference: dict[str, Any], current: dict[str, Any]) -> str:
    known = 0
    for task in TASK_METRICS:
        left = reference["tasks"][task].get("protocol_hash")
        right = current["tasks"][task].get("protocol_hash")
        if left and right:
            known += 1
            if left != right:
                return "mismatch"
    return "match" if known == len(TASK_METRICS) else "unknown"


def delta_pp(current: float | None, reference: float | None) -> float | None:
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
        row["baseline"] = baseline_name
        row["protocol_match"] = protocol_match(reference, row)
        row["macro_delta_pp"] = delta_pp(row["macro_score"], reference["macro_score"])
        if row["macro_score"] is not None and reference["macro_score"] not in (None, 0):
            row["retention_percent"] = 100.0 * row["macro_score"] / reference["macro_score"]
        else:
            row["retention_percent"] = None

        dropped_tasks = 0
        large_drop_tasks = 0
        for task in TASK_METRICS:
            current_task = row["tasks"][task]
            reference_task = reference["tasks"][task]
            change = delta_pp(current_task["score"], reference_task["score"])
            current_task["delta_pp"] = change
            if change is not None and change < 0:
                dropped_tasks += 1
            if change is not None and change <= -threshold_pp:
                large_drop_tasks += 1
        row["dropped_tasks"] = dropped_tasks
        row["large_drop_tasks"] = large_drop_tasks

        if row["name"] == baseline_name:
            flag = "reference"
        elif row["missing_tasks"]:
            flag = "incomplete"
        elif row["protocol_match"] == "mismatch":
            flag = "invalid_protocol_mismatch"
        elif (
            row["macro_delta_pp"] is not None
            and row["macro_delta_pp"] <= -threshold_pp
            and dropped_tasks >= 3
        ):
            flag = "forgetting_warning"
        else:
            flag = "no_large_drop_detected"
        row["screening_flag"] = flag
    return rows


def percent(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.2f}"


def number(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def flat_row(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "model": row["name"],
        "baseline": row["baseline"],
        "macro_raw_percent": percent(row["macro_score"]),
        "macro_delta_pp": number(row["macro_delta_pp"]),
        "retention_percent": number(row["retention_percent"]),
        "protocol_match": row["protocol_match"],
        "dropped_tasks": row["dropped_tasks"],
        "large_drop_tasks": row["large_drop_tasks"],
        "screening_flag": row["screening_flag"],
        "result_file": row["result_file"],
    }
    for task in TASK_METRICS:
        task_row = row["tasks"][task]
        output[f"{task}_percent"] = percent(task_row["score"])
        output[f"{task}_delta_pp"] = number(task_row["delta_pp"])
        output[f"{task}_n"] = task_row["num_samples"]
    return output


def markdown_report(rows: list[dict[str, Any]], threshold_pp: float) -> str:
    lines = [
        "# tinyBenchmarks MCQ catastrophic-forgetting screen",
        "",
        "All values are raw scores from the local tiny sets through ChatTS vLLM. "
        "No GPIRT/IRT++ full-benchmark estimate is claimed.",
        "",
        "| Model | Macro raw (%) | Δ vs base (pp) | Retention (%) | Protocol | Screen |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {percent(row['macro_score']) or 'N/A'} | "
            f"{number(row['macro_delta_pp']) or 'N/A'} | "
            f"{number(row['retention_percent']) or 'N/A'} | "
            f"{row['protocol_match']} | {row['screening_flag']} |"
        )

    lines.extend(
        [
            "",
            "## Per-task raw scores (%)",
            "",
            "| Model | " + " | ".join(TASK_METRICS) + " |",
            "| --- | " + " | ".join(["---:"] * len(TASK_METRICS)) + " |",
        ]
    )
    for row in rows:
        values = [percent(row["tasks"][task]["score"]) or "N/A" for task in TASK_METRICS]
        lines.append(f"| {row['name']} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`tinyTruthfulQA` is its MC2 correct-answer probability mass; the other four "
            "columns are length-normalized multiple-choice accuracy.",
            f"`forgetting_warning` requires a macro decline of at least {threshold_pp:.2f} "
            "percentage points and declines on at least three of five tasks.",
            "This is a low-cost screen, not a statistical proof. Confirm warnings on the "
            "corresponding full benchmarks.",
            "`protocol_match=mismatch` means the local documents/prompts differed, so the "
            "model-to-model delta is not interpretable.",
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
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "benchmark": "tinyBenchmarks local multiple-choice subset",
                "tasks": list(TASK_METRICS),
                "primary_metric": "raw tiny-set macro score",
                "threshold_pp": threshold_pp,
                "rows": rows,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    markdown_path.write_text(markdown_report(rows, threshold_pp), encoding="utf-8")
    return csv_path, json_path, markdown_path


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        specs = []
        for name, offset in (("base", 0.0), ("chatts", -0.08)):
            model_root = root / name
            model_root.mkdir()
            tasks = {}
            for task, metric in TASK_METRICS.items():
                tasks[task] = {
                    "task": task,
                    "metric": metric,
                    "score": 0.70 + offset,
                    "num_samples": 100,
                    "protocol_hash": f"hash-{task}",
                    "dataset_source": f"/{task}.parquet",
                }
            (model_root / "metrics.json").write_text(
                json.dumps(
                    {
                        "evaluator": "ChatTS vLLM prompt_logprobs",
                        "evaluator_version": 1,
                        "tasks": tasks,
                    }
                ),
                encoding="utf-8",
            )
            specs.append((name, model_root))
        rows = enrich_rows(
            [load_model_result(name, path) for name, path in specs], "base", 5.0
        )
        assert rows[0]["screening_flag"] == "reference"
        assert rows[1]["screening_flag"] == "forgetting_warning"
        assert abs(rows[1]["macro_delta_pp"] + 8.0) < 1e-9
        outputs = write_outputs(rows, root / "summary", "test", 5.0)
        assert all(path.is_file() for path in outputs)
    print("Self-test passed: raw metrics, protocol hashes, deltas, and warning rule.")


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
    rows = enrich_rows(
        [load_model_result(name, path) for name, path in specs],
        baseline,
        args.threshold_pp,
    )
    paths = write_outputs(
        rows,
        Path(args.output_dir).expanduser().resolve(),
        args.basename,
        args.threshold_pp,
    )
    print(Path(paths[2]).read_text(encoding="utf-8"))
    print("Saved:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

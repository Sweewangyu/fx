"""Evaluate ChatTS TSRBench result files without a judge model or API."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


TASK_CATEGORIES = {
    "perception": "Perception",
    "causal_reasoning": "Reasoning",
    "inductive_reasoning": "Reasoning",
    "numerical_reasoning": "Reasoning",
    "temporal_relation_reasoning": "Reasoning",
    "etiological_reasoning": "Reasoning",
    "abductive_reasoning": "Reasoning",
    "deductive_reasoning": "Reasoning",
    "time_series_forecasting": "Prediction",
    "event_prediction": "Prediction",
    "qualitative_decision": "Decision-Making",
    "quantitative_decision": "Decision-Making",
}

TASK_ALIASES = {
    "math_reasoning": "numerical_reasoning",
    "event_forecast": "event_prediction",
    "pattern_decision": "qualitative_decision",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover(dataset_root: Path, requested: Iterable[str]) -> list[tuple[str, Path]]:
    by_stem = {path.stem: path for path in dataset_root.rglob("*.jsonl")}
    names = [TASK_ALIASES.get(name, name) for name in requested]
    if not names or "all" in names:
        names = list(TASK_CATEGORIES)
    missing = [name for name in names if name not in by_stem]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing)}")
    return [(name, by_stem[name]) for name in names]


def extract_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("answer")
    text = str(value).strip()
    text = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            match = re.search(r"[A-G]", str(parsed["answer"]), re.IGNORECASE)
            if match:
                return match.group(0).upper()
    patterns = (
        r"<answer>\s*([A-G])\s*</answer>",
        r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([A-G])",
        r"(?:^|\n)\s*(?:final\s+answer\s*[:：]?\s*)?([A-G])[.)]",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def ground_truth(sample: dict[str, Any], dataset_name: str) -> str | None:
    if dataset_name == "abductive_reasoning":
        value = sample.get("multiple_choice_question", {}).get("answer")
    else:
        value = sample.get("answer")
    if value is None:
        return None
    match = re.search(r"[A-G]", str(value), re.IGNORECASE)
    return match.group(0).upper() if match else None


def load_answers(path: Path) -> dict[int, dict[str, Any]]:
    answers: dict[int, dict[str, Any]] = {}
    if not path.is_dir():
        return answers
    for result_file in sorted(path.glob("generated_answer*.json")):
        with result_file.open("r", encoding="utf-8") as stream:
            for item in json.load(stream):
                if isinstance(item, dict) and "idx" in item:
                    answers[int(item["idx"])] = item
    return answers


def metric_row(
    dataset_name: str,
    category: str,
    samples: list[dict[str, Any]],
    answers: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    generated = parsed = correct = 0
    subtask_scores: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        item = answers.get(index)
        if not item or not item.get("response"):
            continue
        generated += 1
        answer = extract_answer(item.get("answer")) or extract_answer(item.get("response"))
        truth = ground_truth(sample, dataset_name)
        if answer is None or truth is None:
            continue
        parsed += 1
        score = int(answer == truth)
        correct += score
        if dataset_name == "perception":
            subtask_scores.setdefault(str(sample.get("category", "Unknown")), []).append(score)

    size = len(samples)
    return {
        "category": category,
        "dataset": dataset_name,
        "dataset_size": size,
        "generated": generated,
        "parsed": parsed,
        "correct": correct,
        "coverage": generated / size if size else None,
        "parse_rate": parsed / generated if generated else None,
        "accuracy_strict": correct / size if size else None,
        "accuracy_parsed": correct / parsed if parsed else None,
    }, subtask_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ChatTS on TSRBench")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    return parser.parse_args()


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    results_root = Path(args.results_root).resolve()
    datasets = discover(dataset_root, args.datasets)

    rows: list[dict[str, Any]] = []
    perception_subtasks: dict[str, float] = {}
    for name, dataset_path in datasets:
        samples = read_jsonl(dataset_path)
        answers = load_answers(results_root / f"{name}_{args.model_name}")
        row, subtasks = metric_row(name, TASK_CATEGORIES[name], samples, answers)
        rows.append(row)
        for subtask, values in subtasks.items():
            if values:
                perception_subtasks[subtask] = sum(values) / len(values)

    total_size = sum(row["dataset_size"] for row in rows)
    total_generated = sum(row["generated"] for row in rows)
    total_parsed = sum(row["parsed"] for row in rows)
    total_correct = sum(row["correct"] for row in rows)
    overall = {
        "dataset_size": total_size,
        "generated": total_generated,
        "parsed": total_parsed,
        "correct": total_correct,
        "coverage": total_generated / total_size if total_size else None,
        "parse_rate": total_parsed / total_generated if total_generated else None,
        "accuracy_strict": total_correct / total_size if total_size else None,
        "accuracy_parsed": total_correct / total_parsed if total_parsed else None,
    }

    print(
        f"{'Category':16} {'Dataset':31} {'Done':>9} {'Parsed':>9} "
        f"{'Strict':>9} {'ParsedAcc':>10}"
    )
    print("-" * 92)
    for row in rows:
        print(
            f"{row['category']:16} {row['dataset']:31} "
            f"{row['generated']:4}/{row['dataset_size']:<4} "
            f"{row['parsed']:4}/{row['generated']:<4} "
            f"{_fmt(row['accuracy_strict']):>9} "
            f"{_fmt(row['accuracy_parsed']):>10}"
        )
    print("-" * 92)
    print(
        f"{'Overall':16} {'all':31} "
        f"{overall['generated']:4}/{overall['dataset_size']:<4} "
        f"{overall['parsed']:4}/{overall['generated']:<4} "
        f"{_fmt(overall['accuracy_strict']):>9} "
        f"{_fmt(overall['accuracy_parsed']):>10}"
    )

    summary = {
        "model": args.model_name,
        "per_dataset": rows,
        "perception_subtasks": perception_subtasks,
        "overall": overall,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    json_path = results_root / f"tsrbench_summary_{args.model_name}.json"
    csv_path = results_root / f"tsrbench_summary_{args.model_name}.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary JSON: {json_path}")
    print(f"Summary CSV:  {csv_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .hashing import canonical_json, hash_object


class MetricsError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_values(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield name.lower(), child
            yield from _walk_values(child, name)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if 1.0 < number <= 100.0:
        number /= 100.0
    return number


def _find_metric(payload: Any, aliases: tuple[str, ...]) -> float | None:
    aliases = tuple(alias.lower() for alias in aliases)
    for name, value in _walk_values(payload):
        leaf = name.rsplit(".", 1)[-1]
        if leaf in aliases:
            number = _numeric(value)
            if number is not None:
                return number
    return None


def _find_mapping(payload: Any, aliases: tuple[str, ...]) -> dict[str, Any] | None:
    wanted = {alias.lower() for alias in aliases}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() in wanted and isinstance(value, dict):
                return value
            if found := _find_mapping(value, aliases):
                return found
    elif isinstance(payload, list):
        for value in payload:
            if found := _find_mapping(value, aliases):
                return found
    return None


SUITE_ALIASES = {
    "tsrbench": ("tsrbench", "tsr_bench", "tsr"),
    "timeseriesexam": ("timeseriesexam", "time_series_exam", "exam"),
    "ts_haystack": ("ts_haystack", "ts-haystack", "haystack"),
    "tinybenchmarks": ("tinybenchmarks", "tinybench", "tiny_benchmarks"),
}


def _suite_from_path(path: Path) -> str | None:
    lowered = "/".join(path.parts).lower()
    for canonical, aliases in SUITE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return canonical
    return None


def _canonical_suite(name: str) -> str | None:
    lowered = name.lower()
    for canonical, aliases in SUITE_ALIASES.items():
        if lowered == canonical or lowered in aliases:
            return canonical
    return None


def _normalize_suite(name: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise MetricsError(f"Metrics for {name} must be a JSON object")
    result: dict[str, Any] = {"raw": payload}
    if name in {"tsrbench", "timeseriesexam"}:
        result["strict_accuracy"] = _find_metric(
            payload, ("strict_accuracy", "strict_acc", "accuracy_strict")
        )
        result["flexible_accuracy"] = _find_metric(
            payload, ("flexible_accuracy", "parsed_accuracy", "flex_acc", "accuracy")
        )
        result["coverage"] = _find_metric(
            payload, ("coverage", "parse_coverage", "parsed_coverage")
        )
    elif name == "ts_haystack":
        result["mean_iou"] = _find_metric(payload, ("mean_iou", "iou", "average_iou"))
    elif name == "tinybenchmarks":
        result["average_accuracy"] = _find_metric(
            payload,
            ("average_accuracy", "mean_accuracy", "accuracy", "avg_accuracy", "macro_score"),
        )
        task_metrics = _find_mapping(payload, ("tasks", "task_metrics", "task_scores")) or {}
        if isinstance(task_metrics, dict):
            result["tasks"] = {
                task: metric
                for task, raw in task_metrics.items()
                if (
                    metric := _numeric(raw)
                    if not isinstance(raw, dict)
                    else _find_metric(raw, ("accuracy", "score", "acc"))
                )
                is not None
            }
    return result


def load_metrics(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    root_metrics = root / "metrics.json"
    suites: dict[str, dict[str, Any]] = {}
    if root_metrics.is_file():
        payload = _read_json(root_metrics)
        candidate_suites = payload.get("suites", payload) if isinstance(payload, dict) else {}
        if isinstance(candidate_suites, dict):
            for raw_name, suite_payload in candidate_suites.items():
                if canonical := _canonical_suite(str(raw_name)):
                    suites[canonical] = _normalize_suite(canonical, suite_payload)
    if len(suites) < 2:
        for path in sorted(root.rglob("*.json")):
            if path == root_metrics or path.name in {"run_manifest.json", "TRAINING_COMPLETE.json"}:
                continue
            suite = _suite_from_path(path.relative_to(root))
            if not suite or suite in suites:
                continue
            if not any(token in path.name.lower() for token in ("metric", "summary", "result", "score")):
                continue
            try:
                payload = _read_json(path)
                normalized = _normalize_suite(suite, payload)
            except (OSError, json.JSONDecodeError, MetricsError):
                continue
            key = (
                "strict_accuracy"
                if suite in {"tsrbench", "timeseriesexam"}
                else "mean_iou" if suite == "ts_haystack" else "average_accuracy"
            )
            if normalized.get(key) is not None:
                suites[suite] = normalized
    for suite in ("tsrbench", "timeseriesexam"):
        if suite in suites and suites[suite].get("strict_accuracy") is None:
            raise MetricsError(f"{suite} metrics lack strict_accuracy")
    result = {"suites": suites}
    if all(name in suites for name in ("tsrbench", "timeseriesexam")):
        result["primary_score"] = (
            suites["tsrbench"]["strict_accuracy"]
            + suites["timeseriesexam"]["strict_accuracy"]
        ) / 2.0
        coverages = [
            suites[name].get("coverage")
            for name in ("tsrbench", "timeseriesexam")
            if suites[name].get("coverage") is not None
        ]
        result["coverage"] = sum(coverages) / len(coverages) if coverages else None
    return result


def apply_gates(
    metrics: dict[str, Any],
    baseline: dict[str, Any] | None,
    thresholds: dict[str, Any],
    *,
    require_guards: bool,
) -> dict[str, Any]:
    result = dict(metrics)
    result["guard_evaluated"] = require_guards
    reasons: list[str] = []
    current_suites = metrics.get("suites", {})
    baseline_suites = (baseline or {}).get("suites", {})
    if metrics.get("primary_score") is None:
        reasons.append("missing primary score")
    if metrics.get("coverage") is None:
        reasons.append("missing main-suite coverage")
    for suite_name in ("tsrbench", "timeseriesexam"):
        if current_suites.get(suite_name, {}).get("coverage") is None:
            reasons.append(f"missing {suite_name} coverage")
    tiny = current_suites.get("tinybenchmarks", {})
    base_tiny = baseline_suites.get("tinybenchmarks", {})
    haystack = current_suites.get("ts_haystack", {}).get("mean_iou")
    base_haystack = baseline_suites.get("ts_haystack", {}).get("mean_iou")
    if require_guards:
        if tiny.get("average_accuracy") is None:
            reasons.append("missing tinyBench average accuracy")
        if not tiny.get("tasks"):
            reasons.append("missing tinyBench task scores")
        expected_tasks = thresholds.get(
            "tiny_expected_tasks",
            [
                "tinyArc",
                "tinyHellaswag",
                "tinyMMLU",
                "tinyTruthfulQA",
                "tinyWinogrande",
            ],
        )
        current_tasks = tiny.get("tasks", {})
        for task in expected_tasks:
            if task not in current_tasks:
                reasons.append(f"missing expected tinyBench task score: {task}")
        if haystack is None:
            reasons.append("missing TS-Haystack mean IoU")
    if baseline is None:
        result["gate_pass"] = not reasons
        result["gate_reasons"] = reasons
        return result
    if tiny.get("average_accuracy") is not None and base_tiny.get("average_accuracy") is not None:
        drop = base_tiny["average_accuracy"] - tiny["average_accuracy"]
        if drop > float(thresholds["tiny_average_max_drop"]):
            reasons.append(f"tiny average drop {drop:.6f}")
        for task, base_value in base_tiny.get("tasks", {}).items():
            if task not in tiny.get("tasks", {}):
                if require_guards:
                    reasons.append(f"missing tinyBench task score: {task}")
                continue
            task_drop = base_value - tiny["tasks"][task]
            if task_drop > float(thresholds["tiny_task_max_drop"]):
                reasons.append(f"tiny {task} drop {task_drop:.6f}")
    if haystack is not None and base_haystack is not None:
        drop = base_haystack - haystack
        if drop > float(thresholds["haystack_iou_max_drop"]):
            reasons.append(f"TS-Haystack IoU drop {drop:.6f}")
    if metrics.get("coverage") is not None and baseline.get("coverage") is not None:
        drop = baseline["coverage"] - metrics["coverage"]
        if drop > float(thresholds["coverage_max_drop"]):
            reasons.append(f"coverage drop {drop:.6f}")
    result["gate_pass"] = not reasons
    result["gate_reasons"] = reasons
    return result


CORRECT_KEYS = (
    "is_correct",
    "correct",
    "strict_correct",
    "official_strict_correct",
)
PREDICTION_KEYS = (
    "prediction",
    "pred",
    "predicted_answer",
    "model_answer",
    "response",
    "answer",
)
GOLD_KEYS = (
    "gold",
    "gold_answer",
    "ground_truth",
    "target",
    "label",
    "correct_answer",
)


def _records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        value = json.loads(line)
                        if isinstance(value, dict):
                            yield value
        else:
            value = _read_json(path)
            if isinstance(value, list):
                yield from (item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                for key in ("samples", "results", "predictions", "items", "data"):
                    if isinstance(value.get(key), list):
                        yield from (item for item in value[key] if isinstance(item, dict))
                        break
    except (OSError, json.JSONDecodeError):
        return


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    return next((record[key] for key in keys if key in record), None)


def _answer_letter(value: Any) -> str | None:
    import re

    if isinstance(value, dict):
        value = value.get("answer")
    if value is None:
        return None
    match = re.search(r"[A-G]", str(value), re.IGNORECASE)
    return match.group(0).upper() if match else None


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    bulky = {"timeseries", "time_series", "series", "context_series"}
    return {
        key: value
        for key, value in record.items()
        if key not in bulky and not (key == "context" and isinstance(value, (list, dict)))
    }


def _tsrbench_joined_badcases(
    root: Path, dataset_root: Path
) -> Iterator[tuple[dict[str, Any], bool]]:
    datasets = {path.stem: path for path in dataset_root.rglob("*.jsonl")}
    result_root = root / "tsrbench"
    if not result_root.is_dir():
        return
    for result_path in sorted(result_root.rglob("generated_answer*.json")):
        matches = [name for name in datasets if result_path.parent.name.startswith(name + "_")]
        if not matches:
            continue
        dataset_name = max(matches, key=len)
        samples = list(_records(datasets[dataset_name]))
        results = {
            int(item["idx"]): item
            for item in _records(result_path)
            if isinstance(item.get("idx"), int)
        }
        for index, sample in enumerate(samples):
            generated = results.get(index, {})
            prediction = _answer_letter(generated.get("answer")) or _answer_letter(
                generated.get("response")
            )
            gold_raw = (
                sample.get("multiple_choice_question", {}).get("answer")
                if dataset_name == "abductive_reasoning"
                else sample.get("answer")
            )
            gold = _answer_letter(gold_raw)
            correct = prediction is not None and gold is not None and prediction == gold
            yield {
                "badcase_id": hash_object(
                    {"suite": "tsrbench", "dataset": dataset_name, "index": index}
                ),
                "suite": "tsrbench",
                "task": dataset_name,
                "difficulty": sample.get("difficulty"),
                "prediction": prediction,
                "gold": gold,
                "question": generated.get("question_text") or sample.get("question"),
                "source_file": str(result_path.relative_to(root)),
                "source_record": _compact_record(sample),
            }, correct


def extract_badcases(
    output_dir: str | Path,
    destination: str | Path,
    dataset_roots: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    total = bad = diagnostic_bad = 0
    suite_counts: dict[str, dict[str, int]] = {}

    def increment(suite: str, key: str) -> None:
        suite_counts.setdefault(
            suite, {"scored_records": 0, "badcases": 0, "diagnostic_badcases": 0}
        )[key] += 1

    with temporary.open("w", encoding="utf-8") as stream:
        tsr_root = (dataset_roots or {}).get("tsrbench")
        if tsr_root and Path(tsr_root).is_dir():
            for payload, correct in _tsrbench_joined_badcases(root, Path(tsr_root)):
                total += 1
                increment("tsrbench", "scored_records")
                if not correct:
                    stream.write(canonical_json(payload) + "\n")
                    bad += 1
                    increment("tsrbench", "badcases")
        for path in sorted([*root.rglob("*.jsonl"), *root.rglob("*.json")]):
            if path.name in {"metrics.json", "run_manifest.json"} or "badcase" in path.name.lower():
                continue
            suite = _suite_from_path(path.relative_to(root)) or "unknown"
            for index, record in enumerate(_records(path)):
                correct = _first(record, CORRECT_KEYS)
                diagnostic_only = False
                if (
                    not isinstance(correct, bool)
                    and suite == "tinybenchmarks"
                    and record.get("metric") == "acc_norm"
                    and isinstance(record.get("score"), (int, float))
                    and not isinstance(record.get("score"), bool)
                    and record.get("score") in {0, 1}
                    and isinstance(record.get("predicted_index"), int)
                    and isinstance(record.get("gold_indices"), list)
                ):
                    correct = float(record["score"]) == 1.0
                elif (
                    not isinstance(correct, bool)
                    and suite == "tinybenchmarks"
                    and record.get("metric") == "mc2"
                    and isinstance(record.get("predicted_index"), int)
                    and isinstance(record.get("gold_indices"), list)
                ):
                    # MC2 is probability mass, not binary accuracy.  A top-1
                    # miss is useful for qualitative diagnosis but must never
                    # be counted as an official scored error.
                    if record["predicted_index"] in record["gold_indices"]:
                        continue
                    diagnostic_only = True
                if not isinstance(correct, bool):
                    if not diagnostic_only:
                        continue
                if not diagnostic_only:
                    total += 1
                    increment(suite, "scored_records")
                if correct is True:
                    continue
                prediction = _first(record, PREDICTION_KEYS)
                gold = _first(record, GOLD_KEYS)
                if suite == "tinybenchmarks":
                    prediction = record.get("predicted_index", prediction)
                    gold = record.get("gold_indices", gold)
                task = record.get("task") or record.get("category")
                if suite == "tinybenchmarks" and task is None and path.stem.startswith(
                    "samples_"
                ):
                    task = path.stem.removeprefix("samples_")
                payload = {
                    "badcase_id": hash_object(
                        {"path": str(path.relative_to(root)), "index": index, "record": record}
                    ),
                    "suite": suite,
                    "task": task,
                    "source": record.get("source") or record.get("dataset_source"),
                    "difficulty": record.get("difficulty"),
                    "prediction": prediction,
                    "gold": gold,
                    "question": record.get("question") or record.get("prompt") or record.get("input"),
                    "source_file": str(path.relative_to(root)),
                    "source_record": _compact_record(record),
                    "accounting": (
                        "diagnostic_top1_only"
                        if diagnostic_only
                        else "official_binary_correctness"
                    ),
                }
                stream.write(canonical_json(payload) + "\n")
                if diagnostic_only:
                    diagnostic_bad += 1
                    increment(suite, "diagnostic_badcases")
                else:
                    bad += 1
                    increment(suite, "badcases")
    temporary.replace(target)
    return {
        "scored_records": total,
        "badcases": bad,
        "diagnostic_badcases": diagnostic_bad,
        "by_suite": suite_counts,
        "path": str(target),
    }


def sample_badcases(path: str | Path, maximum: int = 64) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    source = Path(path)
    if not source.is_file():
        return []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            key = f"{value.get('suite')}|{value.get('task')}|{value.get('difficulty')}"
            buckets.setdefault(key, []).append(value)
    selected: list[dict[str, Any]] = []
    while len(selected) < maximum and buckets:
        for key in sorted(list(buckets)):
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                if len(selected) >= maximum:
                    break
            if not buckets[key]:
                del buckets[key]
    return selected

#!/usr/bin/env python3
"""Local API for browsing large ChatTS-style JSONL datasets.

The browser never reads multi-GB files or calls the model directly. This server
provides random-access records, template members, audit metadata, merged labels,
and a cached Qwen English-to-Chinese translation proxy.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import mmap
import os
import random
import re
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import parse_qs, urlparse

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency boundary
    raise RuntimeError("PyYAML is required: python -m pip install pyyaml") from exc


NUMBER_RE = re.compile(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.I)
TS_TAG_RE = re.compile(r"<ts>.*?<ts\s*/>", re.I | re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.I | re.S)
ANSWER_LABEL_RE = re.compile(r"\s*(?:\(?[A-Za-z]\)?|true|false)\s*", re.I)
LEAK_PATTERNS = (
    ("answer_leakage", re.compile(r"must\s+(?:end|conclude).*?answer\s*:", re.I | re.S)),
    ("answer_leakage", re.compile(r"final sentence must conclude with", re.I)),
    ("answer_leakage", re.compile(r"correct answer (?:is|must be)", re.I)),
)
VISUAL_WORD_RE = re.compile(r"\b(?:graph|plot|chart|x-axis|y-axis|visual(?:ization)?)\b", re.I)


class ApiError(Exception):
    """An expected API failure with an explicit HTTP status."""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = int(status)


class AnnotationConflict(ApiError):
    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.CONFLICT)


def compact_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"configuration must be an object: {path}")
    return payload


def normalize_template(prompt: str) -> str:
    """Match DataTaste's source-aware taxonomy template normalization."""
    text = prompt.lower().replace("<ts><ts/>", " <ts> ")
    text = TS_TAG_RE.sub(" <ts> ", text)
    text = re.sub(r"https?://\S+", "<url>", text)
    text = NUMBER_RE.sub("<num>", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_structure(answer: str) -> str:
    normalized = TS_TAG_RE.sub("<ts>", answer).strip()
    lowered = normalized.lower()
    if "<ts>" in lowered:
        return "tagged_time_series" if "<answer>" in lowered else "time_series"
    inner = ANSWER_RE.findall(normalized)
    candidate = inner[-1].strip() if inner else normalized
    if ANSWER_LABEL_RE.fullmatch(candidate):
        return "choice_label"
    if "<think>" in lowered and "<answer>" in lowered:
        return "reasoning_with_answer"
    if "<answer>" in lowered:
        return "tagged_answer"
    if "\n" in answer:
        return "multiline_text"
    return "text"


def quality_template_id(prompt: str, output: str) -> str:
    question = NUMBER_RE.sub("<num>", TS_TAG_RE.sub("<ts>", prompt))
    payload = json.dumps(
        {"answer_structure": answer_structure(output), "question_template": question},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def extract_field(line: bytes, key: str, from_end: bool = False) -> Optional[str]:
    marker = (f'"{key}"').encode()
    index = line.rfind(marker) if from_end else line.find(marker)
    if index < 0:
        return None
    colon = line.find(b":", index + len(marker))
    if colon < 0:
        return None
    text = line[colon + 1 :].lstrip().decode("utf-8", errors="replace")
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) else None


def extract_answer_class(output: str) -> str:
    matches = ANSWER_RE.findall(output)
    value = matches[-1].strip() if matches else output.strip()
    if len(value) <= 80 and "\n" not in value:
        return value
    return answer_structure(output)


def parse_choice_answer(value: Any) -> Optional[str]:
    """Parse an A-G answer using the same conventions as evaluate_tsrbench.py."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("answer")
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", text, flags=re.I)
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:  # noqa: BLE001 - permissive model-output parser
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            match = re.search(r"[A-G]", str(parsed["answer"]), re.I)
            if match:
                return match.group(0).upper()
    patterns = (
        r"<answer>\s*([A-G])\s*</answer>",
        r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([A-G])",
        r"(?:^|\n)\s*(?:final\s+answer\s*[:：]?\s*)?([A-G])[.)]",
        r"^\s*([A-G])\s*(?:\r?\n|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).upper()
    return None


def string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def strict_nonnegative_int(value: Any, field: str) -> int:
    """Accept an integer or decimal integer string, never bools/floats."""
    if isinstance(value, bool):
        raise ApiError(f"{field} must be a non-negative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        result = int(value.strip())
    else:
        raise ApiError(f"{field} must be a non-negative integer")
    if result < 0:
        raise ApiError(f"{field} must be a non-negative integer")
    return result


def iter_result_rows(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    """Read model results from either a JSON array/object or JSONL."""
    with path.open("r", encoding="utf-8") as stream:
        first = ""
        while True:
            character = stream.read(1)
            if not character:
                return
            if not character.isspace():
                first = character
                break
        stream.seek(0)
        if first in "[{":
            try:
                payload = json.load(stream)
            except json.JSONDecodeError:
                # A JSONL file also begins with `{`; fall through to line mode.
                stream.seek(0)
            else:
                if isinstance(payload, dict):
                    payload = payload.get("results") or payload.get("responses") or [payload]
                if not isinstance(payload, list):
                    raise ValueError(f"result file must contain an array or JSONL objects: {path}")
                for index, item in enumerate(payload):
                    if isinstance(item, dict):
                        yield index, item
                return
        for index, line in enumerate(stream):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"result JSONL row must be an object: {path}:{index + 1}")
            yield index, item


def issue_flags(prompt: str, output: str, audit: Optional[Mapping[str, Any]]) -> list[str]:
    flags: list[str] = []
    for label, pattern in LEAK_PATTERNS:
        if pattern.search(prompt):
            flags.append(label)
            break
    if VISUAL_WORD_RE.search(output) and not VISUAL_WORD_RE.search(prompt):
        flags.append("visual_grounding_mismatch")
    verifier = (audit or {}).get("verifier") or {}
    if isinstance(verifier, dict) and "unverified" in str(verifier.get("reasoning_status", "")):
        flags.append("unverified_reasoning")
    return flags


def line_offsets(path: Path) -> Iterable[tuple[int, int, int]]:
    """Yield record index, byte offset and length for JSONL or a top-level JSON array."""
    with path.open("rb") as stream:
        if path.stat().st_size == 0:
            return
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            first = 0
            while first < len(mapped) and chr(mapped[first]).isspace():
                first += 1
            if first < len(mapped) and mapped[first] == ord("["):
                yield from json_array_offsets(mapped, first)
                return
            start = 0
            index = 0
            size = len(mapped)
            while start < size:
                end = mapped.find(b"\n", start)
                if end < 0:
                    end = size
                length = end - start
                if length:
                    yield index, start, length
                    index += 1
                start = end + 1


def json_array_offsets(mapped: mmap.mmap, array_start: int) -> Iterable[tuple[int, int, int]]:
    """Find top-level JSON array elements without loading a benchmark file in memory."""
    index = 0
    start: Optional[int] = None
    depth = 0
    in_string = False
    escaped = False
    cursor = array_start + 1
    size = len(mapped)
    while cursor < size:
        byte = mapped[cursor]
        if start is None:
            if chr(byte).isspace() or byte == ord(","):
                cursor += 1
                continue
            if byte == ord("]"):
                return
            start = cursor
            depth = 0

        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
        elif byte == ord('"'):
            in_string = True
        elif byte in (ord("{"), ord("[")):
            depth += 1
        elif byte in (ord("}"), ord("]")):
            depth -= 1

        next_byte = mapped[cursor + 1] if cursor + 1 < size else None
        if not in_string and depth == 0 and next_byte in (ord(","), ord("]")):
            assert start is not None
            end = cursor + 1
            yield index, start, end - start
            index += 1
            start = None
        cursor += 1


def read_slice(path: Optional[Path], offset: Optional[int], length: Optional[int]) -> Optional[Dict[str, Any]]:
    if path is None or offset is None or length is None or offset < 0:
        return None
    with path.open("rb") as stream:
        stream.seek(offset)
        line = stream.read(length)
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def largest_triangle_three_buckets(values: list[float], threshold: int) -> list[float]:
    """LTTB-like index-preserving downsampling for chart transport."""
    if threshold >= len(values) or threshold <= 2:
        return values
    every = (len(values) - 2) / (threshold - 2)
    sampled = [values[0]]
    a = 0
    for i in range(threshold - 2):
        avg_start = int(math.floor((i + 1) * every)) + 1
        avg_end = min(int(math.floor((i + 2) * every)) + 1, len(values))
        bucket = values[avg_start:avg_end] or [values[-1]]
        avg_y = sum(bucket) / len(bucket)
        range_start = int(math.floor(i * every)) + 1
        range_end = min(int(math.floor((i + 1) * every)) + 1, len(values) - 1)
        ax, ay = a, values[a]
        max_area = -1.0
        next_a = range_start
        for idx in range(range_start, max(range_start + 1, range_end)):
            area = abs((ax - avg_start) * (values[idx] - ay) - (ax - idx) * (avg_y - ay))
            if area > max_area:
                max_area = area
                next_a = idx
        sampled.append(values[next_a])
        a = next_a
    sampled.append(values[-1])
    return sampled


def summarize_series(series: Any, max_points: int) -> list[Dict[str, Any]]:
    if not isinstance(series, list):
        return []
    result = []
    for index, channel in enumerate(series):
        if not isinstance(channel, list):
            continue
        numeric = [float(value) for value in channel if isinstance(value, (int, float)) and math.isfinite(value)]
        if not numeric:
            result.append({"index": index, "length": len(channel), "values": [], "stats": None})
            continue
        ordered = sorted(numeric)
        mean = sum(numeric) / len(numeric)
        variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
        result.append(
            {
                "index": index,
                "length": len(channel),
                "values": largest_triangle_three_buckets(numeric, max_points),
                "stats": {
                    "min": min(numeric),
                    "max": max(numeric),
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "first": numeric[0],
                    "last": numeric[-1],
                },
            }
        )
    return result


@dataclass(frozen=True)
class Source:
    name: str
    family: str
    split: str
    training_role: str
    path: Path
    audit: Optional[Path]
    annotations: Optional[Path]
    schema: str = "chatts"
    benchmark_task: Optional[str] = None


TSRBENCH_TASKS = {
    "perception": ("perception/perception.jsonl", "Perception"),
    "causal_reasoning": ("reasoning/causal_reasoning.jsonl", "Reasoning"),
    "inductive_reasoning": ("reasoning/inductive_reasoning.jsonl", "Reasoning"),
    "numerical_reasoning": ("reasoning/numerical_reasoning.jsonl", "Reasoning"),
    "temporal_relation_reasoning": ("reasoning/temporal_relation_reasoning.jsonl", "Reasoning"),
    "etiological_reasoning": ("reasoning/etiological_reasoning.jsonl", "Reasoning"),
    "abductive_reasoning": ("reasoning/abductive_reasoning.jsonl", "Reasoning"),
    "deductive_reasoning": ("reasoning/deductive_reasoning.jsonl", "Reasoning"),
    "time_series_forecasting": ("prediction/time_series_forecasting.jsonl", "Prediction"),
    "event_prediction": ("prediction/event_prediction.jsonl", "Prediction"),
    "qualitative_decision": ("decision/qualitative_decision.jsonl", "Decision-Making"),
    "quantitative_decision": ("decision/quantitative_decision.jsonl", "Decision-Making"),
}

TSRBENCH_TASK_ALIASES = {
    "math_reasoning": "numerical_reasoning",
    "event_forecast": "event_prediction",
    "pattern_decision": "qualitative_decision",
}


def format_choices(choices: Any) -> str:
    labels = "ABCDEFG"
    if isinstance(choices, dict):
        return "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))
    if isinstance(choices, list):
        return "\n".join(
            str(value) if re.match(r"^[A-G][.)]\s*", str(value), re.I)
            else f"{labels[index]}. {value}"
            for index, value in enumerate(choices[: len(labels)])
        )
    return str(choices) if choices not in (None, "") else ""


def tsrbench_fields(record: Mapping[str, Any], task: str) -> Dict[str, Any]:
    """Expose TSRBench's standard and abductive schemas through one view model."""
    if task == "abductive_reasoning" or (
        isinstance(record.get("multiple_choice_question"), dict)
        and isinstance(record.get("numerical_time_series"), dict)
    ):
        context = record.get("context") if isinstance(record.get("context"), dict) else {}
        mcq = record.get("multiple_choice_question") or {}
        numerical = record.get("numerical_time_series") or {}
        history_events = list(context.get("history_events") or [])
        history_times = list(context.get("history_times") or [])
        future_events = list(context.get("future_events") or [])
        future_times = list(context.get("future_times") or [])
        past = "\n".join(f"- {when}: {event}" for when, event in zip(history_times[-10:], history_events[-10:]))
        future = "\n".join(f"- {when}: {event}" for when, event in zip(future_times[:10], future_events[:10]))
        question = (
            "Past Events (History):\n" + past
            + "\n\n... [A CRITICAL EVENT HAPPENED HERE] ...\n\nFuture Events:\n" + future
            + "\n\nQuestion: " + str(mcq.get("question", ""))
        )
        choices = mcq.get("choices")
        answer = str(mcq.get("answer", ""))
        series = []
        names = []
        for name, values in numerical.items():
            if not isinstance(values, dict):
                continue
            combined = list(values.get("history") or []) + list(values.get("future") or [])
            if combined:
                names.append(str(name))
                series.append(combined)
        return {
            "input": question,
            "output": answer,
            "timeseries": series,
            "choices": choices,
            "series_names": names,
            "domain": record.get("domain") or "Basketball",
            "category": record.get("category") or "abductive_reasoning",
            "raw_metadata": {
                key: value for key, value in record.items()
                if key not in {"context", "multiple_choice_question", "numerical_time_series"}
            },
        }

    question = str(record.get("question", ""))
    choices = record.get("choices")
    return {
        "input": question,
        "output": str(record.get("answer", "")),
        "timeseries": record.get("timeseries") or [],
        "choices": choices,
        "series_names": record.get("name_of_series") or [],
        "domain": record.get("domain"),
        "category": record.get("category") or task,
        "raw_metadata": {
            key: value for key, value in record.items()
            if key not in {"question", "answer", "timeseries", "choices", "name_of_series"}
        },
    }


def view_fields(record: Mapping[str, Any], source: Source) -> Dict[str, Any]:
    if source.schema == "tsrbench":
        return tsrbench_fields(record, source.benchmark_task or source.name.removeprefix("tsrbench_"))
    return {
        "input": str(record.get("input", "")),
        "output": str(record.get("output", "")),
        "timeseries": record.get("timeseries") or [],
        "choices": record.get("choices"),
        "series_names": record.get("name_of_series") or [],
        "domain": record.get("domain"),
        "category": record.get("category"),
        "raw_metadata": {},
    }


class DatasetStore:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = load_yaml(self.config_path)
        base = self.config_path.parent
        data_config = self.config.get("data") or {}
        self.data_root = resolve_path(base, data_config["root"])
        self.registry_path = resolve_path(base, data_config["registry"])
        self.template_stats_path = resolve_path(base, data_config.get("template_stats", "template_stats.json"))
        self.annotations_dir = resolve_path(base, data_config.get("annotations_dir", "merged_labels/annotations"))
        server = self.config.get("server") or {}
        self.cache_dir = resolve_path(base, server.get("cache_dir", ".cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        review_config = self.config.get("review") or {}
        self.state_db = resolve_path(
            base, review_config.get("state_db", server.get("state_db", "state/inspector-state.sqlite3"))
        )
        self.state_db.parent.mkdir(parents=True, exist_ok=True)
        self._state_lock = threading.RLock()
        self._results_import_lock = threading.RLock()
        self._init_state_db()
        self.max_chart_points = int(server.get("max_chart_points", 900))
        self.template_page_size = int(server.get("template_page_size", 12))
        self.tsrbench_status: Dict[str, Any] = {
            "configured": False,
            "found": False,
            "root": None,
            "checked_paths": [],
            "tasks_found": 0,
            "tasks_expected": len(TSRBENCH_TASKS),
        }
        self.sources = self._load_sources()
        self.template_stats = self._load_template_stats()
        self._locks: Dict[str, threading.Lock] = {name: threading.Lock() for name in self.sources}
        self.translation_db = self.cache_dir / "translations.sqlite"
        self._init_translation_db()
        self.results_root: Optional[Path] = None
        self.results_root_source: Optional[str] = None
        self.results_scanned_at: Optional[int] = None
        self.results_error: Optional[str] = None
        self._load_results_root()
        results_config = self.config.get("evaluation_results") or {}
        if bool(results_config.get("auto_scan", False)) and self.results_root and self.results_root.is_dir():
            try:
                self.import_model_results(self.results_root)
            except Exception as exc:  # noqa: BLE001 - startup diagnostic, server remains usable
                self.results_error = str(exc)

    def _state_connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.state_db), timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def _init_state_db(self) -> None:
        with sqlite3.connect(str(self.state_db), timeout=60) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_runs (
                    run_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    prompt_mode TEXT,
                    root_path TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    imported_at INTEGER NOT NULL,
                    file_count INTEGER NOT NULL,
                    expected_rows INTEGER NOT NULL,
                    matched_rows INTEGER NOT NULL,
                    invalid_rows INTEGER NOT NULL,
                    missing_rows INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS model_responses (
                    run_id TEXT NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
                    dataset_name TEXT NOT NULL,
                    line_index INTEGER NOT NULL,
                    response TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    reported_answer TEXT,
                    parsed_answer TEXT,
                    gold_answer TEXT,
                    correctness TEXT NOT NULL,
                    generation_status TEXT NOT NULL,
                    parse_status TEXT NOT NULL,
                    error TEXT,
                    question_text TEXT,
                    reasoning_path TEXT,
                    input_tokens INTEGER,
                    processed_input_tokens INTEGER,
                    num_tokens INTEGER,
                    source_file TEXT NOT NULL,
                    source_item_index INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (run_id, dataset_name, line_index)
                );
                CREATE INDEX IF NOT EXISTS model_response_lookup
                    ON model_responses(dataset_name, line_index);
                CREATE INDEX IF NOT EXISTS model_badcases
                    ON model_responses(run_id, dataset_name, correctness, parse_status, generation_status);
                CREATE TABLE IF NOT EXISTS human_annotations (
                    dataset_name TEXT NOT NULL,
                    line_index INTEGER NOT NULL,
                    record_hash TEXT NOT NULL,
                    source_signature TEXT NOT NULL DEFAULT '',
                    verdict TEXT NOT NULL CHECK (verdict IN ('good', 'bad')),
                    comment TEXT NOT NULL DEFAULT '',
                    annotator TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (dataset_name, line_index)
                );
                CREATE INDEX IF NOT EXISTS human_annotation_verdict
                    ON human_annotations(dataset_name, verdict, line_index);
                CREATE TABLE IF NOT EXISTS annotation_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    line_index INTEGER NOT NULL,
                    record_hash TEXT NOT NULL,
                    old_verdict TEXT,
                    new_verdict TEXT,
                    comment TEXT NOT NULL DEFAULT '',
                    annotator TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                );
                """
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(human_annotations)")
            }
            if "source_signature" not in columns:
                connection.execute(
                    "ALTER TABLE human_annotations ADD COLUMN source_signature TEXT NOT NULL DEFAULT ''"
                )

    def _setting(self, key: str) -> Optional[str]:
        with self._state_connect() as connection:
            row = connection.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else None

    def _save_setting(self, key: str, value: str) -> None:
        with self._state_lock, self._state_connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO app_settings(key,value,updated_at) VALUES (?,?,?)",
                (key, value, int(time.time())),
            )

    def _load_results_root(self) -> None:
        results_config = self.config.get("evaluation_results") or {}
        env_value = (
            os.getenv("TSRBENCH_RESULTS_ROOT", "").strip()
            or os.getenv("TSR_RESULTS_ROOT", "").strip()
        )
        saved_value = self._setting("results_root")
        configured_value = str(results_config.get("root") or "").strip()
        value = env_value or saved_value or configured_value
        if not value:
            return
        self.results_root = resolve_path(self.config_path.parent, value)
        self.results_root_source = "environment" if env_value else "settings" if saved_value else "config"

    def model_results_settings(self) -> Dict[str, Any]:
        return {
            "results_root": str(self.results_root) if self.results_root else None,
            "source": self.results_root_source,
            "exists": bool(self.results_root and self.results_root.is_dir()),
            "scanned_at": self.results_scanned_at,
            "error": self.results_error,
        }

    def set_results_root(self, value: str, import_now: bool = True) -> Dict[str, Any]:
        if not isinstance(value, str) or not value.strip():
            raise ApiError("root must be a non-empty server filesystem path")
        root = resolve_path(self.config_path.parent, value.strip())
        if not root.is_dir():
            raise ApiError(f"model results directory does not exist: {root}", HTTPStatus.NOT_FOUND)
        self.results_root = root
        self.results_root_source = "settings"
        self._save_setting("results_root", str(root))
        imported = self.import_model_results(root) if import_now else None
        return {"settings": self.model_results_settings(), "import": imported}

    def _result_file_identity(
        self, path: Path, scan_root: Path, model_override: Optional[str]
    ) -> Optional[tuple[str, str, str]]:
        task_names = {**{name: name for name in TSRBENCH_TASKS}, **TSRBENCH_TASK_ALIASES}
        tasks = sorted(task_names, key=len, reverse=True)
        parts = list(path.parent.parts)
        for position in range(len(parts) - 1, -1, -1):
            component = parts[position]
            for task in tasks:
                if component == task or component.startswith(task + "_"):
                    inferred = component[len(task) + 1 :] if component != task else ""
                    anchor = Path(*parts[:position]) if position else path.parent
                    descendants = [part for part in parts[position + 1 :] if part not in {"", os.sep}]
                    if model_override:
                        model_name = str(model_override).strip()
                    elif inferred or descendants:
                        # Some official runners preserve Hugging Face model IDs
                        # as path segments, e.g. task_OpenGVLab/InternVL3-8B.
                        # Keeping every segment prevents different checkpoints
                        # under one organization from collapsing into one run.
                        model_name = "/".join([part for part in [inferred, *descendants] if part])
                    else:
                        model_name = anchor.name if anchor.resolve() != scan_root.resolve() else scan_root.name
                    if not model_name:
                        return None
                    try:
                        scope = str(anchor.resolve().relative_to(scan_root.resolve())) or "."
                    except ValueError:
                        scope = str(anchor.resolve())
                    return task_names[task], model_name, scope
        return None

    def _source_record_fields(self, name: str, index: int) -> tuple[Dict[str, Any], str]:
        source = self.sources[name]
        row = self._row(name, index)
        record = read_slice(source.path, row["byte_offset"], row["byte_length"])
        if record is None:
            raise ValueError(f"invalid JSON at {source.path}:{index + 1}")
        digest = hashlib.sha256(compact_json(record)).hexdigest()
        return view_fields(record, source), digest

    @staticmethod
    def _int_or_none(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None and value != "" else None
        except (TypeError, ValueError):
            return None

    def import_model_results(
        self, path: str | Path | None = None, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._results_import_lock:
            return self._import_model_results_unlocked(path, model_name)

    def _import_model_results_unlocked(
        self, path: str | Path | None = None, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        target = resolve_path(self.config_path.parent, path) if path else self.results_root
        if target is None:
            raise ApiError("model results root is not configured")
        if not target.exists():
            raise ApiError(f"model results path does not exist: {target}", HTTPStatus.NOT_FOUND)
        scan_root = target if target.is_dir() else target.parent
        if target.is_file():
            files = [target]
        else:
            files = sorted(
                {
                    *target.rglob("generated_answer*.json"),
                    *target.rglob("generated_answer*.jsonl"),
                }
            )
        if not files:
            raise ApiError(f"no generated_answer JSON/JSONL files found under: {target}", HTTPStatus.NOT_FOUND)

        groups: Dict[tuple[str, str], list[tuple[str, Path]]] = {}
        ignored = []
        for result_file in files:
            identity = self._result_file_identity(result_file, scan_root, model_name)
            if identity is None:
                ignored.append(str(result_file))
                continue
            task, inferred_model, scope = identity
            groups.setdefault((inferred_model, scope), []).append((task, result_file))
        if not groups:
            raise ApiError("no result directory matched a known TSRBench task name")

        imported_runs = []
        imported_run_ids: set[str] = set()
        scan_had_file_errors = False
        for (inferred_model, scope), group_files in sorted(groups.items()):
            diagnostics: list[Dict[str, Any]] = []
            rows_by_mode: Dict[str, Dict[tuple[str, int], Dict[str, Any]]] = {}
            task_names = sorted({task for task, _ in group_files})
            expected_rows = 0
            for task in task_names:
                dataset_name = f"tsrbench_{task}"
                if dataset_name in self.sources:
                    expected_rows += self.count(dataset_name)
                else:
                    diagnostics.append(
                        {"type": "dataset_missing", "task": task, "dataset": dataset_name}
                    )

            for task, result_file in sorted(group_files, key=lambda item: str(item[1])):
                dataset_name = f"tsrbench_{task}"
                if dataset_name not in self.sources:
                    continue
                dataset_size = self.count(dataset_name)
                try:
                    result_rows = iter_result_rows(result_file)
                    for source_item_index, item in result_rows:
                        raw_index = next(
                            (
                                item[key]
                                for key in ("idx", "index", "sample_index", "sample_idx", "line_index")
                                if key in item
                            ),
                            None,
                        )
                        try:
                            line_index = strict_nonnegative_int(raw_index, "result idx")
                        except ApiError:
                            diagnostics.append(
                                {
                                    "type": "invalid_index",
                                    "file": str(result_file),
                                    "item": source_item_index,
                                    "value": raw_index,
                                }
                            )
                            continue
                        if line_index < 0 or line_index >= dataset_size:
                            diagnostics.append(
                                {
                                    "type": "index_out_of_range",
                                    "file": str(result_file),
                                    "item": source_item_index,
                                    "index": line_index,
                                    "dataset_size": dataset_size,
                                }
                            )
                            continue
                        fields, _ = self._source_record_fields(dataset_name, line_index)
                        response_value = next(
                            (
                                item[key]
                                for key in ("response", "model_response", "completion", "output", "text")
                                if item.get(key) not in (None, "")
                            ),
                            "",
                        )
                        response = string_value(response_value)
                        raw_response = string_value(item.get("raw_response")) or response
                        effective_response = raw_response or response
                        reported = item.get("answer")
                        parsed_answer = parse_choice_answer(reported) or parse_choice_answer(effective_response)
                        gold_answer = parse_choice_answer(fields.get("output"))
                        error = string_value(item.get("error")) or None
                        if error or effective_response.startswith("INPUT_SKIPPED:"):
                            generation_status = "error"
                        elif not effective_response.strip():
                            generation_status = "empty"
                        else:
                            generation_status = "ok"
                        parse_status = "parsed" if parsed_answer else "unparsed"
                        correctness = (
                            "correct"
                            if generation_status == "ok" and parsed_answer and gold_answer and parsed_answer == gold_answer
                            else "incorrect"
                            if generation_status == "ok" and parsed_answer and gold_answer
                            else "unknown"
                        )
                        prompt_mode = str(item.get("prompt_mode") or "unknown")
                        stored_keys = {
                            "idx", "index", "sample_index", "sample_idx", "line_index",
                            "question_text", "response", "model_response", "completion", "output", "text",
                            "raw_response", "answer", "prompt_mode", "reasoning_path", "error",
                            "input_tokens", "processed_input_tokens", "num_tokens",
                        }
                        candidate = {
                            "dataset_name": dataset_name,
                            "line_index": line_index,
                            "response": response,
                            "raw_response": raw_response,
                            "reported_answer": string_value(reported) or None,
                            "parsed_answer": parsed_answer,
                            "gold_answer": gold_answer,
                            "correctness": correctness,
                            "generation_status": generation_status,
                            "parse_status": parse_status,
                            "error": error,
                            "question_text": string_value(item.get("question_text")) or None,
                            "reasoning_path": string_value(item.get("reasoning_path")) or None,
                            "input_tokens": self._int_or_none(item.get("input_tokens")),
                            "processed_input_tokens": self._int_or_none(item.get("processed_input_tokens")),
                            "num_tokens": self._int_or_none(item.get("num_tokens")),
                            "source_file": str(result_file),
                            "source_item_index": source_item_index,
                            "metadata_json": json.dumps(
                                {key: value for key, value in item.items() if key not in stored_keys},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                        rows_by_key = rows_by_mode.setdefault(prompt_mode, {})
                        key = (dataset_name, line_index)
                        previous = rows_by_key.get(key)
                        if previous is not None:
                            conflict = any(
                                previous[field] != candidate[field]
                                for field in ("response", "raw_response", "reported_answer", "error")
                            )
                            diagnostics.append(
                                {
                                    "type": "duplicate_index_conflict" if conflict else "duplicate_index",
                                    "dataset": dataset_name,
                                    "index": line_index,
                                    "prompt_mode": prompt_mode,
                                    "previous_file": previous["source_file"],
                                    "file": str(result_file),
                                }
                            )
                            if conflict:
                                previous["generation_status"] = "error"
                                previous["correctness"] = "unknown"
                                previous["error"] = "conflicting duplicate inference outputs for this task+idx"
                            continue
                        rows_by_key[key] = candidate
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    scan_had_file_errors = True
                    diagnostics.append(
                        {"type": "file_error", "file": str(result_file), "message": str(exc)}
                    )

            run_root = (scan_root / scope).resolve() if scope != "." else scan_root.resolve()
            signatures = []
            for _, result_file in sorted(group_files, key=lambda item: str(item[1])):
                stat = result_file.stat()
                signatures.append(f"{result_file}:{stat.st_size}:{stat.st_mtime_ns}")
            source_signature = hashlib.sha256("\n".join(signatures).encode()).hexdigest()
            for prompt_mode, rows_by_key in sorted(rows_by_mode.items()):
                run_id = hashlib.sha256(
                    f"{run_root}\n{inferred_model}\n{prompt_mode}".encode("utf-8")
                ).hexdigest()[:24]
                matched_rows = len(rows_by_key)
                missing_rows = max(0, expected_rows - matched_rows)
                invalid_rows = len(diagnostics)
                status = "complete" if matched_rows == expected_rows and not diagnostics else "partial"
                now = int(time.time())
                with self._state_lock, self._state_connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "DELETE FROM model_runs WHERE root_path=? AND model_name=? AND prompt_mode=?",
                        (str(run_root), inferred_model, prompt_mode),
                    )
                    connection.execute(
                        """INSERT INTO model_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            run_id,
                            inferred_model,
                            inferred_model,
                            prompt_mode,
                            str(run_root),
                            source_signature,
                            now,
                            len(group_files),
                            expected_rows,
                            matched_rows,
                            invalid_rows,
                            missing_rows,
                            status,
                            json.dumps(diagnostics[:500], ensure_ascii=False, separators=(",", ":")),
                        ),
                    )
                    connection.executemany(
                        """INSERT INTO model_responses (
                            run_id,dataset_name,line_index,response,raw_response,reported_answer,
                            parsed_answer,gold_answer,correctness,generation_status,parse_status,error,
                            question_text,reasoning_path,input_tokens,processed_input_tokens,num_tokens,
                            source_file,source_item_index,metadata_json
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        [
                            (
                                run_id,
                                row["dataset_name"],
                                row["line_index"],
                                row["response"],
                                row["raw_response"],
                                row["reported_answer"],
                                row["parsed_answer"],
                                row["gold_answer"],
                                row["correctness"],
                                row["generation_status"],
                                row["parse_status"],
                                row["error"],
                                row["question_text"],
                                row["reasoning_path"],
                                row["input_tokens"],
                                row["processed_input_tokens"],
                                row["num_tokens"],
                                row["source_file"],
                                row["source_item_index"],
                                row["metadata_json"],
                            )
                            for row in rows_by_key.values()
                        ],
                    )
                imported_run_ids.add(run_id)
                imported_runs.append(
                    {
                        "run_id": run_id,
                        "model_name": inferred_model,
                        "prompt_mode": prompt_mode,
                        "root_path": str(run_root),
                        "files": len(group_files),
                        "expected": expected_rows,
                        "matched": matched_rows,
                        "missing": missing_rows,
                        "invalid": invalid_rows,
                        "status": status,
                        "diagnostics": diagnostics[:100],
                    }
                )

        # A full-directory refresh is authoritative for that subtree. Remove
        # runs whose source directories disappeared, but preserve the last
        # successful import if any file was unreadable during this scan.
        if target.is_dir() and not scan_had_file_errors:
            with self._state_lock, self._state_connect() as connection:
                stale_ids = []
                for row in connection.execute("SELECT run_id,root_path FROM model_runs"):
                    try:
                        in_scope = Path(str(row["root_path"])).resolve().is_relative_to(scan_root.resolve())
                    except (OSError, ValueError):
                        in_scope = False
                    if in_scope and str(row["run_id"]) not in imported_run_ids:
                        stale_ids.append(str(row["run_id"]))
                connection.executemany(
                    "DELETE FROM model_runs WHERE run_id=?", [(run_id,) for run_id in stale_ids]
                )

        self.results_root = scan_root.resolve()
        self.results_scanned_at = int(time.time())
        self.results_error = None
        return {
            "root": str(scan_root.resolve()),
            "files": len(files),
            "ignored_files": ignored,
            "runs": imported_runs,
            "scanned_at": self.results_scanned_at,
        }

    @staticmethod
    def _response_status(row: Mapping[str, Any]) -> str:
        generation = str(row["generation_status"])
        if generation != "ok":
            return generation
        if str(row["parse_status"]) != "parsed":
            return "unparsed"
        return str(row["correctness"])

    def model_runs(self) -> list[Dict[str, Any]]:
        with self._state_connect() as connection:
            rows = connection.execute(
                """SELECT r.*,
                          SUM(CASE WHEN p.generation_status='ok' AND p.parse_status='parsed'
                                        AND p.correctness='correct' THEN 1 ELSE 0 END) AS correct_rows,
                          SUM(CASE WHEN p.generation_status='ok' AND p.parse_status='parsed'
                                        AND p.correctness='incorrect' THEN 1 ELSE 0 END) AS incorrect_rows,
                          SUM(CASE WHEN p.generation_status='ok' AND p.parse_status='unparsed'
                                   THEN 1 ELSE 0 END) AS unparsed_rows,
                          SUM(CASE WHEN p.generation_status IN ('error','empty') THEN 1 ELSE 0 END) AS error_rows
                   FROM model_runs r LEFT JOIN model_responses p ON p.run_id=r.run_id
                   GROUP BY r.run_id ORDER BY r.imported_at DESC, r.display_name"""
            ).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "run_id": row["run_id"],
                    "model_name": row["model_name"],
                    "display_name": row["display_name"],
                    "prompt_mode": row["prompt_mode"],
                    "root_path": row["root_path"],
                    "imported_at": int(row["imported_at"]),
                    "file_count": int(row["file_count"]),
                    "expected": int(row["expected_rows"]),
                    "matched": int(row["matched_rows"]),
                    "total_rows": int(row["expected_rows"]),
                    "matched_rows": int(row["matched_rows"]),
                    "missing": int(row["missing_rows"]),
                    "invalid": int(row["invalid_rows"]),
                    "correct": int(row["correct_rows"] or 0),
                    "incorrect": int(row["incorrect_rows"] or 0),
                    "unparsed": int(row["unparsed_rows"] or 0),
                    "errors": int(row["error_rows"] or 0),
                    "status": row["status"],
                    "source_file": row["root_path"],
                    "diagnostics": json.loads(row["diagnostics_json"] or "[]"),
                }
            )
        return result

    def evaluation_status(self) -> Dict[str, Any]:
        settings = self.model_results_settings()
        scanned_at = settings["scanned_at"]
        return {
            "results_root": settings["results_root"],
            "root_source": settings["source"],
            "exists": settings["exists"],
            "runs": self.model_runs(),
            "scanned_at": (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(scanned_at))
                if scanned_at else None
            ),
            "scanned_at_epoch": scanned_at,
            "error": settings["error"],
        }

    def model_responses(self, name: str, index: int) -> list[Dict[str, Any]]:
        with self._state_connect() as connection:
            runs = connection.execute(
                """SELECT DISTINCT r.* FROM model_runs r
                   JOIN model_responses seen ON seen.run_id=r.run_id
                   WHERE seen.dataset_name=? ORDER BY r.display_name,r.run_id""",
                (name,),
            ).fetchall()
            actual = {
                str(row["run_id"]): row
                for row in connection.execute(
                    "SELECT * FROM model_responses WHERE dataset_name=? AND line_index=?",
                    (name, index),
                )
            }
        gold_answer: Optional[str] = None
        if runs:
            fields, _ = self._source_record_fields(name, index)
            gold_answer = parse_choice_answer(fields.get("output"))
        result = []
        for run in runs:
            row = actual.get(str(run["run_id"]))
            if row is None:
                result.append(
                    {
                        "run_id": run["run_id"],
                        "model_name": run["model_name"],
                        "display_name": run["display_name"],
                        "prompt_mode": run["prompt_mode"],
                        "response": "",
                        "raw_response": "",
                        "reported_answer": None,
                        "parsed_answer": None,
                        "gold_answer": gold_answer,
                        "correctness": "unknown",
                        "generation_status": "missing",
                        "parse_status": "missing",
                        "status": "missing",
                        "error": None,
                        "reasoning_path": None,
                        "input_tokens": None,
                        "processed_input_tokens": None,
                        "num_tokens": None,
                        "source_file": run["root_path"],
                        "latency_ms": None,
                    }
                )
                continue
            result.append(
                {
                    "run_id": row["run_id"],
                    "model_name": run["model_name"],
                    "display_name": run["display_name"],
                    "prompt_mode": run["prompt_mode"],
                    "response": row["response"],
                    "raw_response": row["raw_response"],
                    "reported_answer": row["reported_answer"],
                    "parsed_answer": row["parsed_answer"],
                    "gold_answer": row["gold_answer"],
                    "correctness": row["correctness"],
                    "generation_status": row["generation_status"],
                    "parse_status": row["parse_status"],
                    "status": self._response_status(row),
                    "error": row["error"],
                    "reasoning_path": row["reasoning_path"],
                    "input_tokens": row["input_tokens"],
                    "processed_input_tokens": row["processed_input_tokens"],
                    "num_tokens": row["num_tokens"],
                    "source_file": row["source_file"],
                    "latency_ms": None,
                }
            )
        return result

    def badcases(
        self,
        run_id: str,
        dataset: Optional[str],
        status: str,
        offset: int,
        limit: int,
    ) -> Dict[str, Any]:
        allowed = {"all", "correct", "incorrect", "unparsed", "error", "empty", "unknown"}
        if status not in allowed:
            raise ApiError(f"invalid badcase status: {status}")
        where = ["p.run_id=?"]
        params: list[Any] = [run_id]
        if dataset:
            where.append("p.dataset_name=?")
            params.append(dataset)
        if status in {"correct", "incorrect", "unknown"}:
            where.append("p.correctness=?")
            params.append(status)
        elif status == "unparsed":
            where.extend(["p.parse_status='unparsed'", "p.generation_status='ok'"])
        elif status in {"error", "empty"}:
            where.append("p.generation_status=?")
            params.append(status)
        clause = " WHERE " + " AND ".join(where)
        with self._state_connect() as connection:
            exists = connection.execute("SELECT 1 FROM model_runs WHERE run_id=?", (run_id,)).fetchone()
            if not exists:
                raise ApiError(f"unknown model run: {run_id}", HTTPStatus.NOT_FOUND)
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM model_responses p" + clause, params
                ).fetchone()[0]
            )
            rows = connection.execute(
                """SELECT p.*, r.model_name, r.display_name FROM model_responses p
                   JOIN model_runs r ON r.run_id=p.run_id"""
                + clause
                + " ORDER BY p.dataset_name,p.line_index LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return {
            "run_id": run_id,
            "status": status,
            "dataset": dataset,
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": [
                {
                    "dataset": row["dataset_name"],
                    "index": int(row["line_index"]),
                    "model_name": row["model_name"],
                    "display_name": row["display_name"],
                    "response": str(row["response"])[:1200],
                    "parsed_answer": row["parsed_answer"],
                    "gold_answer": row["gold_answer"],
                    "status": self._response_status(row),
                    "error": row["error"],
                }
                for row in rows
            ],
        }

    def next_badcase(self, run_id: str, dataset: str, after: int) -> Dict[str, Any]:
        total = self.count(dataset)
        with self._state_connect() as connection:
            run = connection.execute("SELECT 1 FROM model_runs WHERE run_id=?", (run_id,)).fetchone()
            if not run:
                raise ApiError(f"unknown model run: {run_id}", HTTPStatus.NOT_FOUND)
            rows = connection.execute(
                """SELECT line_index,correctness,parse_status,generation_status
                   FROM model_responses WHERE run_id=? AND dataset_name=?""",
                (run_id, dataset),
            ).fetchall()
        by_index = {int(row["line_index"]): row for row in rows}
        if not by_index:
            raise ApiError(f"model run has no results for dataset: {dataset}", HTTPStatus.NOT_FOUND)
        next_index = None
        for candidate in range(max(-1, after) + 1, total):
            row = by_index.get(candidate)
            if row is None or (
                str(row["correctness"]) == "incorrect"
                or str(row["parse_status"]) != "parsed"
                or str(row["generation_status"]) != "ok"
            ):
                next_index = candidate
                break
        return {
            "run_id": run_id,
            "dataset": dataset,
            "index": next_index,
            "next_index": next_index,
            "complete": next_index is None,
            "message": None if next_index is not None else "这个模型没有更多 badcase。",
        }

    def _is_training_source(self, name: str) -> bool:
        if name not in self.sources:
            raise ApiError(f"unknown dataset: {name}", HTTPStatus.NOT_FOUND)
        source = self.sources[name]
        return (
            source.schema == "chatts"
            and source.split == "train"
            and source.training_role != "evaluation_only"
        )

    def human_progress(self, name: Optional[str] = None) -> Dict[str, Any]:
        names = [name] if name else [key for key in self.sources if self._is_training_source(key)]
        datasets = []
        with self._state_connect() as connection:
            for dataset_name in names:
                if not self._is_training_source(dataset_name):
                    raise ApiError(
                        f"human review is only available for training data: {dataset_name}",
                        HTTPStatus.FORBIDDEN,
                    )
                total = self.count(dataset_name)
                source_signature = self._data_source_signature(self.sources[dataset_name])
                counts = {
                    str(row[0]): int(row[1])
                    for row in connection.execute(
                        """SELECT verdict,COUNT(*) FROM human_annotations
                           WHERE dataset_name=? AND source_signature=? GROUP BY verdict""",
                        (dataset_name, source_signature),
                    )
                }
                good = counts.get("good", 0)
                bad = counts.get("bad", 0)
                labeled = good + bad
                datasets.append(
                    {
                        "dataset": dataset_name,
                        "total": total,
                        "labeled": labeled,
                        "good": good,
                        "bad": bad,
                        "remaining": max(0, total - labeled),
                        "coverage_percent": round(100 * labeled / total, 4) if total else 0,
                    }
                )
        if name:
            return datasets[0]
        overall = {
            key: sum(int(item[key]) for item in datasets)
            for key in ("total", "labeled", "good", "bad", "remaining")
        }
        overall["coverage_percent"] = (
            round(100 * overall["labeled"] / overall["total"], 4) if overall["total"] else 0
        )
        return {"overall": overall, "datasets": datasets}

    def human_review(self, name: str, index: int, record_hash: str) -> Dict[str, Any]:
        if not self._is_training_source(name):
            return {
                "editable": False,
                "label": None,
                "verdict": None,
                "progress": None,
            }
        with self._state_connect() as connection:
            row = connection.execute(
                "SELECT * FROM human_annotations WHERE dataset_name=? AND line_index=?",
                (name, index),
            ).fetchone()
            event = connection.execute(
                "SELECT MAX(revision) FROM annotation_events WHERE dataset_name=? AND line_index=?",
                (name, index),
            ).fetchone()
        latest_revision = int(event[0] or 0) if event else 0
        current_signature = self._data_source_signature(self.sources[name])
        payload: Dict[str, Any] = {
            "editable": True,
            "label": None,
            "verdict": None,
            "comment": "",
            "annotator": "",
            "revision": latest_revision,
            "created_at": None,
            "updated_at": None,
            "stale": False,
            "record_hash": record_hash,
            "progress": self.human_progress(name),
        }
        if row:
            payload.update(
                {
                    "label": row["verdict"],
                    "verdict": row["verdict"],
                    "comment": row["comment"],
                    "annotator": row["annotator"],
                    "revision": int(row["revision"]),
                    "created_at": int(row["created_at"]),
                    "updated_at": int(row["updated_at"]),
                    "stale": (
                        str(row["record_hash"]) != record_hash
                        or str(row["source_signature"]) != current_signature
                    ),
                    "stored_record_hash": row["record_hash"],
                    "stored_source_signature": row["source_signature"],
                }
            )
        return payload

    def save_human_annotation(
        self,
        name: str,
        index: int,
        verdict: str,
        comment: str = "",
        annotator: str = "",
        expected_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        if verdict not in {"good", "bad"}:
            raise ApiError("verdict must be 'good' or 'bad'")
        if not self._is_training_source(name):
            raise ApiError(
                f"benchmark/evaluation dataset cannot be human-labeled: {name}",
                HTTPStatus.FORBIDDEN,
            )
        _, record_hash = self._source_record_fields(name, index)
        source_signature = self._data_source_signature(self.sources[name])
        now = int(time.time())
        with self._state_lock, self._state_connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM human_annotations WHERE dataset_name=? AND line_index=?",
                (name, index),
            ).fetchone()
            if current:
                current_revision = int(current["revision"])
            else:
                revision_row = connection.execute(
                    "SELECT MAX(revision) FROM annotation_events WHERE dataset_name=? AND line_index=?",
                    (name, index),
                ).fetchone()
                current_revision = int(revision_row[0] or 0)
            if expected_revision is not None and int(expected_revision) != current_revision:
                raise AnnotationConflict(
                    f"annotation revision changed: expected {expected_revision}, current {current_revision}"
                )
            revision = current_revision + 1
            created_at = int(current["created_at"]) if current else now
            connection.execute(
                """INSERT OR REPLACE INTO human_annotations
                   (dataset_name,line_index,record_hash,source_signature,verdict,comment,
                    annotator,revision,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    name,
                    index,
                    record_hash,
                    source_signature,
                    verdict,
                    str(comment or "")[:4000],
                    str(annotator or "")[:200],
                    revision,
                    created_at,
                    now,
                ),
            )
            connection.execute(
                """INSERT INTO annotation_events
                   (dataset_name,line_index,record_hash,old_verdict,new_verdict,comment,annotator,revision,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    name,
                    index,
                    record_hash,
                    current["verdict"] if current else None,
                    verdict,
                    str(comment or "")[:4000],
                    str(annotator or "")[:200],
                    revision,
                    now,
                ),
            )
        return self.human_review(name, index, record_hash)

    def delete_human_annotation(
        self, name: str, index: int, expected_revision: Optional[int] = None
    ) -> Dict[str, Any]:
        if not self._is_training_source(name):
            raise ApiError("benchmark/evaluation dataset cannot be human-labeled", HTTPStatus.FORBIDDEN)
        _, record_hash = self._source_record_fields(name, index)
        with self._state_lock, self._state_connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM human_annotations WHERE dataset_name=? AND line_index=?",
                (name, index),
            ).fetchone()
            if current is None:
                revision_row = connection.execute(
                    "SELECT MAX(revision) FROM annotation_events WHERE dataset_name=? AND line_index=?",
                    (name, index),
                ).fetchone()
                current_revision = int(revision_row[0] or 0)
                if expected_revision is not None and int(expected_revision) != current_revision:
                    raise AnnotationConflict(
                        f"annotation revision changed: expected {expected_revision}, current {current_revision}"
                    )
            else:
                current_revision = int(current["revision"])
                if expected_revision is not None and int(expected_revision) != current_revision:
                    raise AnnotationConflict(
                        f"annotation revision changed: expected {expected_revision}, current {current_revision}"
                    )
                revision = current_revision + 1
                now = int(time.time())
                connection.execute(
                    "DELETE FROM human_annotations WHERE dataset_name=? AND line_index=?", (name, index)
                )
                connection.execute(
                    """INSERT INTO annotation_events
                       (dataset_name,line_index,record_hash,old_verdict,new_verdict,comment,annotator,revision,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        name,
                        index,
                        record_hash,
                        current["verdict"],
                        None,
                        current["comment"],
                        current["annotator"],
                        revision,
                        now,
                    ),
                )
        return self.human_review(name, index, record_hash)

    def next_unlabeled(self, name: str, after: int = -1) -> Dict[str, Any]:
        if not self._is_training_source(name):
            raise ApiError("next-unlabeled is only available for training data", HTTPStatus.FORBIDDEN)
        total = self.count(name)
        source_signature = self._data_source_signature(self.sources[name])
        with self._state_connect() as connection:
            labeled = {
                int(row[0])
                for row in connection.execute(
                    """SELECT line_index FROM human_annotations
                       WHERE dataset_name=? AND source_signature=?""",
                    (name, source_signature),
                )
            }
        order = list(range(max(-1, after) + 1, total)) + list(range(0, min(total, after + 1)))
        index = next((candidate for candidate in order if candidate not in labeled), None)
        return {
            "dataset": name,
            "index": index,
            "remaining": max(0, total - len(labeled)),
            "complete": index is None,
        }

    def export_human_annotations(
        self, output_format: str, dataset: Optional[str] = None, include_stale: bool = False
    ) -> tuple[bytes, str, str]:
        if output_format not in {"jsonl", "csv"}:
            raise ApiError("format must be jsonl or csv")
        if dataset and not self._is_training_source(dataset):
            raise ApiError("only training annotations can be exported", HTTPStatus.FORBIDDEN)
        where = " WHERE dataset_name=?" if dataset else ""
        params = (dataset,) if dataset else ()
        with self._state_connect() as connection:
            rows = connection.execute(
                "SELECT * FROM human_annotations" + where + " ORDER BY dataset_name,line_index",
                params,
            ).fetchall()
        fields = [
            "dataset_name", "line_index", "record_hash", "source_signature", "stale",
            "verdict", "comment", "annotator", "revision", "created_at", "updated_at",
        ]
        current_signatures = {
            name: self._data_source_signature(source)
            for name, source in self.sources.items()
            if self._is_training_source(name)
        }
        records = []
        for row in rows:
            stale = str(row["source_signature"]) != current_signatures.get(str(row["dataset_name"]), "")
            if stale and not include_stale:
                continue
            records.append(
                {
                    key: (bool(stale) if key == "stale" else row[key])
                    for key in fields
                }
            )
        filename = f"human-annotations{('-' + dataset) if dataset else ''}.{output_format}"
        if output_format == "jsonl":
            payload = "".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in records
            )
            return payload.encode("utf-8"), "application/x-ndjson; charset=utf-8", filename
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        return stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8", filename

    def _load_sources(self) -> Dict[str, Source]:
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        result = {}
        for item in payload.get("sources", []):
            name = str(item["name"])
            source_path = resolve_path(self.data_root, item["path"])
            audit_path = resolve_path(self.data_root, item["audit"]) if item.get("audit") else None
            annotation_path = self.annotations_dir / f"{name}.jsonl"
            result[name] = Source(
                name=name,
                family=str(item.get("family", "unknown")),
                split=str(item.get("split", "train")),
                training_role=str(item.get("training_role", "sft")),
                path=source_path,
                audit=audit_path if audit_path and audit_path.is_file() else None,
                annotations=annotation_path if annotation_path.is_file() else None,
            )
        self._load_tsrbench_sources(result)
        return result

    def _load_tsrbench_sources(self, result: Dict[str, Source]) -> None:
        data_config = self.config.get("data") or {}
        configured = data_config.get("tsrbench_root")
        env_root = os.getenv("TSRBENCH_ROOT", "").strip()
        candidates = []
        if env_root:
            candidates.append(resolve_path(self.config_path.parent, env_root))
        if configured:
            values = configured if isinstance(configured, list) else [configured]
            candidates.extend(resolve_path(self.config_path.parent, value) for value in values)

        # Also recognize common layouts when the project was copied next to,
        # inside, or above the official TSRBench repository.
        base = self.config_path.parent
        candidates.extend(
            [
                base / "dataset",
                base / "TSRBench" / "dataset",
                base.parent / "TSRBench" / "dataset",
                base.parent / "TSRBenchmark" / "dataset",
                base.parent / "tsrbench" / "dataset",
                base.parent.parent / "TSRBench" / "dataset",
                base.parent.parent / "TSRBenchmark" / "dataset",
                Path("/workspace/TSRBench/dataset"),
                Path("/workspace/TSRBenchmark/dataset"),
            ]
        )
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(resolved)

        self.tsrbench_status["configured"] = bool(env_root or configured)
        self.tsrbench_status["checked_paths"] = [str(path) for path in unique_candidates]
        root = next((path for path in unique_candidates if path.is_dir()), None)
        if root is None:
            return
        self.tsrbench_status["found"] = True
        self.tsrbench_status["root"] = str(root)
        by_stem = {path.stem: path for path in root.rglob("*.jsonl") if path.is_file()}
        for task, (relative, major) in TSRBENCH_TASKS.items():
            expected = root / relative
            path = expected if expected.is_file() else by_stem.get(task)
            if path is None:
                continue
            name = f"tsrbench_{task}"
            result[name] = Source(
                name=name,
                family=f"TSRBench · {major}",
                split="benchmark",
                training_role="evaluation_only",
                path=path.resolve(),
                audit=None,
                annotations=None,
                schema="tsrbench",
                benchmark_task=task,
            )
            self.tsrbench_status["tasks_found"] += 1

    def _load_template_stats(self) -> Dict[str, Dict[str, Any]]:
        if not self.template_stats_path.is_file():
            return {}
        payload = json.loads(self.template_stats_path.read_text(encoding="utf-8"))
        return {str(item["name"]): item for item in payload.get("datasets", [])}

    def _fallback_row_count(self, source: Source) -> int:
        """Count JSONL rows only when the optional precomputed stats are absent."""
        return sum(1 for _ in line_offsets(source.path))

    def _db_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.index.sqlite"

    def _source_signature(self, source: Source) -> str:
        parts = []
        for path in (source.path, source.audit, source.annotations):
            if path and path.exists():
                stat = path.stat()
                parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    @staticmethod
    def _data_source_signature(source: Source) -> str:
        """Fingerprint only raw QA data, not mutable audit/label sidecars."""
        stat = source.path.stat()
        payload = f"{source.path}:{stat.st_size}:{stat.st_mtime_ns}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _connect(self, name: str) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._db_path(name)), timeout=60)
        connection.row_factory = sqlite3.Row
        return connection

    def _index_valid(self, name: str) -> bool:
        path = self._db_path(name)
        if not path.is_file():
            return False
        try:
            with self._connect(name) as connection:
                row = connection.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
                return bool(row and row[0] == self._source_signature(self.sources[name]))
        except sqlite3.DatabaseError:
            return False

    def ensure_index(self, name: str) -> None:
        if name not in self.sources:
            raise KeyError(f"unknown dataset: {name}")
        if self._index_valid(name):
            return
        with self._locks[name]:
            if self._index_valid(name):
                return
            self._build_index(self.sources[name])

    def _build_index(self, source: Source) -> None:
        descriptor, temp_name = tempfile.mkstemp(prefix=f"{source.name}.", suffix=".sqlite", dir=self.cache_dir)
        os.close(descriptor)
        temp_path = Path(temp_name)
        connection = sqlite3.connect(str(temp_path))
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE records (
                line_index INTEGER PRIMARY KEY,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                audit_offset INTEGER,
                audit_length INTEGER,
                annotation_offset INTEGER,
                annotation_length INTEGER,
                taxonomy_template_id TEXT,
                quality_template_id TEXT,
                raw_input_hash TEXT,
                answer_class TEXT,
                issues TEXT NOT NULL DEFAULT '',
                ability_label TEXT,
                quality TEXT,
                difficulty TEXT
            );
            """
        )
        audit_iter = iter(line_offsets(source.audit)) if source.audit else None
        annotation_stream = source.annotations.open("rb") if source.annotations else None
        audit_stream = source.audit.open("rb") if source.audit else None
        try:
            batch = []
            with source.path.open("rb") as data_stream:
                for index, offset, length in line_offsets(source.path):
                    data_stream.seek(offset)
                    line = data_stream.read(length)
                    try:
                        source_row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON at {source.path}:{index + 1}") from exc
                    fields = view_fields(source_row, source)
                    prompt = str(fields["input"])
                    output = str(fields["output"])
                    audit_offset = audit_length = None
                    audit_row = None
                    if audit_iter is not None:
                        _, audit_offset, audit_length = next(audit_iter)
                        assert audit_stream is not None
                        audit_stream.seek(audit_offset)
                        try:
                            audit_row = json.loads(audit_stream.read(audit_length))
                        except json.JSONDecodeError:
                            audit_row = None

                    annotation_offset = annotation_length = None
                    annotation_row = None
                    if annotation_stream is not None:
                        annotation_offset = annotation_stream.tell()
                        annotation_line = annotation_stream.readline()
                        annotation_length = len(annotation_line.rstrip(b"\n"))
                        try:
                            annotation_row = json.loads(annotation_line)
                        except json.JSONDecodeError:
                            annotation_row = None

                    normalized = normalize_template(prompt)
                    taxonomy_id = hashlib.sha256(
                        f"{source.name}\n{normalized}".encode("utf-8")
                    ).hexdigest()[:24]
                    if annotation_row:
                        taxonomy_id = str(annotation_row.get("taxonomy_cluster_id") or taxonomy_id)
                    q_template = quality_template_id(prompt, output)
                    if annotation_row:
                        q_template = str(annotation_row.get("quality_template_id") or q_template)
                    flags = issue_flags(prompt, output, audit_row)
                    batch.append(
                        (
                            index,
                            offset,
                            length,
                            audit_offset,
                            audit_length,
                            annotation_offset,
                            annotation_length,
                            taxonomy_id,
                            q_template,
                            hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24],
                            extract_answer_class(output)[:160],
                            ",".join(flags),
                            (annotation_row or {}).get("ability_label"),
                            (annotation_row or {}).get("quality"),
                            (annotation_row or {}).get("difficulty"),
                        )
                    )
                    if len(batch) >= 5000:
                        connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                        batch.clear()
                if batch:
                    connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            connection.executescript(
                """
                CREATE INDEX records_taxonomy_template ON records(taxonomy_template_id, line_index);
                CREATE INDEX records_quality_template ON records(quality_template_id, line_index);
                CREATE INDEX records_issues ON records(issues, line_index);
                CREATE INDEX records_ability ON records(ability_label, line_index);
                """
            )
            connection.execute("INSERT INTO meta VALUES ('signature', ?)", (self._source_signature(source),))
            connection.execute("INSERT INTO meta VALUES ('built_at', ?)", (str(int(time.time())),))
            connection.commit()
            connection.close()
            os.replace(temp_path, self._db_path(source.name))
        except Exception:
            connection.close()
            temp_path.unlink(missing_ok=True)
            raise
        finally:
            if audit_stream:
                audit_stream.close()
            if annotation_stream:
                annotation_stream.close()

    def datasets(self) -> Dict[str, Any]:
        rows = []
        with self._state_connect() as state:
            model_run_counts = {
                str(row[0]): int(row[1])
                for row in state.execute(
                    "SELECT dataset_name,COUNT(DISTINCT run_id) FROM model_responses GROUP BY dataset_name"
                )
            }
            review_counts: Dict[str, Dict[str, int]] = {}
            for row in state.execute(
                """SELECT dataset_name,verdict,source_signature,COUNT(*)
                   FROM human_annotations GROUP BY dataset_name,verdict,source_signature"""
            ):
                dataset_name = str(row[0])
                source = self.sources.get(dataset_name)
                if source and str(row[2]) == self._data_source_signature(source):
                    review_counts.setdefault(dataset_name, {})[str(row[1])] = int(row[3])
        for source in self.sources.values():
            stats = self.template_stats.get(source.name, {})
            row_count = int(stats.get("rows", 0))
            if row_count <= 0:
                if self._index_valid(source.name):
                    with self._connect(source.name) as connection:
                        row_count = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
                else:
                    row_count = self._fallback_row_count(source)
            item = {
                    "name": source.name,
                    "family": source.family,
                    "split": source.split,
                    "training_role": source.training_role,
                    "rows": row_count,
                    "templates": int(stats.get("templates", 0)),
                    "compression_ratio": float(stats.get("compression_ratio", 0)),
                    "has_audit": source.audit is not None,
                    "has_annotations": source.annotations is not None,
                    "index_ready": self._index_valid(source.name),
                    "schema": source.schema,
                    "benchmark_task": source.benchmark_task,
                    "model_runs": model_run_counts.get(source.name, 0),
                }
            if self._is_training_source(source.name):
                good = review_counts.get(source.name, {}).get("good", 0)
                bad = review_counts.get(source.name, {}).get("bad", 0)
                labeled = good + bad
                item["human_review"] = {
                    "total": row_count,
                    "labeled": labeled,
                    "good": good,
                    "bad": bad,
                    "remaining": max(0, row_count - labeled),
                    "coverage_percent": round(100 * labeled / row_count, 4) if row_count else 0,
                }
            else:
                item["human_review"] = None
            rows.append(item)
        return {
            "datasets": rows,
            "qwen": self.qwen_public_config(),
            "tsrbench": self.tsrbench_status,
            "evaluation": self.evaluation_status(),
        }

    def qwen_public_config(self) -> Dict[str, Any]:
        qwen = self.config.get("qwen") or {}
        return {
            "model": qwen.get("model"),
            "configured": bool(qwen.get("base_url") and qwen.get("model")),
        }

    def _row(self, name: str, index: int) -> sqlite3.Row:
        self.ensure_index(name)
        with self._connect(name) as connection:
            row = connection.execute("SELECT * FROM records WHERE line_index=?", (index,)).fetchone()
        if row is None:
            raise IndexError(f"record index out of range: {index}")
        return row

    def record(self, name: str, index: int) -> Dict[str, Any]:
        source = self.sources[name]
        row = self._row(name, index)
        record = read_slice(source.path, row["byte_offset"], row["byte_length"])
        if record is None:
            raise ValueError(f"invalid JSON at {source.path}:{index + 1}")
        record_hash = hashlib.sha256(compact_json(record)).hexdigest()
        audit = read_slice(source.audit, row["audit_offset"], row["audit_length"])
        annotation = read_slice(source.annotations, row["annotation_offset"], row["annotation_length"])
        template = self.template_summary(name, str(row["taxonomy_template_id"]))
        fields = view_fields(record, source)
        benchmark_metadata = None
        if source.schema == "tsrbench":
            benchmark_metadata = {
                "task": source.benchmark_task,
                "major": source.family.removeprefix("TSRBench · "),
                "domain": fields["domain"],
                "category": fields["category"],
                "choices": fields["choices"],
                "series_names": fields["series_names"],
                "extra": fields["raw_metadata"],
            }
        return {
            "dataset": name,
            "index": index,
            "schema": source.schema,
            "input": fields["input"],
            "output": fields["output"],
            "series": summarize_series(fields["timeseries"], self.max_chart_points),
            "series_count": len(fields["timeseries"]) if isinstance(fields["timeseries"], list) else 0,
            "series_names": fields["series_names"],
            "choices": fields["choices"],
            "benchmark": benchmark_metadata,
            "audit": audit,
            "annotation": annotation,
            "taxonomy_template_id": row["taxonomy_template_id"],
            "quality_template_id": row["quality_template_id"],
            "answer_class": row["answer_class"],
            "issues": [item for item in str(row["issues"] or "").split(",") if item],
            "normalized_template": normalize_template(str(fields["input"])),
            "template": template,
            "record_hash": record_hash,
            "model_responses": self.model_responses(name, index),
            "human_review": self.human_review(name, index, record_hash),
        }

    def count(self, name: str) -> int:
        self.ensure_index(name)
        with self._connect(name) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def random_index(self, name: str, template_id: Optional[str] = None, issue: Optional[str] = None) -> int:
        self.ensure_index(name)
        where = []
        params: list[Any] = []
        if template_id:
            where.append("taxonomy_template_id=?")
            params.append(template_id)
        if issue:
            where.append("(',' || issues || ',') LIKE ?")
            params.append(f"%,{issue},%")
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self._connect(name) as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM records" + clause, params).fetchone()[0])
            if not total:
                raise LookupError("no matching records")
            offset = random.randrange(total)
            row = connection.execute(
                "SELECT line_index FROM records" + clause + " ORDER BY line_index LIMIT 1 OFFSET ?",
                [*params, offset],
            ).fetchone()
        return int(row[0])

    def template_summary(self, name: str, template_id: str) -> Dict[str, Any]:
        self.ensure_index(name)
        with self._connect(name) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS members,
                       COUNT(DISTINCT raw_input_hash) AS raw_prompts,
                       COUNT(DISTINCT answer_class) AS answer_classes,
                       MIN(line_index) AS first_index,
                       MAX(line_index) AS last_index
                FROM records WHERE taxonomy_template_id=?
                """,
                (template_id,),
            ).fetchone()
            answers = [
                {"value": item[0], "count": int(item[1])}
                for item in connection.execute(
                    """SELECT answer_class, COUNT(*) FROM records
                       WHERE taxonomy_template_id=? GROUP BY answer_class
                       ORDER BY COUNT(*) DESC, answer_class LIMIT 12""",
                    (template_id,),
                )
            ]
        return {
            "members": int(row["members"]),
            "raw_prompts": int(row["raw_prompts"]),
            "answer_classes": int(row["answer_classes"]),
            "first_index": row["first_index"],
            "last_index": row["last_index"],
            "answers": answers,
        }

    def template_members(self, name: str, template_id: str, offset: int, limit: int) -> Dict[str, Any]:
        self.ensure_index(name)
        source = self.sources[name]
        with self._connect(name) as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM records WHERE taxonomy_template_id=?", (template_id,)
                ).fetchone()[0]
            )
            rows = connection.execute(
                """SELECT * FROM records WHERE taxonomy_template_id=?
                   ORDER BY line_index LIMIT ? OFFSET ?""",
                (template_id, limit, offset),
            ).fetchall()
        members = []
        for row in rows:
            record = read_slice(source.path, row["byte_offset"], row["byte_length"]) or {}
            fields = view_fields(record, source)
            members.append(
                {
                    "index": int(row["line_index"]),
                    "input": str(fields["input"])[:360],
                    "output": str(fields["output"])[:360],
                    "answer_class": row["answer_class"],
                    "issues": [item for item in str(row["issues"] or "").split(",") if item],
                    "ability_label": row["ability_label"],
                    "quality": row["quality"],
                    "difficulty": row["difficulty"],
                }
            )
        return {"total": total, "offset": offset, "limit": limit, "members": members}

    def dataset_issues(self, name: str) -> Dict[str, Any]:
        self.ensure_index(name)
        counters = Counter()
        with self._connect(name) as connection:
            for row in connection.execute("SELECT issues, COUNT(*) FROM records GROUP BY issues"):
                for issue in str(row[0] or "").split(","):
                    if issue:
                        counters[issue] += int(row[1])
            total = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        return {
            "total": total,
            "issues": [
                {"name": key, "count": value, "percent": round(100 * value / total, 4) if total else 0}
                for key, value in counters.most_common()
            ],
        }

    def _init_translation_db(self) -> None:
        with sqlite3.connect(str(self.translation_db)) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS translations (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )"""
            )

    def translate(self, texts: Mapping[str, str]) -> Dict[str, Any]:
        clean = {key: value for key, value in texts.items() if isinstance(value, str) and value.strip()}
        if not clean:
            raise ValueError("no text to translate")
        qwen = self.config.get("qwen") or {}
        model = str(qwen.get("model", ""))
        cache_payload = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.sha256((model + "\n" + cache_payload).encode()).hexdigest()
        with sqlite3.connect(str(self.translation_db)) as connection:
            row = connection.execute(
                "SELECT translated_text FROM translations WHERE cache_key=?", (cache_key,)
            ).fetchone()
        if row:
            return {"translations": json.loads(row[0]), "cached": True, "model": model}

        system = (
            "你是时间序列问答数据审查助手。将用户给出的英文字段准确翻译成简体中文。"
            "保留 <ts><ts/>、<think>、<answer> 等标签，保留全部数字、选项字母、单位、变量名和专有名词；"
            "不要解释、评价、纠错或省略内容。只输出一个JSON对象，键与输入完全相同，值为译文。"
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": cache_payload},
            ],
            "temperature": 0,
            "max_tokens": int(qwen.get("max_tokens", 4096)),
            "response_format": {"type": "json_object"},
        }
        base_url = str(qwen.get("base_url", "")).rstrip("/")
        if not base_url or not model:
            raise RuntimeError("Qwen base_url/model is not configured")
        headers = {"Content-Type": "application/json"}
        key = os.getenv(str(qwen.get("api_key_env", "DT_QWEN_API_KEY")), "")
        if key:
            headers["Authorization"] = "Bearer " + key
        elif not bool(qwen.get("allow_no_key", False)):
            raise RuntimeError("Qwen API key is not configured")
        request = urllib.request.Request(
            base_url + "/chat/completions",
            data=compact_json(payload),
            headers=headers,
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def submit(current_request: urllib.request.Request) -> Dict[str, Any]:
            try:
                with opener.open(
                    current_request,
                    timeout=float(qwen.get("timeout_seconds", 180)),
                ) as response:
                    parsed = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace").strip()
                message = f"Qwen HTTP {exc.code}: {details[:1600] or exc.reason}"
                raise RuntimeError(message) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Qwen request failed: {exc.reason}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("Qwen response is not a JSON object")
            return parsed

        try:
            # Cluster deployments commonly set a global HTTP proxy that cannot
            # route private model-service addresses. Match the existing
            # annotation scripts' NO_PROXY behavior explicitly here.
            response_payload = submit(request)
        except RuntimeError as primary_error:
            # Some OpenAI-compatible vLLM gateways reject response_format, or
            # reject a requested completion budget larger than their deployed
            # context limit. Retry once with the conservative settings already
            # used successfully by the taxonomy annotation command.
            if "Qwen HTTP 400:" not in str(primary_error) or not bool(qwen.get("compatibility_retry", True)):
                raise
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            fallback_payload["max_tokens"] = min(int(payload["max_tokens"]), 1024)
            fallback_request = urllib.request.Request(
                base_url + "/chat/completions",
                data=compact_json(fallback_payload),
                headers=headers,
                method="POST",
            )
            try:
                response_payload = submit(fallback_request)
            except RuntimeError as fallback_error:
                raise RuntimeError(
                    f"{fallback_error}（兼容重试仍失败；首次错误：{primary_error}）"
                ) from fallback_error
        content = response_payload["choices"][0]["message"].get("content", "")
        if content.startswith("```"):
            content = "\n".join(content.splitlines()[1:-1]).strip()
            if content.lower().startswith("json"):
                content = content[4:].lstrip()
        try:
            translated = json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Qwen did not return a JSON object")
            translated = json.loads(content[start : end + 1])
        if not isinstance(translated, dict):
            raise ValueError("Qwen translation response is not an object")
        result = {key: str(translated.get(key, "")) for key in clean}
        with sqlite3.connect(str(self.translation_db)) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO translations VALUES (?,?,?,?,?)",
                (cache_key, model, cache_payload, json.dumps(result, ensure_ascii=False), int(time.time())),
            )
        return {"translations": result, "cached": False, "model": model}


class ApiHandler(BaseHTTPRequestHandler):
    store: DatasetStore
    allowed_origin: str

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _headers(
        self,
        status: int = 200,
        content_type: str = "application/json; charset=utf-8",
        disposition: Optional[str] = None,
        content_length: Optional[int] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", self.allowed_origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        if disposition:
            self.send_header("Content-Disposition", disposition)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.end_headers()

    def _json(self, payload: Any, status: int = 200) -> None:
        self._headers(status)
        self.wfile.write(compact_json(payload))

    def _error(self, exc: Exception, status: int = 400) -> None:
        self._json({"error": type(exc).__name__, "message": str(exc)}, status)

    def _download(self, payload: bytes, content_type: str, filename: str) -> None:
        self._headers(
            200,
            content_type=content_type,
            disposition=f'attachment; filename="{filename}"',
            content_length=len(payload),
        )
        self.wfile.write(payload)

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ApiError("request body must be a JSON object")
        return value

    def _handle_expected_error(self, exc: Exception) -> bool:
        if isinstance(exc, ApiError):
            self._error(exc, exc.status)
            return True
        if isinstance(exc, KeyError):
            self._error(exc, HTTPStatus.NOT_FOUND)
            return True
        if isinstance(exc, IndexError):
            self._error(exc, HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return True
        if isinstance(exc, (ValueError, json.JSONDecodeError)):
            self._error(exc, HTTPStatus.BAD_REQUEST)
            return True
        return False

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._headers(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self._json({
                    "ok": True,
                    "datasets": len(self.store.sources),
                    "qwen": self.store.qwen_public_config(),
                    "tsrbench": self.store.tsrbench_status,
                    "evaluation": self.store.evaluation_status(),
                })
            elif parsed.path == "/api/datasets":
                self._json(self.store.datasets())
            elif parsed.path == "/api/record":
                name = query["dataset"][0]
                index = int(query.get("index", [0])[0])
                payload = self.store.record(name, index)
                payload["dataset_total"] = self.store.count(name)
                self._json(payload)
            elif parsed.path == "/api/random":
                name = query["dataset"][0]
                index = self.store.random_index(
                    name,
                    query.get("template_id", [None])[0],
                    query.get("issue", [None])[0],
                )
                self._json({"index": index})
            elif parsed.path == "/api/template-members":
                name = query["dataset"][0]
                template_id = query["template_id"][0]
                offset = max(0, int(query.get("offset", [0])[0]))
                limit = min(50, max(1, int(query.get("limit", [self.store.template_page_size])[0])))
                self._json(self.store.template_members(name, template_id, offset, limit))
            elif parsed.path == "/api/issues":
                self._json(self.store.dataset_issues(query["dataset"][0]))
            elif parsed.path == "/api/model-runs":
                self._json({"runs": self.store.model_runs(), "settings": self.store.model_results_settings()})
            elif parsed.path == "/api/model-results/settings":
                self._json(self.store.model_results_settings())
            elif parsed.path == "/api/model-responses":
                self._json(
                    {
                        "responses": self.store.model_responses(
                            query["dataset"][0], int(query.get("index", [0])[0])
                        )
                    }
                )
            elif parsed.path == "/api/badcases":
                self._json(
                    self.store.badcases(
                        query["run_id"][0],
                        query.get("dataset", [None])[0],
                        query.get("status", ["incorrect"])[0],
                        max(0, int(query.get("offset", [0])[0])),
                        min(200, max(1, int(query.get("limit", [50])[0]))),
                    )
                )
            elif parsed.path == "/api/model-results/badcase":
                self._json(
                    self.store.next_badcase(
                        query["run_id"][0],
                        query["dataset"][0],
                        int(query.get("after", [-1])[0]),
                    )
                )
            elif parsed.path in {"/api/human-annotations/stats", "/api/human-labels/stats"}:
                self._json(self.store.human_progress(query.get("dataset", [None])[0]))
            elif parsed.path in {"/api/human-annotations/next", "/api/human-labels/next"}:
                self._json(
                    self.store.next_unlabeled(
                        query["dataset"][0], int(query.get("after", [-1])[0])
                    )
                )
            elif parsed.path in {"/api/human-annotations/export", "/api/human-labels/export"}:
                payload, content_type, filename = self.store.export_human_annotations(
                    query.get("format", ["jsonl"])[0],
                    query.get("dataset", [None])[0],
                    query.get("include_stale", ["0"])[0].lower() in {"1", "true", "yes"},
                )
                self._download(payload, content_type, filename)
            else:
                self._error(FileNotFoundError(parsed.path), 404)
        except Exception as exc:  # noqa: BLE001 - API boundary
            if not self._handle_expected_error(exc):
                self._error(exc, 500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/translate":
                texts = body.get("texts")
                if not isinstance(texts, dict):
                    raise ValueError("body.texts must be an object")
                self._json(self.store.translate(texts))
            elif parsed.path == "/api/model-runs/import":
                import_path = body.get("path") or body.get("root")
                if import_path:
                    self.store.set_results_root(str(import_path), import_now=False)
                self._json(
                    self.store.import_model_results(
                        import_path, str(body["model_name"]) if body.get("model_name") else None
                    )
                )
            elif parsed.path in {"/api/model-results/settings", "/api/model-results/configure"}:
                self._json(
                    self.store.set_results_root(
                        str(body.get("root") or body.get("path") or ""),
                        bool(body.get("import", parsed.path == "/api/model-results/settings")),
                    )
                )
            elif parsed.path == "/api/model-results/refresh":
                self._json(
                    self.store.import_model_results(
                        body.get("path"), str(body["model_name"]) if body.get("model_name") else None
                    )
                )
            elif parsed.path in {"/api/human-annotation", "/api/human-label"}:
                if "dataset" not in body or not str(body.get("dataset") or "").strip():
                    raise ApiError("body.dataset is required")
                if "index" not in body:
                    raise ApiError("body.index is required")
                if "verdict" not in body and "label" not in body:
                    raise ApiError("body.label or body.verdict is required")
                expected = body.get("expected_revision")
                expected_revision = (
                    strict_nonnegative_int(expected, "body.expected_revision")
                    if expected is not None else None
                )
                dataset = str(body.get("dataset") or "")
                index = strict_nonnegative_int(body["index"], "body.index")
                if body.get("verdict", body.get("label")) is None:
                    review = self.store.delete_human_annotation(
                        dataset, index, expected_revision
                    )
                else:
                    review = self.store.save_human_annotation(
                        str(body.get("dataset") or ""),
                        index,
                        str(body.get("verdict") or body.get("label") or ""),
                        str(body.get("comment") or ""),
                        str(body.get("annotator") or ""),
                        expected_revision,
                    )
                self._json(
                    {
                        "human_review": review,
                        "review": review,
                        "progress": review.get("progress"),
                        "label": review.get("label"),
                        "revision": review.get("revision"),
                    }
                )
            else:
                self._error(FileNotFoundError(parsed.path), 404)
        except Exception as exc:  # noqa: BLE001 - API boundary
            if not self._handle_expected_error(exc):
                self._error(exc, 500)

    def do_PUT(self) -> None:  # noqa: N802
        self.do_POST()

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path in {"/api/human-annotation", "/api/human-label"}:
                if "dataset" not in query or "index" not in query:
                    raise ApiError("dataset and index query parameters are required")
                revision = query.get("expected_revision", [None])[0]
                expected_revision = (
                    strict_nonnegative_int(revision, "expected_revision")
                    if revision is not None else None
                )
                self._json(
                    self.store.delete_human_annotation(
                        query["dataset"][0],
                        strict_nonnegative_int(query["index"][0], "index"),
                        expected_revision,
                    )
                )
            else:
                self._error(FileNotFoundError(parsed.path), 404)
        except Exception as exc:  # noqa: BLE001 - API boundary
            if not self._handle_expected_error(exc):
                self._error(exc, 500)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSQA Lens local data API")
    parser.add_argument("--config", type=Path, default=Path("inspector_config.yaml"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = DatasetStore(args.config)
    server_config = store.config.get("server") or {}
    host = args.host or str(server_config.get("host", "127.0.0.1"))
    port = args.port or int(server_config.get("port", 8765))
    ApiHandler.store = store
    ApiHandler.allowed_origin = str(server_config.get("frontend_origin", "http://localhost:3000"))
    server = ThreadingHTTPServer((host, port), ApiHandler)
    print(json.dumps({"event": "server_ready", "url": f"http://{host}:{port}", "datasets": len(store.sources)}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

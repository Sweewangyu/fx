#!/usr/bin/env python3
"""Local API for browsing large ChatTS-style JSONL datasets.

The browser never reads multi-GB files or calls the model directly. This server
provides random-access records, template members, audit metadata, merged labels,
and a cached Qwen English-to-Chinese translation proxy.
"""

from __future__ import annotations

import argparse
import hashlib
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
        for source in self.sources.values():
            stats = self.template_stats.get(source.name, {})
            row_count = int(stats.get("rows", 0))
            if row_count <= 0:
                if self._index_valid(source.name):
                    with self._connect(source.name) as connection:
                        row_count = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
                else:
                    row_count = self._fallback_row_count(source)
            rows.append(
                {
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
                }
            )
        return {
            "datasets": rows,
            "qwen": self.qwen_public_config(),
            "tsrbench": self.tsrbench_status,
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

    def _headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", self.allowed_origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _json(self, payload: Any, status: int = 200) -> None:
        self._headers(status)
        self.wfile.write(compact_json(payload))

    def _error(self, exc: Exception, status: int = 400) -> None:
        self._json({"error": type(exc).__name__, "message": str(exc)}, status)

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
            else:
                self._error(FileNotFoundError(parsed.path), 404)
        except KeyError as exc:
            self._error(exc, 404)
        except IndexError as exc:
            self._error(exc, 416)
        except Exception as exc:  # noqa: BLE001 - API boundary
            self._error(exc, 500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/translate":
                texts = body.get("texts") if isinstance(body, dict) else None
                if not isinstance(texts, dict):
                    raise ValueError("body.texts must be an object")
                self._json(self.store.translate(texts))
            else:
                self._error(FileNotFoundError(parsed.path), 404)
        except Exception as exc:  # noqa: BLE001 - API boundary
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

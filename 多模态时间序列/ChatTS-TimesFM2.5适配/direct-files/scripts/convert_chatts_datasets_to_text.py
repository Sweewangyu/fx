#!/usr/bin/env python3
# ruff: noqa: PYI041, UP045
"""Convert ChatTS-style time-series datasets into text-only datasets.

The source data is never modified.  Each ``<ts><ts/>`` (or bare ``<ts>``)
placeholder in a prompt is replaced, in order, by a compact numeric list.
Outputs are JSONL files plus a LlamaFactory-compatible ``dataset_info.json``,
length statistics, a manifest, and recoverable rejected rows.

Only the optional exact token counter requires ``transformers``.  Without a
tokenizer, token lengths are conservatively estimated from character counts.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, MutableMapping, MutableSequence, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO, Union


TS_PLACEHOLDER_RE = re.compile(r"<ts>\s*(?:<ts\s*/>)?", re.IGNORECASE)
TS_TOKEN_RE = re.compile(r"<ts(?:\s*/)?\s*>", re.IGNORECASE)
DATA_EXTENSIONS = {".json", ".jsonl"}
ARTIFACT_NAMES = {
    "dataset_info.json",
    "dataset_stats.json",
    "dataset_stats.csv",
    "dropped_rows.jsonl",
    "manifest.json",
}


class ConversionError(ValueError):
    """A row cannot be converted without changing its meaning."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class InputDataset:
    name: str
    files: list[Path]
    config: dict[str, Any]
    source_kind: str


@dataclass
class Record:
    index: int
    source_file: Path
    row: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    raw: Optional[str] = None


@dataclass
class RowLengths:
    prompt_chars: int
    response_chars: int
    total_chars: int
    estimated_tokens: int
    exact_tokens: Optional[int]
    filter_tokens: int
    placeholders: int
    series: int
    points: int


@dataclass
class DatasetResult:
    name: str
    status: str
    source_files: list[str]
    output_file: Optional[str] = None
    source_rows: int = 0
    valid_rows: int = 0
    kept_before_dataset_filter: int = 0
    kept_rows: int = 0
    rejected_rows: int = 0
    invalid_rows: int = 0
    overlength_rows: int = 0
    dataset_dropped_rows: int = 0
    reject_ratio: float = 0.0
    drop_reason: Optional[str] = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    source_sha256: Optional[str] = None
    prompt_chars: dict[str, Any] = field(default_factory=dict)
    total_chars: dict[str, Any] = field(default_factory=dict)
    estimated_tokens: dict[str, Any] = field(default_factory=dict)
    exact_tokens: dict[str, Any] = field(default_factory=dict)
    filter_tokens: dict[str, Any] = field(default_factory=dict)
    points_per_row: dict[str, Any] = field(default_factory=dict)
    placeholders: int = 0
    series: int = 0
    points: int = 0
    schema: dict[str, str] = field(default_factory=dict)


class ReservoirStats:
    """Streaming exact aggregates plus bounded-memory percentile samples."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.capacity = max(1, capacity)
        self.count = 0
        self.total = 0.0
        self.minimum: Optional[float] = None
        self.maximum: Optional[float] = None
        self.samples: list[float] = []
        self._rng = random.Random(seed)

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.samples) < self.capacity:
            self.samples.append(value)
            return

        replacement = self._rng.randrange(self.count)
        if replacement < self.capacity:
            self.samples[replacement] = value

    def percentile(self, percentile: float) -> Optional[float]:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        rank = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[min(rank, len(ordered) - 1)]

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}

        return {
            "count": self.count,
            "min": _clean_number(self.minimum),
            "mean": _clean_number(self.total / self.count),
            "p50": _clean_number(self.percentile(0.50)),
            "p95": _clean_number(self.percentile(0.95)),
            "max": _clean_number(self.maximum),
            "percentiles_approximate": self.count > self.capacity,
            "percentile_sample_size": len(self.samples),
        }


class TokenCounter:
    def __init__(
        self,
        tokenizer_path: Optional[str],
        chars_per_token: float,
        allow_network: bool,
        trust_remote_code: bool,
    ) -> None:
        self.chars_per_token = chars_per_token
        self.tokenizer_path = tokenizer_path
        self.tokenizer: Any = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("--tokenizer requires transformers to be installed") from exc

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=not allow_network,
                trust_remote_code=trust_remote_code,
            )

    @property
    def method(self) -> str:
        if self.tokenizer is not None:
            return f"exact_content_tokens:{self.tokenizer_path}"
        return f"estimated_ceil_chars_div_{self.chars_per_token:g}"

    def count(self, text: str) -> tuple[int, Optional[int], int]:
        estimated = math.ceil(len(text) / self.chars_per_token) if text else 0
        if self.tokenizer is None:
            return estimated, None, estimated

        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        exact = len(encoded)
        return estimated, exact, exact


def _clean_number(value: Optional[float]) -> Optional[float | int]:
    if value is None:
        return None
    if float(value).is_integer():
        return int(value)
    return round(float(value), 4)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(value: Any, stream: TextIO, *, pretty: bool = False) -> None:
    json.dump(
        value,
        stream,
        ensure_ascii=False,
        allow_nan=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        _json_dump(value, stream, pretty=True)
        stream.write("\n")
    os.replace(temporary, path)


def _safe_slug(name: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    if not compact:
        return f"dataset_{digest}"
    if compact != name or len(compact) > 100:
        return f"{compact[:90]}_{digest}"
    return compact


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _first_non_whitespace(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as stream:
        while True:
            char = stream.read(1)
            if not char or not char.isspace():
                return char


def _iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Record]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8-sig") as stream:
        buffer = ""
        position = 0
        eof = False

        def fill() -> bool:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = stream.read(chunk_size)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        fill()
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if not fill():
                raise ValueError(f"Empty JSON file: {path}")

        if buffer[position] != "[":
            raise ValueError(f"Expected a top-level JSON array in {path}")
        position += 1
        index = 0
        expect_value = True
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    break
                fill()

            if position >= len(buffer):
                raise ValueError(f"Unterminated JSON array in {path}")
            if buffer[position] == "]":
                return
            if not expect_value:
                if buffer[position] != ",":
                    raise ValueError(f"Expected ',' between JSON records in {path}")
                position += 1
                expect_value = True
                continue

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Invalid JSON array near record {index + 1} in {path}")
                    fill()

            index += 1
            if isinstance(value, dict):
                yield Record(index=index, source_file=path, row=value)
            else:
                yield Record(index=index, source_file=path, error="row is not a JSON object", raw=json.dumps(value))
            expect_value = False


def _iter_json_object(path: Path) -> Iterator[Record]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)

    if isinstance(value, list):
        for index, row in enumerate(value, start=1):
            if isinstance(row, dict):
                yield Record(index=index, source_file=path, row=row)
            else:
                yield Record(index=index, source_file=path, error="row is not a JSON object", raw=json.dumps(row))
        return

    if isinstance(value, dict):
        for key in ("data", "train", "records", "examples"):
            nested = value.get(key)
            if isinstance(nested, list):
                for index, row in enumerate(nested, start=1):
                    if isinstance(row, dict):
                        yield Record(index=index, source_file=path, row=row)
                    else:
                        yield Record(
                            index=index,
                            source_file=path,
                            error=f"row in top-level '{key}' is not a JSON object",
                            raw=json.dumps(row),
                        )
                return
        yield Record(index=1, source_file=path, row=value)
        return

    yield Record(index=1, source_file=path, error="top-level JSON value is not an object or array")


def iter_records(path: Path) -> Iterator[Record]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as stream:
            for index, line in enumerate(stream, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    yield Record(index=index, source_file=path, error=f"invalid JSONL: {exc}", raw=stripped)
                    continue
                if isinstance(row, dict):
                    yield Record(index=index, source_file=path, row=row)
                else:
                    yield Record(index=index, source_file=path, error="row is not a JSON object", raw=stripped)
        return

    first = _first_non_whitespace(path)
    if first == "[":
        yield from _iter_json_array(path)
    else:
        yield from _iter_json_object(path)


def _data_files_in_directory(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in DATA_EXTENSIONS
        and candidate.name not in ARTIFACT_NAMES
    )


def _resolve_referenced_path(file_name: str, source_root: Path, info_path: Path) -> Optional[Path]:
    raw = Path(file_name).expanduser()
    if raw.is_absolute():
        return raw.resolve() if raw.exists() else None

    candidates = [
        source_root / raw,
        info_path.parent / raw,
        info_path.parent.parent / raw,
        Path.cwd() / raw,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _parse_dataset_selection(value: str) -> Optional[set[str]]:
    names = {item.strip() for item in value.split(",") if item.strip()}
    if not names or names == {"all"}:
        return None
    return names


def discover_datasets(
    input_path: Path,
    dataset_info_arg: Optional[Path],
    selected_names: Optional[set[str]],
) -> tuple[list[InputDataset], Optional[Path], list[dict[str, Any]]]:
    input_path = input_path.resolve()
    if dataset_info_arg:
        info_path = dataset_info_arg.expanduser().resolve()
    elif input_path.is_file() and input_path.name == "dataset_info.json":
        info_path = input_path
    elif input_path.is_dir() and (input_path / "dataset_info.json").is_file():
        info_path = input_path / "dataset_info.json"
    else:
        info_path = None

    source_root = input_path if input_path.is_dir() else input_path.parent
    skipped: list[dict[str, Any]] = []
    datasets: list[InputDataset] = []
    if info_path:
        with info_path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)
        if not isinstance(info, dict):
            raise ValueError(f"dataset_info.json must contain an object: {info_path}")

        missing_requested = set(selected_names or ()) - set(info)
        if missing_requested:
            raise ValueError(f"Datasets not found in {info_path}: {', '.join(sorted(missing_requested))}")

        for name, raw_config in info.items():
            if selected_names is not None and name not in selected_names:
                continue
            if not isinstance(raw_config, dict):
                skipped.append({"name": name, "reason": "invalid_dataset_info_entry"})
                continue
            file_name = raw_config.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                skipped.append({"name": name, "reason": "non_local_or_missing_file_name"})
                continue
            resolved = _resolve_referenced_path(file_name, source_root, info_path)
            if resolved is None:
                skipped.append({"name": name, "reason": "source_not_found", "file_name": file_name})
                continue
            files = _data_files_in_directory(resolved) if resolved.is_dir() else [resolved]
            files = [path for path in files if path.suffix.lower() in DATA_EXTENSIONS]
            if not files:
                skipped.append({"name": name, "reason": "no_json_files", "file_name": file_name})
                continue
            datasets.append(InputDataset(name=name, files=files, config=copy.deepcopy(raw_config), source_kind="info"))
        return datasets, info_path, skipped

    if input_path.is_file():
        if input_path.suffix.lower() not in DATA_EXTENSIONS:
            raise ValueError(f"Input file must be JSON or JSONL: {input_path}")
        datasets.append(InputDataset(name=input_path.stem, files=[input_path], config={}, source_kind="discovered"))
    else:
        for path in _data_files_in_directory(input_path):
            relative = path.relative_to(input_path).with_suffix("")
            name = "__".join(relative.parts)
            if selected_names is None or name in selected_names:
                datasets.append(InputDataset(name=name, files=[path], config={}, source_kind="discovered"))
        if selected_names is not None:
            found = {dataset.name for dataset in datasets}
            missing = selected_names - found
            if missing:
                raise ValueError(f"Discovered datasets not found: {', '.join(sorted(missing))}")
    return datasets, None, skipped


def _columns(config: dict[str, Any]) -> dict[str, str]:
    columns = config.get("columns")
    if not isinstance(columns, dict):
        return {}
    return {str(key): str(value) for key, value in columns.items() if isinstance(value, str) and value}


def _infer_schema(row: dict[str, Any]) -> dict[str, str]:
    schema: dict[str, str] = {}
    for key in ("input", "instruction", "prompt", "question", "query"):
        if isinstance(row.get(key), str):
            schema["prompt"] = key
            break
    for key in ("output", "response", "answer", "target"):
        if isinstance(row.get(key), str):
            schema["response"] = key
            break
    for key in ("timeseries", "time_series", "ts", "series"):
        if key in row:
            schema["timeseries"] = key
            break
    if "messages" in row and isinstance(row["messages"], list):
        schema["messages"] = "messages"
    elif "conversations" in row and isinstance(row["conversations"], list):
        schema["messages"] = "conversations"
    return schema


Location = tuple[Union[MutableMapping[Any, Any], MutableSequence[Any]], Any]


def _add_location(locations: list[Location], seen: set[tuple[int, Any]], container: Any, key: Any) -> None:
    try:
        value = container[key]
    except (KeyError, IndexError, TypeError):
        return
    identity = (id(container), key)
    if identity not in seen and isinstance(value, str):
        locations.append((container, key))
        seen.add(identity)


def _prompt_locations(row: dict[str, Any], config: dict[str, Any], schema: dict[str, str]) -> list[Location]:
    columns = _columns(config)
    formatting = config.get("formatting", "alpaca")
    locations: list[Location] = []
    seen: set[tuple[int, Any]] = set()

    system_key = columns.get("system")
    if system_key:
        _add_location(locations, seen, row, system_key)

    messages_key = columns.get("messages") or schema.get("messages")
    if formatting == "sharegpt" or messages_key:
        messages = row.get(messages_key) if messages_key else None
        tags = config.get("tags") if isinstance(config.get("tags"), dict) else {}
        role_key = tags.get("role_tag", "from")
        content_key = tags.get("content_tag", "value")
        prompt_roles = {
            tags.get("user_tag", "human"),
            tags.get("observation_tag", "observation"),
            tags.get("system_tag", "system"),
            "user",
            "human",
            "observation",
            "system",
        }
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get(role_key, message.get("role"))
                candidate_content_key = content_key if content_key in message else "content"
                if role in prompt_roles:
                    _add_location(locations, seen, message, candidate_content_key)
        return locations

    history_key = columns.get("history")
    if history_key and isinstance(row.get(history_key), list):
        for turn in row[history_key]:
            if isinstance(turn, list) and turn:
                _add_location(locations, seen, turn, 0)
            elif isinstance(turn, dict):
                for candidate in ("prompt", "input", "user"):
                    if candidate in turn:
                        _add_location(locations, seen, turn, candidate)
                        break

    prompt_key = columns.get("prompt") or schema.get("prompt")
    query_key = columns.get("query")
    if prompt_key:
        _add_location(locations, seen, row, prompt_key)
    if query_key:
        _add_location(locations, seen, row, query_key)
    return locations


def _response_texts(row: dict[str, Any], config: dict[str, Any], schema: dict[str, str]) -> list[str]:
    columns = _columns(config)
    texts: list[str] = []
    response_key = columns.get("response") or schema.get("response")
    if response_key and isinstance(row.get(response_key), str):
        texts.append(row[response_key])

    history_key = columns.get("history")
    if history_key and isinstance(row.get(history_key), list):
        for turn in row[history_key]:
            if isinstance(turn, list) and len(turn) > 1 and isinstance(turn[1], str):
                texts.append(turn[1])

    messages_key = columns.get("messages") or schema.get("messages")
    if messages_key and isinstance(row.get(messages_key), list):
        tags = config.get("tags") if isinstance(config.get("tags"), dict) else {}
        role_key = tags.get("role_tag", "from")
        content_key = tags.get("content_tag", "value")
        response_roles = {tags.get("assistant_tag", "gpt"), tags.get("function_tag", "function_call"), "assistant", "gpt"}
        for message in row[messages_key]:
            if not isinstance(message, dict):
                continue
            role = message.get(role_key, message.get("role"))
            candidate_content_key = content_key if content_key in message else "content"
            content = message.get(candidate_content_key)
            if role in response_roles and isinstance(content, str):
                texts.append(content)
    return texts


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _series_values(value: Any) -> list[int | float]:
    if isinstance(value, dict):
        for key in ("values", "data", "timeseries", "series"):
            if key in value:
                return _series_values(value[key])
        raise ConversionError("invalid_timeseries", "time-series object has no values/data field")
    if not isinstance(value, list):
        raise ConversionError("invalid_timeseries", "a time series must be a JSON list")
    if all(_is_numeric(point) for point in value):
        return value
    if value and all(isinstance(point, list) and len(point) == 1 and _is_numeric(point[0]) for point in value):
        return [point[0] for point in value]
    raise ConversionError("invalid_timeseries", "time series must be one-dimensional numeric values")


def _normalise_series(raw: Any, expected: int) -> list[list[int | float]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConversionError("invalid_timeseries", "timeseries string is not valid JSON") from exc
    if isinstance(raw, dict):
        for key in ("timeseries", "series", "values", "data"):
            if key in raw:
                raw = raw[key]
                break

    if expected == 0:
        if raw in (None, [], {}):
            return []
        raise ConversionError("unused_timeseries", "row has timeseries values but no <ts> placeholder")
    if raw is None:
        raise ConversionError("missing_timeseries", "row has <ts> placeholders but no timeseries field")
    if not isinstance(raw, list):
        raise ConversionError("invalid_timeseries", "timeseries field must be a JSON list")

    if expected == 1:
        try:
            return [_series_values(raw)]
        except ConversionError:
            if len(raw) == 1:
                return [_series_values(raw[0])]
            raise ConversionError(
                "placeholder_series_mismatch",
                f"found 1 placeholder but timeseries cannot be interpreted as one series (top-level length={len(raw)})",
            )

    if len(raw) != expected:
        raise ConversionError(
            "placeholder_series_mismatch",
            f"found {expected} placeholders but {len(raw)} top-level series",
        )
    return [_series_values(series) for series in raw]


def _format_number(value: int | float, precision: int) -> str:
    if isinstance(value, int):
        return str(value)
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ConversionError("non_finite_timeseries", f"non-finite time-series value: {value}")
    if numeric == 0:
        return "0"
    return format(numeric, f".{precision}g")


def _format_series(series: Sequence[int | float], precision: int) -> str:
    return "[" + ",".join(_format_number(value, precision) for value in series) + "]"


def transform_row(
    source_row: dict[str, Any],
    config: dict[str, Any],
    token_counter: TokenCounter,
    precision: int,
    inferred_schema: Optional[dict[str, str]] = None,
) -> tuple[dict[str, Any], RowLengths, dict[str, str]]:
    row = copy.deepcopy(source_row)
    schema = dict(inferred_schema or _infer_schema(row))
    columns = _columns(config)
    timeseries_key = columns.get("timeseries") or schema.get("timeseries")
    locations = _prompt_locations(row, config, schema)
    if not locations:
        raise ConversionError("missing_prompt", "could not find a prompt/message text field")

    placeholder_count = sum(len(TS_PLACEHOLDER_RE.findall(container[key])) for container, key in locations)
    raw_timeseries = row.get(timeseries_key) if timeseries_key else None
    series = _normalise_series(raw_timeseries, placeholder_count)
    replacements = iter(_format_series(values, precision) for values in series)

    def replace(match: re.Match[str]) -> str:
        try:
            return next(replacements)
        except StopIteration as exc:
            raise ConversionError("placeholder_series_mismatch", "not enough series for prompt placeholders") from exc

    for container, key in locations:
        container[key] = TS_PLACEHOLDER_RE.sub(replace, container[key])
    try:
        next(replacements)
    except StopIteration:
        pass
    else:
        raise ConversionError("placeholder_series_mismatch", "unused time series remained after replacement")

    if any(TS_TOKEN_RE.search(container[key]) for container, key in locations):
        raise ConversionError("unreplaced_placeholder", "a <ts> token remains in the converted prompt")
    if timeseries_key:
        row.pop(timeseries_key, None)

    prompt_texts = [container[key] for container, key in locations]
    response_texts = _response_texts(row, config, schema)
    if any(TS_TOKEN_RE.search(text) for text in response_texts):
        raise ConversionError(
            "timeseries_token_in_response",
            "an assistant/response field contains a <ts> token; text-only targets must not contain time-series placeholders",
        )
    prompt_chars = sum(len(text) for text in prompt_texts)
    response_chars = sum(len(text) for text in response_texts)
    content = "\n".join(prompt_texts + response_texts)
    estimated, exact, filter_tokens = token_counter.count(content)
    lengths = RowLengths(
        prompt_chars=prompt_chars,
        response_chars=response_chars,
        total_chars=len(content),
        estimated_tokens=estimated,
        exact_tokens=exact,
        filter_tokens=filter_tokens,
        placeholders=placeholder_count,
        series=len(series),
        points=sum(len(values) for values in series),
    )
    return row, lengths, schema


def _output_dataset_config(dataset: InputDataset, output_file: str, schema: dict[str, str]) -> dict[str, Any]:
    output = copy.deepcopy(dataset.config)
    output.pop("hf_hub_url", None)
    output.pop("ms_hub_url", None)
    output.pop("om_hub_url", None)
    output.pop("script_url", None)
    output.pop("cloud_file_name", None)
    output["file_name"] = output_file
    columns = _columns(output) or dict(schema)
    columns.pop("timeseries", None)
    if columns:
        output["columns"] = columns
    description = output.get("description")
    marker = "Text-only copy: <ts> placeholders were replaced by numeric sequences."
    output["description"] = f"{description} {marker}" if description else marker
    return output


def _write_dropped(
    stream: TextIO,
    dataset: str,
    record: Record,
    reason: str,
    message: str,
    row: Any,
    lengths: Optional[RowLengths] = None,
    include_row: bool = False,
) -> None:
    try:
        canonical_row = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        canonical_row = repr(row)

    prompt_preview = ""
    if isinstance(row, dict):
        for key in ("input", "instruction", "prompt", "question", "query"):
            if isinstance(row.get(key), str):
                prompt_preview = row[key]
                break
        if not prompt_preview:
            for messages_key in ("messages", "conversations"):
                messages = row.get(messages_key)
                if not isinstance(messages, list):
                    continue
                for item in messages:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content", item.get("value"))
                    if isinstance(content, str):
                        prompt_preview = content
                        break
                if prompt_preview:
                    break
    elif isinstance(row, str):
        prompt_preview = row

    payload: dict[str, Any] = {
        "dataset": dataset,
        "source_file": str(record.source_file),
        "source_index": record.index,
        "reason": reason,
        "message": message,
        "row_sha256": hashlib.sha256(canonical_row.encode("utf-8")).hexdigest(),
        "prompt_preview": prompt_preview[:500],
    }
    if lengths is not None:
        payload["lengths"] = asdict(lengths)
    if include_row:
        payload["row"] = row
    _json_dump(payload, stream)
    stream.write("\n")


def convert_dataset(
    dataset: InputDataset,
    output_dir: Path,
    dropped_stream: TextIO,
    token_counter: TokenCounter,
    args: argparse.Namespace,
    used_slugs: set[str],
) -> tuple[DatasetResult, Optional[dict[str, Any]]]:
    result = DatasetResult(
        name=dataset.name,
        status="running",
        source_files=[str(path) for path in dataset.files],
        source_sha256=None if args.skip_source_hash else _sha256_files(dataset.files),
    )
    slug = _safe_slug(dataset.name)
    if slug in used_slugs:
        slug = f"{slug}_{hashlib.sha1(dataset.name.encode('utf-8')).hexdigest()[:8]}"
    used_slugs.add(slug)
    relative_output = f"datasets/{slug}.jsonl"
    final_path = output_dir / relative_output
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging_name = tempfile.mkstemp(prefix=f".{slug}.", suffix=".partial", dir=str(final_path.parent))
    os.close(fd)
    staging_path = Path(staging_name)

    prompt_stats = ReservoirStats(args.stats_reservoir_size, seed=101)
    total_char_stats = ReservoirStats(args.stats_reservoir_size, seed=102)
    estimated_stats = ReservoirStats(args.stats_reservoir_size, seed=103)
    exact_stats = ReservoirStats(args.stats_reservoir_size, seed=104)
    filter_stats = ReservoirStats(args.stats_reservoir_size, seed=105)
    points_stats = ReservoirStats(args.stats_reservoir_size, seed=106)
    rejection_reasons: Counter[str] = Counter()
    schema: dict[str, str] = {}

    try:
        with staging_path.open("w", encoding="utf-8") as output_stream:
            for source_file in dataset.files:
                try:
                    records: Iterable[Record] = iter_records(source_file)
                    for record in records:
                        result.source_rows += 1
                        if args.log_every and result.source_rows % args.log_every == 0:
                            print(
                                f"[{dataset.name}] rows={result.source_rows:,} "
                                f"kept={result.kept_before_dataset_filter:,} rejected={result.rejected_rows:,}",
                                file=sys.stderr,
                            )
                        if record.error:
                            rejection_reasons["invalid_json_record"] += 1
                            result.invalid_rows += 1
                            result.rejected_rows += 1
                            _write_dropped(
                                dropped_stream,
                                dataset.name,
                                record,
                                "invalid_json_record",
                                record.error,
                                {"raw": record.raw},
                                include_row=args.include_dropped_row,
                            )
                            continue
                        assert record.row is not None
                        try:
                            converted, lengths, detected_schema = transform_row(
                                record.row,
                                dataset.config,
                                token_counter,
                                args.float_precision,
                                schema or None,
                            )
                            if not schema:
                                schema = detected_schema
                        except (ConversionError, TypeError, ValueError) as exc:
                            reason = exc.reason if isinstance(exc, ConversionError) else "conversion_error"
                            rejection_reasons[reason] += 1
                            result.invalid_rows += 1
                            result.rejected_rows += 1
                            _write_dropped(
                                dropped_stream,
                                dataset.name,
                                record,
                                reason,
                                str(exc),
                                record.row,
                                include_row=args.include_dropped_row,
                            )
                            if args.fail_on_error:
                                raise
                            continue

                        result.valid_rows += 1
                        result.placeholders += lengths.placeholders
                        result.series += lengths.series
                        result.points += lengths.points
                        prompt_stats.add(lengths.prompt_chars)
                        total_char_stats.add(lengths.total_chars)
                        estimated_stats.add(lengths.estimated_tokens)
                        if lengths.exact_tokens is not None:
                            exact_stats.add(lengths.exact_tokens)
                        filter_stats.add(lengths.filter_tokens)
                        points_stats.add(lengths.points)

                        over_chars = args.max_chars > 0 and lengths.total_chars > args.max_chars
                        over_tokens = args.max_tokens > 0 and lengths.filter_tokens > args.max_tokens
                        over_points = args.max_points > 0 and lengths.points > args.max_points
                        if over_chars or over_tokens or over_points:
                            if over_points:
                                reason = "exceeds_max_points"
                                threshold = args.max_points
                                observed = lengths.points
                            elif over_chars:
                                reason = "exceeds_max_chars"
                                threshold = args.max_chars
                                observed = lengths.total_chars
                            else:
                                reason = "exceeds_max_tokens"
                                threshold = args.max_tokens
                                observed = lengths.filter_tokens
                            rejection_reasons[reason] += 1
                            result.overlength_rows += 1
                            result.rejected_rows += 1
                            _write_dropped(
                                dropped_stream,
                                dataset.name,
                                record,
                                reason,
                                f"observed {observed} > limit {threshold}",
                                record.row,
                                lengths,
                                include_row=args.include_dropped_row,
                            )
                            continue

                        _json_dump(converted, output_stream)
                        output_stream.write("\n")
                        result.kept_before_dataset_filter += 1
                except Exception as exc:
                    if args.fail_on_error:
                        raise
                    rejection_reasons["source_read_error"] += 1
                    result.rejected_rows += 1
                    _write_dropped(
                        dropped_stream,
                        dataset.name,
                        Record(index=0, source_file=source_file),
                        "source_read_error",
                        str(exc),
                        None,
                        include_row=args.include_dropped_row,
                    )

        result.reject_ratio = result.rejected_rows / result.source_rows if result.source_rows else 1.0
        drop_reasons = []
        p95_tokens = filter_stats.percentile(0.95)
        if args.dataset_max_reject_ratio <= 1 and result.reject_ratio > args.dataset_max_reject_ratio:
            drop_reasons.append(
                f"reject_ratio={result.reject_ratio:.6f} > dataset_max_reject_ratio={args.dataset_max_reject_ratio:.6f}"
            )
        if args.dataset_max_p95_tokens > 0 and p95_tokens is not None and p95_tokens > args.dataset_max_p95_tokens:
            drop_reasons.append(
                f"p95_tokens={p95_tokens:g} > dataset_max_p95_tokens={args.dataset_max_p95_tokens}"
            )
        p95_points = points_stats.percentile(0.95)
        if args.dataset_max_p95_points > 0 and p95_points is not None and p95_points > args.dataset_max_p95_points:
            drop_reasons.append(
                f"p95_points={p95_points:g} > dataset_max_p95_points={args.dataset_max_p95_points}"
            )
        if result.kept_before_dataset_filter == 0:
            drop_reasons.append("no rows remain after row-level validation/filtering")

        if drop_reasons:
            result.status = "dropped"
            result.drop_reason = "; ".join(drop_reasons)
            result.dataset_dropped_rows = result.kept_before_dataset_filter
            result.kept_rows = 0
            if staging_path.stat().st_size:
                with staging_path.open("r", encoding="utf-8") as stream:
                    for index, line in enumerate(stream, start=1):
                        converted_row = json.loads(line)
                        _write_dropped(
                            dropped_stream,
                            dataset.name,
                            Record(index=index, source_file=staging_path),
                            "dataset_threshold",
                            result.drop_reason,
                            converted_row,
                            include_row=args.include_dropped_row,
                        )
            staging_path.unlink(missing_ok=True)
            output_config = None
        else:
            result.status = "kept"
            result.kept_rows = result.kept_before_dataset_filter
            result.output_file = relative_output
            os.replace(staging_path, final_path)
            output_config = _output_dataset_config(dataset, relative_output, schema)

        result.rejection_reasons = dict(sorted(rejection_reasons.items()))
        result.prompt_chars = prompt_stats.summary()
        result.total_chars = total_char_stats.summary()
        result.estimated_tokens = estimated_stats.summary()
        result.exact_tokens = exact_stats.summary()
        result.filter_tokens = filter_stats.summary()
        result.points_per_row = points_stats.summary()
        result.schema = schema
        return result, output_config
    finally:
        staging_path.unlink(missing_ok=True)


def _stats_csv_row(result: DatasetResult) -> dict[str, Any]:
    def metric(group: dict[str, Any], key: str) -> Any:
        return group.get(key) if group else None

    return {
        "dataset": result.name,
        "status": result.status,
        "source_rows": result.source_rows,
        "valid_rows": result.valid_rows,
        "kept_rows": result.kept_rows,
        "rejected_rows": result.rejected_rows,
        "invalid_rows": result.invalid_rows,
        "overlength_rows": result.overlength_rows,
        "dataset_dropped_rows": result.dataset_dropped_rows,
        "reject_ratio": f"{result.reject_ratio:.8f}",
        "tokens_min": metric(result.filter_tokens, "min"),
        "tokens_mean": metric(result.filter_tokens, "mean"),
        "tokens_p50": metric(result.filter_tokens, "p50"),
        "tokens_p95": metric(result.filter_tokens, "p95"),
        "tokens_max": metric(result.filter_tokens, "max"),
        "points_p50": metric(result.points_per_row, "p50"),
        "points_p95": metric(result.points_per_row, "p95"),
        "points_max": metric(result.points_per_row, "max"),
        "chars_p95": metric(result.total_chars, "p95"),
        "chars_max": metric(result.total_chars, "max"),
        "placeholders": result.placeholders,
        "series": result.series,
        "points": result.points,
        "drop_reason": result.drop_reason or "",
        "output_file": result.output_file or "",
        "source_sha256": result.source_sha256 or "",
    }


def _prepare_output(input_path: Path, output_dir: Path, overwrite: bool) -> None:
    input_resolved = input_path.resolve()
    output_resolved = output_dir.resolve()
    if input_resolved == output_resolved:
        raise ValueError("Output directory must differ from the source path")
    if input_resolved.is_dir() and input_resolved in output_resolved.parents:
        raise ValueError("Output directory must not be nested inside the source directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty (use --overwrite): {output_dir}")
        manifest_path = output_dir / "manifest.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                previous_manifest = json.load(stream)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                "--overwrite refused: output has no valid manifest from this converter; choose a new output directory"
            ) from exc
        if previous_manifest.get("converter") != "convert_chatts_datasets_to_text.py":
            raise ValueError(
                "--overwrite refused: existing output is not identified as a ChatTS text-converter output"
            )
        datasets_dir = output_dir / "datasets"
        if datasets_dir.exists():
            shutil.rmtree(datasets_dir)
        for artifact in ARTIFACT_NAMES:
            target = output_dir / artifact
            if target.is_file():
                target.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace ChatTS <ts> placeholders with numeric sequences and write text-only datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "--input-dir", dest="input_path", required=True, help="Dataset directory/file")
    parser.add_argument("--output", "--output-dir", dest="output_dir", required=True, help="New output directory")
    parser.add_argument("--dataset-info", help="Explicit dataset_info.json (otherwise auto-detected)")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names, or all")
    parser.add_argument("--max-tokens", type=int, default=0, help="Drop a row above this token length; 0 disables")
    parser.add_argument("--max-chars", type=int, default=0, help="Drop a row above this character length; 0 disables")
    parser.add_argument("--max-points", type=int, default=0, help="Drop a row above this total numeric-point count; 0 disables")
    parser.add_argument(
        "--dataset-max-reject-ratio",
        "--drop-dataset-if-reject-ratio",
        type=float,
        default=1.01,
        help="Drop a whole dataset when its row rejection ratio is greater than this; >1 disables",
    )
    parser.add_argument(
        "--dataset-max-p95-tokens",
        type=int,
        default=0,
        help="Drop a whole dataset when its pre-filter p95 token length exceeds this; 0 disables",
    )
    parser.add_argument(
        "--dataset-max-p95-points",
        type=int,
        default=0,
        help="Drop a whole dataset when its p95 numeric-point count exceeds this; 0 disables",
    )
    parser.add_argument("--chars-per-token", type=float, default=3.0, help="Estimator used when --tokenizer is omitted")
    parser.add_argument("--tokenizer", help="Optional local Hugging Face tokenizer path for exact content-token counts")
    parser.add_argument(
        "--allow-estimated-token-filter",
        action="store_true",
        help="Explicitly allow token thresholds to use the chars-per-token estimate without --tokenizer",
    )
    parser.add_argument("--allow-tokenizer-network", action="store_true", help="Allow tokenizer download from the network")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust tokenizer remote code")
    parser.add_argument("--float-precision", type=int, default=6, help="Significant digits used for numeric sequences")
    parser.add_argument("--stats-reservoir-size", type=int, default=100000, help="Max values retained per percentile metric")
    parser.add_argument("--log-every", type=int, default=10000, help="Print progress after this many rows; 0 disables")
    parser.add_argument("--skip-source-hash", action="store_true", help="Skip SHA256 calculation to reduce disk reads")
    parser.add_argument("--fail-on-error", action="store_true", help="Stop on the first malformed row/source")
    parser.add_argument(
        "--include-dropped-row",
        action="store_true",
        help="Store the complete original row in dropped_rows.jsonl (can be very large)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace converter artifacts in a non-empty output directory")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens < 0 or args.max_chars < 0 or args.max_points < 0:
        raise ValueError("--max-tokens, --max-chars and --max-points must be >= 0")
    if args.dataset_max_reject_ratio < 0:
        raise ValueError("--dataset-max-reject-ratio must be >= 0")
    if args.dataset_max_p95_tokens < 0:
        raise ValueError("--dataset-max-p95-tokens must be >= 0")
    if args.dataset_max_p95_points < 0:
        raise ValueError("--dataset-max-p95-points must be >= 0")
    if args.chars_per_token <= 0:
        raise ValueError("--chars-per-token must be > 0")
    if args.float_precision < 1:
        raise ValueError("--float-precision must be >= 1")
    if args.stats_reservoir_size < 1:
        raise ValueError("--stats-reservoir-size must be >= 1")
    if (
        (args.max_tokens > 0 or args.dataset_max_p95_tokens > 0)
        and not args.tokenizer
        and not args.allow_estimated_token_filter
    ):
        raise ValueError(
            "token filtering without --tokenizer is estimated; add --allow-estimated-token-filter or use --max-points"
        )


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    input_path = Path(args.input_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    _prepare_output(input_path, output_dir, args.overwrite)

    selected = _parse_dataset_selection(args.datasets)
    info_arg = Path(args.dataset_info) if args.dataset_info else None
    datasets, info_path, skipped = discover_datasets(input_path, info_arg, selected)
    if not datasets:
        raise ValueError("No local JSON/JSONL datasets were found")

    counter = TokenCounter(
        tokenizer_path=args.tokenizer,
        chars_per_token=args.chars_per_token,
        allow_network=args.allow_tokenizer_network,
        trust_remote_code=args.trust_remote_code,
    )
    started_at = _utc_now()
    results: list[DatasetResult] = []
    output_info: dict[str, Any] = {}
    used_slugs: set[str] = set()
    dropped_path = output_dir / "dropped_rows.jsonl"
    with dropped_path.open("w", encoding="utf-8") as dropped_stream:
        for dataset in datasets:
            print(f"Converting {dataset.name} ({len(dataset.files)} source file(s))...", file=sys.stderr)
            result, config = convert_dataset(dataset, output_dir, dropped_stream, counter, args, used_slugs)
            results.append(result)
            if config is not None:
                output_info[dataset.name] = config
            print(
                f"[{dataset.name}] status={result.status} source={result.source_rows:,} "
                f"kept={result.kept_rows:,} rejected={result.rejected_rows:,} "
                f"p95_tokens={result.filter_tokens.get('p95')}",
                file=sys.stderr,
            )

    _write_json(output_dir / "dataset_info.json", output_info)
    stats_payload = {
        "created_at": _utc_now(),
        "token_count_method": counter.method,
        "length_scope": "prompt/system/history/response content joined with newlines; chat-template overhead is excluded",
        "datasets": [asdict(result) for result in results],
        "skipped_datasets": skipped,
    }
    _write_json(output_dir / "dataset_stats.json", stats_payload)

    csv_rows = [_stats_csv_row(result) for result in results]
    with (output_dir / "dataset_stats.csv").open("w", encoding="utf-8", newline="") as stream:
        if csv_rows:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)

    summary = {
        "datasets_discovered": len(datasets),
        "datasets_kept": sum(result.status == "kept" for result in results),
        "datasets_dropped": sum(result.status == "dropped" for result in results),
        "datasets_skipped": len(skipped),
        "source_rows": sum(result.source_rows for result in results),
        "kept_rows": sum(result.kept_rows for result in results),
        "rejected_rows": sum(result.rejected_rows for result in results),
        "dataset_dropped_rows": sum(result.dataset_dropped_rows for result in results),
    }
    script_path = Path(__file__).resolve()
    manifest = {
        "converter": "convert_chatts_datasets_to_text.py",
        "format_version": 1,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "source_path": str(input_path.resolve()),
        "source_dataset_info": str(info_path) if info_path else None,
        "source_dataset_info_sha256": _sha256_file(info_path) if info_path else None,
        "output_path": str(output_dir.resolve()),
        "script": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"tokenizer"} or value is None or isinstance(value, (str, int, float, bool))
        },
        "token_count_method": counter.method,
        "summary": summary,
        "artifacts": {
            "dataset_info": "dataset_info.json",
            "dataset_stats_json": "dataset_stats.json",
            "dataset_stats_csv": "dataset_stats.csv",
            "dropped_rows": "dropped_rows.jsonl",
        },
        "skipped_datasets": skipped,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

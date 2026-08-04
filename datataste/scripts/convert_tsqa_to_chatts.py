#!/usr/bin/env python3
"""Convert TSAQA and Time-MQA/TSQA into ChatTS SFT JSONL.

The emitted training rows contain exactly the three fields used by the
ChatTS Training Dataset::

    {"input": str, "timeseries": list[list[float]], "output": str}

Every ``<ts><ts/>`` in ``input`` corresponds, in order, to one entry in
``timeseries``.  Source metadata is written to a separate audit JSONL so it
cannot accidentally become a model feature.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

PLACEHOLDER = "<ts><ts/>"
SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet"}
MISSING_STRINGS = {"", "x", "nan", "na", "n/a", "none", "null", "missing", "?"}


class ConversionError(ValueError):
    """A source row cannot be represented safely as a ChatTS sample."""


@dataclass
class ConvertOptions:
    min_inline_length: int = 4
    missing_policy: str = "mask-zero"
    max_series: int = 30
    max_series_length: int = 0
    long_series_policy: str = "drop"


@dataclass
class Stats:
    files_seen: int = 0
    files_skipped_nontrain: int = 0
    files_skipped_contamination: int = 0
    rows_seen: int = 0
    rows_written: int = 0
    rows_duplicate: int = 0
    rows_filtered_contamination: int = 0
    rows_invalid: int = 0
    rows_missing_mask: int = 0
    max_series_per_row: int = 0
    max_series_length: int = 0
    task_counts: Counter = field(default_factory=Counter)
    question_type_counts: Counter = field(default_factory=Counter)
    invalid_reasons: Counter = field(default_factory=Counter)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("task_counts", "question_type_counts", "invalid_reasons"):
            # dataclasses.asdict reconstructs Counter from an iterable of
            # key/value tuples on Python 3.9, which turns each tuple into a
            # Counter key.  Read the original Counter explicitly instead.
            result[key] = dict(sorted(getattr(self, key).items()))
        return result


@dataclass
class Materialized:
    arrays: List[List[float]]
    marker: str
    used_missing_mask: bool = False


def _preview(value: Any, limit: int = 180) -> str:
    text = repr(value).replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _lower_key_map(row: Mapping[str, Any]) -> Dict[str, str]:
    return {str(key).strip().lower(): str(key) for key in row.keys()}


def get_field(row: Mapping[str, Any], aliases: Sequence[str], default: Any = None) -> Any:
    key_map = _lower_key_map(row)
    for alias in aliases:
        original = key_map.get(alias.lower())
        if original is not None:
            value = row.get(original)
            if value is not None and (not isinstance(value, str) or value.strip()):
                return value
    return default


def get_text(row: Mapping[str, Any], aliases: Sequence[str], default: str = "") -> str:
    value = get_field(row, aliases, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _literal_with_missing_tokens(text: str) -> Any:
    """Parse Python/JSON-style arrays, including bare X/NaN markers."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty literal")
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        pass

    normalized = re.sub(
        r"(?i)(?<![\w'\"])(?:x|nan|na|null|none|missing)(?![\w'\"])",
        lambda match: repr(match.group(0)),
        stripped,
    )
    return ast.literal_eval(normalized)


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (list, tuple, dict))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_STRINGS
    if isinstance(value, float):
        return not math.isfinite(value)
    return False


def _number_or_missing(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise ConversionError("boolean value in time series")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"non-numeric time-series value: {value!r}") from exc
    if not math.isfinite(result):
        return None
    return result


def parse_array(value: Any) -> Any:
    if isinstance(value, str):
        return _literal_with_missing_tokens(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def raw_channels(value: Any, orientation: str = "rows") -> List[List[Any]]:
    """Turn a 1-D/2-D array into channels expected by ChatTS.

    TSAQA stores multiple series by row.  ``auto`` additionally recognizes the
    common time-major shape ``[time, channel]`` and transposes it.
    """
    parsed = parse_array(value)
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ConversionError("time series is not a non-empty list")
    parsed = list(parsed)
    if all(_is_scalar(item) for item in parsed):
        return [parsed]
    if not all(isinstance(item, (list, tuple)) for item in parsed):
        raise ConversionError("time series mixes scalar and nested values")

    rows = [list(item) for item in parsed]
    if not rows or any(not row for row in rows):
        raise ConversionError("time series contains an empty channel")
    if any(not all(_is_scalar(item) for item in row) for row in rows):
        raise ConversionError("only 1-D or 2-D time-series arrays are supported")

    if orientation == "auto":
        widths = {len(row) for row in rows}
        if len(widths) == 1:
            width = next(iter(widths))
            # Typical sensor input is T x C (many timestamps, few channels).
            if len(rows) > width and width <= 64:
                rows = [list(channel) for channel in zip(*rows)]
    return rows


def _limit_length(values: List[float], options: ConvertOptions) -> List[float]:
    limit = options.max_series_length
    if limit <= 0 or len(values) <= limit:
        return values
    if options.long_series_policy == "truncate":
        return values[:limit]
    raise ConversionError(f"series length {len(values)} exceeds configured maximum {limit}")


def materialize_channels(channels: Sequence[Sequence[Any]], options: ConvertOptions) -> Materialized:
    arrays: List[List[float]] = []
    markers: List[str] = []
    used_mask = False

    for channel_number, raw in enumerate(channels, start=1):
        converted = [_number_or_missing(value) for value in raw]
        if not converted:
            raise ConversionError("empty time-series channel")
        has_missing = any(value is None for value in converted)
        prefix = f"channel {channel_number}: " if len(channels) > 1 else ""
        if has_missing:
            if options.missing_policy == "drop":
                raise ConversionError("time series contains missing values")
            filled = [0.0 if value is None else value for value in converted]
            mask = [0.0 if value is None else 1.0 for value in converted]
            arrays.extend([_limit_length(filled, options), _limit_length(mask, options)])
            markers.append(
                f"{prefix}{PLACEHOLDER} (values; missing entries filled with 0), "
                f"{PLACEHOLDER} (observation mask; 1=observed, 0=missing)"
            )
            used_mask = True
        else:
            values = [value for value in converted if value is not None]
            arrays.append(_limit_length(values, options))
            markers.append(f"{prefix}{PLACEHOLDER}")

    return Materialized(arrays=arrays, marker="; ".join(markers), used_missing_mask=used_mask)


def _balanced_bracket_spans(text: str) -> List[Tuple[int, int]]:
    """Return outermost square-bracket spans while respecting quoted text."""
    spans: List[Tuple[int, int]] = []
    depth = 0
    start = -1
    quote: Optional[str] = None
    escaped = False

    for index, char in enumerate(text):
        # Apostrophes and quotation marks in ordinary prose (for example,
        # "Yahoo's servers") are not string delimiters for our purposes.
        # Only track quoted strings after entering an array literal.
        if depth == 0:
            if char == "[":
                start = index
                depth = 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, index + 1))
                start = -1
    return spans


def extract_inline_timeseries(
    text: str, options: ConvertOptions, orientation: str = "auto"
) -> Tuple[str, List[List[float]], bool]:
    """Replace numeric array literals with ChatTS placeholders."""
    arrays: List[List[float]] = []
    used_mask = False
    pieces: List[str] = []
    cursor = 0

    for start, end in _balanced_bracket_spans(text):
        literal = text[start:end]
        try:
            channels = raw_channels(literal, orientation=orientation)
            scalar_count = sum(len(channel) for channel in channels)
            if scalar_count < options.min_inline_length:
                continue
            materialized = materialize_channels(channels, options)
        except (ConversionError, ValueError, SyntaxError):
            continue

        pieces.append(text[cursor:start])
        pieces.append(materialized.marker)
        cursor = end
        arrays.extend(materialized.arrays)
        used_mask = used_mask or materialized.used_missing_mask

    if not arrays:
        return text, [], False
    pieces.append(text[cursor:])
    return "".join(pieces), arrays, used_mask


def _clean_context(value: Any) -> str:
    if value is None:
        return ""
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null", "nan"}:
            return ""
        try:
            parsed = _literal_with_missing_tokens(stripped)
        except (ValueError, SyntaxError):
            return stripped
    if isinstance(parsed, (list, tuple)):
        return "; ".join(str(item) for item in parsed if str(item).strip())
    if isinstance(parsed, dict):
        return "; ".join(f"{key}: {value}" for key, value in parsed.items())
    return str(parsed).strip()


def _series_block(materialized: Materialized) -> str:
    return "Input time series: " + materialized.marker + "."


def _validate_sample(sample: Mapping[str, Any], options: ConvertOptions) -> None:
    prompt = sample.get("input")
    output = sample.get("output")
    arrays = sample.get("timeseries")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConversionError("empty input prompt")
    if not isinstance(output, str) or not output.strip():
        raise ConversionError("empty output answer")
    if not isinstance(arrays, list) or not arrays:
        raise ConversionError("no time series extracted")
    count = prompt.count(PLACEHOLDER)
    if count != len(arrays):
        raise ConversionError(f"placeholder/series mismatch: {count} != {len(arrays)}")
    if len(arrays) > options.max_series:
        raise ConversionError(f"sample contains {len(arrays)} series; maximum is {options.max_series}")
    for values in arrays:
        if not isinstance(values, list) or not values:
            raise ConversionError("empty time-series array")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ConversionError("non-finite value remains after conversion")


def convert_tsaqa(row: Mapping[str, Any], options: ConvertOptions) -> Tuple[Dict[str, Any], bool]:
    question = get_text(row, ["question", "query", "prompt", "input"])
    answer = get_text(row, ["answer", "output", "response", "target", "label"])
    input_ts = get_field(row, ["input_ts", "timeseries", "time_series", "series"])
    if not question:
        raise ConversionError("missing TSAQA question")
    if input_ts is None:
        raise ConversionError("missing TSAQA input_ts")

    primary = materialize_channels(raw_channels(input_ts, orientation="rows"), options)
    converted_question, inline_arrays, inline_mask = extract_inline_timeseries(question, options, orientation="auto")

    sections: List[str] = []
    context = _clean_context(get_field(row, ["meta_info", "context", "context_text", "description"]))
    if context:
        sections.append("Context information: " + context)
    sections.append(_series_block(primary))
    sections.append(converted_question.strip())

    sample = {
        "input": "\n\n".join(sections),
        "timeseries": primary.arrays + inline_arrays,
        "output": answer,
    }
    _validate_sample(sample, options)
    return sample, primary.used_missing_mask or inline_mask


_SOURCE_PLACEHOLDER_RE = re.compile(
    r"\[(?:input\s+)?time\s+series(?:\s+data)?\s+points?(?:\s+with\s+missing\s+values)?\]",
    flags=re.IGNORECASE,
)


def _explicit_series(row: Mapping[str, Any]) -> Any:
    return get_field(
        row,
        [
            "timeseries",
            "time_series",
            "input_ts",
            "series",
            "ts_data",
            "data_points",
            "time_series_data",
        ],
    )


def _question_answer_from_combined_text(row: Mapping[str, Any]) -> Tuple[str, str]:
    combined = get_text(row, ["text", "qa", "question_answer", "formatted_text"])
    if not combined:
        return "", ""
    tagged = re.search(
        r"<QUE>\s*(.*?)\s*<ANS>\s*(.*?)\s*(?:</END>|$)",
        combined,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tagged:
        return tagged.group(1).strip(), tagged.group(2).strip()
    labeled = re.search(
        r"(?:^|\n)\s*Question\s*:\s*(.*?)\s*(?:\n\s*)Answer\s*:\s*(.*)$",
        combined,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if labeled:
        return labeled.group(1).strip(), labeled.group(2).strip()
    return "", ""


def _question_answer_from_qa_list(row: Mapping[str, Any]) -> Tuple[str, str]:
    """Parse the fragment stored in the official Time-MQA ``QA_list`` column."""
    value = get_field(row, ["qa_list", "qa_pair", "qa"])
    if value is None:
        return "", ""
    parsed: Any = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "", ""
        candidates = [stripped]
        if not stripped.startswith("{"):
            candidates.insert(0, "{" + stripped + "}")
        parsed = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                break
            except (json.JSONDecodeError, TypeError):
                try:
                    parsed = ast.literal_eval(candidate)
                    break
                except (ValueError, SyntaxError):
                    continue
        # Some finance rows contain long earnings-call transcripts with raw,
        # unescaped quotation marks inside the question.  They are not valid
        # JSON/Python literals, but the official fragment still has stable
        # question/answer delimiters, so split those without evaluating text.
        if parsed is None:
            prefix = '"question": "'
            delimiter = '", "answer": "'
            split_at = stripped.rfind(delimiter)
            if stripped.startswith(prefix) and split_at >= len(prefix):
                question = stripped[len(prefix) : split_at]
                answer = stripped[split_at + len(delimiter) :]
                if answer.endswith('"'):
                    answer = answer[:-1]
                return question.strip(), answer.strip()
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return "", ""
    question = parsed.get("question", parsed.get("Question", ""))
    answer = parsed.get("answer", parsed.get("Answer", ""))
    return str(question).strip(), str(answer).strip()


def convert_time_mqa(row: Mapping[str, Any], options: ConvertOptions) -> Tuple[Dict[str, Any], bool]:
    question = get_text(row, ["question", "query", "prompt", "input", "instruction"])
    answer = get_text(row, ["answer", "output", "response", "target", "label"])
    if not question or not answer:
        qa_list_question, qa_list_answer = _question_answer_from_qa_list(row)
        question = question or qa_list_question
        answer = answer or qa_list_answer
    if not question or not answer:
        combined_question, combined_answer = _question_answer_from_combined_text(row)
        question = question or combined_question
        answer = answer or combined_answer
    if not question:
        raise ConversionError("missing Time-MQA question")

    converted_question, inline_arrays, inline_mask = extract_inline_timeseries(question, options, orientation="auto")
    arrays = inline_arrays
    used_mask = inline_mask

    # Current Time-MQA CSVs embed the numeric values in the question.  The
    # explicit-field branch keeps the converter compatible with repackaged
    # versions without guessing a generic `data` column.
    if not arrays:
        explicit = _explicit_series(row)
        if explicit is None:
            raise ConversionError("no numeric array found in Time-MQA question")
        materialized = materialize_channels(raw_channels(explicit, orientation="auto"), options)
        arrays = materialized.arrays
        used_mask = materialized.used_missing_mask
        if _SOURCE_PLACEHOLDER_RE.search(converted_question):
            converted_question = _SOURCE_PLACEHOLDER_RE.sub(materialized.marker, converted_question, count=1)
        elif converted_question.count(PLACEHOLDER) == len(arrays):
            pass
        else:
            converted_question = _series_block(materialized) + "\n\n" + converted_question

    context = _clean_context(get_field(row, ["context", "context_text", "meta_info", "description"]))
    if context and context not in converted_question:
        converted_question = "Context information: " + context + "\n\n" + converted_question

    sample = {"input": converted_question.strip(), "timeseries": arrays, "output": answer}
    _validate_sample(sample, options)
    return sample, used_mask


def _raise_csv_limit() -> None:
    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def iter_csv(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    _raise_csv_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {path}")
        for row_number, row in enumerate(reader, start=2):
            yield row_number, dict(row)


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"JSONL row {row_number} is not an object: {path}")
            yield row_number, value


def iter_json(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if isinstance(value, dict):
        for candidate in ("train", "data", "rows", "examples"):
            if isinstance(value.get(candidate), list):
                value = value[candidate]
                break
        else:
            value = [value]
    if not isinstance(value, list):
        raise RuntimeError(f"JSON must contain an object or top-level list: {path}")
    for row_number, row in enumerate(value, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"JSON item {row_number} is not an object: {path}")
        yield row_number, row


def iter_parquet(path: Path, batch_size: int = 1024) -> Iterator[Tuple[int, Dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading Parquet requires pyarrow. Install it with: pip install 'pyarrow>=14,<22'") from exc
    parquet_file = pq.ParquetFile(path)
    row_number = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            row_number += 1
            yield row_number, row


def iter_records(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from iter_csv(path)
    elif suffix == ".jsonl":
        yield from iter_jsonl(path)
    elif suffix == ".json":
        yield from iter_json(path)
    elif suffix == ".parquet":
        yield from iter_parquet(path)
    else:
        raise RuntimeError(f"unsupported input format: {path}")


def discover_files(inputs: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"input does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                found.append(path)
        else:
            found.extend(
                child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
            )
    unique = sorted(set(found))
    if not unique:
        raise FileNotFoundError("no CSV, JSON, JSONL, or Parquet inputs found")
    return unique


def _is_nontrain_file(path: Path) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
    return bool(tokens & {"test", "val", "valid", "validation", "dev"})


def _is_time_mqa_classification_file(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return "classification" in lowered_parts or path.stem.lower() == "classification"


def _contains_sunspot(row: Mapping[str, Any]) -> bool:
    value = get_field(row, ["dataset", "dataset_name", "source_dataset"], "")
    return "sunspot" in str(value).lower()


def _sample_hash(sample: Mapping[str, Any]) -> str:
    payload = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_text_paths(paths: Sequence[Path]) -> Tuple[List[Path], List[Any]]:
    temp_paths: List[Path] = []
    streams: List[Any] = []
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        temp_path = Path(name)
        temp_paths.append(temp_path)
        streams.append(os.fdopen(descriptor, "w", encoding="utf-8"))
    return temp_paths, streams


def convert_command(args: argparse.Namespace) -> int:
    options = ConvertOptions(
        min_inline_length=args.min_inline_length,
        missing_policy=args.missing_policy,
        max_series=args.max_series,
        max_series_length=args.max_series_length,
        long_series_policy=args.long_series_policy,
    )
    files = discover_files(args.input)
    output_path = Path(args.output).expanduser().resolve()
    audit_path = (
        Path(args.audit_output).expanduser().resolve()
        if args.audit_output
        else output_path.with_name(output_path.stem + ".audit.jsonl")
    )
    manifest_path = output_path.with_name(output_path.stem + ".manifest.json")
    if len({output_path, audit_path, manifest_path}) != 3:
        raise ValueError("output, audit, and manifest paths must be different")

    stats = Stats()
    seen_hashes = set()
    examples: List[Dict[str, Any]] = []
    temp_paths, streams = _atomic_text_paths([output_path, audit_path])
    output_stream, audit_stream = streams
    converter = convert_tsaqa if args.dataset == "tsaqa" else convert_time_mqa

    try:
        for path in files:
            stats.files_seen += 1
            if not args.include_nontrain and _is_nontrain_file(path):
                stats.files_skipped_nontrain += 1
                continue
            if args.dataset == "time-mqa" and not args.include_contaminated and _is_time_mqa_classification_file(path):
                stats.files_skipped_contamination += 1
                continue

            for row_number, row in iter_records(path):
                stats.rows_seen += 1
                if args.dataset == "tsaqa" and not args.include_contaminated and _contains_sunspot(row):
                    stats.rows_filtered_contamination += 1
                    continue
                try:
                    sample, used_missing_mask = converter(row, options)
                except ConversionError as exc:
                    stats.rows_invalid += 1
                    stats.invalid_reasons[str(exc)] += 1
                    if args.fail_fast:
                        raise RuntimeError(f"{path}:{row_number}: {exc}") from exc
                    continue

                digest = _sample_hash(sample)
                if not args.keep_duplicates and digest in seen_hashes:
                    stats.rows_duplicate += 1
                    continue
                seen_hashes.add(digest)

                task = get_text(row, ["task", "task_type"], path.parent.name or path.stem)
                question_type = get_text(
                    row, ["question_type", "question_format", "type", "qa_type"], "unknown"
                )
                audit = {
                    "sample_index": stats.rows_written,
                    "sample_sha256": digest,
                    "source_dataset": args.dataset,
                    "source_file": str(path),
                    "source_row": row_number,
                    "task": task,
                    "question_type": question_type,
                    "dataset_name": get_text(row, ["dataset", "dataset_name", "source_dataset"]),
                    "domain": get_text(row, ["domain", "application_domain"]),
                    "series_count": len(sample["timeseries"]),
                    "series_lengths": [len(values) for values in sample["timeseries"]],
                    "used_missing_mask": used_missing_mask,
                }
                output_stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                audit_stream.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n")

                stats.rows_written += 1
                stats.rows_missing_mask += int(used_missing_mask)
                stats.max_series_per_row = max(stats.max_series_per_row, len(sample["timeseries"]))
                stats.max_series_length = max(
                    stats.max_series_length, max(len(values) for values in sample["timeseries"])
                )
                stats.task_counts[task] += 1
                stats.question_type_counts[question_type] += 1
                if len(examples) < args.preview:
                    examples.append(sample)
                if args.limit and stats.rows_written >= args.limit:
                    break
            if args.limit and stats.rows_written >= args.limit:
                break

        output_stream.flush()
        audit_stream.flush()
        output_stream.close()
        audit_stream.close()
        for temp_path, final_path in zip(temp_paths, [output_path, audit_path]):
            os.replace(temp_path, final_path)
    except Exception:
        for stream in streams:
            if not stream.closed:
                stream.close()
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
        raise

    manifest = {
        "format": "ChatTS Training Dataset SFT JSONL",
        "schema": {"input": "string", "timeseries": "list[list[float]]", "output": "string"},
        "dataset_adapter": args.dataset,
        "inputs": [str(path) for path in files],
        "output": str(output_path),
        "audit_output": str(audit_path),
        "contamination_filter": not args.include_contaminated,
        "nontrain_filter": not args.include_nontrain,
        "options": asdict(options),
        "stats": stats.to_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nChatTS JSONL: {output_path}")
    print(f"Audit JSONL:  {audit_path}")
    print(f"Manifest:     {manifest_path}")
    for index, sample in enumerate(examples, start=1):
        compact = dict(sample)
        compact["timeseries"] = [f"<series length={len(values)} first={values[:3]}>" for values in sample["timeseries"]]
        print(f"\nPreview {index}:\n{json.dumps(compact, ensure_ascii=False, indent=2)}")
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    files = discover_files(args.input)
    remaining = args.rows
    for path in files:
        print(f"\n=== {path} ===")
        for row_number, row in iter_records(path):
            print(f"row {row_number}; fields={list(row.keys())}")
            for key, value in row.items():
                print(f"  {key}: {_preview(value, args.value_chars)}")
            remaining -= 1
            if remaining <= 0:
                return 0
            print()
    return 0


def download_command(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: pip install huggingface_hub") from exc

    if args.dataset == "tsaqa":
        repo_id = "TSAQA/TSAQA-Benchmark"
        patterns = ["train.parquet", "README.md", "LICENSE*"]
    else:
        repo_id = "Time-MQA/TSQA"
        patterns = [
            "Anomaly_Detection/*.csv",
            "Forecasting+Imputation/**/*.csv",
            "Open_Ended_QA/*.csv",
            "README.md",
            "LICENSE*",
        ]
        if args.include_contaminated:
            patterns.append("Classification/*.csv")

    try:
        result = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(Path(args.output_dir).expanduser().resolve()),
            allow_patterns=patterns,
            token=args.token,
        )
    except Exception as exc:
        if args.dataset == "time-mqa":
            raise RuntimeError(
                "Time-MQA is gated. Accept its Hugging Face terms, then run `hf auth login` "
                "or pass --token/define HF_TOKEN before retrying."
            ) from exc
        raise
    print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download/inspect/convert TSAQA and Time-MQA into ChatTS SFT format.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download train files from Hugging Face")
    download.add_argument("--dataset", choices=["tsaqa", "time-mqa"], required=True)
    download.add_argument("--output-dir", required=True)
    download.add_argument("--revision", default="main", help="pin a commit SHA for reproducibility")
    download.add_argument("--token", default=None, help="HF token; prefer HF_TOKEN or `hf auth login`")
    download.add_argument(
        "--include-contaminated",
        action="store_true",
        help="also download Time-MQA Classification (not recommended for TSRBench training)",
    )
    download.set_defaults(func=download_command)

    inspect = subparsers.add_parser("inspect", help="print source fields and short value previews")
    inspect.add_argument("--input", nargs="+", required=True)
    inspect.add_argument("--rows", type=int, default=3)
    inspect.add_argument("--value-chars", type=int, default=180)
    inspect.set_defaults(func=inspect_command)

    convert = subparsers.add_parser("convert", help="write ChatTS JSONL plus audit/manifest files")
    convert.add_argument("--dataset", choices=["tsaqa", "time-mqa"], required=True)
    convert.add_argument("--input", nargs="+", required=True, help="one or more files/directories")
    convert.add_argument("--output", required=True, help="output ChatTS JSONL")
    convert.add_argument("--audit-output", default=None)
    convert.add_argument("--min-inline-length", type=int, default=4)
    convert.add_argument("--missing-policy", choices=["mask-zero", "drop"], default="mask-zero")
    convert.add_argument("--max-series", type=int, default=30)
    convert.add_argument("--max-series-length", type=int, default=0, help="0 means no limit")
    convert.add_argument("--long-series-policy", choices=["drop", "truncate"], default="drop")
    convert.add_argument("--limit", type=int, default=0, help="write at most N samples; 0 means all")
    convert.add_argument("--preview", type=int, default=2)
    convert.add_argument("--fail-fast", action="store_true")
    convert.add_argument("--keep-duplicates", action="store_true")
    convert.add_argument(
        "--include-nontrain",
        action="store_true",
        help="also read val/dev/test files (unsafe when targeting a benchmark)",
    )
    convert.add_argument(
        "--include-contaminated",
        action="store_true",
        help="disable built-in TSRBench overlap filters",
    )
    convert.set_defaults(func=convert_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "min_inline_length", 1) < 1:
        parser.error("--min-inline-length must be positive")
    if getattr(args, "max_series", 1) < 1:
        parser.error("--max-series must be positive")
    if getattr(args, "max_series_length", 0) < 0:
        parser.error("--max-series-length cannot be negative")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fingerprint, cache, and aggregate ChatTS benchmark artifacts.

This module intentionally uses only the Python standard library so that the
shell evaluation entrypoint can use it in an offline inference environment.
It does not import ChatTS or any benchmark implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

SCHEMA_VERSION = 3
MANIFEST_BASENAME = ".chatts_benchmark_manifest.json"
FILE_DIGEST_CACHE_SCHEMA_VERSION = 1
FILE_DIGEST_CACHE_ENV = "CHATTS_FINGERPRINT_CACHE"
DEFAULT_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".git", ".pytest_cache", "__pycache__"}
)
OUTPUT_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".git", ".pytest_cache", "__pycache__", ".logs", "log", "logs", "temp", "tmp"}
)
OUTPUT_EXCLUDED_FILE_NAMES = frozenset({MANIFEST_BASENAME})
OUTPUT_EXCLUDED_FILE_SUFFIXES = frozenset(
    {".lock", ".log", ".partial", ".swp", ".temp", ".tmp"}
)
SUCCESS_STATUSES = {"PASS", "CACHED"}
SUPPORTED_SUITES = {
    "tsrbench",
    "timeseriesexam",
    "ts_haystack",
    "tinybenchmarks",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return metadata that changes after an ordinary in-place file rewrite.

    ``mtime`` alone is insufficient because callers can restore it with
    ``os.utime``.  ``ctime`` cannot be restored through that API, so it makes
    the digest cache reject the stale entry and recompute the content hash.
    The final benchmark fingerprint still contains the actual SHA256 rather
    than trusting this metadata.
    """

    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _default_digest_cache_path() -> Path | None:
    configured = os.environ.get(FILE_DIGEST_CACHE_ENV)
    if configured is not None:
        configured = configured.strip()
        if configured.lower() in {"", "0", "none", "off", "disabled"}:
            return None
        return Path(configured).expanduser()

    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        base = Path(cache_root).expanduser()
    else:
        try:
            base = Path.home() / ".cache"
        except RuntimeError:
            return None
    return base / "chatts" / "benchmark_file_digests.sqlite3"


class _FileDigestCache:
    """Best-effort persistent SHA256 cache for immutable benchmark inputs.

    A cache failure never weakens fingerprinting: the caller simply hashes the
    file again.  Entries are reused only when path, device, inode, size, mtime,
    and ctime all match.  This lets the four sequential suites share expensive
    model/data reads without reverting to size/mtime-only integrity checks.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._connection: sqlite3.Connection | None = None
        self._memory: dict[tuple[str, tuple[int, int, int, int, int]], str] = {}
        cache_path = _default_digest_cache_path() if path is None else path
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(cache_path), timeout=30)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS file_digests (
                    resolved_path TEXT PRIMARY KEY,
                    device INTEGER NOT NULL,
                    inode INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    ctime_ns INTEGER NOT NULL,
                    sha256 TEXT NOT NULL
                )
                """
            )
            connection.execute(f"PRAGMA user_version={FILE_DIGEST_CACHE_SCHEMA_VERSION}")
            self._connection = connection
        except (OSError, sqlite3.Error):
            if "connection" in locals():
                connection.close()

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.commit()
            except sqlite3.Error:
                pass
            finally:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _disable_persistent_cache(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except sqlite3.Error:
                pass
            self._connection = None

    def _lookup(
        self,
        resolved_path: str,
        identity: tuple[int, int, int, int, int],
    ) -> str | None:
        memory_key = (resolved_path, identity)
        if memory_key in self._memory:
            return self._memory[memory_key]
        if self._connection is None:
            return None
        try:
            row = self._connection.execute(
                """
                SELECT sha256 FROM file_digests
                WHERE resolved_path = ? AND device = ? AND inode = ?
                  AND size = ? AND mtime_ns = ? AND ctime_ns = ?
                """,
                (resolved_path, *identity),
            ).fetchone()
        except sqlite3.Error:
            self._disable_persistent_cache()
            return None
        if row is None:
            return None
        digest = str(row[0])
        self._memory[memory_key] = digest
        return digest

    def _store(
        self,
        resolved_path: str,
        identity: tuple[int, int, int, int, int],
        digest: str,
    ) -> None:
        self._memory[(resolved_path, identity)] = digest
        if self._connection is None:
            return
        try:
            self._connection.execute(
                """
                INSERT INTO file_digests (
                    resolved_path, device, inode, size, mtime_ns, ctime_ns, sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(resolved_path) DO UPDATE SET
                    device = excluded.device,
                    inode = excluded.inode,
                    size = excluded.size,
                    mtime_ns = excluded.mtime_ns,
                    ctime_ns = excluded.ctime_ns,
                    sha256 = excluded.sha256
                """,
                (resolved_path, *identity, digest),
            )
        except sqlite3.Error:
            self._disable_persistent_cache()

    def sha256(self, path: Path) -> str:
        resolved = str(path.resolve(strict=True))
        before = path.stat()
        identity = _stat_identity(before)
        cached = self._lookup(resolved, identity)
        if cached is not None:
            return cached

        # Detect a concurrent replacement/rewrite while the file is being read.
        # Two retries keep a transient writer from producing an unstable digest.
        for _ in range(3):
            digest = _sha256_file(path)
            after = path.stat()
            after_identity = _stat_identity(after)
            if after_identity == identity:
                self._store(resolved, identity, digest)
                return digest
            identity = after_identity
            cached = self._lookup(resolved, identity)
            if cached is not None:
                return cached
        raise RuntimeError(f"File changed repeatedly while hashing: {path}")


def _file_record(
    path: Path,
    logical_name: str,
    *,
    digest_cache: _FileDigestCache,
) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "name": logical_name,
        "type": "file",
        "size": stat.st_size,
        "content_sha256": digest_cache.sha256(path),
    }
    if path.is_symlink():
        record["symlink_target"] = os.readlink(path)
    return record


def _path_inventory(
    path: Path,
    label: str,
    *,
    digest_cache: _FileDigestCache,
    excluded_directory_names: frozenset[str] = DEFAULT_EXCLUDED_DIRECTORY_NAMES,
    excluded_file_names: frozenset[str] = frozenset(),
    excluded_file_suffixes: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Fingerprint input does not exist: {path}")

    if path.is_file():
        return [_file_record(path, label, digest_cache=digest_cache)]

    if not path.is_dir():
        raise ValueError(f"Fingerprint input is neither a file nor directory: {path}")

    records: list[dict[str, Any]] = []
    root = path.resolve()

    def logical_name(relative_parts: tuple[str, ...]) -> str:
        return label if not relative_parts else f"{label}/{'/'.join(relative_parts)}"

    def walk_directory(
        current: Path,
        relative_parts: tuple[str, ...],
        ancestor_directory_ids: frozenset[tuple[int, int]],
    ) -> None:
        directory_stat = current.stat()
        directory_id = (directory_stat.st_dev, directory_stat.st_ino)
        if directory_id in ancestor_directory_ids:
            raise ValueError(
                "Directory symlink cycle while fingerprinting "
                f"{logical_name(relative_parts)}: {current}"
            )
        child_ancestors = ancestor_directory_ids | {directory_id}
        try:
            children = sorted(current.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise OSError(f"Cannot enumerate fingerprint directory: {current}") from exc

        included_children = 0
        for candidate in children:
            name = candidate.name
            child_parts = (*relative_parts, name)
            child_logical_name = logical_name(child_parts)
            if candidate.is_symlink():
                if candidate.is_dir():
                    if name in excluded_directory_names:
                        continue
                    records.append(
                        {
                            "name": child_logical_name,
                            "type": "symlink_dir",
                            "target": os.readlink(candidate),
                        }
                    )
                    included_children += 1
                    walk_directory(candidate, child_parts, child_ancestors)
                    continue
                if not candidate.is_file():
                    raise ValueError(
                        f"Broken or unsupported symlink in fingerprint input: {candidate}"
                    )

            if candidate.is_dir():
                if name in excluded_directory_names:
                    continue
                included_children += 1
                walk_directory(candidate, child_parts, child_ancestors)
                continue

            if candidate.is_file():
                if (
                    name in excluded_file_names
                    or candidate.suffix in excluded_file_suffixes
                ):
                    continue
                included_children += 1
                records.append(
                    _file_record(
                        candidate,
                        child_logical_name,
                        digest_cache=digest_cache,
                    )
                )
                continue

            raise ValueError(f"Unsupported filesystem entry in fingerprint input: {candidate}")

        if included_children == 0:
            records.append(
                {"name": logical_name(relative_parts), "type": "empty_directory"}
            )

    walk_directory(root, (), frozenset())

    if not records:
        records.append({"name": label, "type": "empty_directory"})
    return records


def fingerprint_paths(
    paths: Sequence[str | os.PathLike[str]],
    *,
    hash_all: bool = False,
    _digest_cache: _FileDigestCache | None = None,
) -> str:
    """Return a stable SHA256 over path inventories.

    Every regular file is represented by its content SHA256, including large
    model and benchmark files.  ``hash_all`` remains as a backward-compatible
    keyword but no longer changes the secure behavior.  A persistent digest
    cache avoids rereading unchanged files for every suite/helper invocation.
    """

    del hash_all

    def collect(digest_cache: _FileDigestCache) -> str:
        records: list[dict[str, Any]] = []
        for index, raw_path in enumerate(paths):
            path = Path(raw_path).expanduser()
            records.extend(
                _path_inventory(path, f"input_{index}", digest_cache=digest_cache)
            )
        return _sha256_json({"inventory_version": 2, "records": records})

    if _digest_cache is not None:
        return collect(_digest_cache)
    with _FileDigestCache() as digest_cache:
        return collect(digest_cache)


def fingerprint_output_artifacts(
    output_dir: str | os.PathLike[str],
    *,
    _digest_cache: _FileDigestCache | None = None,
) -> tuple[str, int]:
    """Fingerprint stable suite artifacts while ignoring operational noise.

    The manifest itself, logs, lock files, editor swaps, and common temporary
    files are deliberately excluded.  Prediction JSON/JSONL, evaluator
    intermediates, and the suite summary remain content-hashed.
    """

    path = Path(output_dir).expanduser()

    def collect(digest_cache: _FileDigestCache) -> tuple[str, int]:
        records = _path_inventory(
            path,
            "output",
            digest_cache=digest_cache,
            excluded_directory_names=OUTPUT_EXCLUDED_DIRECTORY_NAMES,
            excluded_file_names=OUTPUT_EXCLUDED_FILE_NAMES,
            excluded_file_suffixes=OUTPUT_EXCLUDED_FILE_SUFFIXES,
        )
        fingerprint = _sha256_json(
            {"output_inventory_version": 1, "records": records}
        )
        artifact_count = sum(record.get("type") == "file" for record in records)
        return fingerprint, artifact_count

    if _digest_cache is not None:
        return collect(_digest_cache)
    with _FileDigestCache() as digest_cache:
        return collect(digest_cache)


def _resolved_paths(paths: Sequence[str]) -> list[str]:
    return [str(Path(path).expanduser().resolve()) for path in paths]


def build_request(
    *,
    suite: str,
    model_path: str,
    model_name: str,
    model_components: Sequence[str] = (),
    data_paths: Sequence[str],
    protocol_files: Sequence[str],
    protocol_items: Sequence[str],
    eval_protocol_hash: str,
) -> dict[str, Any]:
    if suite not in SUPPORTED_SUITES:
        raise ValueError(f"Unsupported suite: {suite}")
    if not data_paths:
        raise ValueError(f"At least one data path is required for {suite}")
    if not protocol_files:
        raise ValueError(f"At least one protocol file is required for {suite}")

    resolved_model = str(Path(model_path).expanduser().resolve())
    resolved_model_components = _resolved_paths(model_components)
    resolved_data = _resolved_paths(data_paths)
    resolved_protocol_files = _resolved_paths(protocol_files)
    command_descriptor = sorted(protocol_items)
    command_fingerprint = _sha256_json(
        {
            "suite": suite,
            "model_name": model_name,
            "items": command_descriptor,
        }
    )
    with _FileDigestCache() as digest_cache:
        protocol_code_fingerprint = fingerprint_paths(
            resolved_protocol_files,
            hash_all=True,
            _digest_cache=digest_cache,
        )
        model_fingerprint = fingerprint_paths(
            [resolved_model, *resolved_model_components],
            _digest_cache=digest_cache,
        )
        data_fingerprint = fingerprint_paths(
            resolved_data,
            _digest_cache=digest_cache,
        )
    protocol_fingerprint = _sha256_json(
        {
            "external_eval_protocol_hash": eval_protocol_hash or None,
            "protocol_code_fingerprint": protocol_code_fingerprint,
            "command_fingerprint": command_fingerprint,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "model_name": model_name,
        "model_path": resolved_model,
        "model_components": resolved_model_components,
        "model_fingerprint": model_fingerprint,
        "data_paths": resolved_data,
        "data_fingerprint": data_fingerprint,
        "protocol_files": resolved_protocol_files,
        "protocol_items": command_descriptor,
        "protocol_code_fingerprint": protocol_code_fingerprint,
        "command_fingerprint": command_fingerprint,
        "external_eval_protocol_hash": eval_protocol_hash or None,
        "protocol_fingerprint": protocol_fingerprint,
    }


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def cache_matches(manifest: Mapping[str, Any], request: Mapping[str, Any]) -> tuple[bool, str]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return False, "manifest schema changed"
    if manifest.get("status") != "pass":
        return False, "manifest status is not pass"
    for key in (
        "suite",
        "model_fingerprint",
        "data_fingerprint",
        "protocol_fingerprint",
        "command_fingerprint",
    ):
        if manifest.get(key) != request.get(key):
            return False, f"{key} changed"
    summary_file = manifest.get("summary_file")
    summary_path = Path(str(summary_file)) if summary_file else None
    if summary_path is None or not summary_path.is_file():
        return False, "source summary is missing"
    expected_summary_hash = manifest.get("summary_sha256")
    output_dir = manifest.get("output_dir")
    output_path = Path(str(output_dir)) if output_dir else None
    if output_path is None or not output_path.is_dir():
        return False, "suite output directory is missing"
    expected_output_fingerprint = manifest.get("output_artifacts_fingerprint")
    if not expected_output_fingerprint:
        return False, "output artifacts fingerprint is missing"
    with _FileDigestCache() as digest_cache:
        if (
            not expected_summary_hash
            or digest_cache.sha256(summary_path) != expected_summary_hash
        ):
            return False, "source summary changed"
        output_fingerprint, _ = fingerprint_output_artifacts(
            output_path,
            _digest_cache=digest_cache,
        )
    if output_fingerprint != expected_output_fingerprint:
        return False, "output artifacts changed"
    return True, "all fingerprints match"


def write_suite_manifest(
    *,
    request: Mapping[str, Any],
    output_dir: str,
    summary_file: str,
    run_id: str,
) -> Path:
    summary_path = Path(summary_file).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Suite completed without its JSON summary: {summary_path}")
    output_path = Path(output_dir).expanduser().resolve()
    if not output_path.is_dir():
        raise FileNotFoundError(f"Suite output directory does not exist: {output_path}")
    with _FileDigestCache() as digest_cache:
        summary_sha256 = digest_cache.sha256(summary_path)
        output_fingerprint, output_artifact_count = fingerprint_output_artifacts(
            output_path,
            _digest_cache=digest_cache,
        )
    manifest_path = output_path / MANIFEST_BASENAME
    payload = dict(request)
    payload.update(
        {
            "status": "pass",
            "run_id": run_id,
            "output_dir": str(output_path),
            "summary_file": str(summary_path),
            "summary_sha256": summary_sha256,
            "output_artifacts_fingerprint": output_fingerprint,
            "output_artifact_count": output_artifact_count,
            "completed_at_utc": _utc_now(),
        }
    )
    _atomic_write_json(manifest_path, payload)
    return manifest_path


def normalized_suite_metrics(suite: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    if suite == "tsrbench":
        overall = dict(summary.get("overall") or {})
        return {
            "dataset_size": overall.get("dataset_size"),
            "generated": overall.get("generated"),
            "parsed": overall.get("parsed"),
            "correct": overall.get("correct"),
            "coverage": overall.get("coverage"),
            "parse_rate": overall.get("parse_rate"),
            "strict_accuracy": overall.get("accuracy_strict"),
            "parsed_accuracy": overall.get("accuracy_parsed"),
        }
    if suite == "timeseriesexam":
        overall = dict(summary.get("overall") or {})
        return {
            "total": overall.get("total"),
            "generated": overall.get("generated"),
            "parsed": overall.get("parsed"),
            "coverage": overall.get("coverage"),
            "parse_rate": overall.get("parse_rate"),
            "flexible_accuracy": overall.get("official_flexible_accuracy"),
            "strict_accuracy": overall.get("official_strict_accuracy"),
            "letter_accuracy": overall.get("letter_accuracy"),
            "letter_accuracy_parsed": overall.get("letter_accuracy_parsed"),
        }
    if suite == "ts_haystack":
        overall = dict(summary.get("overall") or {})
        return {
            "total": overall.get("total"),
            "generated": overall.get("generated"),
            "parsed": overall.get("parsed"),
            "correct": overall.get("correct"),
            "coverage": overall.get("coverage"),
            "parse_rate": overall.get("parse_rate"),
            "strict_accuracy": overall.get("accuracy_strict"),
            "generated_accuracy": overall.get("accuracy_generated"),
            "mean_iou": overall.get("mean_iou"),
            "mean_timestamp_error_s": overall.get("mean_timestamp_error_s"),
        }
    if suite == "tinybenchmarks":
        tasks = summary.get("tasks") or {}
        if not isinstance(tasks, Mapping):
            tasks = {}
        task_scores: dict[str, float | None] = {}
        valid_scores: list[float] = []
        for name, values in tasks.items():
            raw_score = values.get("score") if isinstance(values, Mapping) else None
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = None
            if score is not None and not math.isfinite(score):
                score = None
            task_scores[str(name)] = score
            if score is not None:
                valid_scores.append(score)

        raw_macro = summary.get("macro_score")
        try:
            macro_score = float(raw_macro)
        except (TypeError, ValueError):
            macro_score = None
        if macro_score is not None and not math.isfinite(macro_score):
            macro_score = None
        if macro_score is None and valid_scores:
            macro_score = math.fsum(valid_scores) / len(valid_scores)

        raw_num_tasks = summary.get("num_tasks")
        try:
            num_tasks = int(raw_num_tasks)
        except (TypeError, ValueError):
            num_tasks = len(valid_scores)
        return {
            "macro_score": macro_score,
            "num_tasks": num_tasks,
            "task_scores": task_scores,
        }
    raise ValueError(f"Unsupported suite: {suite}")


def _parse_name_path(values: Iterable[str], label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid {label} value {value!r}; expected NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or name in parsed:
            raise ValueError(f"Invalid or duplicate {label} name: {name!r}")
        parsed[name] = Path(raw_path).expanduser().resolve()
    return parsed


def aggregate_run(
    *,
    status_file: str,
    suite_manifests: Mapping[str, Path],
    metrics_file: str,
    run_manifest_file: str,
    run_id: str,
    model_path: str,
    model_name: str,
    seed: int,
    max_samples: int,
    force_eval: bool,
    output_root: str,
    eval_protocol_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with Path(status_file).open(encoding="utf-8", newline="") as stream:
        statuses = list(csv.DictReader(stream, delimiter="\t"))

    suites: dict[str, Any] = {}
    suite_manifest_payloads: dict[str, Any] = {}
    protocol_fingerprints: dict[str, str] = {}
    data_fingerprints: dict[str, str] = {}
    model_fingerprints: set[str] = set()

    for status in statuses:
        suite = status["suite"]
        entry: dict[str, Any] = {
            "status": status["status"].lower(),
            "exit_code": int(status["exit_code"]),
            "output_dir": status["output_dir"],
            "log_file": status["log_file"],
        }
        manifest_path = suite_manifests.get(suite)
        if manifest_path and manifest_path.is_file() and status["status"] in SUCCESS_STATUSES:
            manifest = _read_json(manifest_path)
            suite_manifest_payloads[suite] = {
                "path": str(manifest_path),
                "model_fingerprint": manifest.get("model_fingerprint"),
                "data_fingerprint": manifest.get("data_fingerprint"),
                "protocol_fingerprint": manifest.get("protocol_fingerprint"),
                "command_fingerprint": manifest.get("command_fingerprint"),
                "summary_file": manifest.get("summary_file"),
                "summary_sha256": manifest.get("summary_sha256"),
                "output_artifacts_fingerprint": manifest.get(
                    "output_artifacts_fingerprint"
                ),
                "output_artifact_count": manifest.get("output_artifact_count"),
            }
            if manifest.get("model_fingerprint"):
                model_fingerprints.add(str(manifest["model_fingerprint"]))
            if manifest.get("data_fingerprint"):
                data_fingerprints[suite] = str(manifest["data_fingerprint"])
            if manifest.get("protocol_fingerprint"):
                protocol_fingerprints[suite] = str(manifest["protocol_fingerprint"])
            summary_path = Path(str(manifest.get("summary_file", "")))
            if summary_path.is_file():
                summary = _read_json(summary_path)
                entry["source_summary_file"] = str(summary_path)
                entry["metrics"] = normalized_suite_metrics(suite, summary)
                entry["summary"] = summary
            elif status["status"] in SUCCESS_STATUSES:
                entry["status"] = "fail"
                entry["error"] = "source summary is missing"
        elif status["status"] in SUCCESS_STATUSES:
            entry["status"] = "fail"
            entry["error"] = "suite manifest is missing"
        elif manifest_path and manifest_path.is_file():
            entry["previous_manifest_ignored"] = str(manifest_path)
        suites[suite] = entry

    effective_protocol_hash = eval_protocol_hash or _sha256_json(protocol_fingerprints)
    integrity_errors: list[str] = []
    if len(model_fingerprints) > 1:
        integrity_errors.append("model fingerprint changed between benchmark suites")
    run_status = "pass" if (
        suites
        and not integrity_errors
        and all(entry["status"] in {"pass", "cached"} for entry in suites.values())
    ) else "fail"
    completed_at = _utc_now()
    common = {
        "schema_version": SCHEMA_VERSION,
        "status": run_status,
        "run_id": run_id,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "model_name": model_name,
        "model_fingerprint": next(iter(model_fingerprints)) if len(model_fingerprints) == 1 else None,
        "ts_encoder_type": "chronos2",
        "seed": seed,
        "max_samples": max_samples,
        "force_eval": force_eval,
        "eval_protocol_hash": effective_protocol_hash,
        "external_eval_protocol_hash": eval_protocol_hash or None,
        "protocol_fingerprints": protocol_fingerprints,
        "data_fingerprints": data_fingerprints,
        "selected_suites": [status["suite"] for status in statuses],
        "integrity_errors": integrity_errors,
        "output_root": str(Path(output_root).expanduser().resolve()),
        "completed_at_utc": completed_at,
    }
    metrics_payload = dict(common)
    metrics_payload["suites"] = suites
    manifest_payload = dict(common)
    manifest_payload.update(
        {
            "scheduling": "sequential",
            "exclusive_gpus_per_suite": 8,
            "suite_statuses": statuses,
            "suite_manifests": suite_manifest_payloads,
            "metrics_file": str(Path(metrics_file).expanduser().resolve()),
        }
    )
    _atomic_write_json(Path(metrics_file), metrics_payload)
    _atomic_write_json(Path(run_manifest_file), manifest_payload)
    return metrics_payload, manifest_payload


def _request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return build_request(
        suite=args.suite,
        model_path=args.model_path,
        model_name=args.model_name,
        model_components=args.model_component,
        data_paths=args.data_path,
        protocol_files=args.protocol_file,
        protocol_items=args.protocol,
        eval_protocol_hash=args.eval_protocol_hash,
    )


def _add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", required=True, choices=sorted(SUPPORTED_SUITES))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-component", action="append", default=[])
    parser.add_argument("--data-path", action="append", required=True)
    parser.add_argument("--protocol-file", action="append", required=True)
    parser.add_argument("--protocol", action="append", default=[])
    parser.add_argument("--eval-protocol-hash", default="")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("cache-status")
    _add_request_arguments(cache_parser)
    cache_parser.add_argument("--manifest", required=True)

    write_parser = subparsers.add_parser("write-suite-manifest")
    _add_request_arguments(write_parser)
    write_parser.add_argument("--output-dir", required=True)
    write_parser.add_argument("--summary-file", required=True)
    write_parser.add_argument("--run-id", required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--status-file", required=True)
    aggregate_parser.add_argument("--suite-manifest", action="append", default=[])
    aggregate_parser.add_argument("--metrics-file", required=True)
    aggregate_parser.add_argument("--run-manifest-file", required=True)
    aggregate_parser.add_argument("--run-id", required=True)
    aggregate_parser.add_argument("--model-path", required=True)
    aggregate_parser.add_argument("--model-name", required=True)
    aggregate_parser.add_argument("--seed", required=True, type=int)
    aggregate_parser.add_argument("--max-samples", required=True, type=int)
    aggregate_parser.add_argument("--force-eval", required=True, choices=("0", "1"))
    aggregate_parser.add_argument("--output-root", required=True)
    aggregate_parser.add_argument("--eval-protocol-hash", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "cache-status":
        request = _request_from_args(args)
        manifest_path = Path(args.manifest).expanduser()
        if not manifest_path.is_file():
            print(f"MISS manifest not found: {manifest_path}")
            return 1
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"MISS unreadable manifest: {exc}")
            return 1
        matches, reason = cache_matches(manifest, request)
        print(f"{'HIT' if matches else 'MISS'} {reason}")
        return 0 if matches else 1

    if args.command == "write-suite-manifest":
        request = _request_from_args(args)
        path = write_suite_manifest(
            request=request,
            output_dir=args.output_dir,
            summary_file=args.summary_file,
            run_id=args.run_id,
        )
        print(path)
        return 0

    if args.command == "aggregate":
        suite_manifests = _parse_name_path(args.suite_manifest, "suite manifest")
        metrics, _ = aggregate_run(
            status_file=args.status_file,
            suite_manifests=suite_manifests,
            metrics_file=args.metrics_file,
            run_manifest_file=args.run_manifest_file,
            run_id=args.run_id,
            model_path=args.model_path,
            model_name=args.model_name,
            seed=args.seed,
            max_samples=args.max_samples,
            force_eval=args.force_eval == "1",
            output_root=args.output_root,
            eval_protocol_hash=args.eval_protocol_hash,
        )
        print(f"metrics={args.metrics_file} status={metrics['status']}")
        return 0 if metrics["status"] == "pass" else 1

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark artifact error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

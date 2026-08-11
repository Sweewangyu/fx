from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .config import Config
from .deepseek import LABEL_RESPONSE_SCHEMA, DeepSeekClient, validate_label
from .hashing import canonical_json, hash_object, sha256_file
from .state import StateStore, utc_now


class DataError(RuntimeError):
    pass


_CATALOG_FINGERPRINT_CACHE: dict[tuple[Any, ...], str] = {}
_ZIP_MISSING = object()


def _zip_equal(*iterables: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    """Backport ``zip(strict=True)`` so local Python 3.9 can run the checks."""

    for values in zip_longest(*iterables, fillvalue=_ZIP_MISSING):
        if any(value is _ZIP_MISSING for value in values):
            raise DataError("Internal data collections have mismatched lengths")
        yield values


@dataclass(frozen=True)
class Source:
    name: str
    path: Path
    split: str
    family: str
    training_role: str


class DataCatalog:
    def __init__(self, config: Config):
        root = Path(str(config.require("paths.datav2_root"))).resolve()
        registry_value = Path(str(config.require("paths.datav2_registry")))
        manifest_value = Path(str(config.require("paths.datav2_manifest")))
        self.root = root
        self.registry_path = registry_value if registry_value.is_absolute() else root / registry_value
        self.manifest_path = manifest_value if manifest_value.is_absolute() else root / manifest_value
        if not self.registry_path.is_file() or not self.manifest_path.is_file():
            raise DataError("datav2 registry or manifest does not exist")
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        if not isinstance(registry.get("sources"), list):
            raise DataError("datav2 sources.json has no sources list")
        self.sources = []
        for item in registry["sources"]:
            path_value = Path(item["path"])
            path = path_value if path_value.is_absolute() else root / path_value
            self.sources.append(
                Source(
                    name=item["name"],
                    path=path.resolve(),
                    split=item.get("split", "train"),
                    family=item.get("family", "unknown"),
                    training_role=item.get("training_role", "unknown"),
                )
            )

    @property
    def fingerprint(self) -> str:
        source_stats = tuple(
            (
                str(source.path),
                source.path.stat().st_dev,
                source.path.stat().st_ino,
                source.path.stat().st_size,
                source.path.stat().st_mtime_ns,
                source.path.stat().st_ctime_ns,
            )
            for source in self.sources
        )
        cache_key = (
            str(self.manifest_path),
            self.manifest_path.stat().st_size,
            self.manifest_path.stat().st_mtime_ns,
            self.manifest_path.stat().st_ctime_ns,
            str(self.registry_path),
            self.registry_path.stat().st_size,
            self.registry_path.stat().st_mtime_ns,
            self.registry_path.stat().st_ctime_ns,
            source_stats,
        )
        if cache_key not in _CATALOG_FINGERPRINT_CACHE:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            _CATALOG_FINGERPRINT_CACHE[cache_key] = hash_object(
                {
                    "published_content_sha256": manifest.get("content_sha256"),
                    "manifest_sha256": sha256_file(self.manifest_path),
                    "registry_sha256": sha256_file(self.registry_path),
                    "sources": {
                        source.name: sha256_file(source.path) for source in self.sources
                    },
                }
            )
        return _CATALOG_FINGERPRINT_CACHE[cache_key]

    def selected(self, names: Iterable[str] | None = None) -> list[Source]:
        wanted = set(names or [])
        sources = [source for source in self.sources if not wanted or source.name in wanted]
        missing = wanted - {source.name for source in sources}
        if missing:
            raise DataError(f"Unknown datav2 sources: {sorted(missing)}")
        for source in sources:
            if not source.path.is_file():
                raise DataError(f"Source file not found: {source.path}")
        return sources

    @staticmethod
    def iter_source(source: Source) -> Iterator[tuple[int, dict[str, Any]]]:
        with source.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"Invalid JSON at {source.path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise DataError(f"Expected JSON object at {source.path}:{line_number}")
                yield line_number, value


def record_hash(record: dict[str, Any]) -> str:
    return hash_object(record)


def sample_id(source: str, digest: str) -> str:
    return hash_object({"source": source, "record_hash": digest})


def _numbers(value: Any) -> Iterator[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            yield number
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)


def _series_summary(value: Any) -> dict[str, Any]:
    count = 0
    total = 0.0
    minimum = math.inf
    maximum = -math.inf
    for number in _numbers(value):
        count += 1
        total += number
        minimum = min(minimum, number)
        maximum = max(maximum, number)
    return {
        "numeric_values": count,
        "minimum": minimum if count else None,
        "maximum": maximum if count else None,
        "mean": total / count if count else None,
        "top_level_series": len(value) if isinstance(value, list) else None,
    }


def label_payload(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    input_limit = int(config["input_char_limit"])
    output_limit = int(config["output_char_limit"])
    return {
        "input": str(record.get("input", ""))[:input_limit],
        "output": str(record.get("output", ""))[:output_limit],
        "timeseries_summary": _series_summary(record.get("timeseries")),
    }


_TEMPLATE_NUMBER = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z_])"
)


def _template_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return _TEMPLATE_NUMBER.sub("<NUM>", text)[:limit]


def _series_shape(value: Any, depth: int = 0) -> list[int | str]:
    if depth >= 4 or not isinstance(value, list):
        return []
    if not value:
        return [0]
    child_shapes = {_series_shape(item, depth + 1).__repr__() for item in value[:8]}
    child = _series_shape(value[0], depth + 1)
    return [len(value), *(child if len(child_shapes) == 1 else ["ragged"])]


def template_payload(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Build a high-reuse template key without leaking concrete series values.

    Numeric literals in questions/answers are placeholders because many ChatTS
    generators instantiate one linguistic template with different locations and
    values. Shape is bucketed so trivial and genuinely long-context examples do
    not share a difficulty label.
    """

    shape = _series_shape(record.get("timeseries"))
    bucketed_shape = [
        0 if value == 0 else 2 ** int(math.log2(value)) if isinstance(value, int) else value
        for value in shape
    ]
    return {
        "input_template": _template_text(record.get("input", ""), int(config["input_char_limit"])),
        "output_template": _template_text(record.get("output", ""), int(config["output_char_limit"])),
        "timeseries_shape_bucket": bucketed_shape,
    }


LABEL_SYSTEM = """You audit normalized templates for supervised time-series reasoning examples. Return exactly one JSON object with keys quality_score, difficulty, taxonomy, rationale. quality_score is a number in [0,1]. difficulty is exactly easy, medium, or hard. Judge only visible clarity, format, and task-level consistency. Concrete time-series values are intentionally hidden, so you must not claim that a numeric answer, location, class, or factual conclusion is correct. taxonomy is a short task label. rationale is a concise note limited to visible template evidence. Do not add Markdown or other keys."""


def _source_taxonomy(source: str, taxonomy: str) -> str:
    """Guarantee a stable ECG domain tag for the previously unlabelled ECG source."""

    if re.search(r"(?:^|[_-])(ecg|electrocard)", source, re.IGNORECASE) and not re.search(
        r"\b(ecg|electrocard)", taxonomy, re.IGNORECASE
    ):
        return f"ECG / {taxonomy}"[:120]
    return taxonomy


def label_catalog(config: Config, state: StateStore, client: DeepSeekClient) -> dict[str, int]:
    catalog = DataCatalog(config)
    label_config = config.data["labeling"]
    names = label_config.get("sources") or []
    limit = int(label_config.get("max_samples", 0))
    prompt_version = str(config.get("deepseek.prompt_version"))
    model = str(config.get("deepseek.model"))
    concurrency = max(1, int(config.get("deepseek.concurrency", 8)))
    submitted_templates = completed_templates = 0
    sample_cache_hits = template_cache_hits = expanded_samples = 0

    def do_label(template_id: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        response = client.complete_json(
            purpose="quality-difficulty-taxonomy",
            system=LABEL_SYSTEM,
            user=canonical_json({"template_id": template_id, "template": payload}),
            validator=validate_label,
            prompt_version=prompt_version,
            response_schema=LABEL_RESPONSE_SCHEMA,
        )
        return template_id, {
            "template_id": template_id,
            "quality": response["quality_score"],
            "difficulty": response["difficulty"],
            "taxonomy": response["taxonomy"],
            "rationale": response["rationale"],
            "prompt_version": prompt_version,
            "model": model,
        }

    pending: dict[str, Future[tuple[str, dict[str, Any]]]] = {}
    waiting_samples: dict[str, list[tuple[str, str, str]]] = {}
    resolved_templates: dict[str, dict[str, Any]] = {}
    cached_label_batch: list[dict[str, Any]] = []
    queued_samples = 0

    def expand(done: set[Future[tuple[str, dict[str, Any]]]]) -> None:
        nonlocal completed_templates, expanded_samples, queued_samples
        by_future = {future: template_id for template_id, future in pending.items()}
        for future in done:
            template_id, template_label = future.result()
            state.template_label_put(template_label)
            resolved_templates[template_id] = template_label
            samples = waiting_samples.pop(template_id, [])
            queued_samples -= len(samples)
            expanded = []
            for sid, source_name, digest in samples:
                sample_label = dict(template_label)
                sample_label["taxonomy"] = _source_taxonomy(
                    source_name, str(template_label["taxonomy"])
                )
                expanded.append(
                    {
                        **sample_label,
                        "sample_id": sid,
                        "source": source_name,
                        "record_hash": digest,
                    }
                )
                expanded_samples += 1
            state.label_put_many(expanded)
            pending.pop(by_future[future], None)
            completed_templates += 1

    scanned = 0
    with state.connect() as lookup_db, ThreadPoolExecutor(max_workers=concurrency) as pool:
        stop = False
        for source in catalog.selected(names):
            if stop:
                break
            for _, record in catalog.iter_source(source):
                scanned += 1
                if limit and scanned > limit:
                    stop = True
                    break
                digest = record_hash(record)
                sid = sample_id(source.name, digest)
                if lookup_db.execute(
                    "SELECT 1 FROM labels WHERE sample_id=? AND prompt_version=? AND model=?",
                    (sid, prompt_version, model),
                ).fetchone() is not None:
                    sample_cache_hits += 1
                    continue
                payload = template_payload(record, label_config)
                template_id = hash_object(payload)
                template_label = resolved_templates.get(template_id)
                if template_label is None:
                    row = lookup_db.execute(
                        "SELECT * FROM template_labels WHERE template_id=? "
                        "AND prompt_version=? AND model=?",
                        (template_id, prompt_version, model),
                    ).fetchone()
                    template_label = dict(row) if row else None
                if template_label is not None:
                    resolved_templates[template_id] = template_label
                    sample_label = dict(template_label)
                    sample_label["taxonomy"] = _source_taxonomy(
                        source.name, str(template_label["taxonomy"])
                    )
                    cached_label_batch.append(
                        {
                            **sample_label,
                            "sample_id": sid,
                            "source": source.name,
                            "record_hash": digest,
                        }
                    )
                    template_cache_hits += 1
                    expanded_samples += 1
                    if len(cached_label_batch) >= 1000:
                        state.label_put_many(cached_label_batch)
                        cached_label_batch.clear()
                else:
                    waiting_samples.setdefault(template_id, []).append((sid, source.name, digest))
                    queued_samples += 1
                    if template_id not in pending:
                        pending[template_id] = pool.submit(do_label, template_id, payload)
                        submitted_templates += 1
                if len(pending) >= concurrency * 4 or queued_samples >= concurrency * 16:
                    done, _ = wait(set(pending.values()), return_when=FIRST_COMPLETED)
                    expand(done)
        while pending:
            done, _ = wait(set(pending.values()), return_when=FIRST_COMPLETED)
            expand(done)
        state.label_put_many(cached_label_batch)
    exported = state.export_labels(
        config.output_root / "labels" / "quality_difficulty_taxonomy.jsonl"
    )
    summary = {
        "submitted_templates": submitted_templates,
        "completed_templates": completed_templates,
        "expanded_samples": expanded_samples,
        "sample_cache_hits": sample_cache_hits,
        "template_cache_hits": template_cache_hits,
        "scanned": min(scanned, limit) if limit else scanned,
        "exported": exported,
    }
    state.metadata_put("label_summary", summary)
    return summary


def _normalized_text(record: dict[str, Any]) -> str:
    text = f"{record.get('input', '')}\n{record.get('output', '')}".lower()
    return re.sub(r"\s+", " ", text).strip()


def _simhash(text: str) -> int:
    tokens = re.findall(r"[\w.%-]+", text)
    if not tokens:
        return 0
    vector = [0] * 64
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


class DuplicateIndex:
    def __init__(self, path: Path):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS exact_seen(
              exact_hash TEXT PRIMARY KEY, sample_id TEXT NOT NULL, source TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bands(
              band_key TEXT NOT NULL, signature TEXT NOT NULL,
              sample_id TEXT NOT NULL, source TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS bands_key_idx ON bands(band_key);
            """
        )

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def inspect(
        self, exact_hash: str, signature: int, sid: str, source: str, max_hamming: int
    ) -> dict[str, Any]:
        exact = self.db.execute(
            "SELECT sample_id,source FROM exact_seen WHERE exact_hash=?", (exact_hash,)
        ).fetchone()
        exact_of = exact["sample_id"] if exact else None
        exact_source = exact["source"] if exact else None
        near_of = near_source = None
        if not exact:
            candidates: dict[str, tuple[int, str]] = {}
            for band in range(4):
                value = (signature >> (band * 16)) & 0xFFFF
                key = f"{band}:{value:04x}"
                for row in self.db.execute(
                    "SELECT signature,sample_id,source FROM bands WHERE band_key=? LIMIT 128", (key,)
                ):
                    candidates[row["sample_id"]] = (int(row["signature"], 16), row["source"])
            for candidate_id, (candidate_signature, candidate_source) in candidates.items():
                if (signature ^ candidate_signature).bit_count() <= max_hamming:
                    near_of, near_source = candidate_id, candidate_source
                    break
            self.db.execute(
                "INSERT INTO exact_seen VALUES(?,?,?)", (exact_hash, sid, source)
            )
            signature_hex = f"{signature:016x}"
            for band in range(4):
                value = (signature >> (band * 16)) & 0xFFFF
                self.db.execute(
                    "INSERT INTO bands VALUES(?,?,?,?)",
                    (f"{band}:{value:04x}", signature_hex, sid, source),
                )
        return {
            "exact_duplicate_of": exact_of,
            "near_duplicate_of": near_of,
            "cross_source_duplicate": bool(
                (exact_source and exact_source != source) or (near_source and near_source != source)
            ),
        }


def _dataset_info_payload(
    sources: list[Source], aliases: dict[str, str]
) -> dict[str, Any]:
    info = {}
    for source in sources:
        alias = aliases.get(source.name, source.name)
        if alias in info:
            raise DataError(f"Duplicate dataset alias: {alias}")
        info[alias] = {
            "file_name": f"data/{source.name}.jsonl",
            "columns": {"prompt": "input", "response": "output", "timeseries": "timeseries"},
            "description": f"Autoresearch snapshot of {source.name}",
        }
    return info


def _write_dataset_info(path: Path, sources: list[Source], aliases: dict[str, str]) -> str:
    info = _dataset_info_payload(sources, aliases)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def _validate_snapshot_files(root: Path, manifest: dict[str, Any]) -> None:
    output_hashes = manifest.get("output_hashes")
    if not isinstance(output_hashes, dict) or not output_hashes:
        raise DataError(f"Snapshot manifest has no output hashes: {root}")
    data_root = root / "data"
    actual_names = {
        path.stem for path in data_root.glob("*.jsonl") if path.is_file()
    }
    if actual_names != set(output_hashes):
        raise DataError(f"Snapshot data files were added, removed, or renamed: {root}")
    for source_name, expected_hash in output_hashes.items():
        path = data_root / f"{source_name}.jsonl"
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise DataError(f"Snapshot data file changed: {path}")
    dataset_info = root / "dataset_info.json"
    if (
        not dataset_info.is_file()
        or sha256_file(dataset_info) != manifest.get("dataset_info_sha256")
    ):
        raise DataError(f"Snapshot dataset_info.json changed: {root}")
    if manifest.get("snapshot") == "raw":
        identity = {
            "dataset_hash": manifest.get("dataset_hash"),
            "alias_hash": manifest.get("alias_hash"),
            "output_hashes": output_hashes,
            "dataset_info_sha256": manifest.get("dataset_info_sha256"),
        }
    else:
        duplicate_labels = root / "duplicate_labels.jsonl"
        if (
            not duplicate_labels.is_file()
            or sha256_file(duplicate_labels)
            != manifest.get("duplicate_labels_sha256")
        ):
            raise DataError(f"Snapshot duplicate audit changed: {root}")
        identity = {
            key: manifest.get(key)
            for key in (
                "source_dataset_hash",
                "snapshot_config_hash",
                "label_fingerprint",
                "output_hashes",
                "dataset_info_sha256",
                "duplicate_labels_sha256",
                "stats",
            )
        }
    if manifest.get("snapshot_hash") != hash_object(identity):
        raise DataError(f"Snapshot manifest hash is invalid: {root}")


def validate_snapshot(config: Config, state: StateStore, root: str | Path) -> dict[str, Any]:
    snapshot_root = Path(root)
    manifest_path = snapshot_root / "manifest.json"
    if not manifest_path.is_file():
        raise DataError(f"Snapshot manifest is missing: {snapshot_root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"Snapshot manifest is not valid JSON: {snapshot_root}") from exc
    if not isinstance(manifest, dict):
        raise DataError(f"Snapshot manifest must be an object: {snapshot_root}")
    catalog = DataCatalog(config)
    if manifest.get("snapshot") == "raw":
        if manifest.get("dataset_hash") != catalog.fingerprint:
            raise DataError("Raw snapshot does not match the current datav2 contents")
        if manifest.get("alias_hash") != hash_object(config.data["data"]["aliases"]):
            raise DataError("Raw snapshot does not match the current dataset aliases")
    else:
        if manifest.get("snapshot") != snapshot_root.name:
            raise DataError("Snapshot name does not match its directory")
        if manifest.get("source_dataset_hash") != catalog.fingerprint:
            raise DataError("Snapshot does not match the current datav2 contents")
        if manifest.get("snapshot_config_hash") != hash_object(config.data["data"]):
            raise DataError("Snapshot does not match the current data configuration")
        current_labels = state.label_fingerprint(
            str(config.get("deepseek.prompt_version")),
            str(config.get("deepseek.model")),
        )
        if manifest.get("label_fingerprint") != current_labels:
            raise DataError("Snapshot does not match the current labels")
    _validate_snapshot_files(snapshot_root, manifest)
    return manifest


def _create_raw_snapshot(config: Config, catalog: DataCatalog) -> Path:
    root = config.output_root / "datasets" / "raw"
    manifest_path = root / "manifest.json"
    alias_hash = hash_object(config.data["data"]["aliases"])
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("dataset_hash") != catalog.fingerprint
            or manifest.get("alias_hash") != alias_hash
        ):
            raise DataError("Existing raw snapshot belongs to a different datav2/alias fingerprint")
        _validate_snapshot_files(root, manifest)
        return root
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for source in catalog.sources:
        destination = data_dir / f"{source.name}.jsonl"
        if not destination.exists():
            relative = os.path.relpath(source.path, destination.parent)
            destination.symlink_to(relative)
    dataset_info_sha256 = _write_dataset_info(
        root / "dataset_info.json", catalog.sources, config.data["data"]["aliases"]
    )
    output_hashes = {
        source.name: sha256_file(source.path) for source in catalog.sources
    }
    payload = {
        "schema_version": "chatts-dataset-snapshot-v1",
        "snapshot": "raw",
        "dataset_hash": catalog.fingerprint,
        "alias_hash": alias_hash,
        "output_hashes": output_hashes,
        "dataset_info_sha256": dataset_info_sha256,
        "snapshot_hash": hash_object(
            {
                "dataset_hash": catalog.fingerprint,
                "alias_hash": alias_hash,
                "output_hashes": output_hashes,
                "dataset_info_sha256": dataset_info_sha256,
            }
        ),
        "storage": "relative symlinks to immutable datav2 files",
        "created_at": utc_now(),
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def _keep_by_weight(sid: str, weight: float) -> int:
    if weight <= 0:
        return 0
    whole = int(weight)
    fraction = weight - whole
    score = int(sid[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return whole + (1 if score < fraction else 0)


def prepare_snapshot(config: Config, state: StateStore) -> dict[str, Any]:
    catalog = DataCatalog(config)
    _create_raw_snapshot(config, catalog)
    data_config = config.data["data"]
    snapshot_name = str(data_config["snapshot_name"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", snapshot_name):
        raise DataError("data.snapshot_name contains unsafe characters")
    final_root = config.output_root / "datasets" / snapshot_name
    manifest_path = final_root / "manifest.json"
    snapshot_config_hash = hash_object(data_config)
    prompt_version = str(config.get("deepseek.prompt_version"))
    label_model = str(config.get("deepseek.model"))
    label_fingerprint = state.label_fingerprint(prompt_version, label_model)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_dataset_hash") == catalog.fingerprint
            and manifest.get("snapshot_config_hash") == snapshot_config_hash
            and manifest.get("label_fingerprint") == label_fingerprint
        ):
            _validate_snapshot_files(final_root, manifest)
            state.metadata_put(f"snapshot.{snapshot_name}", manifest)
            return manifest
        raise DataError("Snapshot path exists with a different data/config fingerprint")
    temporary = final_root.with_name(final_root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "data").mkdir(parents=True)
    duplicate_index = DuplicateIndex(temporary / "duplicate_index.sqlite3")
    audit_path = temporary / "duplicate_labels.jsonl"
    stats: dict[str, dict[str, int]] = {}
    output_hashes: dict[str, str] = {}
    missing_policy = data_config["missing_label_policy"]
    min_quality = float(data_config["minimum_quality"])
    max_hamming = int(data_config["near_duplicate_hamming"])
    try:
        with state.connect() as labels_db, audit_path.open("w", encoding="utf-8") as audit_stream:
            for source in catalog.sources:
                counts = {
                    "read": 0,
                    "written": 0,
                    "label_checked": 0,
                    "labeled": 0,
                    "low_quality": 0,
                    "missing_label": 0,
                    "duplicate": 0,
                }
                stats[source.name] = counts
                destination = temporary / "data" / f"{source.name}.jsonl"
                output_digest = hashlib.sha256()
                with destination.open("w", encoding="utf-8") as output:
                    for line_number, record in catalog.iter_source(source):
                        counts["read"] += 1
                        digest = record_hash(record)
                        sid = sample_id(source.name, digest)
                        duplicate = duplicate_index.inspect(
                            digest, _simhash(_normalized_text(record)), sid, source.name, max_hamming
                        )
                        audit_stream.write(
                            canonical_json(
                                {"sample_id": sid, "source": source.name, "line": line_number, **duplicate}
                            )
                            + "\n"
                        )
                        if duplicate["exact_duplicate_of"] and data_config["drop_exact_duplicates"]:
                            counts["duplicate"] += 1
                            continue
                        if duplicate["cross_source_duplicate"] and data_config["drop_cross_source_duplicates"]:
                            counts["duplicate"] += 1
                            continue
                        if duplicate["near_duplicate_of"] and data_config["drop_near_duplicates"]:
                            counts["duplicate"] += 1
                            continue
                        counts["label_checked"] += 1
                        label = labels_db.execute(
                            "SELECT quality,difficulty FROM labels WHERE sample_id=? "
                            "AND prompt_version=? AND model=?",
                            (sid, prompt_version, label_model),
                        ).fetchone()
                        if label is None:
                            counts["missing_label"] += 1
                            if missing_policy == "error":
                                raise DataError(f"Missing quality label for {source.name}:{line_number}")
                            if missing_policy == "drop":
                                continue
                            difficulty = "medium"
                        else:
                            counts["labeled"] += 1
                            if float(label["quality"]) < min_quality:
                                counts["low_quality"] += 1
                                continue
                            difficulty = label["difficulty"]
                        source_weight = float(data_config["source_weights"].get(source.name, 1.0))
                        difficulty_weight = float(data_config["difficulty_weights"].get(difficulty, 1.0))
                        copies = _keep_by_weight(sid, source_weight * difficulty_weight)
                        encoded = canonical_json(record) + "\n"
                        for _ in range(copies):
                            output.write(encoded)
                            output_digest.update(encoded.encode("utf-8"))
                            counts["written"] += 1
                output_hashes[source.name] = output_digest.hexdigest()
                duplicate_index.db.commit()
    except Exception:
        duplicate_index.close()
        shutil.rmtree(temporary)
        raise
    duplicate_index.close()
    dataset_info_sha256 = _write_dataset_info(
        temporary / "dataset_info.json", catalog.sources, data_config["aliases"]
    )
    duplicate_labels_sha256 = sha256_file(audit_path)
    manifest = {
        "schema_version": "chatts-dataset-snapshot-v1",
        "snapshot": snapshot_name,
        "source_dataset_hash": catalog.fingerprint,
        "snapshot_config_hash": snapshot_config_hash,
        "label_fingerprint": label_fingerprint,
        "output_hashes": output_hashes,
        "dataset_info_sha256": dataset_info_sha256,
        "duplicate_labels_sha256": duplicate_labels_sha256,
        "stats": stats,
        "created_at": utc_now(),
    }
    manifest["snapshot_hash"] = hash_object(
        {
            key: manifest[key]
            for key in (
                "source_dataset_hash",
                "snapshot_config_hash",
                "label_fingerprint",
                "output_hashes",
                "dataset_info_sha256",
                "duplicate_labels_sha256",
                "stats",
            )
        }
    )
    checked = sum(item["label_checked"] for item in stats.values())
    labeled = sum(item["labeled"] for item in stats.values())
    manifest["label_coverage"] = labeled / checked if checked else None
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(final_root)
    state.metadata_put(f"snapshot.{snapshot_name}", manifest)
    state.metadata_put("active_snapshot", {"path": str(final_root), **manifest})
    return manifest


EVAL_SPLIT_ALGORITHM_VERSION = 2
SEARCH_SPLIT = "search-dev"
FINAL_SPLIT = "final-test"


def _eval_record_id(record: dict[str, Any], id_field: str | None = None) -> str:
    if id_field and record.get(id_field) not in (None, ""):
        return str(record[id_field])
    for key in ("id", "sample_id", "question_id", "uid", "uuid"):
        if record.get(key) not in (None, ""):
            return str(record[key])
    return hash_object(record)


def _official_split(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"dev", "validation", "valid", SEARCH_SPLIT}:
        return SEARCH_SPLIT
    if normalized in {"test", FINAL_SPLIT}:
        return FINAL_SPLIT
    return None


def _path_split(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    stem = path.stem.lower()
    if (
        parts & {"dev", "validation", "valid"}
        or stem in {"dev", "validation", "valid"}
        or stem.endswith(("_dev", "_validation"))
    ):
        return SEARCH_SPLIT
    if "test" in parts or stem == "test" or stem.endswith("_test"):
        return FINAL_SPLIT
    return None


def _stratum(record: dict[str, Any]) -> str:
    source = record.get("__split_source__", "")
    category = next(
        (
            record[key]
            for key in ("category", "subject", "subcategory", "task", "dataset")
            if record.get(key) not in (None, "")
        ),
        "",
    )
    difficulty = next(
        (
            record[key]
            for key in ("difficulty", "level")
            if record.get(key) not in (None, "")
        ),
        "",
    )
    return f"{source}|{category}|{difficulty}"


def _hash_stratified_assignments(
    records: list[dict[str, Any]], record_ids: list[str]
) -> list[str]:
    if len(records) < 2:
        raise DataError("At least two records are required for a disjoint 20/80 split")
    grouped: dict[str, list[tuple[str, int]]] = {}
    for index, (record, record_id) in enumerate(_zip_equal(records, record_ids)):
        digest = hash_object(
            {
                "seed": 42,
                "id": record_id,
                "record": record,
                "index": index,
            }
        )
        grouped.setdefault(_stratum(record), []).append((digest, index))

    target = max(1, min(len(records) - 1, int(len(records) * 0.2 + 0.5)))
    quotas = {name: len(items) // 5 for name, items in grouped.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        grouped,
        key=lambda name: (-(len(grouped[name]) % 5), hash_object({"seed": 42, "stratum": name})),
    )
    for name in order:
        if remaining <= 0:
            break
        if quotas[name] < len(grouped[name]):
            quotas[name] += 1
            remaining -= 1
    if remaining:
        raise DataError(f"Could not allocate {target} search-dev records")

    dev_indices: set[int] = set()
    for name, items in grouped.items():
        dev_indices.update(index for _, index in sorted(items)[: quotas[name]])
    return [SEARCH_SPLIT if index in dev_indices else FINAL_SPLIT for index in range(len(records))]


def _assign_eval_records(
    records: list[dict[str, Any]],
    record_ids: list[str],
    explicit: list[str | None],
) -> tuple[list[str], str]:
    official = {value for value in explicit if value is not None}
    if len(explicit) == len(records) and all(value is not None for value in explicit) and official == {
        SEARCH_SPLIT,
        FINAL_SPLIT,
    }:
        return [str(value) for value in explicit], "official-dev-test"
    return _hash_stratified_assignments(records, record_ids), "hash-stratified-20-80"


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise DataError(f"Expected a JSON list of objects: {path}")
        return payload
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"Invalid JSONL {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise DataError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def create_eval_split_manifest(config: Config) -> dict[str, Any]:
    sources = config.get("evaluation.split_sources", []) or []
    destination = config.output_root / "configs" / "eval_splits.json"
    descriptors = []
    for source in sources:
        path = Path(source["path"]).expanduser().resolve()
        if not path.is_file():
            raise DataError(f"Evaluation split source not found: {path}")
        descriptors.append(
            {
                "name": source["name"],
                "path": str(path),
                "sha256": sha256_file(path),
                "id_field": source.get("id_field", "id"),
                "official_split_field": source.get("official_split_field"),
            }
        )
    input_hash = hash_object(descriptors)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if (
            existing.get("algorithm_version") == EVAL_SPLIT_ALGORITHM_VERSION
            and existing.get("input_hash") == input_hash
        ):
            return existing
        if existing.get("algorithm_version") == EVAL_SPLIT_ALGORITHM_VERSION:
            raise DataError("Existing evaluation split manifest belongs to different inputs")

    result: dict[str, Any] = {
        "schema_version": "chatts-eval-split-v2",
        "algorithm_version": EVAL_SPLIT_ALGORITHM_VERSION,
        "seed": 42,
        "input_hash": input_hash,
        "sources": {},
        "created_at": utc_now(),
    }
    for source, descriptor in _zip_equal(sources, descriptors):
        path = Path(descriptor["path"])
        records = _read_records(path)
        id_field = str(descriptor["id_field"])
        record_ids = [_eval_record_id(record, id_field) for record in records]
        official_field = descriptor["official_split_field"]
        explicit = [
            _official_split(record.get(official_field)) if official_field else None
            for record in records
        ]
        assignments, mode = _assign_eval_records(records, record_ids, explicit)
        buckets = {SEARCH_SPLIT: [], FINAL_SPLIT: []}
        for record_id, split in _zip_equal(record_ids, assignments):
            buckets[split].append(record_id)
        result["sources"][source["name"]] = {**descriptor, "mode": mode, **buckets}
    result["split_hash"] = hash_object(
        {
            "algorithm_version": EVAL_SPLIT_ALGORITHM_VERSION,
            "seed": 42,
            "input_hash": input_hash,
            "sources": result["sources"],
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return result


def _view_output_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def create_eval_dataset_views(config: Config) -> dict[str, Any]:
    """Materialize immutable, disjoint benchmark views consumed by ChatTS."""

    tsr_root_raw = config.get("paths.tsrbench_root")
    exam_file_raw = config.get("paths.timeseriesexam_data_file")
    tsr_root = Path(str(tsr_root_raw)).expanduser().resolve() if tsr_root_raw else None
    exam_file = Path(str(exam_file_raw)).expanduser().resolve() if exam_file_raw else None
    available = {
        "tsrbench": bool(tsr_root and tsr_root.is_dir()),
        "timeseriesexam": bool(exam_file and exam_file.is_file()),
    }
    root = config.output_root / "eval_views"
    manifest_path = root / "manifest.json"
    inputs: dict[str, Any] = {}
    if available["tsrbench"]:
        assert tsr_root is not None
        inputs["tsrbench"] = {
            str(path.relative_to(tsr_root)): sha256_file(path)
            for path in sorted(tsr_root.rglob("*"))
            if path.is_file()
        }
    if available["timeseriesexam"]:
        assert exam_file is not None
        inputs["timeseriesexam"] = {"path": str(exam_file), "sha256": sha256_file(exam_file)}
    input_hash = hash_object(inputs)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("algorithm_version") != EVAL_SPLIT_ALGORITHM_VERSION
            or manifest.get("input_hash") != input_hash
        ):
            raise DataError("Existing evaluation views use different inputs or split algorithm")
        if manifest.get("output_hashes") != _view_output_hashes(root):
            raise DataError("Existing evaluation view files were deleted or modified")
        return manifest

    temporary = root.with_name(root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    counts = {
        "tsrbench": {SEARCH_SPLIT: 0, FINAL_SPLIT: 0},
        "timeseriesexam": {SEARCH_SPLIT: 0, FINAL_SPLIT: 0},
    }
    ids = {
        "tsrbench": {SEARCH_SPLIT: [], FINAL_SPLIT: []},
        "timeseriesexam": {SEARCH_SPLIT: [], FINAL_SPLIT: []},
    }
    modes: dict[str, str | None] = {"tsrbench": None, "timeseriesexam": None}
    try:
        if available["tsrbench"]:
            assert tsr_root is not None
            entries: list[tuple[Path, Path, dict[str, Any]]] = []
            passthrough: list[tuple[Path, Path]] = []
            for source in sorted(tsr_root.rglob("*")):
                if not source.is_file():
                    continue
                relative = source.relative_to(tsr_root)
                if source.suffix.lower() != ".jsonl":
                    passthrough.append((source, relative))
                    continue
                for record in _read_records(source):
                    entries.append((source, relative, record))
            records = [entry[2] for entry in entries]
            record_ids = [
                f"{relative.as_posix()}::{_eval_record_id(record)}"
                for _, relative, record in entries
            ]
            explicit = [
                _official_split(record.get("official_split"))
                or _official_split(record.get("data_split"))
                or _official_split(record.get("split"))
                or _path_split(relative)
                for _, relative, record in entries
            ]
            stratified_records = [
                {**record, "__split_source__": relative.as_posix()}
                for _, relative, record in entries
            ]
            assignments, modes["tsrbench"] = _assign_eval_records(
                stratified_records, record_ids, explicit
            )
            outputs: dict[tuple[str, str], Any] = {}
            try:
                for (_, relative, record), record_id, split in _zip_equal(
                    entries, record_ids, assignments
                ):
                    key = (split, relative.as_posix())
                    if key not in outputs:
                        destination = temporary / split / "tsrbench" / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        outputs[key] = destination.open("w", encoding="utf-8")
                    outputs[key].write(canonical_json(record) + "\n")
                    counts["tsrbench"][split] += 1
                    ids["tsrbench"][split].append(record_id)
            finally:
                for output in outputs.values():
                    output.close()
            for source, relative in passthrough:
                for split in (SEARCH_SPLIT, FINAL_SPLIT):
                    destination = temporary / split / "tsrbench" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.symlink_to(os.path.relpath(source, destination.parent))

        if available["timeseriesexam"]:
            assert exam_file is not None
            records = _read_records(exam_file)
            record_ids = [_eval_record_id(record) for record in records]
            explicit = [
                _official_split(record.get("official_split"))
                or _official_split(record.get("data_split"))
                or _official_split(record.get("split"))
                for record in records
            ]
            assignments, modes["timeseriesexam"] = _assign_eval_records(
                records, record_ids, explicit
            )
            buckets: dict[str, list[dict[str, Any]]] = {SEARCH_SPLIT: [], FINAL_SPLIT: []}
            for record, record_id, split in _zip_equal(records, record_ids, assignments):
                buckets[split].append(record)
                counts["timeseriesexam"][split] += 1
                ids["timeseriesexam"][split].append(record_id)
            for split, split_records in buckets.items():
                destination = temporary / split / "timeseriesexam" / "qa_dataset.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(split_records, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

        for benchmark in ids:
            for split in ids[benchmark]:
                destination = temporary / split / benchmark / "split_ids.jsonl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    "".join(canonical_json({"id": rid}) + "\n" for rid in ids[benchmark][split]),
                    encoding="utf-8",
                )
        for benchmark, is_available in available.items():
            if is_available and (
                counts[benchmark][SEARCH_SPLIT] == 0 or counts[benchmark][FINAL_SPLIT] == 0
            ):
                raise DataError(f"{benchmark} must have non-empty search-dev and final-test views")

        manifest = {
            "schema_version": "chatts-eval-dataset-views-v2",
            "algorithm_version": EVAL_SPLIT_ALGORITHM_VERSION,
            "seed": 42,
            "rule": "official dev/test when complete; otherwise exact hash-stratified 20/80",
            "available": available,
            "modes": modes,
            "inputs": inputs,
            "input_hash": input_hash,
            "counts": counts,
            "split_ids_hash": hash_object(ids),
            "created_at": utc_now(),
        }
        manifest["view_hash"] = hash_object(
            {
                "algorithm_version": EVAL_SPLIT_ALGORITHM_VERSION,
                "seed": 42,
                "inputs": inputs,
                "counts": counts,
                "split_ids_hash": manifest["split_ids_hash"],
                "modes": modes,
            }
        )
        temporary.mkdir(parents=True, exist_ok=True)
        manifest["output_hashes"] = _view_output_hashes(temporary)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(root)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

#!/usr/bin/env python3
"""Merge TSR ability labels with template-level quality/difficulty labels.

The script is intentionally standalone.  Configuration lives in one YAML file;
no project package installation is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    TextIO,
)

import yaml
from label_tsqa import (
    DIFFICULTY_LEVELS,
    PROMPT_VERSION,
    QUALITY_LEVELS,
    build_template,
)

UNMAPPED = "UNMAPPED"
MERGE_VERSION = "tsqa-annotation-merge-v1"
VALID_CHATTS_KEYS = {"input", "timeseries", "output"}
ANNOTATION_FIELDS = {
    "annotation_id",
    "annotation_source",
    "source_index",
    "line_number",
    "taxonomy_sample_id",
    "taxonomy_cluster_id",
    "ability_label",
    "ability_bucket",
    "ability_name",
    "ability_major",
    "ability_secondary_labels",
    "ability_label_source",
    "ability_status",
    "ability_confidence",
    "taxonomy_fit",
    "reasoning_subtype",
    "training_role",
    "taxonomy_verifier_status",
    "ability_join_method",
    "quality",
    "difficulty",
    "quality_reason",
    "quality_template_id",
    "quality_model",
    "quality_prompt_version",
}


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def resolve_path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def stat_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def signature_digest(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def load_merge_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    for key in ("data_root", "data_registry", "quality_config", "taxonomy", "output"):
        if key not in config:
            raise ValueError(f"merge config is missing `{key}`")
    taxonomy = config["taxonomy"]
    output = config["output"]
    if (
        not isinstance(taxonomy, dict)
        or not taxonomy.get("manifests")
        or not taxonomy.get("labels")
    ):
        raise ValueError("taxonomy requires non-empty `manifests` and `labels`")
    if not isinstance(output, dict) or not output.get("root"):
        raise ValueError("output.root is required")
    return config


def load_registry(path: Path, data_root: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("sources") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"registry has no sources: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("name", ""))
        if not name or name in result:
            raise ValueError(f"invalid or duplicate source name in {path}: {name!r}")
        resolved = dict(row)
        resolved["path"] = resolve_path(data_root, row["path"])
        resolved["audit"] = (
            resolve_path(data_root, row["audit"]) if row.get("audit") else None
        )
        result[name] = resolved
    return result


def load_quality_config(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    value = read_yaml(path)
    datasets = value.get("datasets")
    settings = value.get("deepseek")
    if not isinstance(datasets, list) or not isinstance(settings, dict):
        raise TypeError(f"quality config needs datasets and deepseek: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in datasets:
        name = str(row.get("name", ""))
        if not name or name in result:
            raise ValueError(f"invalid or duplicate dataset in {path}: {name!r}")
        result[name] = {
            **row,
            "input": resolve_path(path.parent, row["input"]),
            "output": resolve_path(path.parent, row["output"]),
        }
    if not settings.get("model"):
        raise ValueError("quality config deepseek.model is required")
    return result, settings


def load_taxonomy_manifests(
    manifest_paths: Sequence[Path], data_root: Path
) -> tuple[dict[str, dict[str, str]], dict[str, Path]]:
    taxonomy: dict[str, dict[str, str]] | None = None
    source_paths: dict[str, Path] = {}
    for path in manifest_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        current = value.get("taxonomy")
        if not isinstance(current, dict) or len(current) != 15:
            raise ValueError(f"expected a 15-label taxonomy in {path}")
        if taxonomy is None:
            taxonomy = current
        elif compact_json(taxonomy) != compact_json(current):
            raise ValueError(f"taxonomy definitions disagree: {path}")
        for source in value.get("sources", []):
            name = str(source["name"])
            resolved = resolve_path(data_root, source["path"])
            if name in source_paths and source_paths[name] != resolved:
                raise ValueError(f"taxonomy source path disagrees for {name}")
            source_paths[name] = resolved
    assert taxonomy is not None
    return taxonomy, source_paths


def taxonomy_cache_signature(
    label_specs: Sequence[Mapping[str, Path]],
) -> dict[str, Any]:
    files = []
    for spec in label_specs:
        files.extend(
            [stat_signature(spec["provisional"]), stat_signature(spec["final"])]
        )
    return {"schema": "tsqa-taxonomy-cache-v1", "files": files}


def cache_matches(path: Path, signature: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        connection = sqlite3.connect(path)
        row = connection.execute(
            "SELECT value FROM meta WHERE key='signature'"
        ).fetchone()
        connection.close()
        return bool(row and json.loads(row[0]) == signature)
    except (sqlite3.Error, json.JSONDecodeError, OSError):
        return False


def build_taxonomy_cache(
    path: Path,
    label_specs: Sequence[Mapping[str, Path]],
    signature: Mapping[str, Any],
    rebuild: bool,
) -> Path:
    if not rebuild and cache_matches(path, signature):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE labels (
                source TEXT NOT NULL,
                source_index INTEGER NOT NULL,
                sample_id TEXT NOT NULL UNIQUE,
                cluster_id TEXT,
                source_sha256 TEXT,
                provisional_json TEXT,
                final_json TEXT,
                context_json TEXT,
                PRIMARY KEY (source, source_index)
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        inserted = 0
        final_seen = 0
        for spec in label_specs:
            with spec["provisional"].open(encoding="utf-8") as stream:
                batch = []
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    context = {
                        key: row.get(key)
                        for key in (
                            "reasoning_subtype",
                            "verifier_status",
                            "taxonomy_verifier_status",
                            "taxonomy_verifier_method",
                            "training_role",
                        )
                        if row.get(key) is not None
                    }
                    batch.append(
                        (
                            str(row["source"]),
                            int(row["source_index"]),
                            str(row["sample_id"]),
                            str(row.get("cluster_id", "")),
                            str(row.get("source_sha256", "")),
                            compact_json(row.get("provisional") or {}),
                            compact_json(context),
                        )
                    )
                    if len(batch) >= 10000:
                        connection.executemany(
                            "INSERT INTO labels(source,source_index,sample_id,cluster_id,source_sha256,provisional_json,context_json) VALUES (?,?,?,?,?,?,?)",
                            batch,
                        )
                        inserted += len(batch)
                        batch.clear()
                        connection.commit()
                if batch:
                    connection.executemany(
                        "INSERT INTO labels(source,source_index,sample_id,cluster_id,source_sha256,provisional_json,context_json) VALUES (?,?,?,?,?,?,?)",
                        batch,
                    )
                    inserted += len(batch)
                    connection.commit()

            with spec["final"].open(encoding="utf-8") as stream:
                batch = []
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    final_seen += 1
                    batch.append(
                        (compact_json(row.get("final") or {}), str(row["sample_id"]))
                    )
                    if len(batch) >= 10000:
                        connection.executemany(
                            "UPDATE labels SET final_json=? WHERE sample_id=?", batch
                        )
                        batch.clear()
                        connection.commit()
                if batch:
                    connection.executemany(
                        "UPDATE labels SET final_json=? WHERE sample_id=?", batch
                    )
                    connection.commit()

        missing_final = int(
            connection.execute(
                "SELECT COUNT(*) FROM labels WHERE final_json IS NULL"
            ).fetchone()[0]
        )
        if missing_final:
            raise ValueError(
                f"taxonomy cache has {missing_final} provisional rows without final rows"
            )
        if final_seen != inserted:
            raise ValueError(
                f"taxonomy provisional/final row count mismatch: {inserted} != {final_seen}"
            )
        connection.execute(
            "CREATE INDEX labels_source_sha ON labels(source, source_sha256)"
        )
        connection.execute(
            "INSERT INTO meta VALUES ('signature', ?)", (compact_json(signature),)
        )
        connection.execute("INSERT INTO meta VALUES ('rows', ?)", (str(inserted),))
        connection.commit()
        connection.close()
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_taxonomy_source(
    connection: sqlite3.Connection, source: str
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    query = """
        SELECT source_index,sample_id,cluster_id,source_sha256,provisional_json,final_json,context_json
        FROM labels WHERE source=? ORDER BY source_index
    """
    for item in connection.execute(query, (source,)):
        rows[int(item[0])] = {
            "source_index": int(item[0]),
            "sample_id": item[1],
            "cluster_id": item[2],
            "source_sha256": item[3],
            "provisional": json.loads(item[4]),
            "final": json.loads(item[5]),
            "context": json.loads(item[6]),
        }
    return rows


def quality_row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "record_id",
            "line_number",
            "input_hash",
            "template_id",
            "quality",
            "difficulty",
            "reason",
            "model",
            "prompt_version",
        )
    }


def load_quality_labels(
    path: Path, model: str, prompt_version: str
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    rows: dict[int, dict[str, Any]] = {}
    stats = Counter()
    with path.open(encoding="utf-8") as stream:
        for file_line, line in enumerate(stream, 1):
            stats["file_rows"] += 1
            try:
                row = json.loads(line)
                if (
                    row.get("model") != model
                    or row.get("prompt_version") != prompt_version
                ):
                    stats["ignored_other_model_or_prompt"] += 1
                    continue
                line_number = int(row["line_number"])
                if (
                    row.get("quality") not in QUALITY_LEVELS
                    or row.get("difficulty") not in DIFFICULTY_LEVELS
                ):
                    raise ValueError("invalid quality or difficulty")
                if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                    raise ValueError("missing reason")
                payload = quality_row_payload(row)
                if line_number in rows:
                    if compact_json(rows[line_number]) != compact_json(payload):
                        raise ValueError(
                            f"conflicting duplicate label for input line {line_number}"
                        )
                    stats["duplicate_identical"] += 1
                    continue
                rows[line_number] = payload
                stats["usable_rows"] += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid quality label at {path}:{file_line}: {exc}"
                ) from exc
    return rows, dict(stats)


def taxonomy_decision(
    taxonomy_row: Mapping[str, Any] | None,
    audit: Mapping[str, Any] | None,
    taxonomy: Mapping[str, Mapping[str, str]],
    join_method: str,
) -> dict[str, Any]:
    decision: Mapping[str, Any] = {}
    label: str | None = None
    label_source = "unmapped"
    status = "unmapped"
    confidence = None
    taxonomy_fit = None
    context: Mapping[str, Any] = {}
    sample_id = cluster_id = None

    if taxonomy_row:
        final = taxonomy_row["final"]
        provisional = taxonomy_row["provisional"]
        context = taxonomy_row["context"]
        sample_id = taxonomy_row["sample_id"]
        cluster_id = taxonomy_row["cluster_id"]
        if final.get("primary_label"):
            decision, label = final, str(final["primary_label"])
            label_source, status = "final", str(final.get("status", "accepted"))
            confidence = final.get("confidence")
            taxonomy_fit = final.get("taxonomy_fit")
        elif final.get("proposed_primary_label"):
            decision, label = final, str(final["proposed_primary_label"])
            label_source, status = "final_proposed", "human_review"
            confidence = final.get("proposed_confidence")
            taxonomy_fit = final.get("proposed_taxonomy_fit")
        elif provisional.get("primary_label"):
            decision, label = provisional, str(provisional["primary_label"])
            label_source, status = (
                "provisional",
                str(final.get("status", provisional.get("status", "review"))),
            )
            confidence = provisional.get("confidence")
            taxonomy_fit = provisional.get("taxonomy_fit")
        else:
            decision = provisional or final
            confidence = decision.get("confidence")
            taxonomy_fit = decision.get("taxonomy_fit")
            status = (
                "out_of_scope"
                if decision.get("taxonomy_fit") == "out_of_scope"
                else "unmapped"
            )
    elif audit and audit.get("primary_label"):
        decision, label = audit, str(audit["primary_label"])
        label_source = "audit_fallback"
        taxonomy_verifier = audit.get("taxonomy_verifier") or {}
        status = str(taxonomy_verifier.get("status", "audit_fallback"))
        context = {
            "reasoning_subtype": audit.get("reasoning_subtype"),
            "training_role": audit.get("training_role"),
            "taxonomy_verifier_status": taxonomy_verifier.get("status"),
        }
        confidence = audit.get("confidence")
        taxonomy_fit = audit.get("taxonomy_fit")

    if label is not None and label not in taxonomy:
        raise ValueError(f"unknown ability label: {label}")
    definition = taxonomy.get(label or "", {})
    secondary = decision.get("secondary_labels") or []
    return {
        "taxonomy_sample_id": sample_id,
        "taxonomy_cluster_id": cluster_id,
        "ability_label": label,
        "ability_bucket": label or UNMAPPED,
        "ability_name": definition.get("name"),
        "ability_major": definition.get("major"),
        "ability_secondary_labels": secondary if isinstance(secondary, list) else [],
        "ability_label_source": label_source,
        "ability_status": status,
        "ability_confidence": confidence,
        "taxonomy_fit": taxonomy_fit,
        "reasoning_subtype": context.get("reasoning_subtype"),
        "training_role": context.get("training_role"),
        "taxonomy_verifier_status": context.get("taxonomy_verifier_status"),
        "ability_join_method": join_method,
    }


def iter_audit(path: Path | None) -> Iterator[dict[str, Any] | None]:
    if path is None:
        while True:
            yield None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            expected = line_number - 1
            if int(row.get("sample_index", -1)) != expected:
                raise ValueError(f"audit/source index mismatch at {path}:{line_number}")
            yield row


class CanonicalTaxonomyAligner:
    """Align a cleaned canonical source with the raw source used for taxonomy labeling."""

    def __init__(self, path: Path, labels: Mapping[int, Mapping[str, Any]]):
        self.path = path
        self.labels = labels
        self.stream = path.open(encoding="utf-8")
        self.raw_index = -1

    def next(self, current: Mapping[str, Any]) -> Mapping[str, Any]:
        for line in self.stream:
            self.raw_index += 1
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict) or set(raw) != VALID_CHATTS_KEYS:
                    continue
            except json.JSONDecodeError:
                continue
            if compact_json(raw) != compact_json(current):
                raise ValueError(
                    f"clean/raw sequence mismatch: current row does not match {self.path}:{self.raw_index + 1}"
                )
            label = self.labels.get(self.raw_index)
            if not label:
                raise ValueError(
                    f"missing taxonomy label for {self.path}:{self.raw_index + 1}"
                )
            return label
        raise ValueError(f"taxonomy source ended before current source: {self.path}")

    def finish(self) -> None:
        for line in self.stream:
            self.raw_index += 1
            try:
                row = json.loads(line)
                if isinstance(row, dict) and set(row) == VALID_CHATTS_KEYS:
                    raise ValueError(
                        f"taxonomy source has extra valid rows: {self.path}:{self.raw_index + 1}"
                    )
            except json.JSONDecodeError:
                continue
        self.stream.close()

    def close(self) -> None:
        self.stream.close()


def temporary_text_stream(final_path: Path) -> tuple[Path, TextIO]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=final_path.name + ".", suffix=".tmp", dir=final_path.parent
    )
    return Path(name), os.fdopen(descriptor, "w", encoding="utf-8")


def serialize_counter(counter: Counter, fields: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for key, count in sorted(counter.items()):
        values = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(fields, values)), "count": count})
    return rows


def deserialize_counter(
    rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> Counter:
    return Counter(
        {tuple(row[field] for field in fields): int(row["count"]) for row in rows}
    )


def make_source_signature(
    source: Mapping[str, Any],
    quality_path: Path,
    taxonomy_signature: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    files = [stat_signature(source["path"]), stat_signature(quality_path)]
    if source.get("audit"):
        files.append(stat_signature(source["audit"]))
    return {
        "schema": MERGE_VERSION,
        "files": files,
        "taxonomy": signature_digest(taxonomy_signature),
        "write_annotated_qa": bool(output.get("write_annotated_qa", True)),
        "write_annotation_sidecar": bool(output.get("write_annotation_sidecar", True)),
        "quality_prompt_version": PROMPT_VERSION,
    }


def source_outputs_exist(
    stats: Mapping[str, Any],
    annotated_path: Path,
    sidecar_path: Path,
    output: Mapping[str, Any],
) -> bool:
    if bool(output.get("write_annotated_qa", True)) and not annotated_path.is_file():
        return False
    return not (
        bool(output.get("write_annotation_sidecar", True))
        and not sidecar_path.is_file()
    )


def process_source(
    name: str,
    source: Mapping[str, Any],
    quality_spec: Mapping[str, Any],
    settings: Mapping[str, Any],
    taxonomy: Mapping[str, Mapping[str, str]],
    taxonomy_source_path: Path,
    taxonomy_rows: Mapping[int, Mapping[str, Any]],
    taxonomy_signature: Mapping[str, Any],
    output_root: Path,
    output: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    annotated_path = output_root / "annotated" / f"{name}.jsonl"
    sidecar_path = output_root / "annotations" / f"{name}.jsonl"
    stats_path = output_root / "reports" / "sources" / f"{name}.json"
    signature = make_source_signature(
        source, quality_spec["output"], taxonomy_signature, output
    )
    if resume and stats_path.is_file():
        previous = json.loads(stats_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature and source_outputs_exist(
            previous, annotated_path, sidecar_path, output
        ):
            print(
                compact_json({"event": "resume_skip", "dataset": name}),
                file=sys.stderr,
                flush=True,
            )
            return previous

    quality_rows, quality_file_stats = load_quality_labels(
        quality_spec["output"], str(settings["model"]), PROMPT_VERSION
    )
    direct_same_file = source["path"].resolve() == taxonomy_source_path.resolve()
    canonical_aligner = (
        None
        if direct_same_file
        else CanonicalTaxonomyAligner(taxonomy_source_path, taxonomy_rows)
    )
    labels_by_digest: dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
    if direct_same_file:
        for row in taxonomy_rows.values():
            if row.get("source_sha256"):
                labels_by_digest[str(row["source_sha256"])].append(row)
    used_taxonomy_indices = set()
    audit_rows = iter_audit(source.get("audit"))

    write_annotated = bool(output.get("write_annotated_qa", True))
    write_sidecar = bool(output.get("write_annotation_sidecar", True))
    strict_quality = bool(output.get("strict_quality", True))
    strict_ability = bool(output.get("strict_ability_join", True))
    annotated_temp = annotated_stream = sidecar_temp = sidecar_stream = None
    if write_annotated:
        annotated_temp, annotated_stream = temporary_text_stream(annotated_path)
    if write_sidecar:
        sidecar_temp, sidecar_stream = temporary_text_stream(sidecar_path)

    counts = Counter()
    cube = Counter()
    source_cube = Counter()
    join_methods = Counter()
    label_sources = Counter()
    used_quality_lines = set()
    try:
        with source["path"].open("rb") as input_stream:
            for source_index, raw_line in enumerate(input_stream):
                if not raw_line.strip():
                    raise ValueError(
                        f"blank input row at {source['path']}:{source_index + 1}"
                    )
                line_number = source_index + 1
                try:
                    record = json.loads(raw_line)
                    if not isinstance(record, dict) or set(record) != VALID_CHATTS_KEYS:
                        raise ValueError(
                            "expected exact input/timeseries/output ChatTS schema"
                        )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid input at {source['path']}:{line_number}: {exc}"
                    ) from exc
                try:
                    audit = next(audit_rows)
                except StopIteration as exc:
                    raise ValueError(
                        f"audit ended before input source: {source.get('audit')}:{line_number}"
                    ) from exc

                taxonomy_row = None
                join_method = "missing"
                if canonical_aligner is not None:
                    taxonomy_row = canonical_aligner.next(record)
                    join_method = "taxonomy_clean_sequence"
                else:
                    digest = hashlib.sha256(raw_line.rstrip(b"\n")).hexdigest()
                    candidate = taxonomy_rows.get(source_index)
                    if (
                        candidate
                        and source_index not in used_taxonomy_indices
                        and candidate.get("source_sha256") == digest
                    ):
                        taxonomy_row = candidate
                        join_method = "taxonomy_direct_hash"
                    else:
                        queue = labels_by_digest.get(digest)
                        while (
                            queue
                            and int(queue[0]["source_index"]) in used_taxonomy_indices
                        ):
                            queue.popleft()
                        if queue:
                            taxonomy_row = queue.popleft()
                            join_method = "taxonomy_digest_recovery"
                        elif audit and audit.get("primary_label"):
                            join_method = "audit_fallback"
                    if taxonomy_row:
                        used_taxonomy_indices.add(int(taxonomy_row["source_index"]))

                ability = taxonomy_decision(taxonomy_row, audit, taxonomy, join_method)
                if strict_ability and join_method == "missing":
                    raise ValueError(f"no ability annotation for {name}:{line_number}")

                quality = quality_rows.get(line_number)
                if quality is None:
                    if strict_quality:
                        raise ValueError(
                            f"no quality annotation for {name}:{line_number}"
                        )
                    quality = {
                        "record_id": None,
                        "template_id": None,
                        "quality": None,
                        "difficulty": None,
                        "reason": None,
                        "model": None,
                        "prompt_version": None,
                    }
                else:
                    template_id, _, input_hash = build_template(record)
                    if (
                        quality.get("template_id") != template_id
                        or quality.get("input_hash") != input_hash
                    ):
                        raise ValueError(
                            f"quality label/input hash mismatch for {name}:{line_number}"
                        )
                    used_quality_lines.add(line_number)

                annotation = {
                    "annotation_id": f"{name}:{line_number}",
                    "annotation_source": name,
                    "source_index": source_index,
                    "line_number": line_number,
                    **ability,
                    "quality": quality.get("quality"),
                    "difficulty": quality.get("difficulty"),
                    "quality_reason": quality.get("reason"),
                    "quality_template_id": quality.get("template_id"),
                    "quality_model": quality.get("model"),
                    "quality_prompt_version": quality.get("prompt_version"),
                }
                if write_sidecar:
                    assert sidecar_stream is not None
                    sidecar_stream.write(compact_json(annotation) + "\n")
                if write_annotated:
                    assert annotated_stream is not None
                    collisions = ANNOTATION_FIELDS.intersection(record)
                    if collisions:
                        raise ValueError(
                            f"annotation field collision in {name}:{line_number}: {sorted(collisions)}"
                        )
                    annotated_stream.write(
                        compact_json({**record, **annotation}) + "\n"
                    )

                bucket = ability["ability_bucket"]
                quality_label = quality.get("quality") or "MISSING"
                difficulty_label = quality.get("difficulty") or "MISSING"
                counts["total_rows"] += 1
                counts["quality_labeled_rows"] += quality.get("quality") is not None
                counts["ability_15d_rows"] += ability["ability_label"] is not None
                counts["ability_unmapped_rows"] += ability["ability_label"] is None
                cube[(bucket, quality_label, difficulty_label)] += 1
                source_cube[(name, bucket, quality_label, difficulty_label)] += 1
                join_methods[join_method] += 1
                label_sources[str(ability["ability_label_source"])] += 1

        if canonical_aligner is not None:
            canonical_aligner.finish()
        if source.get("audit"):
            try:
                next(audit_rows)
            except StopIteration:
                pass
            else:
                raise ValueError(
                    f"audit has more rows than input source: {source['audit']}"
                )
        unused_quality = sorted(set(quality_rows) - used_quality_lines)
        if strict_quality and unused_quality:
            raise ValueError(
                f"{name} has {len(unused_quality)} quality labels not matched to current input"
            )

        if annotated_stream is not None:
            annotated_stream.flush()
            annotated_stream.close()
            os.replace(annotated_temp, annotated_path)
            os.chmod(annotated_path, 0o644)
        if sidecar_stream is not None:
            sidecar_stream.flush()
            sidecar_stream.close()
            os.replace(sidecar_temp, sidecar_path)
            os.chmod(sidecar_path, 0o644)

        stats = {
            "dataset": name,
            "signature": signature,
            "counts": dict(counts),
            "quality_file": {**quality_file_stats, "unused_rows": len(unused_quality)},
            "taxonomy_rows_available": len(taxonomy_rows),
            "taxonomy_source": str(taxonomy_source_path),
            "join_methods": dict(join_methods),
            "ability_label_sources": dict(label_sources),
            "cube": serialize_counter(cube, ("ability_label", "quality", "difficulty")),
            "source_cube": serialize_counter(
                source_cube, ("source", "ability_label", "quality", "difficulty")
            ),
            "outputs": {
                "annotated": str(annotated_path) if write_annotated else None,
                "sidecar": str(sidecar_path) if write_sidecar else None,
            },
        }
        write_json(stats_path, stats)
        return stats
    except Exception:
        if canonical_aligner is not None:
            canonical_aligner.close()
        for stream in (annotated_stream, sidecar_stream):
            if stream is not None and not stream.closed:
                stream.close()
        for temporary in (annotated_temp, sidecar_temp):
            if isinstance(temporary, Path):
                temporary.unlink(missing_ok=True)
        raise


def percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def write_reports(
    output_root: Path,
    taxonomy: Mapping[str, Mapping[str, str]],
    source_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    cube = Counter()
    source_cube = Counter()
    join_methods = Counter()
    label_sources = Counter()
    totals = Counter()
    for stats in source_stats:
        totals.update(stats["counts"])
        join_methods.update(stats["join_methods"])
        label_sources.update(stats["ability_label_sources"])
        cube.update(
            deserialize_counter(
                stats["cube"], ("ability_label", "quality", "difficulty")
            )
        )
        source_cube.update(
            deserialize_counter(
                stats["source_cube"],
                ("source", "ability_label", "quality", "difficulty"),
            )
        )

    ability_order = list(taxonomy) + [UNMAPPED]
    quality_order = list(QUALITY_LEVELS)
    difficulty_order = list(DIFFICULTY_LEVELS)
    total_rows = int(totals["total_rows"])
    by_ability = Counter()
    by_quality = Counter()
    by_difficulty = Counter()
    for (ability, quality, difficulty), count in cube.items():
        by_ability[ability] += count
        by_quality[quality] += count
        by_difficulty[difficulty] += count

    cube_path = reports / "ability_quality_difficulty.csv"
    with cube_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "ability_label",
                "ability_name",
                "ability_major",
                "quality",
                "difficulty",
                "count",
                "percent_all",
                "percent_within_ability",
            ),
        )
        writer.writeheader()
        for ability in ability_order:
            definition = taxonomy.get(ability, {})
            for quality in quality_order:
                for difficulty in difficulty_order:
                    count = int(cube[(ability, quality, difficulty)])
                    writer.writerow(
                        {
                            "ability_label": ability,
                            "ability_name": definition.get("name", "unmapped"),
                            "ability_major": definition.get("major", "unmapped"),
                            "quality": quality,
                            "difficulty": difficulty,
                            "count": count,
                            "percent_all": percentage(count, total_rows),
                            "percent_within_ability": percentage(
                                count, by_ability[ability]
                            ),
                        }
                    )

    source_cube_path = reports / "source_ability_quality_difficulty.csv"
    source_names = [str(stats["dataset"]) for stats in source_stats]
    with source_cube_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "ability_label", "quality", "difficulty", "count"])
        for source in source_names:
            for ability in ability_order:
                for quality in quality_order:
                    for difficulty in difficulty_order:
                        writer.writerow(
                            [
                                source,
                                ability,
                                quality,
                                difficulty,
                                source_cube[(source, ability, quality, difficulty)],
                            ]
                        )

    coverage_path = reports / "coverage_by_source.csv"
    with coverage_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "source",
                "total_rows",
                "quality_labeled_rows",
                "quality_coverage_percent",
                "ability_15d_rows",
                "ability_15d_coverage_percent",
                "ability_unmapped_rows",
                "join_methods",
                "ability_label_sources",
            ),
        )
        writer.writeheader()
        for stats in source_stats:
            counts = stats["counts"]
            rows = int(counts.get("total_rows", 0))
            writer.writerow(
                {
                    "source": stats["dataset"],
                    "total_rows": rows,
                    "quality_labeled_rows": counts.get("quality_labeled_rows", 0),
                    "quality_coverage_percent": percentage(
                        int(counts.get("quality_labeled_rows", 0)), rows
                    ),
                    "ability_15d_rows": counts.get("ability_15d_rows", 0),
                    "ability_15d_coverage_percent": percentage(
                        int(counts.get("ability_15d_rows", 0)), rows
                    ),
                    "ability_unmapped_rows": counts.get("ability_unmapped_rows", 0),
                    "join_methods": compact_json(stats["join_methods"]),
                    "ability_label_sources": compact_json(
                        stats["ability_label_sources"]
                    ),
                }
            )

    ability_summary = []
    for ability in ability_order:
        count = int(by_ability[ability])
        high_quality = sum(
            cube[(ability, q, d)]
            for q in ("good", "excellent")
            for d in difficulty_order
        )
        hard = sum(
            cube[(ability, q, d)] for q in quality_order for d in ("hard", "very_hard")
        )
        definition = taxonomy.get(ability, {})
        ability_summary.append(
            {
                "ability_label": ability,
                "ability_name": definition.get("name", "unmapped"),
                "ability_major": definition.get("major", "unmapped"),
                "count": count,
                "percent_all": percentage(count, total_rows),
                "good_or_excellent_percent": percentage(high_quality, count),
                "hard_or_very_hard_percent": percentage(hard, count),
            }
        )

    summary = {
        "schema_version": "tsqa-annotation-distribution-v1",
        "total_rows": total_rows,
        "quality_labeled_rows": int(totals["quality_labeled_rows"]),
        "ability_15d_rows": int(totals["ability_15d_rows"]),
        "ability_15d_coverage_percent": percentage(
            int(totals["ability_15d_rows"]), total_rows
        ),
        "ability_unmapped_rows": int(totals["ability_unmapped_rows"]),
        "by_ability": dict(by_ability),
        "by_quality": dict(by_quality),
        "by_difficulty": dict(by_difficulty),
        "join_methods": dict(join_methods),
        "ability_label_sources": dict(label_sources),
        "ability_summary": ability_summary,
        "sources": [
            {
                "dataset": stats["dataset"],
                "counts": stats["counts"],
                "join_methods": stats["join_methods"],
            }
            for stats in source_stats
        ],
    }
    write_json(reports / "distribution.json", summary)

    lines = [
        "# QA 标注联合分布",
        "",
        f"- 总 QA：{total_rows:,}",
        f"- quality / difficulty 覆盖：{int(totals['quality_labeled_rows']):,} ({percentage(int(totals['quality_labeled_rows']), total_rows):.2f}%)",
        f"- 15 维能力有效覆盖：{int(totals['ability_15d_rows']):,} ({percentage(int(totals['ability_15d_rows']), total_rows):.2f}%)",
        f"- 明确未映射或超出 15 维：{int(totals['ability_unmapped_rows']):,}",
        "",
        "## 能力维度汇总",
        "",
        "|维度|名称|大类|数量|占全部|good/excellent|hard/very_hard|",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in ability_summary:
        lines.append(
            "|{ability_label}|{ability_name}|{ability_major}|{count:,}|{percent_all:.2f}%|{good_or_excellent_percent:.2f}%|{hard_or_very_hard_percent:.2f}%|".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- `ability_quality_difficulty.csv`：15维 × 质量 × 难度完整立方体（包含零计数格）",
            "- `source_ability_quality_difficulty.csv`：再按数据源展开",
            "- `coverage_by_source.csv`：逐来源覆盖率和连接方法",
            "- `distribution.json`：机器可读汇总",
            "",
        ]
    )
    (reports / "DISTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def dry_run_report(
    names: Sequence[str],
    registry: Mapping[str, Mapping[str, Any]],
    quality: Mapping[str, Mapping[str, Any]],
    taxonomy_sources: Mapping[str, Path],
) -> int:
    rows = []
    missing = False
    for name in names:
        paths = {
            "input": registry[name]["path"],
            "quality": quality[name]["output"],
            "taxonomy_source": taxonomy_sources[name],
            "audit": registry[name].get("audit"),
        }
        status = {
            key: (value is None or Path(value).is_file())
            for key, value in paths.items()
        }
        missing = missing or not all(status.values())
        rows.append(
            {
                "dataset": name,
                "paths": {k: str(v) if v else None for k, v in paths.items()},
                "exists": status,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 1 if missing else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="合并 15 维能力、质量和难度标注并统计联合分布"
    )
    parser.add_argument("--config", type=Path, default=Path("merge_config.yaml"))
    parser.add_argument("--dataset", action="append", help="只处理指定数据集；可重复")
    parser.add_argument("--dry-run", action="store_true", help="只核对路径，不生成输出")
    parser.add_argument(
        "--force", action="store_true", help="忽略逐来源 resume 状态并重建输出"
    )
    parser.add_argument(
        "--rebuild-taxonomy-cache",
        action="store_true",
        help="强制重建能力标注 SQLite 索引",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="只写轻量 sidecar 和分布，不复制原始 QA",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        config = load_merge_config(config_path)
        data_root = resolve_path(config_path.parent, config["data_root"])
        registry_path = resolve_path(config_path.parent, config["data_registry"])
        quality_config_path = resolve_path(config_path.parent, config["quality_config"])
        registry = load_registry(registry_path, data_root)
        quality, quality_settings = load_quality_config(quality_config_path)

        manifest_paths = [
            resolve_path(config_path.parent, item)
            for item in config["taxonomy"]["manifests"]
        ]
        taxonomy, taxonomy_sources = load_taxonomy_manifests(manifest_paths, data_root)
        label_specs = []
        for item in config["taxonomy"]["labels"]:
            label_specs.append(
                {
                    "provisional": resolve_path(
                        config_path.parent, item["provisional"]
                    ),
                    "final": resolve_path(config_path.parent, item["final"]),
                }
            )

        names = list(quality)
        if args.dataset:
            selected = set(args.dataset)
            unknown = selected - set(names)
            if unknown:
                raise ValueError(f"unknown dataset(s): {', '.join(sorted(unknown))}")
            names = [name for name in names if name in selected]
        missing_registry = set(names) - set(registry)
        missing_taxonomy = set(names) - set(taxonomy_sources)
        if missing_registry or missing_taxonomy:
            raise ValueError(
                f"dataset mapping incomplete; registry={sorted(missing_registry)}, taxonomy={sorted(missing_taxonomy)}"
            )
        for name in names:
            if quality[name]["input"].resolve() != registry[name]["path"].resolve():
                raise ValueError(
                    f"quality/data registry input path mismatch for {name}"
                )
        if args.dry_run:
            return dry_run_report(names, registry, quality, taxonomy_sources)

        for path in manifest_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        for spec in label_specs:
            for path in spec.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
        for name in names:
            for path in (
                registry[name]["path"],
                quality[name]["output"],
                taxonomy_sources[name],
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)

        output = dict(config["output"])
        if args.labels_only:
            output["write_annotated_qa"] = False
            output["write_annotation_sidecar"] = True
        if not bool(output.get("write_annotated_qa", True)) and not bool(
            output.get("write_annotation_sidecar", True)
        ):
            raise ValueError("at least one output stream must be enabled")
        output_root = resolve_path(config_path.parent, output["root"])
        output_root.mkdir(parents=True, exist_ok=True)

        cache_signature = taxonomy_cache_signature(label_specs)
        cache_path = output_root / "taxonomy_labels.sqlite"
        build_taxonomy_cache(
            cache_path, label_specs, cache_signature, args.rebuild_taxonomy_cache
        )
        connection = sqlite3.connect(cache_path)
        source_stats = []
        try:
            for name in names:
                taxonomy_rows = load_taxonomy_source(connection, name)
                if not taxonomy_rows:
                    raise ValueError(f"taxonomy cache has no rows for {name}")
                stats = process_source(
                    name=name,
                    source=registry[name],
                    quality_spec=quality[name],
                    settings=quality_settings,
                    taxonomy=taxonomy,
                    taxonomy_source_path=taxonomy_sources[name],
                    taxonomy_rows=taxonomy_rows,
                    taxonomy_signature=cache_signature,
                    output_root=output_root,
                    output=output,
                    resume=bool(output.get("resume", True)) and not args.force,
                )
                source_stats.append(stats)
                print(
                    compact_json(
                        {"event": "source_complete", "dataset": name, **stats["counts"]}
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            connection.close()
        summary = write_reports(output_root, taxonomy, source_stats)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports a concise failure.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

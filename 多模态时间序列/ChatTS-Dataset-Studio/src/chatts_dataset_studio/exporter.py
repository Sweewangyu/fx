from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .catalog import _validate_annotation
from .models import ANNOTATION_FIELDS, DEFAULT_ALIASES, Source, StageRule, StudioError, safe_name

ProgressCallback = Callable[[dict[str, Any]], None]
_MISSING = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StudioError(f"Record cannot be encoded as strict JSON: {exc}") from exc


def _hash_object(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _iter_jsonl_with_digest(path: Path, digest: Any) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                raise StudioError(f"Blank JSONL line at {path}:{line_number}")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StudioError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise StudioError(f"JSONL row is not an object at {path}:{line_number}")
            yield line_number, value


def _validate_qa(record: dict[str, Any], source: Source, line_number: int) -> dict[str, Any]:
    missing = {key for key in ("input", "timeseries", "output") if key not in record}
    if missing:
        raise StudioError(f"QA row lacks {sorted(missing)} at {source.path}:{line_number}")
    if not isinstance(record["input"], str) or not isinstance(record["output"], str):
        raise StudioError(f"QA input/output must be strings at {source.path}:{line_number}")
    if not isinstance(record["timeseries"], list):
        raise StudioError(f"QA timeseries must be a list at {source.path}:{line_number}")
    return {"input": record["input"], "timeseries": record["timeseries"], "output": record["output"]}


def _annotation_only(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in sorted(ANNOTATION_FIELDS) if key in record}


def _iter_joined(source: Source) -> tuple[Iterator[tuple[int, dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    raw_digest = hashlib.sha256()
    annotation_digest = hashlib.sha256()
    identities: dict[str, Any] = {
        "raw_path": str(source.path),
        "annotation_path": str(source.annotation_path),
        "annotation_mode": source.annotation_mode,
        "raw_sha256": None,
        "annotation_sha256": None,
    }

    def rows() -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
        if source.annotation_mode == "sidecar":
            raw_rows = _iter_jsonl_with_digest(source.path, raw_digest)
            annotation_rows = _iter_jsonl_with_digest(source.annotation_path, annotation_digest)
            for raw_item, annotation_item in zip_longest(
                raw_rows, annotation_rows, fillvalue=_MISSING
            ):
                if raw_item is _MISSING or annotation_item is _MISSING:
                    raise StudioError(
                        f"Raw QA and annotation line counts differ for {source.name}"
                    )
                raw_line, raw_record = raw_item
                annotation_line, annotation = annotation_item
                if raw_line != annotation_line:
                    raise StudioError(f"Raw/annotation line mismatch for {source.name}")
                _validate_annotation(annotation, source, annotation_line)
                yield raw_line, raw_record, annotation
            identities["raw_sha256"] = raw_digest.hexdigest()
            identities["annotation_sha256"] = annotation_digest.hexdigest()
            return
        if source.annotation_mode == "annotated":
            for line_number, record in _iter_jsonl_with_digest(
                source.annotation_path, annotation_digest
            ):
                _validate_annotation(record, source, line_number)
                yield line_number, record, _annotation_only(record)
            identities["annotation_sha256"] = annotation_digest.hexdigest()
            identities["raw_sha256"] = "embedded-in-annotated-file"
            return
        raise StudioError(f"No merged annotations are available for {source.name}")

    return rows(), identities


def parse_rules(
    payload: dict[str, Any],
    sources: list[Source],
    catalog: dict[str, Any] | None = None,
) -> tuple[StageRule, StageRule]:
    available = {source.name for source in sources if source.annotation_mode != "missing"}
    wildcard_sources = available
    if catalog is not None:
        wildcard_sources = {
            item["name"]
            for item in catalog.get("sources", [])
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and item.get("available") is True
            and item["name"] in available
        }

    def expand_all(value: Any) -> Any:
        if not isinstance(value, dict) or "*" not in value.get("sources", []):
            return value
        if value.get("sources") != ["*"]:
            raise StudioError("The '*' source selector must be used alone")
        return {**value, "sources": sorted(wildcard_sources)}

    return (
        StageRule.from_mapping(expand_all(payload.get("stage1")), available),
        StageRule.from_mapping(expand_all(payload.get("stage2")), available),
    )


def preview_selection(
    catalog: dict[str, Any], stage1: StageRule, stage2: StageRule
) -> dict[str, Any]:
    source_summaries = {item["name"]: item for item in catalog["sources"]}
    selected = stage1.sources | stage2.sources
    unavailable = {
        name: source_summaries.get(name, {}).get("errors", ["source is not in catalog"])
        for name in selected
        if not source_summaries.get(name, {}).get("available")
    }
    if unavailable:
        raise StudioError(f"Selected datasets are unavailable: {unavailable}")

    stage_totals = {"stage1": 0, "stage2": 0, "overlap": 0}
    stage_quality = {"stage1": Counter(), "stage2": Counter()}
    stage_difficulty = {"stage1": Counter(), "stage2": Counter()}
    stage_ability = {"stage1": Counter(), "stage2": Counter()}
    rows = []
    for name, summary in source_summaries.items():
        counts = {"stage1": 0, "stage2": 0, "overlap": 0}
        for packed, count in summary.get("cube", {}).items():
            quality, difficulty, ability = packed.split("\u001f", 2)
            annotation = {
                "quality": quality,
                "difficulty": difficulty,
                "ability_bucket": ability,
            }
            in_stage1 = stage1.matches(name, annotation)
            in_stage2 = stage2.matches(name, annotation)
            if in_stage1:
                counts["stage1"] += count
                stage_quality["stage1"][quality] += count
                stage_difficulty["stage1"][difficulty] += count
                stage_ability["stage1"][ability] += count
            if in_stage2:
                counts["stage2"] += count
                stage_quality["stage2"][quality] += count
                stage_difficulty["stage2"][difficulty] += count
                stage_ability["stage2"][ability] += count
            if in_stage1 and in_stage2:
                counts["overlap"] += count
        for key in stage_totals:
            stage_totals[key] += counts[key]
        if name in selected:
            rows.append({"source": name, "source_rows": summary["rows"], **counts})
    if stage1.sources and stage_totals["stage1"] == 0:
        raise StudioError("Stage1 filters select zero rows")
    if stage2.sources and stage_totals["stage2"] == 0:
        raise StudioError("Stage2 filters select zero rows")
    return {
        "counts": stage_totals,
        "by_source": rows,
        "distributions": {
            stage: {
                "quality": dict(stage_quality[stage]),
                "difficulty": dict(stage_difficulty[stage]),
                "ability": dict(stage_ability[stage]),
            }
            for stage in ("stage1", "stage2")
        },
    }


def _dataset_key(stage: str, source_name: str, data_version: str | None = None) -> str:
    if data_version is not None:
        # Versioned snapshots use the complete source name.  The original six
        # aliases are intentionally not used here: an all-source registry can
        # contain both e.g. `chatts_sft` and `sft`, which would otherwise map to
        # the same LLaMAFactory dataset key.
        return f"{data_version}__{stage}__{source_name}"
    return f"{stage}_{DEFAULT_ALIASES.get(source_name, source_name)}"


def _dataset_info_entry(stage: str, source_name: str) -> dict[str, Any]:
    return {
        "file_name": f"{stage}/{source_name}.jsonl",
        "columns": {
            "prompt": "input",
            "response": "output",
            "timeseries": "timeseries",
        },
        "description": (
            f"ChatTS Dataset Studio selection: {source_name} for {stage}; "
            "quality/difficulty labels are retained in a separate audit sidecar"
        ),
    }


def _actual_preview(
    selected_names: list[str],
    source_counts: dict[str, dict[str, int]],
    stage_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = {
        key: sum(source_counts[name][key] for name in selected_names)
        for key in ("stage1", "stage2", "overlap")
    }
    distributions: dict[str, dict[str, dict[str, int]]] = {}
    for stage in ("stage1", "stage2"):
        quality: Counter[str] = Counter()
        difficulty: Counter[str] = Counter()
        ability: Counter[str] = Counter()
        for details in stage_stats[stage].values():
            for packed, row_count in details["cube"].items():
                quality_name, difficulty_name, ability_name = packed.split("\u001f", 2)
                quality[quality_name] += row_count
                difficulty[difficulty_name] += row_count
                ability[ability_name] += row_count
        distributions[stage] = {
            "quality": dict(quality),
            "difficulty": dict(difficulty),
            "ability": dict(ability),
        }
    return {
        "counts": counts,
        "by_source": [
            {
                "source": name,
                "source_rows": source_counts[name]["source_rows"],
                "stage1": source_counts[name]["stage1"],
                "stage2": source_counts[name]["stage2"],
                "overlap": source_counts[name]["overlap"],
            }
            for name in selected_names
        ],
        "distributions": distributions,
    }


def _preview_signature(preview: dict[str, Any]) -> dict[str, Any]:
    return {
        "counts": preview.get("counts"),
        "by_source": sorted(preview.get("by_source", []), key=lambda row: row["source"]),
        "distributions": preview.get("distributions"),
    }


def export_selection(
    payload: dict[str, Any],
    sources: list[Source],
    catalog: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    stage1, stage2 = parse_rules(payload, sources, catalog)
    preview = preview_selection(catalog, stage1, stage2)
    run_name = safe_name(payload.get("run_name"), "run_name")
    data_version_value = payload.get("data_version")
    data_version = None
    if data_version_value is not None:
        data_version = safe_name(data_version_value, "data_version")
        if data_version != run_name or not re.fullmatch(r"datav[1-9][0-9]*", data_version):
            raise StudioError("A versioned export must use the same canonical datavN run_name")
    output_root_value = payload.get("output_root")
    if not isinstance(output_root_value, str) or not output_root_value.strip():
        raise StudioError("output_root must be a non-empty path")
    output_root = Path(output_root_value).expanduser().resolve()
    final_root = output_root / run_name
    if final_root.exists():
        raise StudioError(f"Export destination already exists; choose a new run_name: {final_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{run_name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    temporary.mkdir()

    source_map = {source.name: source for source in sources}
    selected_names = [
        source.name for source in sources if source.name in stage1.sources | stage2.sources
    ]
    dataset_info: dict[str, Any] = {}
    dataset_names = {"stage1": [], "stage2": []}
    stage_stats: dict[str, dict[str, Any]] = {"stage1": {}, "stage2": {}}
    actual_source_counts: dict[str, dict[str, int]] = {}
    input_identities: dict[str, Any] = {}
    file_hashes: dict[str, str] = {}

    for stage in ("stage1", "stage2"):
        (temporary / stage).mkdir()
        (temporary / f"{stage}_annotations").mkdir()

    total_input_rows = sum(
        next(item["rows"] for item in catalog["sources"] if item["name"] == name)
        for name in selected_names
    )
    processed = 0
    try:
        for source_index, name in enumerate(selected_names, 1):
            source = source_map[name]
            joined, identities = _iter_joined(source)
            outputs: dict[str, Any] = {}
            annotation_outputs: dict[str, Any] = {}
            digests: dict[str, Any] = {}
            annotation_digests: dict[str, Any] = {}
            counts = {"stage1": 0, "stage2": 0, "overlap": 0}
            source_rows = 0
            cubes = {"stage1": Counter(), "stage2": Counter()}
            for stage, rule in (("stage1", stage1), ("stage2", stage2)):
                if name not in rule.sources:
                    continue
                output_path = temporary / stage / f"{name}.jsonl"
                annotation_output_path = temporary / f"{stage}_annotations" / f"{name}.jsonl"
                outputs[stage] = output_path.open("w", encoding="utf-8")
                annotation_outputs[stage] = annotation_output_path.open("w", encoding="utf-8")
                digests[stage] = hashlib.sha256()
                annotation_digests[stage] = hashlib.sha256()
            try:
                for line_number, raw_record, annotation in joined:
                    source_rows += 1
                    qa = _validate_qa(raw_record, source, line_number)
                    matches = {
                        "stage1": stage1.matches(name, annotation),
                        "stage2": stage2.matches(name, annotation),
                    }
                    if matches["stage1"] and matches["stage2"]:
                        counts["overlap"] += 1
                    for stage in ("stage1", "stage2"):
                        if not matches[stage]:
                            continue
                        qa_line = _canonical(qa) + "\n"
                        annotation_line = _canonical(annotation) + "\n"
                        outputs[stage].write(qa_line)
                        annotation_outputs[stage].write(annotation_line)
                        digests[stage].update(qa_line.encode())
                        annotation_digests[stage].update(annotation_line.encode())
                        counts[stage] += 1
                        cubes[stage][
                            f"{annotation['quality']}\u001f{annotation['difficulty']}\u001f"
                            f"{annotation.get('ability_bucket') or annotation.get('ability_label') or 'UNMAPPED'}"
                        ] += 1
                    processed += 1
                    if progress and (processed % 1000 == 0 or processed == total_input_rows):
                        progress(
                            {
                                "phase": "exporting",
                                "source": name,
                                "source_index": source_index,
                                "source_count": len(selected_names),
                                "processed_rows": processed,
                                "total_rows": total_input_rows,
                            }
                        )
            finally:
                for stream in (*outputs.values(), *annotation_outputs.values()):
                    stream.close()

            input_identities[name] = identities
            actual_source_counts[name] = {"source_rows": source_rows, **counts}
            for stage in ("stage1", "stage2"):
                if stage not in outputs:
                    continue
                qa_relative = f"{stage}/{name}.jsonl"
                annotation_relative = f"{stage}_annotations/{name}.jsonl"
                file_hashes[qa_relative] = digests[stage].hexdigest()
                file_hashes[annotation_relative] = annotation_digests[stage].hexdigest()
                stage_stats[stage][name] = {
                    "rows": counts[stage],
                    "overlap_rows": counts["overlap"],
                    "cube": dict(cubes[stage]),
                    "qa_sha256": digests[stage].hexdigest(),
                    "annotation_sha256": annotation_digests[stage].hexdigest(),
                }
                if counts[stage] > 0:
                    key = _dataset_key(stage, name, data_version)
                    if key in dataset_info:
                        raise StudioError(
                            f"Dataset key collision for {name}: {key}; source names must be unique"
                        )
                    dataset_info[key] = _dataset_info_entry(stage, name)
                    dataset_names[stage].append(key)

        actual_preview = _actual_preview(selected_names, actual_source_counts, stage_stats)
        if _preview_signature(actual_preview) != _preview_signature(preview):
            raise StudioError(
                "Source data changed after the catalog preview; rescan the catalog and retry"
            )
        # Manifest counts and distributions are always produced from the rows
        # actually read during export, even after the equality check above.
        preview = actual_preview

        selection = {
            "stage1": {
                "sources": sorted(stage1.sources),
                "qualities": sorted(stage1.qualities),
                "difficulties": sorted(stage1.difficulties),
                "abilities": sorted(stage1.abilities),
            },
            "stage2": {
                "sources": sorted(stage2.sources),
                "qualities": sorted(stage2.qualities),
                "difficulties": sorted(stage2.difficulties),
                "abilities": sorted(stage2.abilities),
            },
        }
        selection_hash = _hash_object(selection)
        content_identities = {
            name: {
                "annotation_mode": identity["annotation_mode"],
                "raw_sha256": identity["raw_sha256"],
                "annotation_sha256": identity["annotation_sha256"],
            }
            for name, identity in input_identities.items()
        }
        snapshot_payload = {
            "schema_version": "chatts-dataset-snapshot-v2",
            "selection": selection,
            # Absolute source paths are audit metadata, not content identity.
            # This keeps a snapshot hash stable after copying the same inputs
            # between local and HPC workspaces.
            "inputs": content_identities,
            "selected_outputs": {
                path: digest
                for path, digest in file_hashes.items()
                if path.endswith(".jsonl")
            },
            "counts": preview["counts"],
        }
        dataset_snapshot_hash = _hash_object(snapshot_payload)
        for stage in ("stage1", "stage2"):
            stage_manifest = {
                "schema_version": "chatts-dataset-stage-v1",
                "stage": stage,
                "selection_hash": selection_hash,
                "dataset_snapshot_hash": dataset_snapshot_hash,
                "rule": selection[stage],
                "dataset_names": dataset_names[stage],
                "total_rows": sum(item["rows"] for item in stage_stats[stage].values()),
                "sources": stage_stats[stage],
            }
            path = temporary / stage / "manifest.json"
            encoded = json.dumps(stage_manifest, ensure_ascii=False, indent=2) + "\n"
            path.write_text(encoded, encoding="utf-8")
            file_hashes[f"{stage}/manifest.json"] = hashlib.sha256(encoded.encode()).hexdigest()
        dataset_info_path = temporary / "dataset_info.json"
        dataset_info_encoded = json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n"
        dataset_info_path.write_text(dataset_info_encoded, encoding="utf-8")
        file_hashes["dataset_info.json"] = hashlib.sha256(dataset_info_encoded.encode()).hexdigest()
        env_lines = [
            f"DATASET_DIR={shlex.quote(str(final_root))}",
            f"STAGE1_DATASETS={shlex.quote(','.join(dataset_names['stage1']))}",
            f"STAGE2_DATASETS={shlex.quote(','.join(dataset_names['stage2']))}",
            "STAGE1_MIX_STRATEGY=concat",
            "STAGE2_MIX_STRATEGY=concat",
            "STAGE1_INTERLEAVE_PROBS=''",
            "STAGE2_INTERLEAVE_PROBS=''",
            f"DATASET_SNAPSHOT_HASH={dataset_snapshot_hash}",
        ]
        if data_version is not None:
            env_lines.append(f"DATA_VERSION={data_version}")
        env_encoded = "\n".join(env_lines) + "\n"
        (temporary / "training.env").write_text(env_encoded, encoding="utf-8")
        file_hashes["training.env"] = hashlib.sha256(env_encoded.encode()).hexdigest()
        manifest = {
            "schema_version": "chatts-dataset-studio-export-v1",
            "run_name": run_name,
            "data_version": data_version,
            "created_at": _utc_now(),
            "selection_hash": selection_hash,
            "dataset_snapshot_hash": dataset_snapshot_hash,
            "snapshot_hash_schema": "chatts-dataset-snapshot-v2",
            "selection": selection,
            "preview": preview,
            "dataset_names": dataset_names,
            "input_identities": input_identities,
            "files": file_hashes,
        }
        manifest["manifest_hash"] = _hash_object(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(final_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    result = {
        "status": "completed",
        "output_dir": str(final_root),
        "manifest": str(final_root / "manifest.json"),
        "dataset_info": str(final_root / "dataset_info.json"),
        "training_env": str(final_root / "training.env"),
        "counts": preview["counts"],
        "dataset_names": dataset_names,
    }
    if progress:
        progress({"phase": "completed", **result})
    return result

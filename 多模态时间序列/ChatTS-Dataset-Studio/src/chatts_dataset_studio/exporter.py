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
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .catalog import _validate_annotation, source_signatures
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

    source_dimensions: dict[str, dict[str, set[str]]] | None = None
    if catalog is not None:
        source_dimensions = {
            item["name"]: {
                "qualities": set(item.get("quality", {})),
                "difficulties": set(item.get("difficulty", {})),
                "abilities": set(item.get("ability", {})),
            }
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
        StageRule.from_mapping(
            expand_all(payload.get("stage1")), available, source_dimensions
        ),
        StageRule.from_mapping(
            expand_all(payload.get("stage2")), available, source_dimensions
        ),
    )


def canonical_selection(stage1: StageRule, stage2: StageRule) -> dict[str, Any]:
    """Return the single canonical selection representation used by every manifest."""

    return {"stage1": stage1.to_mapping(), "stage2": stage2.to_mapping()}


def _sample_size(total: int, sample_percent: int | float) -> int:
    if sample_percent == 100:
        return total
    scaled = Decimal(total) * Decimal(str(sample_percent)) / Decimal(100)
    rounded = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
    # A selected source with a non-empty filter must never silently disappear
    # just because its percentage rounds below one row.
    return min(total, max(1, rounded))


def _quota_tie_key(stage: str, source: str, bucket: str) -> str:
    value = f"chatts-dataset-quota-v1\0{stage}\0{source}\0{bucket}"
    return hashlib.sha256(value.encode()).hexdigest()


def _allocate_bucket_quotas(
    counts: dict[str, int],
    sample_percent: int | float,
    *,
    stage: str,
    source: str,
) -> dict[str, int]:
    total = sum(counts.values())
    if total == 0:
        return {}
    target = _sample_size(total, sample_percent)
    if target == total:
        return dict(counts)

    quotas: dict[str, int] = {}
    remainders: list[tuple[Decimal, str, str]] = []
    for bucket, count in sorted(counts.items()):
        exact = Decimal(count) * Decimal(target) / Decimal(total)
        base = int(exact.to_integral_value(rounding=ROUND_FLOOR))
        quotas[bucket] = base
        remainders.append((exact - Decimal(base), _quota_tie_key(stage, source, bucket), bucket))
    remaining = target - sum(quotas.values())
    for _, _, bucket in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        quotas[bucket] += 1
    return quotas


def _sampling_plan(
    catalog: dict[str, Any], stage1: StageRule, stage2: StageRule
) -> dict[str, dict[str, dict[str, Any]]]:
    source_summaries = {item["name"]: item for item in catalog["sources"]}
    selected = stage1.sources | stage2.sources
    unavailable = {
        name: source_summaries.get(name, {}).get("errors", ["source is not in catalog"])
        for name in selected
        if not source_summaries.get(name, {}).get("available")
    }
    if unavailable:
        raise StudioError(f"Selected datasets are unavailable: {unavailable}")

    plan: dict[str, dict[str, dict[str, Any]]] = {"stage1": {}, "stage2": {}}
    for stage, rule in (("stage1", stage1), ("stage2", stage2)):
        empty_sources = []
        for source in sorted(rule.sources):
            filtered = {
                packed: count
                for packed, count in source_summaries[source].get("cube", {}).items()
                if rule.matches(
                    source,
                    {
                        "quality": packed.split("\u001f", 2)[0],
                        "difficulty": packed.split("\u001f", 2)[1],
                        "ability_bucket": packed.split("\u001f", 2)[2],
                    },
                )
            }
            if not filtered:
                empty_sources.append(source)
                continue
            effective = rule.effective_rule(source)
            plan[stage][source] = {
                "filtered": filtered,
                "selected": _allocate_bucket_quotas(
                    filtered,
                    effective.sample_percent,
                    stage=stage,
                    source=source,
                ),
                "sample_percent": effective.sample_percent,
            }
        if empty_sources:
            raise StudioError(
                f"{stage.capitalize()} filters select zero rows for selected datasets: "
                f"{empty_sources}"
            )
    return plan


def _percentages(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {
        name: round(count * 100 / total, 6)
        for name, count in sorted(counts.items())
    }


def _preview_from_plan(
    catalog: dict[str, Any],
    stage1: StageRule,
    stage2: StageRule,
    plan: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    source_summaries = {item["name"]: item for item in catalog["sources"]}
    selected_sources = stage1.sources | stage2.sources
    stage_totals = {"stage1": 0, "stage2": 0, "overlap": 0}
    filtered_totals = {"stage1": 0, "stage2": 0}
    stage_quality = {"stage1": Counter(), "stage2": Counter()}
    stage_difficulty = {"stage1": Counter(), "stage2": Counter()}
    stage_ability = {"stage1": Counter(), "stage2": Counter()}
    rows = []

    for name, summary in source_summaries.items():
        if name not in selected_sources:
            continue
        counts = {"stage1": 0, "stage2": 0, "overlap": 0}
        filtered_counts = {"stage1": 0, "stage2": 0}
        selected_by_stage: dict[str, dict[str, int]] = {}
        for stage in ("stage1", "stage2"):
            details = plan[stage].get(name)
            selected_by_stage[stage] = details["selected"] if details else {}
            if not details:
                continue
            filtered_counts[stage] = sum(details["filtered"].values())
            filtered_totals[stage] += filtered_counts[stage]
            for packed, count in details["selected"].items():
                quality, difficulty, ability = packed.split("\u001f", 2)
                counts[stage] += count
                stage_quality[stage][quality] += count
                stage_difficulty[stage][difficulty] += count
                stage_ability[stage][ability] += count
        counts["overlap"] = sum(
            min(count, selected_by_stage["stage2"].get(packed, 0))
            for packed, count in selected_by_stage["stage1"].items()
        )
        for key in stage_totals:
            stage_totals[key] += counts[key]
        rows.append(
            {
                "source": name,
                "source_rows": summary["rows"],
                "stage1_filtered": filtered_counts["stage1"],
                "stage2_filtered": filtered_counts["stage2"],
                **counts,
            }
        )

    return {
        "counts": stage_totals,
        "filtered_counts": filtered_totals,
        "by_source": rows,
        "distributions": {
            stage: {
                "quality": dict(stage_quality[stage]),
                "difficulty": dict(stage_difficulty[stage]),
                "ability": dict(stage_ability[stage]),
                "ability_percentages": _percentages(stage_ability[stage]),
            }
            for stage in ("stage1", "stage2")
        },
    }


def preview_selection(
    catalog: dict[str, Any], stage1: StageRule, stage2: StageRule
) -> dict[str, Any]:
    plan = _sampling_plan(catalog, stage1, stage2)
    return _preview_from_plan(catalog, stage1, stage2, plan)


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
    filtered_counts = {
        stage: sum(source_counts[name][f"{stage}_filtered"] for name in selected_names)
        for stage in ("stage1", "stage2")
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
            "ability_percentages": _percentages(ability),
        }
    return {
        "counts": counts,
        "filtered_counts": filtered_counts,
        "by_source": [
            {
                "source": name,
                "source_rows": source_counts[name]["source_rows"],
                "stage1_filtered": source_counts[name]["stage1_filtered"],
                "stage2_filtered": source_counts[name]["stage2_filtered"],
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
        "filtered_counts": preview.get("filtered_counts"),
        "by_source": sorted(preview.get("by_source", []), key=lambda row: row["source"]),
        "distributions": preview.get("distributions"),
    }


def _annotation_bucket(annotation: dict[str, Any]) -> str:
    ability = annotation.get("ability_bucket") or annotation.get("ability_label") or "UNMAPPED"
    return f"{annotation['quality']}\u001f{annotation['difficulty']}\u001f{ability}"


_HASH_SPACE = 1 << 256


def _sampling_hash(
    stream: str,
    source: str,
    bucket: str,
    line_number: int,
) -> int:
    value = (
        f"chatts-dataset-online-sampler-v2\0{stream}\0{source}\0"
        f"{bucket}\0{line_number}"
    )
    return int.from_bytes(hashlib.sha256(value.encode()).digest(), "big")


@dataclass
class _ExactOnlineChooser:
    population: int
    target: int
    stream: str
    source: str
    bucket: str
    seen: int = 0
    selected: int = 0

    def choose(self, line_number: int) -> bool:
        if self.seen >= self.population:
            raise StudioError(
                "Source data changed after the catalog preview; rescan the catalog and retry"
            )
        remaining_population = self.population - self.seen
        remaining_target = self.target - self.selected
        if remaining_target == 0:
            choose = False
        elif remaining_target == remaining_population:
            choose = True
        else:
            random_value = _sampling_hash(
                self.stream,
                self.source,
                self.bucket,
                line_number,
            )
            choose = (
                random_value * remaining_population
                < remaining_target * _HASH_SPACE
            )
        self.seen += 1
        if choose:
            self.selected += 1
        return choose

    def finalize(self) -> None:
        if self.seen != self.population or self.selected != self.target:
            raise StudioError(
                "Source data changed after the catalog preview; rescan the catalog and retry"
            )


@dataclass
class _NestedBucketSampler:
    stage1_quota: int
    stage2_quota: int
    primary: _ExactOnlineChooser
    secondary: _ExactOnlineChooser | None

    def choose(self, line_number: int) -> tuple[bool, bool]:
        if not self.primary.choose(line_number):
            return False, False
        if self.stage1_quota == self.stage2_quota:
            return True, True
        assert self.secondary is not None
        selected_for_smaller = self.secondary.choose(line_number)
        if self.stage1_quota > self.stage2_quota:
            return True, selected_for_smaller
        return selected_for_smaller, True

    def finalize(self) -> None:
        self.primary.finalize()
        if self.secondary is not None:
            self.secondary.finalize()


class _OnlineNestedSampler:
    """One-pass exact sampler with O(number of selected cubes) scalar state."""

    def __init__(
        self,
        source: str,
        plan: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        self.source = source
        self.bucket_states: dict[str, _NestedBucketSampler] = {}
        stage1 = plan["stage1"].get(source)
        stage2 = plan["stage2"].get(source)
        buckets = set(stage1["filtered"] if stage1 else ()) | set(
            stage2["filtered"] if stage2 else ()
        )
        for bucket in sorted(buckets):
            populations = {
                details["filtered"][bucket]
                for details in (stage1, stage2)
                if details is not None and bucket in details["filtered"]
            }
            if len(populations) != 1:
                raise StudioError(f"Inconsistent catalog cube count for {source}: {bucket}")
            population = populations.pop()
            stage1_quota = stage1["selected"].get(bucket, 0) if stage1 else 0
            stage2_quota = stage2["selected"].get(bucket, 0) if stage2 else 0
            maximum = max(stage1_quota, stage2_quota)
            minimum = min(stage1_quota, stage2_quota)
            primary = _ExactOnlineChooser(
                population,
                maximum,
                "primary",
                source,
                bucket,
            )
            secondary = None
            if stage1_quota != stage2_quota:
                secondary = _ExactOnlineChooser(
                    maximum,
                    minimum,
                    "secondary",
                    source,
                    bucket,
                )
            self.bucket_states[bucket] = _NestedBucketSampler(
                stage1_quota,
                stage2_quota,
                primary,
                secondary,
            )

    @property
    def state_size(self) -> int:
        return len(self.bucket_states)

    def choose(self, bucket: str, line_number: int) -> tuple[bool, bool]:
        state = self.bucket_states.get(bucket)
        if state is None:
            return False, False
        return state.choose(line_number)

    def finalize(self) -> None:
        for state in self.bucket_states.values():
            state.finalize()


def export_selection(
    payload: dict[str, Any],
    sources: list[Source],
    catalog: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    stage1, stage2 = parse_rules(payload, sources, catalog)
    sampling_plan = _sampling_plan(catalog, stage1, stage2)
    preview = _preview_from_plan(catalog, stage1, stage2, sampling_plan)
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
    source_summary_map = {item["name"]: item for item in catalog["sources"]}
    expected_signatures = {
        item["name"]: item
        for item in catalog.get("signatures", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
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
            expected_signature = expected_signatures.get(name)
            if expected_signature is None or source_signatures([source])[0] != expected_signature:
                raise StudioError(
                    f"Source identity changed after the catalog preview for {name}; "
                    "rescan the catalog and retry"
                )
            sampler = _OnlineNestedSampler(name, sampling_plan)
            joined, identities = _iter_joined(source)
            outputs: dict[str, Any] = {}
            annotation_outputs: dict[str, Any] = {}
            digests: dict[str, Any] = {}
            annotation_digests: dict[str, Any] = {}
            counts = {"stage1": 0, "stage2": 0, "overlap": 0}
            filtered_counts = {"stage1": 0, "stage2": 0}
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
                    bucket = _annotation_bucket(annotation)
                    filter_matches = {
                        "stage1": stage1.matches(name, annotation),
                        "stage2": stage2.matches(name, annotation),
                    }
                    sampled_stage1, sampled_stage2 = sampler.choose(bucket, line_number)
                    matches = {
                        "stage1": sampled_stage1,
                        "stage2": sampled_stage2,
                    }
                    for stage in ("stage1", "stage2"):
                        if filter_matches[stage]:
                            filtered_counts[stage] += 1
                        if matches[stage] and not filter_matches[stage]:
                            raise StudioError(
                                f"Sampling plan selected a row outside {stage} filters for {name}"
                            )
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
                        cubes[stage][bucket] += 1
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

            sampler.finalize()
            if source_rows != source_summary_map[name]["rows"]:
                raise StudioError(
                    "Source data changed after the catalog preview; rescan the catalog and retry"
                )
            for stage in ("stage1", "stage2"):
                details = sampling_plan[stage].get(name)
                expected_filtered = sum(details["filtered"].values()) if details else 0
                if filtered_counts[stage] != expected_filtered:
                    raise StudioError(
                        "Source data changed after the catalog preview; "
                        "rescan the catalog and retry"
                    )
            if source_signatures([source])[0] != expected_signature:
                raise StudioError(
                    f"Source identity changed during export for {name}; retry the export"
                )
            input_identities[name] = identities
            actual_source_counts[name] = {
                "source_rows": source_rows,
                "stage1_filtered": filtered_counts["stage1"],
                "stage2_filtered": filtered_counts["stage2"],
                **counts,
            }
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

        selection = canonical_selection(stage1, stage2)
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

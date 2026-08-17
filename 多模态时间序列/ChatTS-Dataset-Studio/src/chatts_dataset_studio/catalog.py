from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .models import DIFFICULTY_LEVELS, QUALITY_LEVELS, Source, StudioError

ABILITY_CODES = (
    "PR",
    "NU",
    "AD",
    "CA",
    "ER",
    "CD",
    "AR",
    "TR",
    "NR",
    "DR",
    "IR",
    "TSF",
    "EP",
    "QualDM",
    "QuantDM",
)
ABILITY_NAMES = (
    "pattern_recognition",
    "noise_understanding",
    "anomaly_detection",
    "comparative_analysis",
    "etiological_reasoning",
    "causal_discovery",
    "abductive_reasoning",
    "temporal_relation_reasoning",
    "numerical_reasoning",
    "deductive_reasoning",
    "inductive_reasoning",
    "time_series_forecasting",
    "event_prediction",
    "qualitative_decision_making",
    "quantitative_decision_making",
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StudioError(f"Invalid JSON in {path}: {exc}") from exc


def _resolve_source_path(registry_path: Path, raw_path: str, data_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates: list[Path] = []
    if data_root is not None:
        candidates.append(data_root / path)
    candidates.extend(parent / path for parent in registry_path.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    root_hint = data_root if data_root is not None else registry_path.parent
    return (root_hint / path).resolve()


def _annotation_location(root: Path, name: str) -> tuple[Path, str]:
    if root.name == "annotations":
        sidecar = root / f"{name}.jsonl"
        annotated = root.parent / "annotated" / f"{name}.jsonl"
    elif root.name == "annotated":
        annotated = root / f"{name}.jsonl"
        sidecar = root.parent / "annotations" / f"{name}.jsonl"
    else:
        sidecar = root / "annotations" / f"{name}.jsonl"
        annotated = root / "annotated" / f"{name}.jsonl"
    if sidecar.is_file():
        return sidecar.resolve(), "sidecar"
    if annotated.is_file():
        return annotated.resolve(), "annotated"
    return sidecar.resolve(), "missing"


def load_sources(
    registry_path: str | Path,
    annotations_root: str | Path,
    data_root: str | Path | None = None,
) -> list[Source]:
    if not isinstance(registry_path, (str, Path)) or not str(registry_path).strip():
        raise StudioError("registry_path is required")
    if not isinstance(annotations_root, (str, Path)) or not str(annotations_root).strip():
        raise StudioError("annotations_root is required")
    registry = Path(registry_path).expanduser().resolve()
    annotation_root = Path(annotations_root).expanduser().resolve()
    explicit_root = Path(data_root).expanduser().resolve() if data_root else None
    if not registry.is_file():
        raise StudioError(f"Dataset registry does not exist: {registry}")
    payload = _load_json(registry)
    rows = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise StudioError(f"Dataset registry has no sources list: {registry}")
    result: list[Source] = []
    names: set[str] = set()
    for item in rows:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise StudioError("Every registry source must contain a string name")
        name = item["name"]
        if name in names:
            raise StudioError(f"Duplicate source name in registry: {name}")
        names.add(name)
        if not isinstance(item.get("path"), str):
            raise StudioError(f"Registry source {name} has no string path")
        source_path = _resolve_source_path(registry, item["path"], explicit_root)
        annotation_path, mode = _annotation_location(annotation_root, name)
        result.append(
            Source(
                name=name,
                path=source_path,
                family=str(item.get("family", "unknown")),
                split=str(item.get("split", "train")),
                training_role=str(item.get("training_role", "unknown")),
                annotation_path=annotation_path,
                annotation_mode=mode,
            )
        )
    return result


def _file_signature(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
    }


def source_signatures(sources: list[Source]) -> list[dict[str, Any]]:
    return [
        {
            "name": source.name,
            "raw": _file_signature(source.path),
            "annotation": _file_signature(source.annotation_path),
            "annotation_mode": source.annotation_mode,
        }
        for source in sources
    ]


def _validate_annotation(
    annotation: Any, source: Source, line_number: int
) -> tuple[str, str, str]:
    if not isinstance(annotation, dict):
        raise StudioError(f"Annotation is not an object at {source.annotation_path}:{line_number}")
    declared_line = annotation.get("line_number")
    if declared_line is not None and int(declared_line) != line_number:
        raise StudioError(
            f"Annotation line mismatch for {source.name}: file line {line_number}, "
            f"declared {declared_line}"
        )
    declared_source = annotation.get("annotation_source")
    if declared_source is not None and declared_source != source.name:
        raise StudioError(
            f"Annotation source mismatch at {source.annotation_path}:{line_number}: "
            f"{declared_source!r}"
        )
    declared_index = annotation.get("source_index")
    if declared_index is not None:
        try:
            normalized_index = int(declared_index)
        except (TypeError, ValueError) as exc:
            raise StudioError(
                f"Invalid source_index at {source.annotation_path}:{line_number}: "
                f"{declared_index!r}"
            ) from exc
        if normalized_index != line_number - 1:
            raise StudioError(
                f"Annotation source_index mismatch for {source.name}: file line "
                f"{line_number}, declared {declared_index}"
            )
    annotation_id = annotation.get("annotation_id")
    expected_id = f"{source.name}:{line_number}"
    if annotation_id is not None and annotation_id != expected_id:
        raise StudioError(
            f"Annotation id mismatch at {source.annotation_path}:{line_number}: "
            f"expected {expected_id!r}, got {annotation_id!r}"
        )
    quality = annotation.get("quality")
    difficulty = annotation.get("difficulty")
    ability = annotation.get("ability_bucket") or annotation.get("ability_label") or "UNMAPPED"
    if quality not in QUALITY_LEVELS:
        raise StudioError(
            f"Invalid or missing quality at {source.annotation_path}:{line_number}: {quality!r}"
        )
    if difficulty not in DIFFICULTY_LEVELS:
        raise StudioError(
            f"Invalid or missing difficulty at {source.annotation_path}:{line_number}: "
            f"{difficulty!r}"
        )
    if not isinstance(ability, str) or not ability:
        raise StudioError(
            f"Invalid ability bucket at {source.annotation_path}:{line_number}: {ability!r}"
        )
    return quality, difficulty, ability


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise StudioError(f"Blank JSONL line at {path}:{line_number}")
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise StudioError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def scan_source(source: Source) -> dict[str, Any]:
    base = {
        "name": source.name,
        "family": source.family,
        "split": source.split,
        "training_role": source.training_role,
        "raw_path": str(source.path),
        "annotation_path": str(source.annotation_path),
        "annotation_mode": source.annotation_mode,
        "available": False,
        "rows": 0,
        "quality": {},
        "difficulty": {},
        "ability": {},
        "cube": {},
    }
    errors = []
    if source.annotation_mode == "sidecar" and not source.path.is_file():
        errors.append(f"raw QA file is missing: {source.path}")
    if source.annotation_mode == "missing" or not source.annotation_path.is_file():
        errors.append(f"merged annotation file is missing: {source.annotation_path}")
    if errors:
        return {**base, "errors": errors}

    quality_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    ability_counts: Counter[str] = Counter()
    cube: Counter[str] = Counter()
    rows = 0
    try:
        for line_number, annotation in iter_jsonl(source.annotation_path):
            quality, difficulty, ability = _validate_annotation(annotation, source, line_number)
            if source.annotation_mode == "annotated":
                missing = {key for key in ("input", "timeseries", "output") if key not in annotation}
                if missing:
                    raise StudioError(
                        f"Annotated QA lacks {sorted(missing)} at "
                        f"{source.annotation_path}:{line_number}"
                    )
            quality_counts[quality] += 1
            difficulty_counts[difficulty] += 1
            ability_counts[ability] += 1
            cube[f"{quality}\u001f{difficulty}\u001f{ability}"] += 1
            rows += 1
    except (OSError, StudioError) as exc:
        return {**base, "errors": [str(exc)]}
    return {
        **base,
        "available": True,
        "rows": rows,
        "quality": dict(quality_counts),
        "difficulty": dict(difficulty_counts),
        "ability": dict(ability_counts),
        "cube": dict(cube),
        "errors": [],
    }


def scan_catalog(sources: list[Source]) -> dict[str, Any]:
    summaries = [scan_source(source) for source in sources]
    abilities = sorted(
        {
            ability
            for summary in summaries
            if summary["available"]
            for ability in summary["ability"]
        }
    )
    observed_abilities = set(abilities)
    code_matches = len(observed_abilities & set(ABILITY_CODES))
    name_matches = len(observed_abilities & set(ABILITY_NAMES))
    ability_level_mode = "codes" if code_matches > name_matches else "names"
    ability_levels = ABILITY_CODES if ability_level_mode == "codes" else ABILITY_NAMES
    return {
        "sources": summaries,
        "abilities": abilities,
        "ability_levels": list(ability_levels),
        "ability_level_mode": ability_level_mode,
        "ability_extras": sorted(observed_abilities - set(ability_levels)),
        "quality_levels": list(QUALITY_LEVELS),
        "difficulty_levels": list(DIFFICULTY_LEVELS),
        "total_sources": len(summaries),
        "available_sources": sum(summary["available"] for summary in summaries),
        "total_rows": sum(summary["rows"] for summary in summaries if summary["available"]),
        "signatures": source_signatures(sources),
    }


class CatalogCache:
    def __init__(self) -> None:
        self._key: str | None = None
        self._catalog: dict[str, Any] | None = None
        self._sources: list[Source] | None = None

    def get(
        self,
        registry_path: str | Path,
        annotations_root: str | Path,
        data_root: str | Path | None = None,
    ) -> tuple[list[Source], dict[str, Any]]:
        sources = load_sources(registry_path, annotations_root, data_root)
        key = json.dumps(source_signatures(sources), sort_keys=True, separators=(",", ":"))
        if key != self._key:
            self._catalog = scan_catalog(sources)
            self._sources = sources
            self._key = key
        assert self._catalog is not None and self._sources is not None
        return self._sources, self._catalog

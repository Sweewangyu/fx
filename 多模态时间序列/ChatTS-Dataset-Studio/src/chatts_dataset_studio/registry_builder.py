from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import StudioError, safe_name


def _load_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise StudioError(f"Metadata registry does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StudioError(f"Invalid metadata registry JSON in {path}: {exc}") from exc
    rows = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise StudioError(f"Metadata registry has no sources list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise StudioError(f"Invalid source metadata row in {path}")
        result[row["name"]] = row
    return result


def _validate_annotated_file(path: Path) -> None:
    try:
        with path.open(encoding="utf-8") as stream:
            first_line = stream.readline()
    except OSError as exc:
        raise StudioError(f"Cannot read annotated dataset {path}: {exc}") from exc
    if not first_line:
        raise StudioError(f"Annotated dataset is empty: {path}")
    if not first_line.strip():
        raise StudioError(f"Annotated dataset starts with a blank JSONL line: {path}")
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise StudioError(f"Invalid first JSONL row in {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise StudioError(f"First JSONL row is not an object: {path}")
    missing = {field for field in ("input", "timeseries", "output") if field not in record}
    if missing:
        raise StudioError(f"Annotated dataset {path} lacks fields: {sorted(missing)}")


def _registry_path(path: Path, data_root: Path | None) -> str:
    resolved = path.resolve()
    if data_root is None:
        return str(resolved)
    try:
        return resolved.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise StudioError(
            f"Annotated dataset is outside data_root: {resolved} (root: {data_root})"
        ) from exc


def build_registry(
    merged_labels_root: str | Path,
    output_path: str | Path,
    *,
    data_root: str | Path | None = None,
    metadata_registry: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(merged_labels_root).expanduser().resolve()
    annotated_dir = root if root.name == "annotated" else root / "annotated"
    merged_root = root.parent if root.name == "annotated" else root
    sidecar_dir = merged_root / "annotations"
    output = Path(output_path).expanduser().resolve()
    resolved_data_root = Path(data_root).expanduser().resolve() if data_root else None
    metadata_path = (
        Path(metadata_registry).expanduser().resolve() if metadata_registry else None
    )

    if not annotated_dir.is_dir():
        raise StudioError(f"Annotated dataset directory does not exist: {annotated_dir}")
    if resolved_data_root is not None and not resolved_data_root.is_dir():
        raise StudioError(f"data_root does not exist: {resolved_data_root}")
    if output.exists() and not force:
        raise StudioError(f"Registry already exists; pass --force to replace it: {output}")

    annotated_files = sorted(
        path for path in annotated_dir.glob("*.jsonl") if path.is_file()
    )
    if not annotated_files:
        raise StudioError(f"No annotated JSONL files found in {annotated_dir}")

    metadata = _load_metadata(metadata_path)
    sources = []
    with_sidecar = []
    without_sidecar = []
    for path in annotated_files:
        name = safe_name(path.stem, "dataset filename")
        _validate_annotated_file(path)
        prior = metadata.get(name, {})
        sources.append(
            {
                "name": name,
                "path": _registry_path(path, resolved_data_root),
                "family": str(prior.get("family", "merged_labels")),
                "split": str(prior.get("split", "train")),
                "training_role": str(prior.get("training_role", "annotated_qa")),
            }
        )
        if (sidecar_dir / f"{name}.jsonl").is_file():
            with_sidecar.append(name)
        else:
            without_sidecar.append(name)

    payload = {
        "schema_version": "chatts-dataset-studio-registry-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "merged_labels_root": str(merged_root),
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "status": "completed",
        "output": str(output),
        "annotated_dir": str(annotated_dir),
        "source_count": len(sources),
        "with_sidecar_count": len(with_sidecar),
        "without_sidecar": without_sidecar,
        "metadata_reused_count": sum(source["name"] in metadata for source in sources),
    }

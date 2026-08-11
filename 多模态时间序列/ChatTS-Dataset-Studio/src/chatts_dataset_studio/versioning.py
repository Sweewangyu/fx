from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import StudioError

LEDGER_SCHEMA = "chatts-data-version-ledger-v1"
VERSION_RE = re.compile(r"^(?:data-?v)([0-9]+)$", re.IGNORECASE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StudioError(f"Value cannot be encoded as canonical JSON: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Durably replace a file without exposing a partially written value."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_write_bytes(path, encoded.encode("utf-8"))


def normalize_data_version(value: str | int) -> str:
    if isinstance(value, bool):
        raise StudioError("Dataset version must be datavN or data-vN")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        match = VERSION_RE.fullmatch(value.strip())
        if match is None:
            raise StudioError("Dataset version must be datavN or data-vN")
        number = int(match.group(1))
    else:
        raise StudioError("Dataset version must be datavN or data-vN")
    if number < 0:
        raise StudioError("Dataset version number must be non-negative")
    return f"datav{number}"


def version_number(value: str | int) -> int:
    return int(normalize_data_version(value)[5:])


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudioError(f"{label} does not exist: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StudioError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudioError(f"{label} must be one JSON object: {path}")
    return value


def _safe_snapshot_file(snapshot_root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise StudioError(f"Snapshot manifest contains an unsafe file path: {relative!r}")
    root = snapshot_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise StudioError(f"Snapshot file escapes its root: {relative}") from exc
    return candidate


def _validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise StudioError(f"{label} must be a lowercase SHA256 digest")
    return value


def _recompute_snapshot_hash(manifest: dict[str, Any]) -> str:
    files = manifest.get("files")
    preview = manifest.get("preview")
    if not isinstance(files, dict) or not isinstance(preview, dict):
        raise StudioError("Snapshot manifest lacks files or preview")
    counts = preview.get("counts")
    if not isinstance(counts, dict):
        raise StudioError("Snapshot manifest preview lacks counts")
    selected_outputs = {
        path: digest
        for path, digest in files.items()
        if isinstance(path, str) and path.endswith(".jsonl")
    }
    schema = manifest.get("snapshot_hash_schema", "chatts-dataset-snapshot-v1")
    inputs = manifest.get("input_identities")
    if schema == "chatts-dataset-snapshot-v2":
        if not isinstance(inputs, dict):
            raise StudioError("Snapshot manifest has invalid input identities")
        inputs = {
            name: {
                key: identity.get(key)
                for key in ("annotation_mode", "raw_sha256", "annotation_sha256")
            }
            for name, identity in inputs.items()
            if isinstance(name, str) and isinstance(identity, dict)
        }
        if len(inputs) != len(manifest["input_identities"]):
            raise StudioError("Snapshot manifest has invalid input identities")
    elif schema != "chatts-dataset-snapshot-v1":
        raise StudioError(f"Unsupported snapshot hash schema: {schema!r}")
    payload = {
        "schema_version": schema,
        "selection": manifest.get("selection"),
        "inputs": inputs,
        "selected_outputs": selected_outputs,
        "counts": counts,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _stage_composition(
    snapshot_root: Path,
    manifest: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    stage_path = snapshot_root / stage / "manifest.json"
    stage_manifest = _load_json_object(stage_path, f"{stage} manifest")
    if stage_manifest.get("stage") != stage:
        raise StudioError(f"Invalid stage marker in {stage_path}")
    if stage_manifest.get("dataset_snapshot_hash") != manifest.get("dataset_snapshot_hash"):
        raise StudioError(f"{stage} manifest has a different dataset snapshot hash")
    if stage_manifest.get("selection_hash") != manifest.get("selection_hash"):
        raise StudioError(f"{stage} manifest has a different selection hash")
    sources = stage_manifest.get("sources")
    dataset_names = stage_manifest.get("dataset_names")
    rule = stage_manifest.get("rule")
    if not isinstance(sources, dict) or not isinstance(dataset_names, list) or not isinstance(rule, dict):
        raise StudioError(f"{stage} manifest has an invalid composition")
    if any(not isinstance(name, str) for name in dataset_names):
        raise StudioError(f"{stage} dataset names must be strings")
    source_rows = []
    calculated_total = 0
    for source_name in sorted(sources):
        details = sources[source_name]
        if not isinstance(source_name, str) or not isinstance(details, dict):
            raise StudioError(f"{stage} source composition is invalid")
        rows = details.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise StudioError(f"{stage} source {source_name} has an invalid row count")
        calculated_total += rows
        source_rows.append({**details, "source": source_name})
    if stage_manifest.get("total_rows") != calculated_total:
        raise StudioError(f"{stage} manifest total_rows does not match its sources")
    return {
        "rule": rule,
        "dataset_names": list(dataset_names),
        "total_rows": calculated_total,
        "sources": source_rows,
    }


def verify_snapshot(snapshot_dir: str | Path) -> dict[str, Any]:
    """Verify a Studio export and return its content identity and full composition."""

    root = Path(snapshot_dir).expanduser().resolve()
    if not root.is_dir():
        raise StudioError(f"Dataset snapshot directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    manifest = _load_json_object(manifest_path, "snapshot manifest")
    if manifest.get("schema_version") != "chatts-dataset-studio-export-v1":
        raise StudioError(f"Unsupported snapshot manifest schema: {manifest.get('schema_version')!r}")

    expected_manifest_hash = _validate_hash(manifest.get("manifest_hash"), "manifest_hash")
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("manifest_hash", None)
    actual_manifest_hash = hashlib.sha256(_canonical_bytes(unhashed_manifest)).hexdigest()
    if actual_manifest_hash != expected_manifest_hash:
        raise StudioError("Snapshot manifest_hash is invalid")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise StudioError("Snapshot manifest has no files map")
    for relative, expected in files.items():
        expected_digest = _validate_hash(expected, f"files[{relative!r}]")
        path = _safe_snapshot_file(root, relative)
        if not path.is_file():
            raise StudioError(f"Snapshot file is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected_digest:
            raise StudioError(f"Snapshot file SHA256 mismatch: {relative}")

    expected_snapshot_hash = _validate_hash(
        manifest.get("dataset_snapshot_hash"), "dataset_snapshot_hash"
    )
    if _recompute_snapshot_hash(manifest) != expected_snapshot_hash:
        raise StudioError("dataset_snapshot_hash is invalid")

    selection_hash = _validate_hash(manifest.get("selection_hash"), "selection_hash")
    selection = manifest.get("selection")
    if hashlib.sha256(_canonical_bytes(selection)).hexdigest() != selection_hash:
        raise StudioError("selection_hash is invalid")

    dataset_info = _load_json_object(root / "dataset_info.json", "dataset_info")
    dataset_names = manifest.get("dataset_names")
    if not isinstance(dataset_names, dict):
        raise StudioError("Snapshot manifest has no dataset_names map")
    expected_names: set[str] = set()
    for stage in ("stage1", "stage2"):
        names = dataset_names.get(stage)
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise StudioError(f"Snapshot manifest has invalid {stage} dataset names")
        expected_names.update(names)
    if set(dataset_info) != expected_names:
        raise StudioError("dataset_info keys do not match snapshot dataset_names")
    for name, details in dataset_info.items():
        if not isinstance(details, dict) or not isinstance(details.get("file_name"), str):
            raise StudioError(f"dataset_info entry is invalid: {name}")
        data_path = _safe_snapshot_file(root, details["file_name"])
        if not data_path.is_file():
            raise StudioError(f"Registered dataset file is missing: {data_path}")

    composition = {
        stage: _stage_composition(root, manifest, stage) for stage in ("stage1", "stage2")
    }
    for stage in ("stage1", "stage2"):
        if composition[stage]["dataset_names"] != dataset_names[stage]:
            raise StudioError(f"{stage} dataset names disagree across manifests")

    return {
        "status": "verified",
        "snapshot_path": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "dataset_info_path": str(root / "dataset_info.json"),
        "dataset_info_sha256": _sha256_file(root / "dataset_info.json"),
        "training_env_path": str(root / "training.env"),
        "dataset_snapshot_hash": expected_snapshot_hash,
        "selection_hash": selection_hash,
        "run_name": manifest.get("run_name"),
        "data_version": manifest.get("data_version"),
        "selection": selection,
        "composition": composition,
    }


def resolve_verified_snapshot(
    snapshot_dir: str | Path,
    verified_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a snapshot, or safely reuse a verification from this operation.

    ``verified_snapshot`` is an internal optimization for a single publish or
    registration transaction.  It deliberately remains explicit at every call
    site: callers must first obtain it from :func:`verify_snapshot` (or from a
    ledger verification that used such a result).  Structural and path checks
    prevent accidentally reusing a verification for another snapshot.
    """

    root = Path(snapshot_dir).expanduser().resolve()
    if verified_snapshot is None:
        return verify_snapshot(root)
    if not isinstance(verified_snapshot, dict):
        raise StudioError("verified_snapshot must be a verified snapshot object")

    required_paths = {
        "snapshot_path": root,
        "manifest_path": root / "manifest.json",
        "dataset_info_path": root / "dataset_info.json",
        "training_env_path": root / "training.env",
    }
    for key, expected in required_paths.items():
        value = verified_snapshot.get(key)
        if not isinstance(value, str) or Path(value).expanduser().resolve() != expected:
            raise StudioError(f"verified_snapshot {key} does not match {expected}")

    for key in (
        "manifest_sha256",
        "dataset_info_sha256",
        "dataset_snapshot_hash",
        "selection_hash",
    ):
        _validate_hash(verified_snapshot.get(key), f"verified_snapshot {key}")
    composition = verified_snapshot.get("composition")
    if not isinstance(composition, dict) or set(composition) != {"stage1", "stage2"}:
        raise StudioError("verified_snapshot has an invalid composition")
    return dict(verified_snapshot)


class VersionLedger:
    def __init__(self, root: str | Path, *, start_number: int = 3):
        if isinstance(start_number, bool) or not isinstance(start_number, int) or start_number < 0:
            raise StudioError("start_number must be a non-negative integer")
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / "ledger.json"
        self.lock_path = self.root / ".ledger.lock"
        self.start_number = start_number

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA,
            "revision": 0,
            "next_number": self.start_number,
            "active_version": None,
            "versions": [],
        }

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        ledger = _load_json_object(self.path, "dataset version ledger")
        if ledger.get("schema_version") != LEDGER_SCHEMA:
            raise StudioError(f"Unsupported dataset ledger schema: {ledger.get('schema_version')!r}")
        versions = ledger.get("versions")
        if not isinstance(versions, list):
            raise StudioError("Dataset version ledger has no versions list")
        normalized: set[str] = set()
        hashes: set[str] = set()
        for entry in versions:
            if not isinstance(entry, dict):
                raise StudioError("Dataset version ledger contains a non-object entry")
            version = normalize_data_version(entry.get("version"))
            if version != entry.get("version") or version in normalized:
                raise StudioError("Dataset version ledger contains invalid or duplicate versions")
            normalized.add(version)
            digest = _validate_hash(entry.get("dataset_snapshot_hash"), "dataset_snapshot_hash")
            if digest in hashes:
                raise StudioError("Dataset version ledger contains duplicate snapshot hashes")
            hashes.add(digest)
        return ledger

    def _write_unlocked(self, ledger: dict[str, Any]) -> None:
        ledger["revision"] = int(ledger.get("revision", 0)) + 1
        atomic_write_json(self.path, ledger)

    def list(self) -> list[dict[str, Any]]:
        with self._locked(exclusive=False):
            ledger = self._read_unlocked()
            return [dict(entry) for entry in ledger["versions"]]

    def state(self) -> dict[str, Any]:
        with self._locked(exclusive=False):
            return dict(self._read_unlocked())

    def next(self) -> str:
        with self._locked(exclusive=False):
            ledger = self._read_unlocked()
            used = [version_number(entry["version"]) for entry in ledger["versions"]]
            number = max([self.start_number - 1, int(ledger["next_number"]) - 1, *used]) + 1
            return f"datav{number}"

    def record(
        self,
        snapshot_dir: str | Path,
        *,
        version: str | int | None = None,
        activate: bool = False,
        notes: str = "",
        parent: str | int | None = None,
        verified_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(notes, str) or len(notes) > 4_000:
            raise StudioError("Dataset version notes must be a string up to 4000 characters")
        canonical_parent = normalize_data_version(parent) if parent is not None else None
        verified = resolve_verified_snapshot(snapshot_dir, verified_snapshot)
        digest = verified["dataset_snapshot_hash"]
        with self._locked(exclusive=True):
            ledger = self._read_unlocked()
            for prior in ledger["versions"]:
                if prior["dataset_snapshot_hash"] == digest:
                    if version is not None and normalize_data_version(version) != prior["version"]:
                        raise StudioError(
                            f"Snapshot is already recorded as {prior['version']}; refusing a duplicate version"
                        )
                    if activate and ledger.get("active_version") != prior["version"]:
                        ledger["active_version"] = prior["version"]
                        self._write_unlocked(ledger)
                    return {**prior, "idempotent": True}

            if canonical_parent is not None and not any(
                entry["version"] == canonical_parent for entry in ledger["versions"]
            ):
                raise StudioError(f"Unknown parent dataset version: {canonical_parent}")

            if version is None:
                used = [version_number(entry["version"]) for entry in ledger["versions"]]
                number = max(
                    [self.start_number - 1, int(ledger.get("next_number", self.start_number)) - 1, *used]
                ) + 1
                canonical = f"datav{number}"
            else:
                canonical = normalize_data_version(version)
                number = version_number(canonical)
                if number < self.start_number:
                    raise StudioError(
                        f"Dataset versions managed here must start at datav{self.start_number}"
                    )
            for prior in ledger["versions"]:
                if prior["version"] == canonical:
                    raise StudioError(f"Dataset version already exists with different content: {canonical}")

            entry = {
                "version": canonical,
                "number": number,
                "created_at": _utc_now(),
                "parent": canonical_parent,
                "notes": notes.strip(),
                **verified,
                "status": "ready",
            }
            ledger["versions"].append(entry)
            ledger["versions"].sort(key=lambda item: item["number"])
            ledger["next_number"] = max(int(ledger.get("next_number", self.start_number)), number + 1)
            if activate:
                ledger["active_version"] = canonical
            self._write_unlocked(ledger)
            return {**entry, "idempotent": False}

    def get(self, version: str | int) -> dict[str, Any]:
        canonical = normalize_data_version(version)
        with self._locked(exclusive=False):
            ledger = self._read_unlocked()
            for entry in ledger["versions"]:
                if entry["version"] == canonical:
                    return dict(entry)
        raise StudioError(f"Unknown dataset version: {canonical}")

    def verify(
        self,
        version: str | int,
        *,
        verified_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = self.get(version)
        verified = resolve_verified_snapshot(entry["snapshot_path"], verified_snapshot)
        for key in (
            "dataset_snapshot_hash",
            "manifest_sha256",
            "dataset_info_sha256",
            "selection_hash",
        ):
            if verified[key] != entry[key]:
                raise StudioError(f"Recorded {entry['version']} no longer matches its {key}")
        if verified["composition"] != entry["composition"]:
            raise StudioError(f"Recorded {entry['version']} composition has changed")
        return {**entry, "verified_at": _utc_now()}

    def activate(
        self,
        version: str | int,
        *,
        verified_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        canonical = normalize_data_version(version)
        verified = self.verify(canonical, verified_snapshot=verified_snapshot)
        with self._locked(exclusive=True):
            ledger = self._read_unlocked()
            if not any(entry["version"] == canonical for entry in ledger["versions"]):
                raise StudioError(f"Unknown dataset version: {canonical}")
            if ledger.get("active_version") != canonical:
                ledger["active_version"] = canonical
                self._write_unlocked(ledger)
        return verified


def list_versions(root: str | Path, *, start_number: int = 3) -> list[dict[str, Any]]:
    return VersionLedger(root, start_number=start_number).list()


def next_version(root: str | Path, *, start_number: int = 3) -> str:
    return VersionLedger(root, start_number=start_number).next()


def record_version(
    root: str | Path,
    snapshot_dir: str | Path,
    *,
    version: str | int | None = None,
    activate: bool = False,
    notes: str = "",
    parent: str | int | None = None,
    start_number: int = 3,
    verified_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return VersionLedger(root, start_number=start_number).record(
        snapshot_dir,
        version=version,
        activate=activate,
        notes=notes,
        parent=parent,
        verified_snapshot=verified_snapshot,
    )


def activate_version(
    root: str | Path,
    version: str | int,
    *,
    start_number: int = 3,
    verified_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return VersionLedger(root, start_number=start_number).activate(
        version, verified_snapshot=verified_snapshot
    )


def verify_version(
    root: str | Path,
    version: str | int,
    *,
    start_number: int = 3,
    verified_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return VersionLedger(root, start_number=start_number).verify(
        version, verified_snapshot=verified_snapshot
    )

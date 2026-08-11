from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import StudioError
from .versioning import (
    VersionLedger,
    atomic_write_bytes,
    atomic_write_json,
    normalize_data_version,
    resolve_verified_snapshot,
)

PROFILE_SCHEMA = "chatts-training-dataset-profile-v1"
ACTIVE_SCHEMA = "chatts-training-active-dataset-v1"
_VERSION_SUFFIX_RE = re.compile(r"(?:[-_]?data-?v[0-9]+)$", re.IGNORECASE)
_REQUIRED_EXPORTED_ENV_KEYS = {
    "DATASET_DIR",
    "STAGE1_DATASETS",
    "STAGE2_DATASETS",
    "STAGE1_MIX_STRATEGY",
    "STAGE2_MIX_STRATEGY",
    "STAGE1_INTERLEAVE_PROBS",
    "STAGE2_INTERLEAVE_PROBS",
    "DATASET_SNAPSHOT_HASH",
}
_OPTIONAL_EXPORTED_ENV_KEYS = {"DATA_VERSION"}
_EXPORTED_ENV_KEYS = _REQUIRED_EXPORTED_ENV_KEYS | _OPTIONAL_EXPORTED_ENV_KEYS


def versioned_output_root(base: str | Path, version: str | int) -> Path:
    canonical = normalize_data_version(version)
    path = Path(base).expanduser()
    name = _VERSION_SUFFIX_RE.sub("", path.name).rstrip("-_")
    if not name:
        raise StudioError("Model output base must have a non-version name")
    return (path.parent / f"{name}-{canonical}").resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry_root(training_root: str | Path) -> tuple[Path, Path]:
    root = Path(training_root).expanduser().resolve()
    if not root.is_dir():
        raise StudioError(f"ChatTS-Training root does not exist: {root}")
    data_root = root / "data"
    if data_root.exists() or data_root.is_symlink():
        try:
            data_root.resolve().relative_to(root)
        except ValueError as exc:
            raise StudioError("Training data directory escapes ChatTS-Training root") from exc
    else:
        data_root.mkdir()
    registry = data_root / "studio_versions"
    if registry.exists() or registry.is_symlink():
        try:
            registry.resolve().relative_to(root)
        except ValueError as exc:
            raise StudioError("Training version registry escapes ChatTS-Training root") from exc
    registry.mkdir(parents=True, exist_ok=True)
    try:
        registry.resolve().relative_to(root)
    except ValueError as exc:
        raise StudioError("Training version registry escapes ChatTS-Training root") from exc
    return root, registry


@contextmanager
def _registry_lock(registry: Path, *, exclusive: bool) -> Iterator[None]:
    lock_path = registry / ".registry.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _parse_export_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise StudioError(f"Cannot read snapshot training.env {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            fields = shlex.split(line, comments=False, posix=True)
        except ValueError as exc:
            raise StudioError(f"Invalid training.env line {line_number}: {exc}") from exc
        if len(fields) != 1 or "=" not in fields[0]:
            raise StudioError(f"Invalid training.env assignment at line {line_number}")
        key, value = fields[0].split("=", 1)
        if key not in _EXPORTED_ENV_KEYS:
            raise StudioError(f"Unexpected training.env key: {key}")
        if key in result:
            raise StudioError(f"Duplicate training.env key: {key}")
        result[key] = value
    missing = _REQUIRED_EXPORTED_ENV_KEYS - result.keys()
    if missing:
        raise StudioError(f"training.env lacks required keys: {sorted(missing)}")
    return result


def _validated_environment(
    training_root: Path,
    version: str,
    verified: dict[str, Any],
    model_output_base: str | Path | None,
) -> dict[str, str]:
    snapshot_root = Path(verified["snapshot_path"])
    exported = _parse_export_env(Path(verified["training_env_path"]))
    if Path(exported["DATASET_DIR"]).expanduser().resolve() != snapshot_root:
        raise StudioError("training.env DATASET_DIR does not point to its snapshot")
    if exported["DATASET_SNAPSHOT_HASH"] != verified["dataset_snapshot_hash"]:
        raise StudioError("training.env DATASET_SNAPSHOT_HASH does not match its manifest")
    if exported.get("DATA_VERSION") not in (None, version):
        raise StudioError("training.env DATA_VERSION does not match the registered version")
    for stage in ("stage1", "stage2"):
        expected = ",".join(verified["composition"][stage]["dataset_names"])
        if exported[f"{stage.upper()}_DATASETS"] != expected:
            raise StudioError(f"training.env {stage.upper()}_DATASETS does not match its manifest")
        strategy = exported[f"{stage.upper()}_MIX_STRATEGY"]
        if strategy not in {"concat", "interleave_under", "interleave_over"}:
            raise StudioError(f"Unsupported {stage} mix strategy: {strategy}")

    environment = {
        "PROJECT_ROOT": str(training_root),
        "DATA_VERSION": version,
        "DATASET_VERSION": version,
        "DATASET_DIR": str(snapshot_root),
        "DATASET_MANIFEST_PATH": verified["manifest_path"],
        "DATASET_MANIFEST_SHA256": verified["manifest_sha256"],
        "DATASET_SNAPSHOT_HASH": verified["dataset_snapshot_hash"],
        "STAGE1_DATASETS": exported["STAGE1_DATASETS"],
        "STAGE2_DATASETS": exported["STAGE2_DATASETS"],
        "STAGE1_MIX_STRATEGY": exported["STAGE1_MIX_STRATEGY"],
        "STAGE2_MIX_STRATEGY": exported["STAGE2_MIX_STRATEGY"],
        "STAGE1_INTERLEAVE_PROBS": exported["STAGE1_INTERLEAVE_PROBS"],
        "STAGE2_INTERLEAVE_PROBS": exported["STAGE2_INTERLEAVE_PROBS"],
    }
    if model_output_base is not None:
        environment["OUTPUT_ROOT"] = str(versioned_output_root(model_output_base, version))
    return environment


def _env_bytes(environment: dict[str, str]) -> bytes:
    return (
        "\n".join(f"{key}={shlex.quote(value)}" for key, value in environment.items()) + "\n"
    ).encode("utf-8")


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_immutable(path: Path, content: bytes) -> bool:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise StudioError(f"Training registration conflicts with existing file: {path}")
        return False
    atomic_write_bytes(path, content)
    return True


def _check_immutable(path: Path, content: bytes) -> bool:
    if not path.exists():
        return False
    if not path.is_file() or path.read_bytes() != content:
        raise StudioError(f"Training registration conflicts with existing file: {path}")
    return True


def write_training_registration(
    training_root: str | Path,
    version_entry: dict[str, Any],
    *,
    model_output_base: str | Path | None = None,
    activate: bool = False,
    verified_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    version = normalize_data_version(version_entry.get("version"))
    root, registry = _registry_root(training_root)
    verified = resolve_verified_snapshot(
        version_entry.get("snapshot_path"), verified_snapshot
    )
    for key in ("dataset_snapshot_hash", "manifest_sha256", "selection_hash"):
        if version_entry.get(key) != verified[key]:
            raise StudioError(f"Version entry does not match verified snapshot {key}")

    environment = _validated_environment(root, version, verified, model_output_base)
    profile_path = registry / f"{version}.json"
    env_path = registry / f"{version}.env"
    profile = {
        "schema_version": PROFILE_SCHEMA,
        "version": version,
        "snapshot_path": verified["snapshot_path"],
        "dataset_snapshot_hash": verified["dataset_snapshot_hash"],
        "selection_hash": verified["selection_hash"],
        "manifest_path": verified["manifest_path"],
        "manifest_sha256": verified["manifest_sha256"],
        "dataset_info_path": verified["dataset_info_path"],
        "dataset_info_sha256": verified["dataset_info_sha256"],
        "composition": verified["composition"],
        "environment": environment,
    }
    profile_content = _json_bytes(profile)
    env_content = _env_bytes(environment)
    with _registry_lock(registry, exclusive=True):
        profile_exists = _check_immutable(profile_path, profile_content)
        env_exists = _check_immutable(env_path, env_content)
        profile_created = not profile_exists and _write_immutable(profile_path, profile_content)
        env_created = not env_exists and _write_immutable(env_path, env_content)
        if activate:
            active = {
                "schema_version": ACTIVE_SCHEMA,
                "version": version,
                "profile_path": str(profile_path),
                "profile_sha256": hashlib.sha256(profile_content).hexdigest(),
                "env_path": str(env_path),
                "env_sha256": hashlib.sha256(env_content).hexdigest(),
                "dataset_snapshot_hash": verified["dataset_snapshot_hash"],
            }
            atomic_write_json(registry / "active.json", active)
    return {
        "status": "registered",
        "version": version,
        "profile_path": str(profile_path),
        "env_path": str(env_path),
        "active_path": str(registry / "active.json") if activate else None,
        "created": profile_created or env_created,
        "environment": environment,
    }


def register_training_version(
    training_root: str | Path,
    ledger_root: str | Path,
    version: str | int,
    *,
    model_output_base: str | Path | None = None,
    activate: bool = False,
    verified_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = VersionLedger(ledger_root)
    entry = ledger.verify(version, verified_snapshot=verified_snapshot)
    result = write_training_registration(
        training_root,
        entry,
        model_output_base=model_output_base,
        activate=False,
        verified_snapshot=entry,
    )
    if activate:
        # The ledger's active version is the externally visible readiness
        # pointer.  Publish the Training pointer first, then atomically advance
        # the ledger so readers never observe a ledger-active version that
        # Training cannot load yet.
        activation = activate_training_version(
            training_root, version, verified_snapshot=entry
        )
        ledger.activate(version, verified_snapshot=entry)
        result["active_path"] = activation["active_path"]
    return result


def activate_training_version(
    training_root: str | Path,
    version: str | int,
    *,
    verified_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = normalize_data_version(version)
    _, registry = _registry_root(training_root)
    profile_path = registry / f"{canonical}.json"
    env_path = registry / f"{canonical}.env"
    with _registry_lock(registry, exclusive=True):
        if not profile_path.is_file() or not env_path.is_file():
            raise StudioError(f"Training dataset version is not registered: {canonical}")
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StudioError(f"Cannot read training profile {profile_path}: {exc}") from exc
        if not isinstance(profile, dict) or profile.get("schema_version") != PROFILE_SCHEMA:
            raise StudioError(f"Invalid training profile: {profile_path}")
        if profile.get("version") != canonical:
            raise StudioError(f"Training profile version mismatch: {profile_path}")
        environment = profile.get("environment")
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise StudioError(f"Training profile {canonical} has an invalid environment")
        if env_path.read_bytes() != _env_bytes(environment):
            raise StudioError(f"Training environment has changed: {canonical}")
        verified = resolve_verified_snapshot(
            profile.get("snapshot_path"), verified_snapshot
        )
        for key in (
            "dataset_snapshot_hash",
            "manifest_sha256",
            "dataset_info_sha256",
            "selection_hash",
        ):
            if verified[key] != profile.get(key):
                raise StudioError(f"Training profile snapshot {key} has changed: {canonical}")
        if verified["composition"] != profile.get("composition"):
            raise StudioError(f"Training profile snapshot composition has changed: {canonical}")
        profile_content = profile_path.read_bytes()
        env_content = env_path.read_bytes()
        active = {
            "schema_version": ACTIVE_SCHEMA,
            "version": canonical,
            "profile_path": str(profile_path),
            "profile_sha256": hashlib.sha256(profile_content).hexdigest(),
            "env_path": str(env_path),
            "env_sha256": hashlib.sha256(env_content).hexdigest(),
            "dataset_snapshot_hash": verified["dataset_snapshot_hash"],
        }
        active_path = registry / "active.json"
        atomic_write_json(active_path, active)
    return {
        "status": "active",
        "version": canonical,
        "active_path": str(active_path),
    }


def verify_training_registration(
    training_root: str | Path,
    version: str | int,
) -> dict[str, Any]:
    canonical = normalize_data_version(version)
    _, registry = _registry_root(training_root)
    profile_path = registry / f"{canonical}.json"
    env_path = registry / f"{canonical}.env"
    with _registry_lock(registry, exclusive=False):
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise StudioError(f"Training dataset version is not registered: {canonical}") from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StudioError(f"Cannot read training profile {profile_path}: {exc}") from exc
        if not isinstance(profile, dict) or profile.get("schema_version") != PROFILE_SCHEMA:
            raise StudioError(f"Invalid training profile: {profile_path}")
        verified = resolve_verified_snapshot(profile.get("snapshot_path"))
        for key in ("dataset_snapshot_hash", "manifest_sha256", "selection_hash"):
            if profile.get(key) != verified[key]:
                raise StudioError(f"Training profile {canonical} has a stale {key}")
        expected_environment = profile.get("environment")
        if not isinstance(expected_environment, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in expected_environment.items()
        ):
            raise StudioError(f"Training profile {canonical} has an invalid environment")
        if env_path.read_bytes() != _env_bytes(expected_environment):
            raise StudioError(f"Training environment has changed: {canonical}")
    return {
        "status": "verified",
        "version": canonical,
        "profile_path": str(profile_path),
        "env_path": str(env_path),
        "dataset_snapshot_hash": verified["dataset_snapshot_hash"],
    }

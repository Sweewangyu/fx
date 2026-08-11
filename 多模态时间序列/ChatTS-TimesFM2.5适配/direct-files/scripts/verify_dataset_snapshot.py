#!/usr/bin/env python3
"""Verify a ChatTS Dataset Studio snapshot before training.

This module intentionally uses only the Python standard library so the training
runner can validate copied snapshots without installing Dataset Studio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

EXPORT_SCHEMA = "chatts-dataset-studio-export-v1"
SNAPSHOT_SCHEMAS = {
    "chatts-dataset-snapshot-v1",
    "chatts-dataset-snapshot-v2",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class VerificationError(ValueError):
    """Raised when a dataset snapshot violates its integrity contract."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"value cannot be canonically encoded: {exc}") from exc
    return encoded.encode("utf-8")


def _hash_object(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VerificationError(f"{label} does not exist: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be one JSON object: {path}")
    return value


def _safe_snapshot_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise VerificationError(f"snapshot manifest contains an unsafe file path: {relative!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise VerificationError(f"snapshot file escapes its root: {relative}") from exc
    return candidate


def _validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise VerificationError(f"{label} must be a lowercase SHA256 digest")
    return value


def _parse_dataset_names(value: str, label: str) -> list[str]:
    names = value.split(",") if value else []
    if not names or any(not name or name != name.strip() for name in names):
        raise VerificationError(f"{label} must be a comma-separated list of dataset keys")
    if len(set(names)) != len(names):
        raise VerificationError(f"{label} contains duplicate dataset keys")
    return names


def _recompute_snapshot_hash(manifest: dict[str, Any]) -> str:
    files = manifest.get("files")
    preview = manifest.get("preview")
    if not isinstance(files, dict) or not isinstance(preview, dict):
        raise VerificationError("snapshot manifest lacks files or preview")
    counts = preview.get("counts")
    if not isinstance(counts, dict):
        raise VerificationError("snapshot manifest preview lacks counts")
    selected_outputs = {
        path: digest
        for path, digest in files.items()
        if isinstance(path, str) and path.endswith(".jsonl")
    }
    schema = manifest.get("snapshot_hash_schema", "chatts-dataset-snapshot-v1")
    if schema not in SNAPSHOT_SCHEMAS:
        raise VerificationError(f"unsupported snapshot hash schema: {schema!r}")
    inputs = manifest.get("input_identities")
    if schema == "chatts-dataset-snapshot-v2":
        if not isinstance(inputs, dict):
            raise VerificationError("snapshot manifest has invalid input identities")
        content_inputs: dict[str, dict[str, Any]] = {}
        for name, identity in inputs.items():
            if not isinstance(name, str) or not isinstance(identity, dict):
                raise VerificationError("snapshot manifest has invalid input identities")
            content_inputs[name] = {
                key: identity.get(key)
                for key in ("annotation_mode", "raw_sha256", "annotation_sha256")
            }
        inputs = content_inputs
    payload = {
        "schema_version": schema,
        "selection": manifest.get("selection"),
        "inputs": inputs,
        "selected_outputs": selected_outputs,
        "counts": counts,
    }
    return _hash_object(payload)


def _verify_stage(
    root: Path,
    manifest: dict[str, Any],
    stage: str,
    expected_names: list[str],
) -> None:
    stage_manifest = _load_json_object(root / stage / "manifest.json", f"{stage} manifest")
    if stage_manifest.get("stage") != stage:
        raise VerificationError(f"{stage} manifest has an invalid stage marker")
    if stage_manifest.get("dataset_snapshot_hash") != manifest.get("dataset_snapshot_hash"):
        raise VerificationError(f"{stage} manifest has a different dataset snapshot hash")
    if stage_manifest.get("selection_hash") != manifest.get("selection_hash"):
        raise VerificationError(f"{stage} manifest has a different selection hash")
    sources = stage_manifest.get("sources")
    names = stage_manifest.get("dataset_names")
    rule = stage_manifest.get("rule")
    if not isinstance(sources, dict) or not isinstance(names, list) or not isinstance(rule, dict):
        raise VerificationError(f"{stage} manifest has an invalid composition")
    if any(not isinstance(name, str) for name in names):
        raise VerificationError(f"{stage} dataset names must be strings")
    total_rows = 0
    for source_name, details in sources.items():
        if not isinstance(source_name, str) or not isinstance(details, dict):
            raise VerificationError(f"{stage} source composition is invalid")
        rows = details.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise VerificationError(f"{stage} source {source_name} has an invalid row count")
        total_rows += rows
    if stage_manifest.get("total_rows") != total_rows:
        raise VerificationError(f"{stage} manifest total_rows does not match its sources")
    if names != expected_names:
        raise VerificationError(f"{stage} dataset names disagree across manifests")


def verify_snapshot(
    dataset_dir: Path,
    *,
    stage1_datasets: str,
    stage2_datasets: str,
    expected_snapshot_hash: str = "",
    expected_data_version: str = "",
) -> str | None:
    """Verify a Studio snapshot, or allow an unidentified legacy directory."""
    root = dataset_dir.expanduser().resolve()
    if not root.is_dir():
        raise VerificationError(f"dataset directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        if expected_snapshot_hash or expected_data_version:
            raise VerificationError(
                "dataset identity was supplied, but Dataset Studio manifest.json is missing"
            )
        return None

    manifest = _load_json_object(manifest_path, "snapshot manifest")
    if manifest.get("schema_version") != EXPORT_SCHEMA:
        raise VerificationError(
            f"unsupported snapshot manifest schema: {manifest.get('schema_version')!r}"
        )
    expected_manifest_hash = _validate_hash(manifest.get("manifest_hash"), "manifest_hash")
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("manifest_hash", None)
    if _hash_object(unhashed_manifest) != expected_manifest_hash:
        raise VerificationError("snapshot manifest_hash is invalid")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise VerificationError("snapshot manifest has no files map")
    for relative, expected in files.items():
        expected_digest = _validate_hash(expected, f"files[{relative!r}]")
        path = _safe_snapshot_file(root, relative)
        if not path.is_file():
            raise VerificationError(f"snapshot file is missing: {path}")
        if _sha256_file(path) != expected_digest:
            raise VerificationError(f"snapshot file SHA256 mismatch: {relative}")

    snapshot_hash = _validate_hash(
        manifest.get("dataset_snapshot_hash"), "dataset_snapshot_hash"
    )
    if _recompute_snapshot_hash(manifest) != snapshot_hash:
        raise VerificationError("dataset_snapshot_hash is invalid")
    selection_hash = _validate_hash(manifest.get("selection_hash"), "selection_hash")
    if _hash_object(manifest.get("selection")) != selection_hash:
        raise VerificationError("selection_hash is invalid")

    manifest_names = manifest.get("dataset_names")
    if not isinstance(manifest_names, dict):
        raise VerificationError("snapshot manifest has no dataset_names map")
    names_by_stage: dict[str, list[str]] = {}
    for stage in ("stage1", "stage2"):
        names = manifest_names.get(stage)
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise VerificationError(f"snapshot manifest has invalid {stage} dataset names")
        names_by_stage[stage] = names

    dataset_info = _load_json_object(root / "dataset_info.json", "dataset_info")
    expected_info_names = set(names_by_stage["stage1"]) | set(names_by_stage["stage2"])
    if set(dataset_info) != expected_info_names:
        raise VerificationError("dataset_info keys do not match snapshot dataset_names")
    for name, details in dataset_info.items():
        if not isinstance(details, dict) or not isinstance(details.get("file_name"), str):
            raise VerificationError(f"dataset_info entry is invalid: {name}")
        data_path = _safe_snapshot_file(root, details["file_name"])
        if not data_path.is_file():
            raise VerificationError(f"registered dataset file is missing: {data_path}")

    for stage in ("stage1", "stage2"):
        _verify_stage(root, manifest, stage, names_by_stage[stage])
    configured_names = {
        "stage1": _parse_dataset_names(stage1_datasets, "STAGE1_DATASETS"),
        "stage2": _parse_dataset_names(stage2_datasets, "STAGE2_DATASETS"),
    }
    for stage in ("stage1", "stage2"):
        if configured_names[stage] != names_by_stage[stage]:
            raise VerificationError(
                f"configured {stage} dataset keys do not match the snapshot manifest: "
                f"configured={configured_names[stage]!r}, manifest={names_by_stage[stage]!r}"
            )

    if expected_snapshot_hash and expected_snapshot_hash.lower() != snapshot_hash:
        raise VerificationError(
            "DATASET_SNAPSHOT_HASH does not match the verified snapshot manifest"
        )
    if expected_data_version and manifest.get("data_version") != expected_data_version:
        raise VerificationError(
            "DATA_VERSION does not match the verified snapshot manifest: "
            f"expected={expected_data_version!r}, manifest={manifest.get('data_version')!r}"
        )
    return snapshot_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--stage1-datasets", required=True)
    parser.add_argument("--stage2-datasets", required=True)
    parser.add_argument("--expected-snapshot-hash", default="")
    parser.add_argument("--expected-data-version", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        snapshot_hash = verify_snapshot(
            args.dataset_dir,
            stage1_datasets=args.stage1_datasets,
            stage2_datasets=args.stage2_datasets,
            expected_snapshot_hash=args.expected_snapshot_hash,
            expected_data_version=args.expected_data_version,
        )
    except VerificationError as exc:
        print(f"Dataset snapshot verification failed: {exc}", file=sys.stderr)
        return 1
    if snapshot_hash is None:
        print("Legacy dataset directory has no Studio manifest; identity checks were not requested.")
    else:
        print(f"Dataset Studio snapshot verified: {snapshot_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

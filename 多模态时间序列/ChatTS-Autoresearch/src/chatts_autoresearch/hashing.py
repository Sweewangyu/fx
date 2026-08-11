from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def hash_object(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def command_fingerprint(
    argv: Iterable[str], cwd: str | Path | None, env: Mapping[str, str]
) -> str:
    return hash_object(
        {
            "argv": [str(item) for item in argv],
            "cwd": str(Path(cwd).resolve()) if cwd else None,
            "env": {key: str(value) for key, value in sorted(env.items())},
        }
    )


def dataset_fingerprint(manifest_path: str | Path, registry_path: str | Path) -> str:
    """Fingerprint a versioned catalog without re-reading multi-GB JSONL files.

    datav2 already publishes a content SHA in its immutable manifest. We bind that
    digest to the exact manifest and source registry bytes. For an unversioned
    fixture, the manifest/registry bytes themselves remain the source of truth.
    """

    manifest_path = Path(manifest_path)
    registry_path = Path(registry_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return hash_object(
        {
            "published_content_sha256": manifest.get("content_sha256"),
            "manifest_sha256": sha256_file(manifest_path),
            "registry_sha256": sha256_file(registry_path),
        }
    )

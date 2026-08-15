from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

from .models import StudioError

STUDIO_SBATCH_MARKER = "# CHATTS_STUDIO_SBATCH_API=1"
DEFAULT_SBATCH_NAME = "run_chatts_studio_pipeline.sbatch"
SLURM_JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")


def _slurm_root(integration: dict[str, Any], *, strict: bool) -> Path:
    value = integration.get("slurm_root")
    if not isinstance(value, str) or not value:
        training_root = integration.get("training_root")
        if not isinstance(training_root, str) or not training_root:
            raise StudioError("integration.slurm_root or integration.training_root is required")
        value = str(Path(training_root) / "slurm")
    root = Path(value).expanduser()
    try:
        return root.resolve(strict=strict)
    except OSError as exc:
        raise StudioError(f"Slurm launcher root is unavailable: {root}: {exc}") from exc


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _inspect_launcher(path: Path, root: Path) -> dict[str, Any]:
    if path.suffix != ".sbatch":
        raise StudioError("Slurm launcher must use the .sbatch suffix")
    if path.is_symlink():
        raise StudioError("Slurm launcher must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StudioError(f"Slurm launcher is unavailable: {path}: {exc}") from exc
    if not _is_relative_to(resolved, root):
        raise StudioError(f"Slurm launcher must stay under the trusted root: {root}")
    if not resolved.is_file():
        raise StudioError(f"Slurm launcher is not a regular file: {resolved}")
    encoded = resolved.read_bytes()
    header = encoded[:8192].decode("utf-8", errors="replace")
    if STUDIO_SBATCH_MARKER not in header.splitlines():
        raise StudioError(
            f"Slurm launcher does not implement the Dataset Studio contract: {resolved}"
        )
    return {
        "path": str(resolved),
        "relative_path": resolved.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def resolve_slurm_launcher(
    integration: dict[str, Any], requested: Any = None
) -> dict[str, Any]:
    root = _slurm_root(integration, strict=True)
    if not root.is_dir():
        raise StudioError(f"Slurm launcher root is not a directory: {root}")
    default_value = integration.get("slurm_sbatch", DEFAULT_SBATCH_NAME)
    value = default_value if requested in (None, "") else requested
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value or "\r" in value:
        raise StudioError("execution.sbatch_path must be a non-empty safe path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    inspected = _inspect_launcher(candidate, root)
    return {"root": str(root), **inspected}


def list_slurm_launchers(integration: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        root = _slurm_root(integration, strict=True)
    except StudioError:
        return []
    if not root.is_dir():
        return []
    launchers: list[dict[str, Any]] = []
    for candidate in sorted(root.glob("*.sbatch")):
        try:
            launchers.append(_inspect_launcher(candidate, root))
        except (OSError, StudioError):
            continue
    return launchers


def slurm_readiness(integration: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for command in ("sbatch", "squeue", "sacct"):
        if shutil.which(command) is None:
            reasons.append(f"{command} CLI is unavailable to Dataset Studio")
    try:
        root = _slurm_root(integration, strict=True)
        if not root.is_dir():
            reasons.append(f"integration.slurm_root is not a directory: {root}")
    except StudioError as exc:
        root = None
        reasons.append(str(exc))
    launchers = list_slurm_launchers(integration)
    if root is not None and not launchers:
        reasons.append(
            f"No trusted Dataset Studio .sbatch launcher exists under: {root}"
        )
    default: dict[str, Any] | None = None
    if launchers:
        try:
            default = resolve_slurm_launcher(integration)
        except StudioError as exc:
            reasons.append(str(exc))
    return {
        "enabled": not reasons,
        "disabled_reasons": reasons,
        "root": str(root) if root is not None else None,
        "launchers": [
            {"relative_path": item["relative_path"], "sha256": item["sha256"]}
            for item in launchers
        ],
        "default_sbatch": default["relative_path"] if default else None,
    }


def parse_slurm_job_id(value: str) -> str:
    token = value.strip().split(";", 1)[0]
    if not SLURM_JOB_ID_RE.fullmatch(token):
        raise StudioError(f"sbatch returned an invalid job id: {value!r}")
    return token

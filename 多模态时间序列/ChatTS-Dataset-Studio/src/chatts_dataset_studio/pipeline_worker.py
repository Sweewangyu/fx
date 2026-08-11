from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Durable ChatTS pipeline worker")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--lock-path", required=True, type=Path)
    parser.add_argument("--status-path", required=True, type=Path)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_path.open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _write_json(
                args.status_path,
                {
                    "schema_version": "chatts-dataset-studio-worker-v1",
                    "job_id": args.job_id,
                    "status": "failed",
                    "pid": os.getpid(),
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                    "duration_seconds": 0.0,
                    "exit_code": 73,
                    "error": "Another ChatTS training/evaluation pipeline holds the run lock",
                },
            )
            return 73

        _write_json(
            args.status_path,
            {
                "schema_version": "chatts-dataset-studio-worker-v1",
                "job_id": args.job_id,
                "status": "running",
                "pid": os.getpid(),
                "started_at": started_at,
            },
        )
        environment = dict(os.environ)
        environment["CONFIG_FILE"] = str(args.config)
        error: str | None = None
        try:
            completed = subprocess.run(  # noqa: S603 - fixed server-side script.
                ["bash", str(args.script)],
                cwd=args.cwd,
                env=environment,
                check=False,
            )
            exit_code = completed.returncode
        except OSError as exc:
            exit_code = 126
            error = f"Could not execute fixed pipeline script: {exc}"

        result = {
            "schema_version": "chatts-dataset-studio-worker-v1",
            "job_id": args.job_id,
            "status": "completed" if exit_code == 0 else "failed",
            "pid": os.getpid(),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_seconds": round(time.monotonic() - started_monotonic, 3),
            "exit_code": exit_code,
        }
        if error:
            result["error"] = error
        _write_json(args.status_path, result)
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

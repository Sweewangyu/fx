from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
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
    # Docker jobs share a local lock. Slurm jobs intentionally omit it because
    # GPU admission and PENDING/RUNNING state belong to the cluster scheduler.
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument("--status-path", required=True, type=Path)
    parser.add_argument("--backend", choices=("docker_host", "slurm"), default="docker_host")
    parser.add_argument("--script", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--sbatch-path", type=Path)
    parser.add_argument("--sbatch-sha256")
    parser.add_argument("--scheduler-job-id")
    parser.add_argument("--scheduler-stdout", type=Path)
    parser.add_argument("--scheduler-stderr", type=Path)
    parser.add_argument("--slurm-preflight", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--accounting-grace-seconds", type=float, default=120.0)
    return parser


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - every executable and argument is fixed/validated.
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _echo_completed(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)


def _slurm_job_id(value: str) -> str:
    token = value.strip().split(";", 1)[0]
    if not re.fullmatch(r"[1-9][0-9]*", token):
        raise ValueError(f"sbatch returned an invalid job id: {value!r}")
    return token


def _slurm_state(value: str) -> tuple[str, bool]:
    normalized = value.strip().upper().split("+", 1)[0].split(" ", 1)[0]
    if normalized in {"PENDING", "CONFIGURING", "REQUEUED", "RESIZING", "SUSPENDED"}:
        return "scheduled", False
    if normalized in {"RUNNING", "COMPLETING", "STAGE_OUT"}:
        return "running", False
    if normalized == "COMPLETED":
        return "completed", True
    if normalized.startswith("CANCELLED"):
        return "canceled", True
    if normalized in {
        "FAILED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
    }:
        return "failed", True
    return "scheduled", False


def _sacct_state(job_id: str, cwd: Path) -> tuple[str, str | None] | None:
    completed = _run(
        [
            "sacct",
            "-n",
            "-P",
            "-X",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw",
        ],
        cwd,
    )
    if completed.returncode != 0:
        _echo_completed(completed)
        return None
    for raw in completed.stdout.splitlines():
        fields = raw.split("|")
        if len(fields) >= 3 and fields[0].strip() == job_id:
            return fields[1].strip(), fields[2].strip() or None
    return None


def _write_running_status(
    args: argparse.Namespace,
    started_at: str,
    scheduler_job_id: str | None = None,
    *,
    status: str = "running",
    scheduler_state: str | None = None,
) -> None:
    value: dict[str, Any] = {
        "schema_version": "chatts-dataset-studio-worker-v1",
        "job_id": args.job_id,
        "status": status,
        "pid": os.getpid(),
        "started_at": started_at,
        "execution_backend": args.backend,
    }
    if scheduler_job_id:
        value["scheduler_job_id"] = scheduler_job_id
    if scheduler_state:
        value["scheduler_state"] = scheduler_state
    _write_json(args.status_path, value)


def _execute_slurm(
    args: argparse.Namespace, started_at: str, started_monotonic: float
) -> tuple[int, str | None, str | None, str | None]:
    if args.sbatch_path is None or args.sbatch_sha256 is None:
        return 126, "Slurm worker is missing its trusted sbatch path/hash", None, None
    try:
        encoded = args.sbatch_path.read_bytes()
    except OSError as exc:
        return 126, f"Could not read trusted Slurm launcher: {exc}", None, None
    actual_hash = hashlib.sha256(encoded).hexdigest()
    if actual_hash != args.sbatch_sha256:
        return 126, "Trusted Slurm launcher changed after the job was frozen", None, None
    if not args.config.is_file():
        return 126, f"Frozen pipeline config is missing: {args.config}", None, None

    scheduler_job_id = args.scheduler_job_id
    scheduler_state: str | None = None
    if args.slurm_preflight:
        completed = _run(
            [
                "sbatch",
                "--test-only",
                f"--chdir={args.cwd}",
                str(args.sbatch_path),
                str(args.config),
                args.job_id,
            ],
            args.cwd,
        )
        _echo_completed(completed)
        error = None if completed.returncode == 0 else "Slurm rejected the frozen job"
        return completed.returncode, error, None, None

    if scheduler_job_id is None:
        if args.scheduler_stdout is None or args.scheduler_stderr is None:
            return 126, "Slurm worker is missing scheduler log paths", None, None
        args.scheduler_stdout.parent.mkdir(parents=True, exist_ok=True)
        args.scheduler_stderr.parent.mkdir(parents=True, exist_ok=True)
        completed = _run(
            [
                "sbatch",
                "--parsable",
                f"--chdir={args.cwd}",
                f"--output={args.scheduler_stdout}",
                f"--error={args.scheduler_stderr}",
                str(args.sbatch_path),
                str(args.config),
                args.job_id,
            ],
            args.cwd,
        )
        _echo_completed(completed)
        if completed.returncode != 0:
            return completed.returncode, "sbatch submission failed", None, None
        try:
            scheduler_job_id = _slurm_job_id(completed.stdout)
        except ValueError as exc:
            return 126, str(exc), None, None
        print(f"Dataset Studio submitted Slurm job {scheduler_job_id}", flush=True)

    _write_running_status(
        args,
        started_at,
        scheduler_job_id,
        status="scheduled",
        scheduler_state="PENDING",
    )
    missing_since: float | None = None
    while True:
        completed = _run(
            ["squeue", "-h", "-j", scheduler_job_id, "-o", "%T"], args.cwd
        )
        live_state = completed.stdout.strip().splitlines()[0] if completed.returncode == 0 and completed.stdout.strip() else None
        exit_code_text: str | None = None
        if live_state is None:
            accounting = _sacct_state(scheduler_job_id, args.cwd)
            if accounting is not None:
                live_state, exit_code_text = accounting
            else:
                if missing_since is None:
                    missing_since = time.monotonic()
                if time.monotonic() - missing_since > args.accounting_grace_seconds:
                    return (
                        125,
                        "Slurm job disappeared and sacct had no top-level record after the accounting grace period",
                        scheduler_job_id,
                        scheduler_state,
                    )
                time.sleep(max(args.poll_seconds, 0.05))
                continue
        else:
            missing_since = None

        scheduler_state = live_state
        status, terminal = _slurm_state(live_state)
        _write_running_status(
            args,
            started_at,
            scheduler_job_id,
            status=status,
            scheduler_state=live_state,
        )
        if terminal:
            if status == "completed":
                return 0, None, scheduler_job_id, scheduler_state
            if status == "canceled":
                return 130, "Slurm job was canceled", scheduler_job_id, scheduler_state
            parsed_exit = 1
            if exit_code_text:
                try:
                    parsed_exit = int(exit_code_text.split(":", 1)[0]) or 1
                except ValueError:
                    parsed_exit = 1
            return (
                parsed_exit,
                f"Slurm job finished in state {live_state}",
                scheduler_job_id,
                scheduler_state,
            )
        time.sleep(max(args.poll_seconds, 0.05))


def _execute_worker(
    args: argparse.Namespace, started_at: str, started_monotonic: float
) -> int:
    _write_running_status(args, started_at)
    error: str | None = None
    scheduler_job_id: str | None = None
    scheduler_state: str | None = None
    if args.backend == "slurm":
        try:
            exit_code, error, scheduler_job_id, scheduler_state = _execute_slurm(
                args, started_at, started_monotonic
            )
        except OSError as exc:
            exit_code = 126
            error = f"Could not execute Slurm command: {exc}"
    elif args.script is None:
        exit_code = 126
        error = "Docker-host worker is missing its fixed pipeline script"
    else:
        environment = dict(os.environ)
        environment["CONFIG_FILE"] = str(args.config)
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
        "status": (
            "completed"
            if exit_code == 0
            else "canceled"
            if scheduler_state and scheduler_state.upper().startswith("CANCELLED")
            else "failed"
        ),
        "pid": os.getpid(),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started_monotonic, 3),
        "exit_code": exit_code,
        "execution_backend": args.backend,
    }
    if scheduler_job_id:
        result["scheduler_job_id"] = scheduler_job_id
    if scheduler_state:
        result["scheduler_state"] = scheduler_state
    if error:
        result["error"] = error
    _write_json(args.status_path, result)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    if args.lock_path is None:
        return _execute_worker(args, started_at, started_monotonic)

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
                    "error": "Another Docker training/evaluation pipeline holds the run lock",
                },
            )
            return 73
        return _execute_worker(args, started_at, started_monotonic)


if __name__ == "__main__":
    raise SystemExit(main())

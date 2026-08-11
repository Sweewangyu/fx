from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .hashing import command_fingerprint


class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunResult:
    argv: tuple[str, ...]
    cwd: str
    command_hash: str
    returncode: int
    duration_seconds: float
    log_path: str


class BlackBoxRunner:
    """Run external repositories without importing them or invoking a shell parser."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(
        self,
        argv: Sequence[str],
        cwd: str | Path,
        env: Mapping[str, str],
        log_path: str | Path,
    ) -> RunResult:
        clean_argv = tuple(str(item) for item in argv)
        clean_env = {str(key): str(value) for key, value in env.items()}
        cwd_path = Path(cwd).resolve()
        log = Path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = command_fingerprint(clean_argv, cwd_path, clean_env)
        if self.dry_run:
            log.write_text(
                "DRY RUN\n" + "argv=" + repr(clean_argv) + "\nenv_keys=" + repr(sorted(clean_env)) + "\n",
                encoding="utf-8",
            )
            return RunResult(clean_argv, str(cwd_path), fingerprint, 0, 0.0, str(log))
        process_env = os.environ.copy()
        process_env.update(clean_env)
        started = time.monotonic()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"[chatts-autoresearch] command_hash={fingerprint}\n")
            stream.flush()
            process = subprocess.Popen(
                clean_argv,
                cwd=cwd_path,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
            returncode = process.wait()
        duration = time.monotonic() - started
        result = RunResult(
            clean_argv, str(cwd_path), fingerprint, returncode, duration, str(log)
        )
        if returncode:
            raise RunnerError(
                f"Command failed with exit code {returncode}; see {log} (hash {fingerprint})"
            )
        return result

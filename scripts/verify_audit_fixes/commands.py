"""Run a shell command and retain its raw evidence (stdout/stderr/exit code).

This is the primitive every gate is built on: no gate is allowed to summarize
a command's outcome in prose without also writing the raw output to disk.
"""
from __future__ import annotations

import dataclasses
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


@dataclasses.dataclass
class CommandResult:
    command: str
    started_at: str
    duration_seconds: float
    exit_code: int
    stdout: str
    stderr: str

    def to_manifest_entry(self, stdout_file: str, stderr_file: str | None = None) -> dict:
        return {
            "command": self.command,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "stdout_file": stdout_file,
            "stderr_file": stderr_file or stdout_file,
        }


def run_command(cmd: list[str], cwd: Path, timeout: int = 1800) -> CommandResult:
    """Run ``cmd`` in ``cwd`` and capture everything needed for evidence.

    Never raises on a nonzero exit code, a missing executable, or a timeout —
    those are all recorded as evidence (exit_code -1 for timeout, -2 for
    command-not-found) rather than concealed by an exception bubbling up.
    """
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        exit_code = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = -1
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or "") + f"\n[TIMEOUT after {timeout}s]"
    except FileNotFoundError as exc:
        exit_code = -2
        stdout = ""
        stderr = f"[COMMAND NOT FOUND] {exc}"

    duration = time.monotonic() - t0
    return CommandResult(
        command=" ".join(cmd),
        started_at=started.isoformat(),
        duration_seconds=round(duration, 3),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )


def write_command_output(evidence_dir: Path, filename: str, result: CommandResult) -> Path:
    """Write one command's raw evidence to ``evidence_dir / filename``."""
    path = evidence_dir / filename
    path.write_text(
        f"$ {result.command}\n"
        f"started_at: {result.started_at}\n"
        f"duration_seconds: {result.duration_seconds}\n"
        f"exit_code: {result.exit_code}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n",
        encoding="utf-8",
    )
    return path


def append_command_output(evidence_dir: Path, filename: str, result: CommandResult) -> Path:
    """Append one command's raw evidence to an existing evidence file."""
    path = evidence_dir / filename
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n$ {result.command}\n"
            f"started_at: {result.started_at}\n"
            f"duration_seconds: {result.duration_seconds}\n"
            f"exit_code: {result.exit_code}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )
    return path

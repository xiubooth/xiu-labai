from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time


@dataclass(frozen=True)
class RuntimeExecResult:
    command: tuple[str, ...]
    cwd: str
    env_overrides: dict[str, str]
    stdout: str
    stderr: str
    exit_code: int
    timeout: bool
    duration_ms: int
    output_truncated: bool
    started_at: str
    finished_at: str
    error: str = ""

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def run_runtime_command(
    command: tuple[str, ...] | list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    env_overrides: dict[str, str] | None = None,
    truncate_output_at: int = 12000,
) -> RuntimeExecResult:
    merged_env = dict(os.environ)
    overrides = dict(env_overrides or {})
    merged_env.update(overrides)
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")
    started = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=merged_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        output_truncated = False
        if len(stdout) > truncate_output_at:
            stdout = stdout[:truncate_output_at]
            output_truncated = True
        if len(stderr) > truncate_output_at:
            stderr = stderr[:truncate_output_at]
            output_truncated = True
        finished = datetime.now(timezone.utc)
        return RuntimeExecResult(
            command=tuple(command),
            cwd=str(cwd),
            env_overrides=overrides,
            stdout=stdout,
            stderr=stderr,
            exit_code=int(completed.returncode),
            timeout=False,
            duration_ms=int((time.perf_counter() - started_monotonic) * 1000),
            output_truncated=output_truncated,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
        )
    except subprocess.TimeoutExpired as exc:
        finished = datetime.now(timezone.utc)
        return RuntimeExecResult(
            command=tuple(command),
            cwd=str(cwd),
            env_overrides=overrides,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
            exit_code=-1,
            timeout=True,
            duration_ms=int((time.perf_counter() - started_monotonic) * 1000),
            output_truncated=False,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            error=f"Local process timed out after {timeout_seconds}s.",
        )
    except OSError as exc:
        finished = datetime.now(timezone.utc)
        return RuntimeExecResult(
            command=tuple(command),
            cwd=str(cwd),
            env_overrides=overrides,
            stdout="",
            stderr="",
            exit_code=-1,
            timeout=False,
            duration_ms=int((time.perf_counter() - started_monotonic) * 1000),
            output_truncated=False,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            error=f"Could not launch local process: {exc}",
        )

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
import threading
import time
from typing import Iterator, Mapping, TextIO


DEFAULT_PROGRESS_MODE = "auto"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0
DEFAULT_HEARTBEAT_INITIAL_DELAY_SECONDS = 5.0


def create_progress_reporter(
    *,
    explicit_enabled: bool | None = None,
    env: Mapping[str, str] | None = None,
    stream: TextIO | None = None,
) -> "ProgressReporter":
    resolved_stream = stream or sys.stderr
    resolved_env = env or os.environ
    raw_mode = str(resolved_env.get("LABAI_PROGRESS", DEFAULT_PROGRESS_MODE)).strip().lower()
    if raw_mode not in {"auto", "on", "off"}:
        raw_mode = DEFAULT_PROGRESS_MODE

    if explicit_enabled is not None:
        enabled = explicit_enabled
    elif raw_mode == "on":
        enabled = True
    elif raw_mode == "off":
        enabled = False
    else:
        enabled = bool(getattr(resolved_stream, "isatty", lambda: False)())

    heartbeat_interval_seconds = _read_float_env(
        resolved_env,
        "LABAI_PROGRESS_HEARTBEAT_SECONDS",
        DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    )
    heartbeat_initial_delay_seconds = _read_float_env(
        resolved_env,
        "LABAI_PROGRESS_HEARTBEAT_INITIAL_DELAY_SECONDS",
        DEFAULT_HEARTBEAT_INITIAL_DELAY_SECONDS,
    )
    return ProgressReporter(
        enabled=enabled,
        stream=resolved_stream,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        heartbeat_initial_delay_seconds=heartbeat_initial_delay_seconds,
    )


def _read_float_env(env: Mapping[str, str], key: str, default: float) -> float:
    raw_value = str(env.get(key, "")).strip()
    if not raw_value:
        return default
    try:
        parsed = float(raw_value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


@dataclass
class ProgressReporter:
    enabled: bool
    stream: TextIO
    prefix: str = "[labai]"
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    heartbeat_initial_delay_seconds: float = DEFAULT_HEARTBEAT_INITIAL_DELAY_SECONDS

    def emit(self, message: str) -> None:
        if not self.enabled:
            return
        cleaned = str(message).strip()
        if not cleaned:
            return
        self.stream.write(f"{self.prefix} {cleaned}\n")
        self.stream.flush()

    @contextmanager
    def heartbeat(
        self,
        *,
        start_message: str | None = None,
        waiting_message: str = "still waiting... {elapsed:.0f}s",
        completion_message: str | None = None,
        failure_message: str | None = None,
        interval_seconds: float | None = None,
        initial_delay_seconds: float | None = None,
    ) -> Iterator[None]:
        if start_message:
            self.emit(start_message)

        if not self.enabled:
            try:
                yield
            except Exception:
                if failure_message:
                    self.emit(failure_message)
                raise
            else:
                if completion_message:
                    self.emit(completion_message)
            return

        stop_event = threading.Event()
        start_time = time.monotonic()
        interval = interval_seconds or self.heartbeat_interval_seconds
        initial_delay = initial_delay_seconds or self.heartbeat_initial_delay_seconds

        def run() -> None:
            if stop_event.wait(initial_delay):
                return
            while not stop_event.is_set():
                elapsed = time.monotonic() - start_time
                self.emit(waiting_message.format(elapsed=elapsed))
                if stop_event.wait(interval):
                    return

        heartbeat_thread = threading.Thread(
            target=run,
            name="labai-progress-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            yield
        except Exception:
            stop_event.set()
            heartbeat_thread.join(timeout=0.2)
            if failure_message:
                self.emit(failure_message)
            raise
        else:
            stop_event.set()
            heartbeat_thread.join(timeout=0.2)
            if completion_message:
                self.emit(completion_message)

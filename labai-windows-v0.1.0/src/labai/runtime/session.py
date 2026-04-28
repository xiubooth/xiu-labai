from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


class RuntimeNotReadyError(RuntimeError):
    """Raised when runtime persistence is invoked before implementation."""


class RuntimePersistenceError(RuntimeError):
    """Raised when runtime persistence cannot complete safely."""


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    command: str
    started_at: str
    completed_at: str
    prompt: str
    mode: str
    mode_reason: str
    answer_schema: str
    read_strategy: str
    read_strategy_reason: str
    response_style: str
    response_language: str
    output_intent: str
    output_intent_reason: str
    operational_status: str
    requested_runtime: str
    runtime: str
    requested_provider: str
    provider: str
    evidence_refs: list[str] = field(default_factory=list)
    runtime_fallback: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    embedding_model: str | None = None
    status: str = "ok"
    fallback: dict[str, Any] = field(default_factory=dict)
    tools_used: bool = False
    tool_decisions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    paper_trace: dict[str, Any] = field(default_factory=dict)
    workspace_trace: dict[str, Any] = field(default_factory=dict)
    workflow_trace: dict[str, Any] = field(default_factory=dict)
    answer_artifact: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    outcome_summary: str = ""
    error: str = ""


class SessionManager:
    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = Path(sessions_dir).resolve()

    def session_path(self, session_id: str) -> Path:
        safe_session_id = _validate_session_id(session_id)
        return self.sessions_dir / f"{safe_session_id}.json"

    def write(self, record: SessionRecord) -> Path:
        path = self.session_path(record.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimePersistenceError(
                f"Session file already exists for session_id '{record.session_id}'."
            )
        payload = json.dumps(
            asdict(record),
            indent=2,
            sort_keys=True,
        )
        path.write_text(f"{payload}\n", encoding="utf-8")
        return path


def _validate_session_id(session_id: str) -> str:
    normalized = session_id.strip()
    if not normalized:
        raise RuntimePersistenceError("session_id must not be empty.")
    if any(separator in normalized for separator in ("/", "\\", ":")):
        raise RuntimePersistenceError(
            "session_id must not contain path separators or drive markers."
        )
    return normalized

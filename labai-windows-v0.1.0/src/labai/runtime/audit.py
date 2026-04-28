from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .session import RuntimePersistenceError


@dataclass(frozen=True)
class AuditRecord:
    timestamp: str
    command: str
    session_id: str
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
    model: str
    status: str
    tool_count: int
    outcome_summary: str
    prompt_preview: str
    evidence_refs: list[str] = field(default_factory=list)
    fallback: dict[str, Any] = field(default_factory=dict)
    runtime_fallback: dict[str, Any] = field(default_factory=dict)
    embedding_model: str = ""
    paper_trace: dict[str, Any] = field(default_factory=dict)
    workspace_trace: dict[str, Any] = field(default_factory=dict)
    workflow_trace: dict[str, Any] = field(default_factory=dict)
    answer_artifact: dict[str, Any] = field(default_factory=dict)
    answer_preview: str = ""
    error: str = ""


class AuditLogger:
    def __init__(self, audit_log: Path) -> None:
        self.audit_log = Path(audit_log).resolve()

    def log_path(self) -> Path:
        return self.audit_log

    def append(self, record: AuditRecord) -> Path:
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.audit_log.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(asdict(record), sort_keys=True))
                handle.write("\n")
        except OSError as exc:
            raise RuntimePersistenceError(
                f"Could not append audit record to {self.audit_log}"
            ) from exc
        return self.audit_log

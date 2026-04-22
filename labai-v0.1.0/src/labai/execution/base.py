from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from labai.config import LabaiConfig


class RuntimeAdapterError(RuntimeError):
    """Raised when an external runtime cannot execute safely."""


RuntimeStatus = Literal["ready", "unavailable", "invalid_config", "error"]


@dataclass(frozen=True)
class RuntimeHealth:
    status: RuntimeStatus
    detail: str
    available: bool
    binary_path: str = ""
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeRequest:
    prompt: str
    session_id: str
    observations: tuple[str, ...] = ()
    preferred_model: str | None = None
    mode: str = "repo_overview"
    mode_reason: str = ""
    answer_schema: str = "repo_overview_sections"
    read_strategy: str = "none"
    read_strategy_reason: str = ""
    response_style: str = "structured"
    include_explicit_evidence_refs: bool = False
    response_language: str = "en"
    evidence_refs: tuple[str, ...] = ()
    grounded_draft: str | None = None


@dataclass(frozen=True)
class RuntimeResponse:
    text: str
    runtime_name: str
    provider_name: str
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeAdapter(Protocol):
    name: str

    def healthcheck(self, config: LabaiConfig) -> RuntimeHealth:
        """Report whether the runtime is ready for one-shot execution."""

    def ask(self, config: LabaiConfig, request: RuntimeRequest) -> RuntimeResponse:
        """Execute one prompt against the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from labai.config import LabaiConfig
    from labai.runtime.progress import ProgressReporter


class ProviderError(RuntimeError):
    """Raised when provider execution fails."""


class ProviderNotImplementedError(ProviderError):
    """Raised when a configured provider is intentionally out of scope."""


class ProviderNotReadyError(ProviderError):
    """Raised when scaffold-only providers are invoked before implementation."""


@dataclass(frozen=True)
class ProviderRequest:
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
    progress_reporter: "ProgressReporter | None" = None


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    provider_name: str
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

ProviderStatus = Literal["ready", "unreachable", "invalid_config", "error"]


@dataclass(frozen=True)
class ProviderHealth:
    status: ProviderStatus
    detail: str
    available: bool
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    name: str

    def healthcheck(self, config: LabaiConfig) -> ProviderHealth:
        """Return health information for the provider."""

    def ask(self, config: LabaiConfig, request: ProviderRequest) -> ProviderResponse:
        """Handle a provider prompt."""

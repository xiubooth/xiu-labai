from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .base import (
    ProviderHealth,
    ProviderRequest,
    ProviderResponse,
)

if TYPE_CHECKING:
    from labai.config import LabaiConfig


_STRICT_LITERAL_PROMPT = re.compile(
    r"^\s*(?:say|print|output|return)\s+exactly\s+(?P<literal>.+?)\s+and\s+nothing\s+else\.?\s*$",
    re.IGNORECASE,
)


class MockProvider:
    name = "mock"
    model_name = "mock-static"

    def healthcheck(self, config: LabaiConfig) -> ProviderHealth:
        return ProviderHealth(
            status="ready",
            detail="Mock provider is ready for deterministic local responses.",
            available=True,
            model=self.model_name,
            metadata={"deterministic": True},
        )

    def ask(self, config: LabaiConfig, request: ProviderRequest) -> ProviderResponse:
        literal_response = _extract_strict_literal_response(request.prompt)
        if literal_response is not None:
            text = literal_response
        else:
            text = f"{config.mock.response_prefix}: {request.prompt}"
        if request.observations and literal_response is None:
            observation_lines = [
                f"- {observation}"
                for observation in request.observations[:8]
            ]
            if len(request.observations) > 8:
                observation_lines.append(
                    f"- ... {len(request.observations) - 8} more observations omitted"
                )
            text = "\n".join(
                [
                    text,
                    "",
                    "Repository observations:",
                    *observation_lines,
                ]
            )

        return ProviderResponse(
            text=text,
            provider_name=self.name,
            model=self.model_name,
            metadata={
                "deterministic": True,
                "observation_count": len(request.observations),
                "session_id": request.session_id,
                "mode": request.mode,
                "answer_schema": request.answer_schema,
            },
        )


def _extract_strict_literal_response(prompt: str) -> str | None:
    match = _STRICT_LITERAL_PROMPT.match(prompt)
    if not match:
        return None
    literal = match.group("literal").strip()
    if len(literal) >= 2 and literal[0] == literal[-1] and literal[0] in {'"', "'"}:
        literal = literal[1:-1].strip()
    return literal or None

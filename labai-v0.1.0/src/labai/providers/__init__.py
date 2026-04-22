from __future__ import annotations

from .base import (
    Provider,
    ProviderError,
    ProviderHealth,
    ProviderRequest,
    ProviderResponse,
)
from .deepseek import DeepSeekProvider
from .mock import MockProvider
from .ollama import OllamaProvider

_PROVIDERS: dict[str, Provider] = {
    "deepseek": DeepSeekProvider(),
    "mock": MockProvider(),
    "ollama": OllamaProvider(),
}


def get_provider(name: str) -> Provider:
    try:
        return _PROVIDERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PROVIDERS))
        raise ProviderError(
            f"Unknown provider '{name}'. Available providers: {available}"
        ) from exc


def list_provider_names() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


def get_default_provider(config: "LabaiConfig") -> Provider:
    if config.generation.active_provider == "deepseek":
        return get_provider("deepseek")
    return get_provider(config.default_provider)


__all__ = [
    "Provider",
    "ProviderError",
    "ProviderHealth",
    "ProviderRequest",
    "ProviderResponse",
    "get_default_provider",
    "get_provider",
    "list_provider_names",
]

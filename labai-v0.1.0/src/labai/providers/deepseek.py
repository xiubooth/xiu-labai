from __future__ import annotations

from contextlib import nullcontext
import json
import os
from typing import Any, TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from labai.config import get_deepseek_api_key
from labai.runtime.answer_style import normalize_answer_text

from .base import (
    ProviderError,
    ProviderHealth,
    ProviderRequest,
    ProviderResponse,
)
from .ollama import _compose_prompt

if TYPE_CHECKING:
    from labai.config import LabaiConfig


class DeepSeekProvider:
    name = "deepseek"

    def healthcheck(self, config: LabaiConfig) -> ProviderHealth:
        validation_error = _validate_deepseek_config(config)
        if validation_error is not None:
            return ProviderHealth(
                status="invalid_config",
                detail=validation_error,
                available=False,
                model=config.deepseek.general_model,
                metadata={"base_url": config.deepseek.base_url},
            )

        api_key = get_deepseek_api_key(config)
        if not api_key:
            return ProviderHealth(
                status="invalid_config",
                detail=f"missing `{config.deepseek.api_key_env}`",
                available=False,
                model=config.deepseek.general_model,
                metadata={"base_url": config.deepseek.base_url},
            )

        request = Request(
            deepseek_openai_url(config.deepseek.base_url, "/models"),
            headers=_deepseek_headers(api_key),
            method="GET",
        )
        try:
            with urlopen(request, timeout=config.deepseek.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            return ProviderHealth(
                status="unreachable",
                detail=(
                    "DeepSeek OpenAI-compatible endpoint is not reachable at "
                    f"{config.deepseek.base_url}: {exc}"
                ),
                available=False,
                model=config.deepseek.general_model,
                metadata={"base_url": config.deepseek.base_url},
            )

        data = payload.get("data", []) if isinstance(payload, dict) else []
        return ProviderHealth(
            status="ready",
            detail=f"DeepSeek OpenAI-compatible endpoint is reachable at {config.deepseek.base_url}.",
            available=True,
            model=config.deepseek.general_model,
            metadata={
                "base_url": config.deepseek.base_url,
                "model_count": len(data) if isinstance(data, list) else 0,
            },
        )

    def ask(self, config: LabaiConfig, request: ProviderRequest) -> ProviderResponse:
        validation_error = _validate_deepseek_config(config)
        if validation_error is not None:
            raise ProviderError(validation_error)

        api_key = get_deepseek_api_key(config)
        if not api_key:
            raise ProviderError(f"missing `{config.deepseek.api_key_env}`")

        selected_model = select_deepseek_model(
            config,
            prompt=request.prompt,
            preferred_model=request.preferred_model,
        )
        progress_reporter = request.progress_reporter
        if progress_reporter is not None:
            progress_reporter.emit(f"model selected: provider=deepseek model={selected_model}")
        payload = {
            "model": selected_model,
            "messages": [
                {
                    "role": "user",
                    "content": _compose_prompt(request),
                }
            ],
            "stream": False,
            "max_tokens": config.deepseek.max_tokens,
        }
        http_request = Request(
            deepseek_openai_url(config.deepseek.base_url, "/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers=_deepseek_headers(api_key),
            method="POST",
        )

        if progress_reporter is not None:
            progress_reporter.emit(
                f"model call started: provider=deepseek model={selected_model}"
            )
        try:
            with (
                progress_reporter.heartbeat(
                    waiting_message="still waiting for model response... {elapsed:.0f}s",
                    failure_message="model call failed: provider=deepseek",
                    completion_message="model call completed: provider=deepseek",
                )
                if progress_reporter is not None
                else nullcontext()
            ):
                with urlopen(http_request, timeout=config.deepseek.timeout_seconds) as response:
                    raw_response = response.read().decode("utf-8")
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            raise ProviderError(
                f"DeepSeek request failed at {config.deepseek.base_url}: {exc}"
            ) from exc

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ProviderError("DeepSeek returned invalid JSON.") from exc

        message = _first_message(data)
        text = _message_text(message.get("content"))
        if not text:
            raise ProviderError("DeepSeek returned an empty response.")

        text = normalize_answer_text(
            text,
            response_language=request.response_language,
            response_style=request.response_style,
            include_explicit_evidence_refs=request.include_explicit_evidence_refs,
        )
        return ProviderResponse(
            text=text,
            provider_name=self.name,
            model=_response_model(data) or selected_model,
            metadata={
                "base_url": config.deepseek.base_url,
                "reasoning_content_present": bool(message.get("reasoning_content")),
                "finish_reason": _first_choice(data).get("finish_reason", ""),
            },
        )


def run_deepseek_direct_smoke(
    config: LabaiConfig,
    *,
    prompt: str = "reply with the word ready",
    model: str | None = None,
) -> dict[str, Any]:
    validation_error = _validate_deepseek_config(config)
    if validation_error is not None:
        return {
            "status": "invalid_config",
            "available": False,
            "detail": validation_error,
            "base_url": config.deepseek.base_url,
            "model": model or config.deepseek.general_model,
            "key_present": False,
            "failure_kind": "invalid_config",
            "max_tokens": config.deepseek.smoke_max_tokens,
        }

    api_key = get_deepseek_api_key(config)
    if not api_key:
        return {
            "status": "invalid_config",
            "available": False,
            "detail": f"missing `{config.deepseek.api_key_env}`",
            "base_url": config.deepseek.base_url,
            "model": model or config.deepseek.general_model,
            "key_present": False,
            "failure_kind": "missing_api_key",
            "max_tokens": config.deepseek.smoke_max_tokens,
        }

    selected_model = model or config.deepseek.general_model
    payload = {
        "model": selected_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": config.deepseek.smoke_max_tokens,
    }
    http_request = Request(
        deepseek_openai_url(config.deepseek.base_url, "/chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers=_deepseek_headers(api_key),
        method="POST",
    )

    try:
        with urlopen(http_request, timeout=config.deepseek.timeout_seconds) as response:
            raw_response = response.read().decode("utf-8")
    except HTTPError as exc:
        return _deepseek_http_failure(
            config,
            exc,
            model=selected_model,
            failure_context="direct chat completion",
        )
    except (URLError, OSError, TimeoutError) as exc:
        return {
            "status": "unreachable",
            "available": False,
            "detail": (
                "DeepSeek direct chat completion is not reachable at "
                f"{config.deepseek.base_url}: {exc}"
            ),
            "base_url": config.deepseek.base_url,
            "model": selected_model,
            "key_present": True,
            "failure_kind": _classify_deepseek_failure(str(exc), status_code=None),
            "max_tokens": config.deepseek.smoke_max_tokens,
        }

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "available": False,
            "detail": "DeepSeek direct chat completion returned invalid JSON.",
            "base_url": config.deepseek.base_url,
            "model": selected_model,
            "key_present": True,
            "failure_kind": "invalid_json",
            "max_tokens": config.deepseek.smoke_max_tokens,
        }

    message = _first_message(data)
    text = _message_text(message.get("content"))
    if not text:
        return {
            "status": "error",
            "available": False,
            "detail": "DeepSeek direct chat completion returned empty content.",
            "base_url": config.deepseek.base_url,
            "model": _response_model(data) or selected_model,
            "key_present": True,
            "failure_kind": "empty_response",
            "reasoning_content_present": bool(message.get("reasoning_content")),
            "max_tokens": config.deepseek.smoke_max_tokens,
        }

    return {
        "status": "ready",
        "available": True,
        "detail": (
            "DeepSeek direct chat completion succeeded at "
            f"{config.deepseek.base_url}."
        ),
        "base_url": config.deepseek.base_url,
        "model": _response_model(data) or selected_model,
        "key_present": True,
        "failure_kind": "",
        "reasoning_content_present": bool(message.get("reasoning_content")),
        "max_tokens": config.deepseek.smoke_max_tokens,
    }


def deepseek_openai_url(base_url: str, path: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}{path}"
    return f"{normalized}/v1{path}"


def select_deepseek_model(
    config: LabaiConfig,
    *,
    prompt: str,
    preferred_model: str | None = None,
) -> str:
    override = os.environ.get("LABAI_MODEL_PROFILE", "").strip()
    supported = {
        config.deepseek.general_model,
        config.deepseek.code_model,
        config.deepseek.reasoning_model,
    }
    if override in supported:
        return override
    if preferred_model and preferred_model in supported:
        return preferred_model
    prompt_lower = prompt.lower()
    if any(token in prompt_lower for token in ("code", "python", "function", "class", "module", "test")):
        return config.deepseek.code_model
    return config.deepseek.general_model


def _validate_deepseek_config(config: LabaiConfig) -> str | None:
    parsed = urlparse(config.deepseek.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "DeepSeek base_url must be an absolute URL."
    if not config.deepseek.api_key_env.strip():
        return "DeepSeek api_key_env must not be empty."
    return None


def _deepseek_headers(api_key: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _deepseek_http_failure(
    config: LabaiConfig,
    exc: HTTPError,
    *,
    model: str,
    failure_context: str,
) -> dict[str, Any]:
    response_text = ""
    try:
        response_text = exc.read().decode("utf-8", errors="replace")
    except Exception:
        response_text = ""
    detail = (
        f"DeepSeek {failure_context} failed at {config.deepseek.base_url}: "
        f"HTTP {exc.code} {exc.reason}"
    )
    body_message = _extract_error_message(response_text)
    if body_message:
        detail = f"{detail} | {body_message}"
    return {
        "status": "unreachable",
        "available": False,
        "detail": detail,
        "base_url": config.deepseek.base_url,
        "model": model,
        "key_present": True,
        "max_tokens": config.deepseek.smoke_max_tokens,
        "failure_kind": _classify_deepseek_failure(
            body_message or str(exc),
            status_code=exc.code,
        ),
    }


def _classify_deepseek_failure(message: str, *, status_code: int | None) -> str:
    lowered = message.lower()
    if status_code in {401, 403} or any(
        token in lowered for token in ("unauthorized", "invalid api key", "authentication", "forbidden")
    ):
        return "auth_failed"
    if status_code == 400 and "max_tokens" in lowered:
        return "max_tokens_invalid"
    if status_code == 404 or "not found" in lowered:
        return "base_url_or_route_error"
    if status_code == 400 and "model" in lowered:
        return "model_rejected"
    if any(
        token in lowered
        for token in (
            "model not found",
            "unsupported model",
            "unknown model",
            "does not exist",
        )
    ):
        return "model_rejected"
    if any(
        token in lowered
        for token in (
            "timed out",
            "timeout",
            "connection refused",
            "name resolution",
            "temporary failure in name resolution",
            "network",
        )
    ):
        return "network_error"
    return "request_failed"


def _extract_error_message(response_text: str) -> str:
    if not response_text.strip():
        return ""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text.strip()[:200]
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message.strip()
    message = payload.get("message")
    if isinstance(message, str):
        return message.strip()
    return ""


def _first_choice(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    return first if isinstance(first, dict) else {}


def _first_message(payload: object) -> dict[str, Any]:
    message = _first_choice(payload).get("message")
    return message if isinstance(message, dict) else {}


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks).strip()
    return ""


def _response_model(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return str(model) if model else None

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import MappingProxyType
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from labai.config import LabaiConfig, get_deepseek_api_key
from labai.providers.deepseek import deepseek_openai_url, select_deepseek_model
from labai.runtime.answer_style import (
    build_rewrite_requirements,
    needs_style_repair,
    normalize_answer_text,
)

from .base import RuntimeAdapterError, RuntimeHealth, RuntimeRequest, RuntimeResponse

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_UNSET_PROVIDER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENROUTER_API_KEY",
    "XAI_API_KEY",
)
_TEXT_KEYS = (
    "text",
    "result",
    "response",
    "content",
    "message",
    "output_text",
)
_MODEL_KEYS = ("model", "model_name")
_STRICT_LITERAL_PROMPT = re.compile(
    r"^\s*(?:say|print|output|return)\s+exactly\s+(?P<literal>.+?)\s+and\s+nothing\s+else\.?\s*$",
    re.IGNORECASE,
)

CLAW_REFERENCE_FACTS: Mapping[str, Any] = MappingProxyType(
    {
        "canonical_runtime_dir": "rust/",
        "doctor_first_check": "claw doctor",
        "one_shot_prompt_shape": 'claw prompt "..."',
        "json_output_flag": "--output-format json",
        "permission_mode_flag": "--permission-mode read-only",
        "allowed_tools_flag": "--allowedTools read,glob",
        "local_endpoint_env": "OPENAI_BASE_URL",
        "ollama_local_endpoint": "http://127.0.0.1:11434/v1",
        "deepseek_endpoint": "https://api.deepseek.com/v1",
        "qwen_local_guidance": (
            "LabAI config keeps plain local Ollama names such as qwen2.5:7b, "
            "then passes them to Claw as openai/<model> with OPENAI_BASE_URL "
            "pointing at the local Ollama OpenAI-compatible endpoint. Avoid "
            "qwen/ and qwen- prefixes because those route to cloud Qwen/DashScope."
        ),
    }
)


@dataclass(frozen=True)
class LocalProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class ClawRuntimeAdapter:
    name = "claw"

    def healthcheck(self, config: LabaiConfig) -> RuntimeHealth:
        endpoint_health = _check_generation_endpoint(config)
        binary_path = resolve_claw_binary(config)
        metadata = {
            "reference": dict(CLAW_REFERENCE_FACTS),
            "doctor_command": build_claw_doctor_command(config, binary_path),
            "endpoint_health": endpoint_health,
            "allowed_tools": list(config.claw.allowed_tools),
            "permission_mode": config.claw.permission_mode,
            "output_format": config.claw.output_format,
            "selected_model": _select_model(config, "", preferred_model=None),
        }
        if config.generation.active_provider == "deepseek":
            metadata["prompt_smoke"] = {
                "status": "skipped",
                "available": False,
                "detail": "DeepSeek prompt smoke did not run yet.",
                "selected_model": _select_model(config, "", preferred_model=None),
                "base_url": deepseek_openai_url(config.deepseek.base_url, ""),
                "key_present": bool(get_deepseek_api_key(config)),
                "failure_kind": "",
                "max_output_tokens": config.deepseek.smoke_max_tokens,
            }

        if binary_path is None:
            return RuntimeHealth(
                status="unavailable",
                detail=(
                    f"Claw binary '{config.claw.binary}' was not found. "
                    "Native fallback should be used."
                ),
                available=False,
                model=str(metadata["selected_model"]),
                metadata=metadata,
            )

        if not endpoint_health["available"]:
            return RuntimeHealth(
                status="invalid_config" if endpoint_health["status"] == "invalid_config" else "unavailable",
                detail=str(endpoint_health["detail"]),
                available=False,
                binary_path=str(binary_path),
                model=str(metadata["selected_model"]),
                metadata=metadata,
            )

        command = build_claw_doctor_command(config, binary_path)
        try:
            completed = _run_command(
                command,
                cwd=config.workspace.active_workspace_root,
                env=_build_runtime_env(
                    config,
                    max_output_tokens_override=(
                        config.deepseek.smoke_max_tokens
                        if config.generation.active_provider == "deepseek"
                        else None
                    ),
                ),
                timeout_seconds=config.claw.health_timeout_seconds,
            )
        except RuntimeAdapterError as exc:
            return RuntimeHealth(
                status="error",
                detail=f"Claw doctor preflight failed: {exc}",
                available=False,
                binary_path=str(binary_path),
                model=str(metadata["selected_model"]),
                metadata=metadata,
            )

        if completed.returncode != 0:
            failure = _render_process_failure(completed)
            return RuntimeHealth(
                status="error",
                detail=f"Claw doctor preflight returned a non-zero exit code. {failure}",
                available=False,
                binary_path=str(binary_path),
                model=str(metadata["selected_model"]),
                metadata=metadata,
            )

        if config.generation.active_provider == "deepseek":
            prompt_smoke = run_claw_prompt_smoke(
                config,
                binary_path=binary_path,
                endpoint_health=endpoint_health,
            )
            metadata["prompt_smoke"] = prompt_smoke
            if not prompt_smoke["available"]:
                prompt_status = str(prompt_smoke.get("status", "error"))
                if prompt_status not in {"unavailable", "invalid_config", "error"}:
                    prompt_status = "error"
                return RuntimeHealth(
                    status=prompt_status,
                    detail=f"Claw DeepSeek prompt smoke failed: {prompt_smoke['detail']}",
                    available=False,
                    binary_path=str(binary_path),
                    model=str(prompt_smoke.get("selected_model") or metadata["selected_model"]),
                    metadata=metadata,
                )

        return RuntimeHealth(
            status="ready",
            detail=(
                "Claw runtime and the selected OpenAI-compatible generation backend are ready."
            ),
            available=True,
            binary_path=str(binary_path),
            model=_select_model(config, "", preferred_model=None),
            metadata=metadata,
        )

    def ask(self, config: LabaiConfig, request: RuntimeRequest) -> RuntimeResponse:
        binary_path = resolve_claw_binary(config)
        if binary_path is None:
            raise RuntimeAdapterError(
                f"Claw binary '{config.claw.binary}' was not found."
            )

        endpoint_health = _check_generation_endpoint(config)
        if not endpoint_health["available"]:
            raise RuntimeAdapterError(str(endpoint_health["detail"]))

        selected_model = _select_model(
            config,
            request.prompt,
            preferred_model=request.preferred_model,
        )
        progress_reporter = request.progress_reporter
        if progress_reporter is not None:
            progress_reporter.emit(
                f"model selected: runtime=claw model={selected_model}"
            )
        command = build_claw_prompt_command(
            config,
            binary_path,
            prompt=_compose_prompt(request),
            model=selected_model,
        )

        if progress_reporter is not None:
            progress_reporter.emit(f"model call started: runtime=claw model={selected_model}")
        try:
            with (
                progress_reporter.heartbeat(
                    waiting_message="still waiting for model response... {elapsed:.0f}s",
                )
                if progress_reporter is not None
                else nullcontext()
            ):
                completed = _run_command(
                    command,
                    cwd=config.workspace.active_workspace_root,
                    env=_build_runtime_env(config),
                    timeout_seconds=config.claw.ask_timeout_seconds,
                )
            if completed.returncode != 0:
                raise RuntimeAdapterError(_render_process_failure(completed))

            payload = _load_json_payload(completed.stdout)
            text = _extract_text(payload)
            if not text:
                raise RuntimeAdapterError(
                    "Claw returned JSON output, but no answer text could be extracted."
                )
            text = normalize_answer_text(
                text,
                response_language=request.response_language,
                response_style=request.response_style,
                include_explicit_evidence_refs=request.include_explicit_evidence_refs,
            )
            if needs_style_repair(
                text,
                prompt=request.prompt,
                response_language=request.response_language,
                response_style=request.response_style,
                include_explicit_evidence_refs=request.include_explicit_evidence_refs,
            ):
                text = _repair_text_response(
                    config,
                    binary_path,
                    model=selected_model,
                    request=request,
                    answer_text=text,
                )
            text = normalize_answer_text(
                text,
                response_language=request.response_language,
                response_style=request.response_style,
                include_explicit_evidence_refs=request.include_explicit_evidence_refs,
            )
        except RuntimeAdapterError as exc:
            if progress_reporter is not None:
                progress_reporter.emit(f"model call failed: runtime=claw reason={_snippet(str(exc), limit=180)}")
            raise
        if progress_reporter is not None:
            progress_reporter.emit("model call completed: runtime=claw fallback=none")
        strict_literal = _extract_strict_literal_response(request.prompt)
        if strict_literal is not None:
            text = strict_literal

        model_name = _extract_model(payload) or selected_model
        metadata = {
            "reference": dict(CLAW_REFERENCE_FACTS),
            "command": command,
            "selected_model": selected_model,
            "payload_kind": type(payload).__name__,
        }

        return RuntimeResponse(
            text=text,
            runtime_name=self.name,
            provider_name=self.name,
            model=model_name,
            metadata=metadata,
        )


def resolve_claw_binary(config: LabaiConfig) -> Path | None:
    configured = os.environ.get("LABAI_CLAW_BINARY", "").strip() or config.claw.binary.strip()
    explicit_path = Path(configured)

    if _looks_like_path(configured):
        candidate = explicit_path
        if not candidate.is_absolute():
            candidate = config.project_root / candidate
        resolved = candidate.resolve()
        return resolved if resolved.is_file() else None

    resolved_which = shutil.which(configured)
    if resolved_which:
        return Path(resolved_which).resolve()

    for candidate in _iter_binary_candidates(config):
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def build_claw_doctor_command(config: LabaiConfig, binary_path: Path | None) -> list[str]:
    command = [str(binary_path or config.claw.binary), "doctor"]
    if config.claw.output_format:
        command.extend(["--output-format", config.claw.output_format])
    return command


def build_claw_prompt_command(
    config: LabaiConfig,
    binary_path: Path,
    *,
    prompt: str,
    model: str,
) -> list[str]:
    return [
        str(binary_path),
        "prompt",
        prompt,
        "--model",
        model,
        "--output-format",
        config.claw.output_format,
        "--permission-mode",
        config.claw.permission_mode,
        "--allowedTools",
        ",".join(config.claw.allowed_tools),
    ]


def _check_local_openai_endpoint(config: LabaiConfig) -> dict[str, Any]:
    parsed = urlparse(config.ollama.openai_base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {
            "status": "invalid_config",
            "available": False,
            "detail": "Ollama OpenAI-compatible base URL must be absolute.",
            "base_url": config.ollama.openai_base_url,
        }

    host = (parsed.hostname or "").lower()
    if host not in _LOCAL_HOSTS:
        return {
            "status": "invalid_config",
            "available": False,
            "detail": "Ollama OpenAI-compatible base URL must stay on a local loopback host.",
            "base_url": config.ollama.openai_base_url,
        }

    request = Request(
        _openai_models_url(config.ollama.openai_base_url),
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=config.ollama.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "available": False,
            "detail": (
                "Local Ollama OpenAI-compatible endpoint is not reachable at "
                f"{config.ollama.openai_base_url}: {exc}"
            ),
            "base_url": config.ollama.openai_base_url,
        }

    model_count = len(payload.get("data", [])) if isinstance(payload, dict) else 0
    return {
        "status": "ready",
        "available": True,
        "detail": (
            "Local Ollama OpenAI-compatible endpoint is reachable at "
            f"{config.ollama.openai_base_url}."
        ),
        "base_url": config.ollama.openai_base_url,
        "model_count": model_count,
    }


def _check_generation_endpoint(config: LabaiConfig) -> dict[str, Any]:
    if config.generation.active_provider == "deepseek":
        return _check_deepseek_openai_endpoint(config)
    return _check_local_openai_endpoint(config)


def _check_deepseek_openai_endpoint(config: LabaiConfig) -> dict[str, Any]:
    api_key = get_deepseek_api_key(config)
    if not api_key:
        return {
            "status": "invalid_config",
            "available": False,
            "detail": f"missing `{config.deepseek.api_key_env}`",
            "base_url": config.deepseek.base_url,
        }

    parsed = urlparse(config.deepseek.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {
            "status": "invalid_config",
            "available": False,
            "detail": "DeepSeek OpenAI-compatible base URL must be absolute.",
            "base_url": config.deepseek.base_url,
        }

    request = Request(
        deepseek_openai_url(config.deepseek.base_url, "/models"),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=config.deepseek.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "available": False,
            "detail": (
                "DeepSeek OpenAI-compatible endpoint is not reachable at "
                f"{config.deepseek.base_url}: {exc}"
            ),
            "base_url": config.deepseek.base_url,
        }

    model_count = len(payload.get("data", [])) if isinstance(payload, dict) else 0
    return {
        "status": "ready",
        "available": True,
        "detail": (
            "DeepSeek OpenAI-compatible endpoint is reachable at "
            f"{config.deepseek.base_url}."
        ),
        "base_url": config.deepseek.base_url,
        "model_count": model_count,
    }


def _openai_models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def run_claw_prompt_smoke(
    config: LabaiConfig,
    *,
    prompt: str = "reply with the word ready",
    preferred_model: str | None = None,
    binary_path: Path | None = None,
    endpoint_health: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_binary = binary_path or resolve_claw_binary(config)
    if resolved_binary is None:
        return {
            "status": "unavailable",
            "available": False,
            "detail": f"Claw binary '{config.claw.binary}' was not found.",
            "selected_model": _select_model(config, prompt, preferred_model=preferred_model),
            "base_url": "",
            "key_present": False,
            "failure_kind": "binary_missing",
            "exit_code": None,
            "stdout_snippet": "",
            "stderr_snippet": "",
        }

    observed_endpoint = dict(endpoint_health or _check_generation_endpoint(config))
    selected_model = _select_model(config, prompt, preferred_model=preferred_model)
    smoke_max_output_tokens = (
        config.deepseek.smoke_max_tokens
        if config.generation.active_provider == "deepseek"
        else None
    )
    env = _build_runtime_env(
        config,
        max_output_tokens_override=smoke_max_output_tokens,
    )
    base_url = env.get("OPENAI_BASE_URL", "")
    key_present = bool(env.get("OPENAI_API_KEY"))
    if not observed_endpoint.get("available", False):
        return {
            "status": str(observed_endpoint.get("status", "invalid_config")),
            "available": False,
            "detail": str(observed_endpoint.get("detail", "Generation endpoint is not ready.")),
            "selected_model": selected_model,
            "base_url": base_url,
            "key_present": key_present,
            "failure_kind": str(observed_endpoint.get("failure_kind", "endpoint_not_ready")),
            "exit_code": None,
            "stdout_snippet": "",
            "stderr_snippet": "",
        }

    command = build_claw_prompt_command(
        config,
        resolved_binary,
        prompt=prompt,
        model=selected_model,
    )
    try:
        completed = _run_command(
            command,
            cwd=config.workspace.active_workspace_root,
            env=env,
            timeout_seconds=_claw_smoke_timeout_seconds(config),
        )
    except RuntimeAdapterError as exc:
        detail = str(exc)
        return {
            "status": "error",
            "available": False,
            "detail": detail,
            "selected_model": selected_model,
            "base_url": base_url,
            "key_present": key_present,
            "failure_kind": _classify_claw_smoke_failure(detail),
            "exit_code": None,
            "stdout_snippet": "",
            "stderr_snippet": "",
            "max_output_tokens": (
                smoke_max_output_tokens or 0
            ),
        }

    stdout_snippet = _snippet(completed.stdout)
    stderr_snippet = _snippet(completed.stderr)
    if completed.returncode != 0:
        detail = _render_process_failure(completed)
        return {
            "status": "error",
            "available": False,
            "detail": detail,
            "selected_model": selected_model,
            "base_url": base_url,
            "key_present": key_present,
            "failure_kind": _classify_claw_smoke_failure(detail),
            "exit_code": completed.returncode,
            "stdout_snippet": stdout_snippet,
            "stderr_snippet": stderr_snippet,
            "max_output_tokens": (
                smoke_max_output_tokens or 0
            ),
        }

    try:
        payload = _load_json_payload(completed.stdout)
        text = _extract_text(payload)
        response_model = _extract_model(payload) or selected_model
    except RuntimeAdapterError as exc:
        return {
            "status": "error",
            "available": False,
            "detail": str(exc),
            "selected_model": selected_model,
            "base_url": base_url,
            "key_present": key_present,
            "failure_kind": "invalid_json",
            "exit_code": completed.returncode,
            "stdout_snippet": stdout_snippet,
            "stderr_snippet": stderr_snippet,
            "max_output_tokens": (
                smoke_max_output_tokens or 0
            ),
        }

    if not text:
        return {
            "status": "error",
            "available": False,
            "detail": "Claw prompt smoke returned no answer text.",
            "selected_model": response_model,
            "base_url": base_url,
            "key_present": key_present,
            "failure_kind": "empty_response",
            "exit_code": completed.returncode,
            "stdout_snippet": stdout_snippet,
            "stderr_snippet": stderr_snippet,
            "max_output_tokens": (
                smoke_max_output_tokens or 0
            ),
        }

    return {
        "status": "ready",
        "available": True,
        "detail": "Claw prompt smoke succeeded with the configured OpenAI-compatible backend.",
        "selected_model": response_model,
        "base_url": base_url,
        "key_present": key_present,
        "failure_kind": "",
        "exit_code": completed.returncode,
        "stdout_snippet": stdout_snippet,
        "stderr_snippet": stderr_snippet,
        "max_output_tokens": (
            smoke_max_output_tokens or 0
        ),
    }


def _claw_smoke_timeout_seconds(config: LabaiConfig) -> int:
    return min(config.claw.ask_timeout_seconds, max(config.claw.health_timeout_seconds, 30))


def _classify_claw_smoke_failure(detail: str) -> str:
    lowered = detail.lower()
    if "invalid model syntax" in lowered or "expected provider/model" in lowered:
        return "claw_model_syntax_failed"
    if "max_tokens" in lowered:
        return "max_tokens_invalid"
    if any(
        token in lowered
        for token in (
            "anthropic",
            "dashscope",
            "qwen/",
            "qwen-",
        )
    ):
        return "provider_routing_conflict"
    if any(
        token in lowered for token in ("unauthorized", "authentication", "forbidden", "invalid api key")
    ):
        return "auth_failed"
    if any(
        token in lowered
        for token in (
            "model not found",
            "unsupported model",
            "unknown model",
            "does not exist",
            "invalid model",
        )
    ):
        return "model_rejected"
    if any(
        token in lowered
        for token in (
            "connection refused",
            "timed out",
            "timeout",
            "name resolution",
            "network",
            "dns",
            "failed to connect",
        )
    ):
        return "network_error"
    if any(
        token in lowered
        for token in (
            "openai_base_url",
            "openai base url",
            "base url",
            "404",
        )
    ):
        return "base_url_or_route_error"
    if "invalid json" in lowered:
        return "invalid_json"
    return "prompt_failed"


def _snippet(value: str, *, limit: int = 240) -> str:
    normalized = " ".join((value or "").strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _build_runtime_env(
    config: LabaiConfig,
    *,
    max_output_tokens_override: int | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if config.generation.active_provider == "deepseek":
        env["OPENAI_BASE_URL"] = deepseek_openai_url(config.deepseek.base_url, "")
    else:
        env["OPENAI_BASE_URL"] = config.ollama.openai_base_url

    if not env.get("HOME"):
        user_profile = env.get("USERPROFILE", "").strip()
        if user_profile:
            env["HOME"] = user_profile

    if config.generation.active_provider == "deepseek":
        api_key = get_deepseek_api_key(config)
    else:
        api_key = config.ollama.openai_api_key.strip()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    else:
        env.pop("OPENAI_API_KEY", None)

    if config.generation.active_provider == "deepseek":
        override = max_output_tokens_override or config.deepseek.max_tokens
        env["CLAW_CONFIG_HOME"] = str(
            _ensure_deepseek_claw_config_home(config, max_output_tokens=override)
        )

    for variable_name in _UNSET_PROVIDER_ENV_VARS:
        env.pop(variable_name, None)

    return env


def _ensure_deepseek_claw_config_home(
    config: LabaiConfig,
    *,
    max_output_tokens: int,
) -> Path:
    config_home = (
        config.project_root
        / ".labai"
        / "runtime"
        / "claw"
        / f"deepseek-max-{int(max_output_tokens)}"
    )
    config_home.mkdir(parents=True, exist_ok=True)
    settings_path = config_home / "settings.json"
    payload = {
        "plugins": {
            "maxOutputTokens": int(max_output_tokens),
        }
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if not settings_path.exists() or settings_path.read_text(encoding="utf-8") != rendered:
        settings_path.write_text(rendered, encoding="utf-8")
    return config_home


def run_local_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    env: Mapping[str, str] | None = None,
) -> LocalProcessResult:
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(dict(env))
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")
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
    except subprocess.TimeoutExpired as exc:
        raise RuntimeAdapterError(
            f"Local process timed out after {timeout_seconds}s."
        ) from exc
    except OSError as exc:
        raise RuntimeAdapterError(f"Could not launch local process: {exc}") from exc
    return LocalProcessResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    merged_env = dict(os.environ)
    merged_env.update(dict(env))
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        return subprocess.run(
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
    except subprocess.TimeoutExpired as exc:
        raise RuntimeAdapterError(
            f"Claw command timed out after {timeout_seconds}s."
        ) from exc
    except OSError as exc:
        raise RuntimeAdapterError(f"Could not launch Claw: {exc}") from exc


def _render_process_failure(result: subprocess.CompletedProcess[str]) -> str:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or f"exit_code={result.returncode}"
    return detail


def _load_json_payload(stdout: str) -> object:
    content = stdout.strip()
    if not content:
        raise RuntimeAdapterError("Claw returned no JSON output.")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        events: list[object] = []
        for line in content.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            try:
                events.append(json.loads(candidate))
            except json.JSONDecodeError:
                continue
        if events:
            return events
        raise RuntimeAdapterError("Claw returned invalid JSON output.")


def _extract_text(payload: object) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, Mapping):
        direct = _extract_mapping_text(payload)
        if direct:
            return direct
        for value in payload.values():
            nested = _extract_text(value)
            if nested:
                return nested
        return ""
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray, str)):
        for item in reversed(list(payload)):
            nested = _extract_text(item)
            if nested:
                return nested
    return ""


def _extract_mapping_text(payload: Mapping[str, Any]) -> str:
    for key in _TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_model(payload: object) -> str | None:
    if isinstance(payload, Mapping):
        for key in _MODEL_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            nested = _extract_model(value)
            if nested:
                return nested
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray, str)):
        for item in payload:
            nested = _extract_model(item)
            if nested:
                return nested
    return None


def _compact_prompt_lines(
    lines: Sequence[str],
    *,
    max_items: int,
    max_chars: int,
    omission_label: str,
) -> list[str]:
    kept: list[str] = []
    char_count = 0
    for line in lines:
        projected = char_count + len(line)
        if len(kept) >= max_items or projected > max_chars:
            break
        kept.append(line)
        char_count = projected
    omitted = len(lines) - len(kept)
    if omitted > 0:
        kept.append(f"- {omitted} additional {omission_label} omitted for prompt compactness.")
    return kept or [f"- No {omission_label} retained after prompt compaction."]


def _compose_prompt(request: RuntimeRequest) -> str:
    if request.answer_schema == "brief_response" and not request.observations:
        return request.prompt
    if request.mode == "general_chat":
        return _compose_general_chat_prompt(request)

    evidence_lines = [f"- {item}" for item in request.evidence_refs] or ["- None yet"]
    observation_lines = [f"- {observation}" for observation in request.observations] or ["- None yet"]
    section_lines = [f"- {item}" for item in _mode_outline(request.mode)]
    constraint_lines = [f"- {item}" for item in _mode_constraints(request.mode)]
    draft_lines = request.grounded_draft.splitlines() if request.grounded_draft else []
    recurring_limitations_prompt = request.mode == "paper_grounded_qa" and "recurring limitations" in request.prompt.lower()
    if recurring_limitations_prompt:
        evidence_lines = _compact_prompt_lines(
            evidence_lines,
            max_items=8,
            max_chars=1200,
            omission_label="evidence refs",
        )
        observation_lines = _compact_prompt_lines(
            observation_lines,
            max_items=12,
            max_chars=2600,
            omission_label="observations",
        )
        draft_lines = _compact_prompt_lines(
            draft_lines,
            max_items=24,
            max_chars=7000,
            omission_label="grounded draft lines",
        )
    if request.response_style == "continuous_prose":
        section_intro = "Cover these points in continuous prose when they are relevant:"
        style_lines = [
            "Return continuous prose rather than bullets, outline blocks, or rigid section headings.",
            "Do not print runtime metadata, internal labels, or evidence appendices unless the user explicitly asked for them.",
            "Do not return JSON, YAML, code fences, or key-value objects.",
        ]
    else:
        section_intro = "Use natural section headings and cover these points when they are relevant:"
        style_lines = [
            "Use a readable structured answer shape when it helps the user.",
            "Prefer natural-language sections or paragraphs instead of JSON, YAML, or key-value objects unless the user explicitly asked for them.",
        ]
    evidence_output_line = (
        "Include compact evidence references in the answer when they materially support the claim."
        if request.include_explicit_evidence_refs
        else "Do not add a dedicated evidence/files consulted section unless the user explicitly asked for citations or grounded support."
    )
    sections = [
        "You are operating through the labai control plane for a single-turn local research request.",
        f"Internal mode: {request.mode}",
        f"Mode rationale: {request.mode_reason}",
        f"Answer schema: {request.answer_schema}",
        f"Read strategy: {request.read_strategy}",
        f"Read strategy rationale: {request.read_strategy_reason}",
        f"Requested response style: {request.response_style}",
        f"Target response language: {request.response_language}",
        "Preserve the user's language in the answer.",
        "If the target language is zh-CN, translate all prose and section headings into Simplified Chinese. Keep only code identifiers, file paths, and CLI literals in English.",
        "Do not alter or invent file paths, function names, class names, config keys, or CLI literals. Reproduce identifiers exactly as they appear in the evidence or grounded draft.",
        "In this project, `claw` means the local external runtime CLI integrated by the repository. Do not reinterpret it as a cloud acronym or SaaS product.",
        "Use the provided repository observations and evidence first.",
        "Base concrete claims on the consulted evidence. If something is not confirmed by the consulted files, say that it is not confirmed.",
        "Do not ask the user to provide repository paths, file paths, or glob patterns.",
        "If you need more context, stay inside the current repository, start from the current workspace root, and use only read-only inspection.",
        "If the evidence below is sufficient, answer directly without asking follow-up questions.",
        "Do not print raw schema identifiers such as repo_overview_sections or architecture_review_sections.",
        *style_lines,
        evidence_output_line,
        section_intro,
        *section_lines,
        "Respect these project constraints:",
        *constraint_lines,
        "",
        "User prompt:",
        request.prompt,
        "",
        "Evidence consulted so far:",
        *evidence_lines,
        "",
        "Repository observations:",
        *observation_lines,
        "",
        "Grounded draft (treat this as the factual scaffold; rewrite it for clarity, but do not add new facts):",
        *(draft_lines or ["- No grounded draft provided."]),
        "",
        "Do not mention session ids, audit paths, runtime names, selected_mode, or other operational metadata in the answer body.",
    ]
    prompt_text = "\n".join(sections)
    if len(prompt_text) <= 18000:
        return prompt_text

    evidence_lines = _compact_prompt_lines(
        [f"- {item}" for item in request.evidence_refs] or ["- None yet"],
        max_items=6,
        max_chars=900,
        omission_label="evidence refs",
    )
    observation_lines = _compact_prompt_lines(
        [f"- {observation}" for observation in request.observations] or ["- None yet"],
        max_items=8,
        max_chars=1800,
        omission_label="observations",
    )
    draft_lines = _compact_prompt_lines(
        request.grounded_draft.splitlines() if request.grounded_draft else [],
        max_items=18,
        max_chars=5000,
        omission_label="grounded draft lines",
    )
    sections = [
        "You are operating through the labai control plane for a single-turn local research request.",
        f"Internal mode: {request.mode}",
        f"Mode rationale: {request.mode_reason}",
        f"Answer schema: {request.answer_schema}",
        f"Read strategy: {request.read_strategy}",
        f"Read strategy rationale: {request.read_strategy_reason}",
        f"Requested response style: {request.response_style}",
        f"Target response language: {request.response_language}",
        "Preserve the user's language in the answer.",
        "If the target language is zh-CN, translate all prose and section headings into Simplified Chinese. Keep only code identifiers, file paths, and CLI literals in English.",
        "Do not alter or invent file paths, function names, class names, config keys, or CLI literals. Reproduce identifiers exactly as they appear in the evidence or grounded draft.",
        "In this project, `claw` means the local external runtime CLI integrated by the repository. Do not reinterpret it as a cloud acronym or SaaS product.",
        "Use the provided repository observations and evidence first.",
        "Base concrete claims on the consulted evidence. If something is not confirmed by the consulted files, say that it is not confirmed.",
        "Do not ask the user to provide repository paths, file paths, or glob patterns.",
        "If the evidence below is sufficient, answer directly without asking follow-up questions.",
        "Do not print raw schema identifiers such as repo_overview_sections or architecture_review_sections.",
        *style_lines,
        evidence_output_line,
        section_intro,
        *section_lines,
        "Respect these project constraints:",
        *constraint_lines,
        "",
        "User prompt:",
        request.prompt,
        "",
        "Evidence consulted so far:",
        *evidence_lines,
        "",
        "Repository observations:",
        *observation_lines,
        "",
        "Grounded draft (compact factual scaffold):",
        *(draft_lines or ["- No grounded draft provided."]),
        "",
        "Do not mention session ids, audit paths, runtime names, selected_mode, or other operational metadata in the answer body.",
    ]
    return "\n".join(sections)


def _compose_general_chat_prompt(request: RuntimeRequest) -> str:
    return "\n".join(
        (
            "You are answering a single-turn direct question through `labai ask`.",
            "Treat the user prompt as self-contained text.",
            "Do not inspect repositories, files, workspaces, PDFs, or papers.",
            "Do not claim to have read, scanned, edited, verified, or compared any local resource.",
            "If the prompt requests actual file, repository, PDF, or workspace operations, do not execute them. Briefly point the user to the appropriate `labai workflow ...` command when it is obvious.",
            "Preserve the user's language in the answer.",
            "If the target language is zh-CN, answer in Simplified Chinese. Keep only code identifiers, file paths, and CLI literals in English.",
            "Obey exact output instructions such as only output the answer, do not explain, or only output the translation.",
            "Do not include operational metadata, evidence sections, or workflow traces in the answer body.",
            "",
            "User prompt:",
            request.prompt,
        )
    )


def _mode_outline(mode: str) -> tuple[str, ...]:
    if mode == "repo_overview":
        return (
            "Purpose",
            "Main directories/modules",
            "Important entry points",
            "Current runtime path",
            "Key risks or caveats",
            "Evidence/files consulted",
        )
    if mode == "workspace_verification":
        return (
            "Readiness status",
            "Why this status was chosen",
            "What is clearly present",
            "Likely entry points or run surfaces",
            "Config/env and dependency signals",
            "Missing pieces or blockers",
            "Risks or uncertainty",
            "What to read first",
            "First three practical next steps",
            "Evidence/files consulted",
        )
    if mode == "project_onboarding":
        return (
            "Project purpose",
            "Main directories/modules",
            "Likely entry points",
            "Config/env and dependency signals",
            "What to read first",
            "Risks or missing pieces",
            "Practical next steps",
            "Evidence/files consulted",
        )
    if mode == "file_explain":
        return (
            "File purpose",
            "Key functions/classes",
            "Inputs and outputs",
            "Dependencies",
            "Risks or confusing spots",
            "Evidence/files consulted",
        )
    if mode == "architecture_review":
        return (
            "Relevant components",
            "Data/control flow",
            "Runtime path and fallback path",
            "Interaction points",
            "Risks and hidden assumptions",
            "Evidence/files consulted",
        )
    if mode == "implementation_plan":
        return (
            "Goal",
            "Proposed steps",
            "Likely files/modules to change",
            "Risks",
            "Validation plan",
            "Evidence/files consulted",
        )
    if mode == "workspace_edit":
        return (
            "Short plan",
            "Structured file blocks",
            "Summary",
        )
    if mode == "prompt_compiler":
        return (
            "Strong prompt",
            "Constraints",
            "Acceptance criteria",
            "Missing assumptions or open questions",
            "Recommendation",
            "Compact variant",
            "Strict executable variant",
        )
    if mode == "paper_summary":
        return (
            "Document identity",
            "Main contribution / purpose",
            "Method / structure",
            "Key caveats",
            "Evidence refs with file + page + chunk",
        )
    if mode == "paper_compare":
        return (
            "Documents compared",
            "Commonalities",
            "Differences",
            "Strengths / weaknesses / limitations",
            "Recommendation or synthesis",
            "Evidence refs with file + page + chunk",
        )
    if mode == "paper_grounded_qa":
        return (
            "Direct answer",
            "Grounded supporting evidence",
            "Uncertainty when evidence is weak",
            "Evidence refs with file + page + chunk",
        )
    return (
        "Options being compared",
        "Strengths",
        "Weaknesses",
        "Tradeoffs",
        "Recommendation",
        "Evidence/files consulted",
    )


def _mode_constraints(mode: str) -> tuple[str, ...]:
    common = (
        "Keep the public CLI unchanged: labai doctor, labai tools, labai ask <prompt>.",
        "Stay local-first. Do not introduce remote cloud model APIs, MCP, shell execution tools, or write-file tools.",
        "Do not mention remote providers, shell features, databases, or extra commands unless the consulted evidence explicitly supports them.",
    )
    if mode == "workspace_verification":
        return common + (
            "Treat this as a readiness decision for a new RA who may need to work in the project today, not as a generic repo summary.",
            "Lead with a practical readiness classification and explain why that status was chosen from visible evidence.",
            "Prioritize concrete run surfaces, config or dependency assumptions, missing pieces, risks, and the first practical next steps.",
            "If the grounded draft or observations mention workspace coverage, reflect that broader coverage instead of sounding like only a few files were sampled.",
            "Be explicit about what is not confirmed; do not invent a clean setup story when the workspace is under-documented.",
        )
    if mode == "project_onboarding":
        return common + (
            "Treat this as a practical handoff for a new RA, not as an architecture comparison or paper summary.",
            "Prioritize concrete modules, entry points, config signals, reading order, and next-step guidance over generic technology commentary.",
            "Do not over-weight README prose when source files, package layout, or config files provide stronger onboarding evidence.",
            "If the grounded draft or observations mention workspace coverage, reflect that broader coverage in the answer instead of sounding like only a few files were sampled.",
            "When a repo contains repeated sibling workspaces or many orchestration scripts, say so explicitly and explain how that affects a new RA's reading order.",
            "If the project purpose is only weakly visible, be explicit about the uncertainty instead of inventing a cleaner story.",
        )
    if mode == "architecture_review":
        return common + (
            "Discuss the implemented local runtime path, native fallback path, and repo-local tracing only.",
            "Name concrete modules or functions from the evidence when describing control flow.",
            "Avoid generic remote-service or performance commentary unless the evidence explicitly mentions it.",
        )
    if mode == "implementation_plan":
        return common + (
            "When planning future work, do not propose databases, Docker, or new public CLI commands unless the evidence explicitly allows them.",
            "Prefer filesystem-backed and repo-local changes when discussing the next phase.",
            "Prefer modifying existing modules from the consulted evidence over inventing brand-new top-level modules.",
            "Do not recommend third-party libraries unless they are already mentioned in the consulted evidence.",
        )
    if mode == "workspace_edit":
        return common + (
            "This is a coding-task workflow, not a roadmap-planning task.",
            "Follow the structured file-block format from the user prompt for every file that should change.",
            "Do not return only a plan or prose memo when the task requires concrete file edits.",
            "Keep unrelated files untouched and keep the change set scoped to the requested task.",
            "When the prompt names checks or tests, prioritize getting those checks to pass.",
            "Emit one FILE block for each locked target that should change, and do not invent collision-suffixed filenames or placeholder paths.",
        )
    if mode == "prompt_compiler":
        return common + (
            "Do not solve the task itself. Rewrite it into a stronger handoff prompt for another agent.",
            "Do not inspect or rely on repository evidence unless the user explicitly included repo-specific context in the rough need.",
            "Do not invent file paths, commands, modules, workflows, or deliverables that the user did not ask for.",
            "Keep the final section headings exactly as requested for the prompt-compilation output contract.",
            "Prefer practical agent instructions over abstract prompt-engineering commentary.",
        )
    if mode == "compare_options":
        return common + (
            "Interpret Claw as the local runtime adapter implemented in this repository, not as a cloud product.",
            "Compare the concrete code paths and tradeoffs visible in the consulted files.",
            "Avoid generic vendor, pricing, or cloud-service tradeoffs unless the evidence explicitly mentions them.",
        )
    if mode.startswith("paper_"):
        return common + (
            "For paper prompts, answer only from the provided grounded draft, whole-document coverage notes, aggregated slot summaries, and retrieved PDF evidence.",
            "Treat the grounded draft as a semantic-slot scaffold rather than a loose summary impression.",
            "If the read strategy is full_document or hybrid, whole-document coverage notes and slot summaries are part of the factual evidence and should not be ignored.",
            "Preserve exact file/page/chunk or file/pages/window evidence references when citing support.",
            "If a document is extraction-poor or OCR is required, say so explicitly instead of inferring missing text.",
            "Only state a method, limitation, contribution, or comparison point if it appears in the grounded draft or retrieved evidence.",
            "If a requested detail is weakly supported, use restrained wording instead of broad generalization.",
            "If a requested detail is not supported by the PDF evidence, say that it is not confirmed from the PDF evidence or that it is not clearly stated in the paper.",
            "Do not add broad textbook commentary about machine learning, finance, or investment implications unless those exact ideas appear in the provided evidence.",
        )
    return common
def _repair_text_response(
    config: LabaiConfig,
    binary_path: Path,
    *,
    model: str,
    request: RuntimeRequest,
    answer_text: str,
) -> str:
    repair_prompt = _compose_rewrite_prompt(request, answer_text)
    command = build_claw_prompt_command(
        config,
        binary_path,
        prompt=repair_prompt,
        model=model,
    )
    completed = _run_command(
        command,
        cwd=config.workspace.active_workspace_root,
        env=_build_runtime_env(config),
        timeout_seconds=config.claw.ask_timeout_seconds,
    )
    if completed.returncode != 0:
        return answer_text
    try:
        payload = _load_json_payload(completed.stdout)
        repaired = _extract_text(payload).strip()
    except RuntimeAdapterError:
        return answer_text
    return repaired or answer_text


def _compose_rewrite_prompt(request: RuntimeRequest, answer_text: str) -> str:
    paper_requirements: list[str] = []
    if request.mode.startswith("paper_"):
        paper_requirements = [
            "- Use only the grounded slot scaffold and retrieved PDF evidence as factual support.",
            "- Remove generic domain commentary that is not clearly supported by the paper evidence.",
            "- If a requested detail is weak or missing, say it is not clearly stated in the paper instead of guessing.",
        ]
    sections = [
        "The previous draft answer violated the requested presentation rules. Rewrite it so it strictly obeys them.",
        f"Original user prompt: {request.prompt}",
        "Hard requirements:",
        *[
            f"- {item}"
            for item in build_rewrite_requirements(
                response_language=request.response_language,
                response_style=request.response_style,
                include_explicit_evidence_refs=request.include_explicit_evidence_refs,
            )
        ],
        *paper_requirements,
        "",
        "Draft answer to rewrite:",
        answer_text,
        "",
        "Return only the rewritten final answer body.",
    ]
    return "\n".join(sections)


def _select_model(
    config: LabaiConfig,
    prompt: str,
    *,
    preferred_model: str | None,
) -> str:
    if config.generation.active_provider == "deepseek":
        return select_deepseek_model(
            config,
            prompt=prompt,
            preferred_model=preferred_model,
        )
    if preferred_model:
        return _normalize_local_claw_model(preferred_model)
    prompt_lower = prompt.lower()
    if any(token in prompt_lower for token in ("code", "python", "function", "class", "module", "test")):
        return _normalize_local_claw_model(config.models.code_model)
    return _normalize_local_claw_model(config.models.general_model)


def _normalize_local_claw_model(model: str) -> str:
    normalized = model.strip()
    lowered = normalized.lower()
    if not normalized:
        return normalized
    if lowered.startswith("openai/"):
        return normalized
    if lowered.startswith("qwen/") or lowered.startswith("qwen-"):
        raise RuntimeAdapterError(
            "Local Claw/Ollama mode must not use qwen/ or qwen- cloud model syntax. "
            "Use a plain Ollama model name such as qwen2.5:7b; LabAI will pass it "
            "to Claw as openai/qwen2.5:7b against the local OPENAI_BASE_URL."
        )
    if "/" in normalized:
        return normalized
    return f"openai/{normalized}"


def _looks_like_path(command: str) -> bool:
    return any(separator in command for separator in ("/", "\\")) or bool(Path(command).suffix)


def _extract_strict_literal_response(prompt: str) -> str | None:
    match = _STRICT_LITERAL_PROMPT.match(prompt)
    if not match:
        return None
    literal = match.group("literal").strip()
    if len(literal) >= 2 and literal[0] == literal[-1] and literal[0] in {'"', "'"}:
        literal = literal[1:-1].strip()
    return literal or None


def _iter_binary_candidates(config: LabaiConfig) -> tuple[Path, ...]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_root(root: Path) -> None:
        for profile in _ordered_profiles(config.claw.build_profile):
            for name in ("claw.exe", "claw"):
                candidate = (root / "target" / profile / name).resolve()
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)

    add_root(config.project_root / "rust")
    add_root(config.project_root)

    if config.claw.source_repo_path is not None:
        add_root(config.claw.source_repo_path / "rust")
        add_root(config.claw.source_repo_path)

    if config.claw.workspace_path is not None:
        add_root(config.claw.workspace_path)
        add_root(config.claw.workspace_path / "rust")

    return tuple(candidates)


def _ordered_profiles(build_profile: str) -> tuple[str, ...]:
    other = "release" if build_profile == "debug" else "debug"
    return (build_profile, other)

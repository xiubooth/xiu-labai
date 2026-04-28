from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from labai.config import LabaiConfig
from labai.runtime.platform import detect_platform, get_platform_paths

from .claw import _check_local_openai_endpoint, build_claw_doctor_command, resolve_claw_binary

DiagnosticStatus = Literal[
    "ready",
    "detected",
    "not_installed",
    "not_built",
    "not_running",
    "misconfigured",
    "not_configured",
    "missing_models",
]
DoctorStatus = Literal["ready", "ready_with_fallback", "guided_not_ready"]
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class DiagnosticItem:
    key: str
    status: DiagnosticStatus
    detail: str
    location: str = ""
    next_step: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalRuntimeReport:
    doctor_status: DoctorStatus
    summary: str
    diagnostics: tuple[DiagnosticItem, ...]
    next_steps: tuple[str, ...]

    def item(self, key: str) -> DiagnosticItem | None:
        for diagnostic in self.diagnostics:
            if diagnostic.key == key:
                return diagnostic
        return None


def build_local_runtime_report(config: LabaiConfig) -> LocalRuntimeReport:
    platform_name = detect_platform()
    setup_script = _platform_setup_script(platform_name)
    source_repo_item = _detect_source_repo(config)
    workspace_item = _detect_workspace(config)
    git_item = _detect_command(
        "git",
        key="git",
        install_hint=_git_install_hint(platform_name),
    )
    cargo_item = _detect_command(
        "cargo",
        key="cargo",
        install_hint=_cargo_install_hint(platform_name),
    )
    rustup_item = _detect_command(
        "rustup",
        key="rustup",
        install_hint=_rustup_install_hint(platform_name),
    )
    claw_item = _detect_claw_binary(config, cargo_item, source_repo_item, workspace_item)
    if _uses_managed_claw_runtime(config, claw_item):
        git_item = DiagnosticItem(
            key="git",
            status="detected",
            detail="Git is optional for the bundled Claw runtime path.",
        )
        cargo_item = DiagnosticItem(
            key="cargo",
            status="detected",
            detail="Cargo is optional because a managed Claw binary is already provisioned.",
        )
        rustup_item = DiagnosticItem(
            key="rustup",
            status="detected",
            detail="rustup is optional because a managed Claw binary is already provisioned.",
        )
        if source_repo_item.status == "not_configured":
            source_repo_item = DiagnosticItem(
                key="claw_source_repo",
                status="detected",
                detail="No Claw source repo is configured, which is expected for the managed bundled runtime.",
            )
        if workspace_item.status == "not_configured":
            workspace_item = DiagnosticItem(
                key="claw_workspace",
                status="detected",
                detail="No Claw source workspace is configured, which is expected for the managed bundled runtime.",
            )
    ollama_command_item = _detect_command(
        config.ollama.command,
        key="ollama_command",
        install_hint=(
            f"Install Ollama for {platform_name}, start it, then rerun {setup_script} "
            "or `labai doctor`."
        ),
    )
    ollama_service_item = _detect_ollama_service(config)
    endpoint_item = _detect_openai_endpoint(config)
    model_item = _detect_required_models(config, endpoint_item, ollama_service_item)

    diagnostics = (
        git_item,
        cargo_item,
        rustup_item,
        claw_item,
        source_repo_item,
        workspace_item,
        ollama_command_item,
        ollama_service_item,
        endpoint_item,
        model_item,
    )
    status, summary = _overall_status(config, claw_item, endpoint_item, model_item)
    next_steps = _collect_next_steps(diagnostics)
    return LocalRuntimeReport(
        doctor_status=status,
        summary=summary,
        diagnostics=diagnostics,
        next_steps=next_steps,
    )


def _detect_command(
    command_name: str,
    *,
    key: str,
    install_hint: str | None = None,
) -> DiagnosticItem:
    explicit_path = Path(command_name) if any(separator in command_name for separator in ("/", "\\")) else None
    if explicit_path is not None or Path(command_name).suffix:
        resolved = explicit_path or Path(command_name)
        resolved = resolved.resolve()
        if resolved.is_file():
            return DiagnosticItem(
                key=key,
                status="ready",
                detail=f"{resolved.name} is available.",
                location=str(resolved),
            )
        return DiagnosticItem(
            key=key,
            status="misconfigured",
            detail=f"Configured command path was not found: {resolved}",
            location=str(resolved),
            next_step=install_hint or f"Fix the configured path for {command_name}.",
        )

    resolved = shutil.which(command_name)
    if resolved:
        return DiagnosticItem(
            key=key,
            status="ready",
            detail=f"{command_name} is available on PATH.",
            location=str(Path(resolved).resolve()),
        )

    windows_fallback = _resolve_windows_command_path(command_name)
    if windows_fallback is not None:
        return DiagnosticItem(
            key=key,
            status="ready",
            detail=f"{command_name} is available from a standard Windows install location.",
            location=str(windows_fallback),
        )

    return DiagnosticItem(
        key=key,
        status="not_installed",
        detail=f"{command_name} was not found on PATH.",
        next_step=install_hint or f"Install {command_name} or add it to PATH.",
    )


def _detect_claw_binary(
    config: LabaiConfig,
    cargo_item: DiagnosticItem,
    source_repo_item: DiagnosticItem,
    workspace_item: DiagnosticItem,
) -> DiagnosticItem:
    configured_binary = config.claw.binary.strip()
    binary_path = resolve_claw_binary(config)
    if binary_path is not None:
        return DiagnosticItem(
            key="claw_binary",
            status="ready",
            detail="A usable Claw binary was found.",
            location=str(binary_path),
            metadata={
                "doctor_command": build_claw_doctor_command(config, binary_path),
            },
        )

    if _looks_like_path(configured_binary):
        candidate = _resolve_configured_path(config.project_root, configured_binary)
        return DiagnosticItem(
            key="claw_binary",
            status="misconfigured",
            detail=f"Configured Claw binary path was not found: {candidate}",
            location=str(candidate),
            next_step=_claw_missing_next_step(),
        )

    if source_repo_item.status == "misconfigured":
        return DiagnosticItem(
            key="claw_binary",
            status="misconfigured",
            detail="Claw source repo configuration is invalid, so no build output can be discovered.",
            location=source_repo_item.location,
            next_step=source_repo_item.next_step,
        )

    if workspace_item.status == "misconfigured":
        return DiagnosticItem(
            key="claw_binary",
            status="misconfigured",
            detail="Claw workspace configuration is invalid, so no build output can be discovered.",
            location=workspace_item.location,
            next_step=workspace_item.next_step,
        )

    if source_repo_item.status == "detected" or workspace_item.status == "detected":
        next_step = (
            "Use the platform bootstrap path first, then use the explicit source-build fallback only "
            "after the Rust toolchain and source repo are ready."
        )
        if cargo_item.status != "ready":
            next_step = (
                "Install Rust tooling (rustup/cargo), then use the explicit developer source-build "
                "fallback for Claw."
            )
        return DiagnosticItem(
            key="claw_binary",
            status="not_built",
            detail=(
                "Claw source path is configured, but no built Claw binary was found under the "
                f"{config.claw.build_profile} profile."
            ),
            next_step=next_step,
            metadata={"build_profile": config.claw.build_profile},
        )

    return DiagnosticItem(
        key="claw_binary",
        status="not_installed",
        detail="No discoverable Claw binary was found on PATH or in configured build outputs.",
        next_step=_claw_missing_next_step(),
    )


def _claw_missing_next_step() -> str:
    platform_name = detect_platform()
    if platform_name == "macos":
        return (
            "Expected a bundled macOS Claw binary at runtime-assets/claw/macos-arm64/claw "
            "or runtime-assets/claw/macos-x64/claw. Rerun scripts/mac/bootstrap-mac.sh "
            "after adding the correct asset, or set LABAI_CLAW_BINARY to a real executable "
            "macOS Claw binary. Rust source builds are maintainer-only."
        )
    return (
        f"Run {_platform_setup_script(platform_name)} to provision the managed Claw runtime. "
        "Use source-build settings only if you intentionally want the developer fallback path."
    )


def _detect_source_repo(config: LabaiConfig) -> DiagnosticItem:
    if config.claw.source_repo_path is None:
        return DiagnosticItem(
            key="claw_source_repo",
            status="not_configured",
            detail="No Claw source repo path is configured.",
            next_step=(
                "This is expected for bundled release installs. Set [claw].source_repo_path only if you want the optional developer source-build path."
            ),
        )

    if not config.claw.source_repo_path.exists():
        return DiagnosticItem(
            key="claw_source_repo",
            status="misconfigured",
            detail="Configured Claw source repo path does not exist.",
            location=str(config.claw.source_repo_path),
            next_step="Fix [claw].source_repo_path or remove it until you have a local clone.",
        )

    rust_dir = (config.claw.source_repo_path / "rust").resolve()
    if not rust_dir.is_dir():
        return DiagnosticItem(
            key="claw_source_repo",
            status="misconfigured",
            detail="Configured Claw source repo path does not contain the canonical rust/ runtime directory.",
            location=str(config.claw.source_repo_path),
            next_step=(
                "Point [claw].source_repo_path at the root of a local ultraworkers/claw-code clone "
                "that contains rust/."
            ),
        )

    return DiagnosticItem(
        key="claw_source_repo",
        status="detected",
        detail="Configured Claw source repo path exists and contains rust/.",
        location=str(config.claw.source_repo_path),
        metadata={"rust_dir": str(rust_dir)},
    )


def _detect_workspace(config: LabaiConfig) -> DiagnosticItem:
    if config.claw.workspace_path is None:
        return DiagnosticItem(
            key="claw_workspace",
            status="not_configured",
            detail="No explicit Rust workspace path is configured for Claw builds.",
            next_step=(
                "This is expected for bundled release installs. Set [claw].workspace_path only if you want the optional developer source-build path."
            ),
        )

    if not config.claw.workspace_path.exists():
        return DiagnosticItem(
            key="claw_workspace",
            status="misconfigured",
            detail="Configured Rust workspace path does not exist.",
            location=str(config.claw.workspace_path),
            next_step="Fix [claw].workspace_path or remove it until the workspace exists.",
        )

    cargo_manifest = (config.claw.workspace_path / "Cargo.toml").resolve()
    if not cargo_manifest.is_file():
        return DiagnosticItem(
            key="claw_workspace",
            status="misconfigured",
            detail="Configured Rust workspace path does not contain Cargo.toml.",
            location=str(config.claw.workspace_path),
            next_step=(
                "Point [claw].workspace_path at the Rust workspace directory that contains Cargo.toml, "
                "or remove it if [claw].source_repo_path is enough."
            ),
        )

    return DiagnosticItem(
        key="claw_workspace",
        status="detected",
        detail="Configured Rust workspace path exists and contains Cargo.toml.",
        location=str(config.claw.workspace_path),
        metadata={"cargo_toml": str(cargo_manifest)},
    )


def _detect_ollama_service(config: LabaiConfig) -> DiagnosticItem:
    parsed = urlparse(config.ollama.base_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or host not in _LOCAL_HOSTS:
        return DiagnosticItem(
            key="ollama_service",
            status="misconfigured",
            detail="Ollama base_url must point at a local loopback host.",
            location=config.ollama.base_url,
            next_step=(
                "Set [ollama].base_url to a local loopback address such as "
                "http://127.0.0.1:11434."
            ),
        )

    request = Request(
        f"{config.ollama.base_url.rstrip('/')}/api/version",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=config.ollama.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return DiagnosticItem(
            key="ollama_service",
            status="not_running",
            detail=f"Ollama service is not reachable at {config.ollama.base_url}: {exc}",
            next_step=(
                f"Start Ollama locally, then rerun {_platform_verify_script(detect_platform())} "
                "or `labai doctor`."
            ),
        )

    version = str(payload.get("version", "unknown"))
    return DiagnosticItem(
        key="ollama_service",
        status="ready",
        detail=f"Ollama service is reachable at {config.ollama.base_url}.",
        location=config.ollama.base_url,
        metadata={"version": version},
    )


def _detect_openai_endpoint(config: LabaiConfig) -> DiagnosticItem:
    endpoint_health = _check_local_openai_endpoint(config)
    status_map = {
        "ready": "ready",
        "invalid_config": "misconfigured",
        "unavailable": "not_running",
    }
    status = status_map.get(str(endpoint_health.get("status")), "misconfigured")
    return DiagnosticItem(
        key="ollama_endpoint",
        status=status,  # type: ignore[arg-type]
        detail=str(endpoint_health.get("detail", "OpenAI-compatible endpoint state unknown.")),
        location=str(endpoint_health.get("base_url", config.ollama.openai_base_url)),
        next_step=_endpoint_next_step(status, config),
        metadata={key: value for key, value in endpoint_health.items() if key not in {"detail", "base_url", "status"}},
    )


def _endpoint_next_step(status: str, config: LabaiConfig) -> str:
    if status == "ready":
        return ""
    if status == "misconfigured":
        return (
            "Fix [ollama].openai_base_url so it points at the local loopback endpoint, "
            f"typically {config.ollama.openai_base_url}."
        )
    return (
        "Start Ollama locally and confirm the OpenAI-compatible endpoint is reachable at "
        f"{config.ollama.openai_base_url}."
    )


def _detect_required_models(
    config: LabaiConfig,
    endpoint_item: DiagnosticItem,
    service_item: DiagnosticItem,
) -> DiagnosticItem:
    if endpoint_item.status != "ready" and service_item.status != "ready":
        return DiagnosticItem(
            key="ollama_models",
            status="not_running",
            detail="Required models cannot be verified until Ollama is reachable.",
            next_step=(
                f"Start Ollama first, then run {_platform_setup_script(detect_platform())} "
                "to see the exact model pull commands."
            ),
            metadata={"required_models": config.ollama.required_models},
        )

    available_models = _fetch_available_models(config)
    missing_models = tuple(
        model_name
        for model_name in config.ollama.required_models
        if model_name not in available_models
    )
    if missing_models:
        return DiagnosticItem(
            key="ollama_models",
            status="missing_models",
            detail=(
                "Local Ollama is reachable, but one or more required Qwen models are missing: "
                f"{', '.join(missing_models)}"
            ),
            next_step=(
                f"Run {_platform_ollama_script(detect_platform())} for pull commands, "
                "then rerun it with -Apply once you are ready."
            ),
            metadata={
                "required_models": config.ollama.required_models,
                "available_models": available_models,
                "missing_models": missing_models,
            },
        )

    return DiagnosticItem(
        key="ollama_models",
        status="ready",
        detail="Required local Qwen models are available to Ollama.",
        metadata={
            "required_models": config.ollama.required_models,
            "available_models": available_models,
        },
    )


def _fetch_available_models(config: LabaiConfig) -> tuple[str, ...]:
    request = Request(
        f"{config.ollama.base_url.rstrip('/')}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=config.ollama.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError):
        return ()

    models = payload.get("models", [])
    discovered: list[str] = []
    if isinstance(models, list):
        for item in models:
            if not isinstance(item, dict):
                continue
            model_name = str(item.get("model") or item.get("name") or "").strip()
            if model_name:
                discovered.append(model_name)
                discovered.append(model_name.split(":", maxsplit=1)[0])

    deduped = tuple(dict.fromkeys(discovered))
    return deduped


def _overall_status(
    config: LabaiConfig,
    claw_item: DiagnosticItem,
    endpoint_item: DiagnosticItem,
    model_item: DiagnosticItem,
) -> tuple[DoctorStatus, str]:
    if config.runtime.runtime == "native":
        return "ready_with_fallback", "Native runtime is selected; Claw-first local setup remains optional."

    if claw_item.status == "ready" and endpoint_item.status == "ready" and model_item.status == "ready":
        return "ready", "Claw and local Ollama/Qwen runtime are ready."

    if config.runtime.fallback_runtime == "native":
        return (
            "guided_not_ready",
            "Preferred local runtime is not fully ready yet, but native fallback remains available.",
        )

    return "ready_with_fallback", "Preferred local runtime is not ready."


def _collect_next_steps(diagnostics: tuple[DiagnosticItem, ...]) -> tuple[str, ...]:
    steps: list[str] = []
    for diagnostic in diagnostics:
        if diagnostic.next_step and diagnostic.next_step not in steps:
            steps.append(diagnostic.next_step)
    return tuple(steps[:6])


def _looks_like_path(command: str) -> bool:
    return any(separator in command for separator in ("/", "\\")) or bool(Path(command).suffix)


def _resolve_configured_path(project_root: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _resolve_windows_command_path(command_name: str) -> Path | None:
    if os.name != "nt":
        return None

    normalized = command_name.strip().lower()
    user_profile = Path(os.environ.get("USERPROFILE", ""))
    local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))

    candidate_map = {
        "cargo": (user_profile / ".cargo" / "bin" / "cargo.exe",),
        "rustup": (user_profile / ".cargo" / "bin" / "rustup.exe",),
        "ollama": (local_appdata / "Programs" / "Ollama" / "ollama.exe",),
    }

    for candidate in candidate_map.get(normalized, ()):
        if candidate.is_file():
            return candidate.resolve()

    return None


def _uses_managed_claw_runtime(config: LabaiConfig, claw_item: DiagnosticItem) -> bool:
    if claw_item.status != "ready":
        return False

    configured_binary = Path(os.path.expandvars(os.path.expanduser(config.claw.binary)))
    managed_claw = get_platform_paths(project_root=config.project_root).managed_claw_path
    return _same_config_path(configured_binary, managed_claw)


def _same_config_path(left: Path, right: Path) -> bool:
    left_text = left.as_posix().lower() if os.name == "nt" else left.as_posix()
    right_text = right.as_posix().lower() if os.name == "nt" else right.as_posix()
    return left_text == right_text


def _platform_setup_script(platform_name: str) -> str:
    if platform_name == "macos":
        return "scripts/mac/bootstrap-mac.sh"
    return "Launch-LabAI-Setup.cmd or scripts/windows/bootstrap-windows.ps1"


def _platform_verify_script(platform_name: str) -> str:
    if platform_name == "macos":
        return "scripts/mac/verify-install.sh"
    return "scripts/windows/verify-install.ps1"


def _platform_ollama_script(platform_name: str) -> str:
    if platform_name == "macos":
        return "scripts/mac/setup-local-ollama.sh"
    return "scripts/windows/setup-local-ollama.ps1"


def _git_install_hint(platform_name: str) -> str:
    if platform_name == "macos":
        return "Install Git with Xcode Command Line Tools or Homebrew, then reopen Terminal."
    return "Install Git for Windows or add git.exe to PATH, then reopen PowerShell."


def _cargo_install_hint(platform_name: str) -> str:
    if platform_name == "macos":
        return "Install Rust tooling with rustup if you intentionally use the Claw source-build fallback."
    return "Install Rust tooling with rustup, then reopen PowerShell so cargo is on PATH."


def _rustup_install_hint(platform_name: str) -> str:
    if platform_name == "macos":
        return "Install rustup only for the explicit developer Claw source-build fallback."
    return "Install rustup-init for Windows so Claw can be built from source."

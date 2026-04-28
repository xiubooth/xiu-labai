from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal
import tomllib
from urllib.parse import urlparse


DEFAULT_CONFIG_PATH = Path(".labai/config.toml")
DEFAULT_SESSIONS_DIR = Path(".labai/sessions")
DEFAULT_AUDIT_DIR = Path(".labai/audit")
DEFAULT_AUDIT_LOG = Path(".labai/audit/audit.jsonl")
DEFAULT_OUTPUTS_DIR = Path(".labai/outputs")
DEFAULT_LIBRARY_ROOT = Path(".labai/library")
DEFAULT_LIBRARY_MANIFESTS_DIR = Path(".labai/library/manifests")
DEFAULT_LIBRARY_EXTRACTED_DIR = Path(".labai/library/extracted")
DEFAULT_LIBRARY_CHUNKS_DIR = Path(".labai/library/chunks")
DEFAULT_LIBRARY_INDEX_DIR = Path(".labai/library/index")
DEFAULT_PAPER_LIBRARY_ROOT = Path("examples/papers")
SUPPORTED_PROVIDERS = frozenset({"deepseek", "mock", "ollama"})
SUPPORTED_GENERATION_PROVIDERS = frozenset({"local", "deepseek"})
SUPPORTED_FALLBACK_POLICIES = frozenset({"fallback_to_mock", "fail"})
SUPPORTED_RUNTIMES = frozenset({"claw", "native"})
SUPPORTED_FALLBACK_RUNTIMES = frozenset({"native"})
SUPPORTED_MODEL_FAMILIES = frozenset({"qwen", "gpt-oss"})
SUPPORTED_RESEARCH_MODES = frozenset(
    {
        "general_chat",
        "repo_overview",
        "workspace_verification",
        "file_explain",
        "architecture_review",
        "implementation_plan",
        "workspace_edit",
        "prompt_compiler",
        "compare_options",
        "paper_summary",
        "paper_compare",
        "paper_grounded_qa",
    }
)
SUPPORTED_OUTPUT_FORMATS = frozenset({"json", "text", "stream-json"})
SUPPORTED_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "danger-full-access"})
SUPPORTED_BUILD_PROFILES = frozenset({"debug", "release"})
SUPPORTED_BOOTSTRAP_POLICIES = frozenset({"guided_setup", "manual_only"})
SUPPORTED_NOT_READY_POLICIES = frozenset({"fallback_to_native", "fail_closed"})
SUPPORTED_PARSER_PREFERENCES = frozenset({"pymupdf_then_pypdf", "pypdf_then_pymupdf"})
SUPPORTED_REINGEST_POLICIES = frozenset({"if_changed", "always"})
SUPPORTED_LOW_TEXT_POLICIES = frozenset({"mark_ocr_required", "skip_document"})
SUPPORTED_ARTIFACT_FORMATS = frozenset({"markdown"})
SUPPORTED_ARTIFACT_EXPORT_POLICIES = frozenset({"always", "explicit_only", "never"})
SUPPORTED_CONSOLE_MODES = frozenset({"verbose", "answer_only"})
SUPPORTED_WORKSPACE_EDIT_MODES = frozenset({"suggest", "auto_edit"})
SUPPORTED_WORKSPACE_ACCESS_POLICIES = frozenset({"allowlisted_workspace_rw"})
DEEPSEEK_MAX_OUTPUT_TOKENS_LIMIT = 8192
DEEPSEEK_DEFAULT_SMOKE_MAX_TOKENS = 256

FallbackPolicy = Literal["fallback_to_mock", "fail"]
RuntimeName = Literal["claw", "native"]


class LabaiConfigError(RuntimeError):
    """Raised when the project configuration cannot be loaded."""


class LabaiConfigNotFoundError(LabaiConfigError):
    """Raised when `.labai/config.toml` cannot be found."""


class LabaiConfigValidationError(LabaiConfigError):
    """Raised when the project configuration is invalid."""


@dataclass(frozen=True)
class PathSettings:
    sessions_dir: Path
    audit_dir: Path
    audit_log: Path
    outputs_dir: Path


@dataclass(frozen=True)
class RuntimeSettings:
    runtime: RuntimeName = "claw"
    fallback_runtime: str = "native"
    bootstrap_policy: str = "guided_setup"
    not_ready_policy: str = "fallback_to_native"


@dataclass(frozen=True)
class ModelSettings:
    default_model_family: str = "qwen"
    general_model: str = "qwen2.5"
    code_model: str = "qwen2.5-coder"


@dataclass(frozen=True)
class GenerationSettings:
    active_provider: str = "local"


@dataclass(frozen=True)
class MockProviderSettings:
    response_prefix: str = "Mock response"


@dataclass(frozen=True)
class ResearchSettings:
    mode_override: str | None = None


@dataclass(frozen=True)
class ArtifactSettings:
    auto_export_markdown: bool = True
    export_policy: str = "explicit_only"
    format: str = "markdown"
    include_metadata_comment: bool = False


@dataclass(frozen=True)
class OutputSettings:
    console_mode: str = "verbose"


@dataclass(frozen=True)
class WorkspaceSettings:
    access_policy: str = "allowlisted_workspace_rw"
    auto_detect_cwd: bool = True
    active_workspace_root: Path = Path(".")
    allowed_workspace_roots: tuple[Path, ...] = ()
    allowed_paper_roots: tuple[Path, ...] = ()
    allow_absolute_paths: bool = True
    edit_mode: str = "suggest"
    same_folder_deliverables: bool = True
    deny_roots: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ClawRuntimeSettings:
    binary: str = "claw"
    source_repo_path: Path | None = None
    workspace_path: Path | None = None
    build_profile: str = "debug"
    output_format: str = "json"
    permission_mode: str = "read-only"
    allowed_tools: tuple[str, ...] = ("read", "glob")
    health_timeout_seconds: int = 10
    ask_timeout_seconds: int = 600


@dataclass(frozen=True)
class OllamaProviderSettings:
    command: str = "ollama"
    model: str = "qwen2.5"
    required_models: tuple[str, ...] = ("qwen2.5", "qwen2.5-coder")
    base_url: str = "http://127.0.0.1:11434"
    openai_base_url: str = "http://127.0.0.1:11434/v1"
    openai_api_key: str = ""
    timeout_seconds: int = 30


@dataclass(frozen=True)
class DeepSeekProviderSettings:
    enabled: bool = False
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    general_model: str = "deepseek-chat"
    code_model: str = "deepseek-chat"
    reasoning_model: str = "deepseek-reasoner"
    timeout_seconds: int = 120
    max_tokens: int = 8192
    smoke_max_tokens: int = DEEPSEEK_DEFAULT_SMOKE_MAX_TOKENS


@dataclass(frozen=True)
class PaperSettings:
    library_roots: tuple[Path, ...]
    runtime_root: Path
    manifests_dir: Path
    extracted_dir: Path
    chunks_dir: Path
    index_dir: Path
    parser_preference: str = "pymupdf_then_pypdf"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_fallback_model: str = "embeddinggemma"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_top_k: int = 6
    reingest_policy: str = "if_changed"
    low_text_policy: str = "mark_ocr_required"
    min_page_text_chars: int = 80


@dataclass(frozen=True)
class LabaiConfig:
    project_root: Path
    config_path: Path
    runtime: RuntimeSettings
    active_profile: str
    default_provider: str
    fallback_policy: FallbackPolicy
    generation: GenerationSettings
    paths: PathSettings
    models: ModelSettings
    mock: MockProviderSettings
    research: ResearchSettings
    artifacts: ArtifactSettings
    output: OutputSettings
    workspace: WorkspaceSettings
    claw: ClawRuntimeSettings
    ollama: OllamaProviderSettings
    deepseek: DeepSeekProviderSettings
    papers: PaperSettings


def default_project_root(start: Path | None = None) -> Path:
    return (start or Path.cwd()).resolve()


def default_config_path(project_root: Path) -> Path:
    return (project_root / DEFAULT_CONFIG_PATH).resolve()


def default_path_settings(project_root: Path) -> PathSettings:
    return PathSettings(
        sessions_dir=(project_root / DEFAULT_SESSIONS_DIR).resolve(),
        audit_dir=(project_root / DEFAULT_AUDIT_DIR).resolve(),
        audit_log=(project_root / DEFAULT_AUDIT_LOG).resolve(),
        outputs_dir=(project_root / DEFAULT_OUTPUTS_DIR).resolve(),
    )


def scaffold_config(start: Path | None = None) -> LabaiConfig:
    project_root = default_project_root(start)
    return LabaiConfig(
        project_root=project_root,
        config_path=default_config_path(project_root),
        runtime=RuntimeSettings(),
        active_profile="local",
        default_provider="mock",
        fallback_policy="fallback_to_mock",
        generation=GenerationSettings(),
        paths=default_path_settings(project_root),
        models=ModelSettings(),
        mock=MockProviderSettings(),
        research=ResearchSettings(),
        artifacts=ArtifactSettings(),
        output=OutputSettings(),
        workspace=WorkspaceSettings(
            active_workspace_root=project_root,
        ),
        claw=ClawRuntimeSettings(),
        ollama=OllamaProviderSettings(),
        deepseek=DeepSeekProviderSettings(),
        papers=default_paper_settings(project_root),
    )


def discover_config_path(start: Path | None = None) -> Path:
    env_override = os.environ.get("LABAI_CONFIG_PATH", "").strip()
    if env_override:
        candidate = Path(_expand_path_like_string(env_override)).resolve()
        if candidate.is_file():
            return candidate
        raise LabaiConfigNotFoundError(f"Config file not found: {candidate}")

    current = default_project_root(start)
    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        candidate = (directory / DEFAULT_CONFIG_PATH).resolve()
        if candidate.is_file():
            return candidate

    package_root_candidate = (Path(__file__).resolve().parents[2] / DEFAULT_CONFIG_PATH).resolve()
    if package_root_candidate.is_file():
        return package_root_candidate

    raise LabaiConfigNotFoundError(
        f"Could not find {DEFAULT_CONFIG_PATH.as_posix()} from {current}"
    )


def load_config(start: Path | None = None) -> LabaiConfig:
    return load_config_from_path(discover_config_path(start), start=start)


def load_config_from_path(config_path: Path, start: Path | None = None) -> LabaiConfig:
    resolved_config_path = Path(config_path).resolve()
    if not resolved_config_path.is_file():
        raise LabaiConfigNotFoundError(
            f"Config file not found: {resolved_config_path}"
        )

    project_root = _derive_project_root(resolved_config_path)
    invocation_root = default_project_root(start)
    if invocation_root.is_file():
        invocation_root = invocation_root.parent
    raw = tomllib.loads(resolved_config_path.read_text(encoding="utf-8"))

    app_data = raw.get("app", {})
    runtime_data = raw.get("runtime", {})
    path_data = raw.get("paths", {})
    model_data = raw.get("models", {})
    mock_data = raw.get("mock", {})
    research_data = raw.get("research", {})
    artifact_data = raw.get("artifacts", {})
    output_data = raw.get("output", {})
    workspace_data = raw.get("workspace", {})
    claw_data = raw.get("claw", {})
    ollama_data = raw.get("ollama", {})
    providers_data = raw.get("providers", {})
    deepseek_data = providers_data.get("deepseek", {}) if isinstance(providers_data, dict) else {}
    papers_data = raw.get("papers", {})
    deepseek_max_tokens = _validate_deepseek_output_tokens(
        deepseek_data.get("max_tokens", DEEPSEEK_MAX_OUTPUT_TOKENS_LIMIT),
        field_name="providers.deepseek.max_tokens",
    )
    deepseek_smoke_max_tokens = min(
        _validate_deepseek_output_tokens(
            deepseek_data.get("smoke_max_tokens", DEEPSEEK_DEFAULT_SMOKE_MAX_TOKENS),
            field_name="providers.deepseek.smoke_max_tokens",
        ),
        deepseek_max_tokens,
    )

    default_provider = _validate_provider_name(
        str(app_data.get("default_provider", "mock"))
    )
    active_profile = _validate_profile_name(
        str(app_data.get("active_profile", "local"))
    )
    active_generation_provider = _validate_generation_provider(
        _env_override(
            "LABAI_GENERATION_PROVIDER",
            str(app_data.get("active_generation_provider", "local")),
        )
    )
    fallback_policy = _validate_fallback_policy(
        str(app_data.get("fallback_policy", "fallback_to_mock"))
    )
    path_settings = _build_path_settings(project_root, path_data)

    return LabaiConfig(
        project_root=project_root,
        config_path=resolved_config_path,
        runtime=RuntimeSettings(
            runtime=_validate_runtime_name(
                str(runtime_data.get("runtime", "claw"))
            ),
            fallback_runtime=_validate_fallback_runtime(
                str(runtime_data.get("fallback_runtime", "native"))
            ),
            bootstrap_policy=_validate_bootstrap_policy(
                str(runtime_data.get("bootstrap_policy", "guided_setup"))
            ),
            not_ready_policy=_validate_not_ready_policy(
                str(runtime_data.get("not_ready_policy", "fallback_to_native"))
            ),
        ),
        active_profile=active_profile,
        default_provider=default_provider,
        fallback_policy=fallback_policy,
        generation=GenerationSettings(
            active_provider=active_generation_provider,
        ),
        paths=path_settings,
        models=ModelSettings(
            default_model_family=_validate_model_family(
                str(model_data.get("default_model_family", "qwen"))
            ),
            general_model=_validate_local_model_name(
                _require_non_empty_string(
                    model_data.get("general_model"),
                    default="qwen2.5",
                    field_name="models.general_model",
                ),
                field_name="models.general_model",
            ),
            code_model=_validate_local_model_name(
                _require_non_empty_string(
                    model_data.get("code_model"),
                    default="qwen2.5-coder",
                    field_name="models.code_model",
                ),
                field_name="models.code_model",
            ),
        ),
        mock=MockProviderSettings(
            response_prefix=str(mock_data.get("response_prefix", "Mock response"))
        ),
        research=ResearchSettings(
            mode_override=_validate_optional_research_mode(
                research_data.get("mode_override")
            )
        ),
        artifacts=_build_artifact_settings(artifact_data),
        output=OutputSettings(
            console_mode=_validate_console_mode(
                str(output_data.get("console_mode", "verbose"))
            )
        ),
        workspace=_build_workspace_settings(
            project_root,
            invocation_root,
            workspace_data,
        ),
        claw=ClawRuntimeSettings(
            binary=_expand_path_like_string(
                _require_non_empty_string(
                    claw_data.get("binary"),
                    default="claw",
                    field_name="claw.binary",
                )
            ),
            source_repo_path=_resolve_optional_external_path(
                project_root,
                claw_data.get("source_repo_path"),
            ),
            workspace_path=_resolve_optional_external_path(
                project_root,
                claw_data.get("workspace_path"),
            ),
            build_profile=_validate_build_profile(
                str(claw_data.get("build_profile", "debug"))
            ),
            output_format=_validate_output_format(
                str(claw_data.get("output_format", "json"))
            ),
            permission_mode=_validate_permission_mode(
                str(claw_data.get("permission_mode", "read-only"))
            ),
            allowed_tools=_validate_allowed_tools(claw_data.get("allowed_tools")),
            health_timeout_seconds=_validate_positive_int(
                claw_data.get("health_timeout_seconds", 10),
                field_name="claw.health_timeout_seconds",
            ),
            ask_timeout_seconds=_validate_positive_int(
                claw_data.get("ask_timeout_seconds", 600),
                field_name="claw.ask_timeout_seconds",
            ),
        ),
        ollama=OllamaProviderSettings(
            command=_expand_path_like_string(
                _require_non_empty_string(
                    ollama_data.get("command"),
                    default="ollama",
                    field_name="ollama.command",
                )
            ),
            model=_validate_local_model_name(
                _require_non_empty_string(
                    ollama_data.get("model"),
                    default="qwen2.5",
                    field_name="ollama.model",
                ),
                field_name="ollama.model",
            ),
            required_models=_validate_required_models(
                ollama_data.get("required_models"),
            ),
            base_url=str(ollama_data.get("base_url", "http://127.0.0.1:11434")),
            openai_base_url=str(
                ollama_data.get("openai_base_url", "http://127.0.0.1:11434/v1")
            ),
            openai_api_key=str(ollama_data.get("openai_api_key", "")),
            timeout_seconds=int(ollama_data.get("timeout_seconds", 30)),
        ),
        deepseek=DeepSeekProviderSettings(
            enabled=_validate_bool(
                deepseek_data.get("enabled", False)
                or active_generation_provider == "deepseek"
                or default_provider == "deepseek",
                field_name="providers.deepseek.enabled",
            ),
            base_url=_validate_absolute_url(
                _require_non_empty_string(
                    deepseek_data.get("base_url"),
                    default="https://api.deepseek.com",
                    field_name="providers.deepseek.base_url",
                ),
                field_name="providers.deepseek.base_url",
            ),
            api_key_env=_require_non_empty_string(
                deepseek_data.get("api_key_env"),
                default="DEEPSEEK_API_KEY",
                field_name="providers.deepseek.api_key_env",
            ),
            general_model=_require_non_empty_string(
                deepseek_data.get("general_model"),
                default="deepseek-chat",
                field_name="providers.deepseek.general_model",
            ),
            code_model=_require_non_empty_string(
                deepseek_data.get("code_model"),
                default="deepseek-chat",
                field_name="providers.deepseek.code_model",
            ),
            reasoning_model=_require_non_empty_string(
                deepseek_data.get("reasoning_model"),
                default="deepseek-reasoner",
                field_name="providers.deepseek.reasoning_model",
            ),
            timeout_seconds=_validate_positive_int(
                deepseek_data.get("timeout_seconds", 120),
                field_name="providers.deepseek.timeout_seconds",
            ),
            max_tokens=deepseek_max_tokens,
            smoke_max_tokens=deepseek_smoke_max_tokens,
        ),
        papers=_build_paper_settings(project_root, papers_data),
    )


def format_project_path(path: Path, project_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        return str(resolved_path)
    return relative.as_posix() or "."


def _derive_project_root(config_path: Path) -> Path:
    if config_path.parent.name == ".labai":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()


def _build_path_settings(project_root: Path, raw_paths: dict[str, object]) -> PathSettings:
    sessions_dir = _resolve_project_path(
        project_root,
        raw_paths.get("sessions_dir"),
        DEFAULT_SESSIONS_DIR,
    )
    audit_log = _resolve_project_path(
        project_root,
        raw_paths.get("audit_log"),
        DEFAULT_AUDIT_LOG,
    )
    outputs_dir = _resolve_project_path(
        project_root,
        raw_paths.get("outputs_dir"),
        DEFAULT_OUTPUTS_DIR,
    )
    audit_dir = audit_log.parent

    _validate_runtime_path(project_root, sessions_dir, "sessions_dir")
    _validate_runtime_path(project_root, audit_dir, "audit_log")
    _validate_runtime_path(project_root, outputs_dir, "outputs_dir")

    return PathSettings(
        sessions_dir=sessions_dir,
        audit_dir=audit_dir,
        audit_log=audit_log,
        outputs_dir=outputs_dir,
    )


def default_paper_settings(project_root: Path) -> PaperSettings:
    runtime_root = (project_root / DEFAULT_LIBRARY_ROOT).resolve()
    manifests_dir = (project_root / DEFAULT_LIBRARY_MANIFESTS_DIR).resolve()
    extracted_dir = (project_root / DEFAULT_LIBRARY_EXTRACTED_DIR).resolve()
    chunks_dir = (project_root / DEFAULT_LIBRARY_CHUNKS_DIR).resolve()
    index_dir = (project_root / DEFAULT_LIBRARY_INDEX_DIR).resolve()
    return PaperSettings(
        library_roots=((project_root / DEFAULT_PAPER_LIBRARY_ROOT).resolve(),),
        runtime_root=runtime_root,
        manifests_dir=manifests_dir,
        extracted_dir=extracted_dir,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
    )


def _build_workspace_settings(
    project_root: Path,
    invocation_root: Path,
    raw_workspace: dict[str, object],
) -> WorkspaceSettings:
    auto_detect_cwd = _validate_bool(
        raw_workspace.get("auto_detect_cwd", True),
        field_name="workspace.auto_detect_cwd",
    )
    active_workspace_root = invocation_root.resolve() if auto_detect_cwd else project_root.resolve()
    return WorkspaceSettings(
        access_policy=_validate_workspace_access_policy(
            str(raw_workspace.get("access_policy", "allowlisted_workspace_rw"))
        ),
        auto_detect_cwd=auto_detect_cwd,
        active_workspace_root=active_workspace_root,
        allowed_workspace_roots=_validate_optional_path_list(
            project_root,
            raw_workspace.get("allowed_workspace_roots"),
            field_name="workspace.allowed_workspace_roots",
        ),
        allowed_paper_roots=_validate_optional_path_list(
            project_root,
            raw_workspace.get("allowed_paper_roots"),
            field_name="workspace.allowed_paper_roots",
        ),
        allow_absolute_paths=_validate_bool(
            raw_workspace.get("allow_absolute_paths", True),
            field_name="workspace.allow_absolute_paths",
        ),
        edit_mode=_validate_workspace_edit_mode(
            str(raw_workspace.get("edit_mode", "suggest"))
        ),
        same_folder_deliverables=_validate_bool(
            raw_workspace.get("same_folder_deliverables", True),
            field_name="workspace.same_folder_deliverables",
        ),
        deny_roots=_validate_optional_path_list(
            project_root,
            raw_workspace.get("deny_roots"),
            field_name="workspace.deny_roots",
        ),
    )


def _build_artifact_settings(raw_artifacts: dict[str, object]) -> ArtifactSettings:
    raw_export_policy = raw_artifacts.get("export_policy")
    if raw_export_policy is None:
        raw_auto_export = raw_artifacts.get("auto_export_markdown")
        if raw_auto_export is None:
            export_policy = "explicit_only"
        else:
            auto_export_markdown = _validate_bool(
                raw_auto_export,
                field_name="artifacts.auto_export_markdown",
            )
            export_policy = "always" if auto_export_markdown else "never"
    else:
        export_policy = _validate_artifact_export_policy(str(raw_export_policy))

    auto_export_markdown = export_policy != "never"
    return ArtifactSettings(
        auto_export_markdown=auto_export_markdown,
        export_policy=export_policy,
        format=_validate_artifact_format(
            str(raw_artifacts.get("format", "markdown"))
        ),
        include_metadata_comment=_validate_bool(
            raw_artifacts.get("include_metadata_comment", False),
            field_name="artifacts.include_metadata_comment",
        ),
    )


def _build_paper_settings(project_root: Path, raw_papers: dict[str, object]) -> PaperSettings:
    defaults = default_paper_settings(project_root)
    runtime_root = _resolve_project_path(
        project_root,
        raw_papers.get("runtime_root"),
        DEFAULT_LIBRARY_ROOT,
    )
    manifests_dir = _resolve_project_path(
        project_root,
        raw_papers.get("manifests_dir"),
        DEFAULT_LIBRARY_MANIFESTS_DIR,
    )
    extracted_dir = _resolve_project_path(
        project_root,
        raw_papers.get("extracted_dir"),
        DEFAULT_LIBRARY_EXTRACTED_DIR,
    )
    chunks_dir = _resolve_project_path(
        project_root,
        raw_papers.get("chunks_dir"),
        DEFAULT_LIBRARY_CHUNKS_DIR,
    )
    index_dir = _resolve_project_path(
        project_root,
        raw_papers.get("index_dir"),
        DEFAULT_LIBRARY_INDEX_DIR,
    )
    for setting_name, path in (
        ("papers.runtime_root", runtime_root),
        ("papers.manifests_dir", manifests_dir),
        ("papers.extracted_dir", extracted_dir),
        ("papers.chunks_dir", chunks_dir),
        ("papers.index_dir", index_dir),
    ):
        _validate_runtime_path(project_root, path, setting_name)

    chunk_size = _validate_positive_int(
        raw_papers.get("chunk_size", defaults.chunk_size),
        field_name="papers.chunk_size",
    )
    chunk_overlap = _validate_non_negative_int(
        raw_papers.get("chunk_overlap", defaults.chunk_overlap),
        field_name="papers.chunk_overlap",
    )
    if chunk_overlap >= chunk_size:
        raise LabaiConfigValidationError(
            "papers.chunk_overlap must be smaller than papers.chunk_size."
        )

    return PaperSettings(
        library_roots=_validate_library_roots(
            project_root,
            raw_papers.get("library_roots"),
            defaults.library_roots,
        ),
        runtime_root=runtime_root,
        manifests_dir=manifests_dir,
        extracted_dir=extracted_dir,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        parser_preference=_validate_parser_preference(
            str(raw_papers.get("parser_preference", defaults.parser_preference))
        ),
        embedding_model=_validate_local_model_name(
            _require_non_empty_string(
                raw_papers.get("embedding_model"),
                default=defaults.embedding_model,
                field_name="papers.embedding_model",
            ),
            field_name="papers.embedding_model",
        ),
        embedding_fallback_model=_validate_local_model_name(
            _require_non_empty_string(
                raw_papers.get("embedding_fallback_model"),
                default=defaults.embedding_fallback_model,
                field_name="papers.embedding_fallback_model",
            ),
            field_name="papers.embedding_fallback_model",
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_top_k=_validate_positive_int(
            raw_papers.get("retrieval_top_k", defaults.retrieval_top_k),
            field_name="papers.retrieval_top_k",
        ),
        reingest_policy=_validate_reingest_policy(
            str(raw_papers.get("reingest_policy", defaults.reingest_policy))
        ),
        low_text_policy=_validate_low_text_policy(
            str(raw_papers.get("low_text_policy", defaults.low_text_policy))
        ),
        min_page_text_chars=_validate_non_negative_int(
            raw_papers.get("min_page_text_chars", defaults.min_page_text_chars),
            field_name="papers.min_page_text_chars",
        ),
    )


def _resolve_project_path(
    project_root: Path,
    raw_value: object,
    default_relative_path: Path,
) -> Path:
    candidate = (
        Path(_expand_path_like_string(str(raw_value)))
        if raw_value is not None
        else default_relative_path
    )
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _validate_provider_name(raw_value: str) -> str:
    provider_name = raw_value.strip().lower()
    if provider_name not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise LabaiConfigValidationError(
            f"Unsupported provider '{provider_name}'. Expected one of: {supported}"
        )
    return provider_name


def _validate_generation_provider(raw_value: str) -> str:
    provider_name = raw_value.strip().lower()
    if provider_name not in SUPPORTED_GENERATION_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_GENERATION_PROVIDERS))
        raise LabaiConfigValidationError(
            f"Unsupported active_generation_provider '{provider_name}'. Expected one of: {supported}"
        )
    return provider_name


def _validate_profile_name(raw_value: str) -> str:
    profile_name = raw_value.strip().lower()
    if not profile_name:
        raise LabaiConfigValidationError("app.active_profile must not be empty.")
    return profile_name


def _validate_fallback_policy(raw_value: str) -> FallbackPolicy:
    fallback_policy = raw_value.strip().lower()
    if fallback_policy not in SUPPORTED_FALLBACK_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_FALLBACK_POLICIES))
        raise LabaiConfigValidationError(
            f"Unsupported fallback policy '{fallback_policy}'. "
            f"Expected one of: {supported}"
        )
    return fallback_policy  # type: ignore[return-value]


def _validate_runtime_name(raw_value: str) -> RuntimeName:
    runtime_name = raw_value.strip().lower()
    if runtime_name not in SUPPORTED_RUNTIMES:
        supported = ", ".join(sorted(SUPPORTED_RUNTIMES))
        raise LabaiConfigValidationError(
            f"Unsupported runtime '{runtime_name}'. Expected one of: {supported}"
        )
    return runtime_name  # type: ignore[return-value]


def _validate_fallback_runtime(raw_value: str) -> str:
    fallback_runtime = raw_value.strip().lower()
    if fallback_runtime not in SUPPORTED_FALLBACK_RUNTIMES:
        supported = ", ".join(sorted(SUPPORTED_FALLBACK_RUNTIMES))
        raise LabaiConfigValidationError(
            f"Unsupported fallback runtime '{fallback_runtime}'. Expected one of: {supported}"
        )
    return fallback_runtime


def _validate_bootstrap_policy(raw_value: str) -> str:
    bootstrap_policy = raw_value.strip().lower()
    if bootstrap_policy not in SUPPORTED_BOOTSTRAP_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_BOOTSTRAP_POLICIES))
        raise LabaiConfigValidationError(
            f"Unsupported runtime bootstrap_policy '{bootstrap_policy}'. Expected one of: {supported}"
        )
    return bootstrap_policy


def _validate_not_ready_policy(raw_value: str) -> str:
    not_ready_policy = raw_value.strip().lower()
    if not_ready_policy not in SUPPORTED_NOT_READY_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_NOT_READY_POLICIES))
        raise LabaiConfigValidationError(
            f"Unsupported runtime not_ready_policy '{not_ready_policy}'. Expected one of: {supported}"
        )
    return not_ready_policy


def _validate_model_family(raw_value: str) -> str:
    model_family = raw_value.strip().lower()
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
        raise LabaiConfigValidationError(
            f"Unsupported default model family '{model_family}'. Expected one of: {supported}"
        )
    return model_family


def _validate_optional_research_mode(raw_value: object) -> str | None:
    if raw_value is None:
        return None

    normalized = str(raw_value).strip().lower()
    if not normalized:
        return None
    if normalized not in SUPPORTED_RESEARCH_MODES:
        supported = ", ".join(sorted(SUPPORTED_RESEARCH_MODES))
        raise LabaiConfigValidationError(
            f"Unsupported research mode override '{normalized}'. Expected one of: {supported}"
        )
    return normalized


def _validate_artifact_format(raw_value: str) -> str:
    artifact_format = raw_value.strip().lower()
    if artifact_format not in SUPPORTED_ARTIFACT_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_ARTIFACT_FORMATS))
        raise LabaiConfigValidationError(
            f"Unsupported artifacts.format '{artifact_format}'. Expected one of: {supported}"
        )
    return artifact_format


def _validate_artifact_export_policy(raw_value: str) -> str:
    export_policy = raw_value.strip().lower()
    if export_policy not in SUPPORTED_ARTIFACT_EXPORT_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_ARTIFACT_EXPORT_POLICIES))
        raise LabaiConfigValidationError(
            f"Unsupported artifacts.export_policy '{export_policy}'. Expected one of: {supported}"
        )
    return export_policy


def _validate_console_mode(raw_value: str) -> str:
    console_mode = raw_value.strip().lower()
    if console_mode not in SUPPORTED_CONSOLE_MODES:
        supported = ", ".join(sorted(SUPPORTED_CONSOLE_MODES))
        raise LabaiConfigValidationError(
            f"Unsupported output.console_mode '{console_mode}'. Expected one of: {supported}"
        )
    return console_mode


def _validate_workspace_access_policy(raw_value: str) -> str:
    policy = raw_value.strip().lower()
    if policy not in SUPPORTED_WORKSPACE_ACCESS_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_WORKSPACE_ACCESS_POLICIES))
        raise LabaiConfigValidationError(
            f"Unsupported workspace.access_policy '{policy}'. Expected one of: {supported}"
        )
    return policy


def _validate_workspace_edit_mode(raw_value: str) -> str:
    mode = raw_value.strip().lower()
    if mode not in SUPPORTED_WORKSPACE_EDIT_MODES:
        supported = ", ".join(sorted(SUPPORTED_WORKSPACE_EDIT_MODES))
        raise LabaiConfigValidationError(
            f"Unsupported workspace.edit_mode '{mode}'. Expected one of: {supported}"
        )
    return mode


def _validate_output_format(raw_value: str) -> str:
    output_format = raw_value.strip().lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_FORMATS))
        raise LabaiConfigValidationError(
            f"Unsupported claw output_format '{output_format}'. Expected one of: {supported}"
        )
    return output_format


def _validate_build_profile(raw_value: str) -> str:
    build_profile = raw_value.strip().lower()
    if build_profile not in SUPPORTED_BUILD_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_BUILD_PROFILES))
        raise LabaiConfigValidationError(
            f"Unsupported claw build_profile '{build_profile}'. Expected one of: {supported}"
        )
    return build_profile


def _validate_local_model_name(raw_value: str, *, field_name: str) -> str:
    normalized = raw_value.strip()
    lowered = normalized.lower()
    if lowered.startswith("qwen/") or lowered.startswith("qwen-"):
        raise LabaiConfigValidationError(
            f"{field_name} must use a plain local model name and avoid qwen/ or qwen- prefixes."
        )
    return normalized


def _validate_absolute_url(raw_value: str, *, field_name: str) -> str:
    normalized = raw_value.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LabaiConfigValidationError(f"{field_name} must be an absolute URL.")
    return normalized


def _validate_permission_mode(raw_value: str) -> str:
    permission_mode = raw_value.strip().lower()
    if permission_mode not in SUPPORTED_PERMISSION_MODES:
        supported = ", ".join(sorted(SUPPORTED_PERMISSION_MODES))
        raise LabaiConfigValidationError(
            f"Unsupported claw permission_mode '{permission_mode}'. Expected one of: {supported}"
        )
    return permission_mode


def _validate_allowed_tools(raw_value: object) -> tuple[str, ...]:
    if raw_value is None:
        return ("read", "glob")

    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",")]
    elif isinstance(raw_value, list):
        items = [str(item).strip() for item in raw_value]
    else:
        raise LabaiConfigValidationError(
            "claw.allowed_tools must be a list of tool names or a comma-separated string."
        )

    normalized = tuple(item for item in items if item)
    if not normalized:
        raise LabaiConfigValidationError("claw.allowed_tools must not be empty.")
    return normalized


def _validate_required_models(raw_value: object) -> tuple[str, ...]:
    if raw_value is None:
        return ("qwen2.5", "qwen2.5-coder")

    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",")]
    elif isinstance(raw_value, list):
        items = [str(item).strip() for item in raw_value]
    else:
        raise LabaiConfigValidationError(
            "ollama.required_models must be a list of model names or a comma-separated string."
        )

    normalized = tuple(
        _validate_local_model_name(item, field_name="ollama.required_models")
        for item in items
        if item
    )
    if not normalized:
        raise LabaiConfigValidationError("ollama.required_models must not be empty.")
    return normalized


def _validate_positive_int(raw_value: object, *, field_name: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise LabaiConfigValidationError(
            f"{field_name} must be a positive integer."
        ) from exc
    if value <= 0:
        raise LabaiConfigValidationError(
            f"{field_name} must be a positive integer."
        )
    return value


def _validate_deepseek_output_tokens(raw_value: object, *, field_name: str) -> int:
    value = _validate_positive_int(raw_value, field_name=field_name)
    return min(value, DEEPSEEK_MAX_OUTPUT_TOKENS_LIMIT)


def _validate_non_negative_int(raw_value: object, *, field_name: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise LabaiConfigValidationError(
            f"{field_name} must be a non-negative integer."
        ) from exc
    if value < 0:
        raise LabaiConfigValidationError(
            f"{field_name} must be a non-negative integer."
        )
    return value


def _validate_bool(raw_value: object, *, field_name: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise LabaiConfigValidationError(f"{field_name} must be a boolean.")


def _require_non_empty_string(
    raw_value: object,
    *,
    default: str,
    field_name: str,
) -> str:
    value = default if raw_value is None else str(raw_value)
    normalized = value.strip()
    if not normalized:
        raise LabaiConfigValidationError(f"{field_name} must not be empty.")
    return normalized


def _env_override(env_name: str, default: str) -> str:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    normalized = raw.strip()
    return normalized or default


def get_deepseek_api_key(config: LabaiConfig) -> str:
    return os.environ.get(config.deepseek.api_key_env, "").strip()


def _resolve_optional_external_path(
    project_root: Path,
    raw_value: object,
) -> Path | None:
    if raw_value is None:
        return None

    normalized = _expand_path_like_string(str(raw_value).strip())
    if not normalized:
        return None

    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _validate_runtime_path(project_root: Path, path: Path, setting_name: str) -> None:
    relative_path = _relative_to_project_root(project_root, path, setting_name)
    if not relative_path.parts or relative_path.parts[0] != ".labai":
        raise LabaiConfigValidationError(
            f"{setting_name} must stay under .labai/: {relative_path.as_posix()}"
        )


def _validate_library_roots(
    project_root: Path,
    raw_value: object,
    default_value: tuple[Path, ...],
) -> tuple[Path, ...]:
    if raw_value is None:
        return default_value

    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",")]
    elif isinstance(raw_value, list):
        items = [str(item).strip() for item in raw_value]
    else:
        raise LabaiConfigValidationError(
            "papers.library_roots must be a list of paths or a comma-separated string."
        )

    resolved: list[Path] = []
    for item in items:
        if not item:
            continue
        candidate = _resolve_project_path(project_root, item, DEFAULT_PAPER_LIBRARY_ROOT)
        _validate_project_relative_path(project_root, candidate, "papers.library_roots")
        resolved.append(candidate)

    if not resolved:
        raise LabaiConfigValidationError("papers.library_roots must not be empty.")
    return tuple(dict.fromkeys(resolved))


def _validate_optional_path_list(
    project_root: Path,
    raw_value: object,
    *,
    field_name: str,
) -> tuple[Path, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",")]
    elif isinstance(raw_value, list):
        items = [str(item).strip() for item in raw_value]
    else:
        raise LabaiConfigValidationError(
            f"{field_name} must be a list of paths or a comma-separated string."
        )
    resolved: list[Path] = []
    for item in items:
        if not item:
            continue
        candidate = _resolve_optional_external_path(project_root, item)
        if candidate is None:
            continue
        resolved.append(candidate)
    return tuple(dict.fromkeys(resolved))


def _validate_project_relative_path(project_root: Path, path: Path, setting_name: str) -> None:
    _relative_to_project_root(project_root, path, setting_name)


def _validate_parser_preference(raw_value: str) -> str:
    parser_preference = raw_value.strip().lower()
    if parser_preference not in SUPPORTED_PARSER_PREFERENCES:
        supported = ", ".join(sorted(SUPPORTED_PARSER_PREFERENCES))
        raise LabaiConfigValidationError(
            f"Unsupported papers.parser_preference '{parser_preference}'. Expected one of: {supported}"
        )
    return parser_preference


def _validate_reingest_policy(raw_value: str) -> str:
    reingest_policy = raw_value.strip().lower()
    if reingest_policy not in SUPPORTED_REINGEST_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_REINGEST_POLICIES))
        raise LabaiConfigValidationError(
            f"Unsupported papers.reingest_policy '{reingest_policy}'. Expected one of: {supported}"
        )
    return reingest_policy


def _validate_low_text_policy(raw_value: str) -> str:
    low_text_policy = raw_value.strip().lower()
    if low_text_policy not in SUPPORTED_LOW_TEXT_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_LOW_TEXT_POLICIES))
        raise LabaiConfigValidationError(
            f"Unsupported papers.low_text_policy '{low_text_policy}'. Expected one of: {supported}"
        )
    return low_text_policy


def _relative_to_project_root(
    project_root: Path,
    path: Path,
    setting_name: str,
) -> Path:
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise LabaiConfigValidationError(
            f"{setting_name} must stay within the project root: {path.resolve()}"
        ) from exc


def _expand_path_like_string(raw_value: str) -> str:
    return os.path.expanduser(os.path.expandvars(raw_value))

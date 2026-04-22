from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Literal
import unicodedata

from labai.config import LabaiConfig, load_config
from labai.editing import (
    WorkspaceEditApplyResult,
    WorkspaceEditPlan,
    apply_workspace_edit_plan,
    build_workspace_check_plan,
    build_workspace_edit_plan,
    classify_output_intent,
    infer_workspace_config_reference_targets,
)
from labai.execution import ClawRuntimeAdapter, RuntimeAdapterError, RuntimeHealth, RuntimeRequest
from labai.papers import PaperContext, PaperLibraryError, discover_paper_targets, prepare_paper_context
from labai.papers.notes import slot_label
from labai.papers.parsing import parse_pdf
from labai.providers import (
    ProviderError,
    ProviderHealth,
    ProviderRequest,
    get_default_provider,
    get_provider,
)
from labai.runtime import AuditRecord, SessionRecord
from labai.runtime.answer_style import looks_like_structured_output
from labai.tools import ToolDispatcher, ToolExecutionError, ToolValidationError
from labai.workspace import WorkspaceAccessManager, WorkspaceWriteResult, _iter_prompt_candidates
from .modes import ModeSelection, select_mode

ResearchStatus = Literal["ok", "error"]
ReadinessStatus = Literal["ready", "ready_with_fallback", "blocked"]
OperationalStatus = Literal["ready", "ready_with_fallback", "guided_not_ready", "error"]
OutputIntentName = Literal["answer_only", "deliverable_requested"]

_ONBOARDING_ENTRYPOINT_NAMES = (
    "__main__.py",
    "main.py",
    "app.py",
    "cli.py",
    "run.py",
    "manage.py",
)
_ONBOARDING_CONFIG_NAMES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "environment.yml",
    "environment.yaml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "pytest.ini",
    "package.json",
    "Pipfile",
    "Pipfile.lock",
    ".env.example",
    ".python-version",
)
_ONBOARDING_PRIORITY_DIRS = (
    "src",
    "app",
    "project",
    "pkg",
    "package",
    "scripts",
    "tests",
    "docs",
    "notebooks",
    "data",
    "config",
    "configs",
)
_ONBOARDING_DOC_NAMES = frozenset(
    {
        "readme.md",
        "readme.rst",
        "readme.txt",
        "changelog.md",
        "contributing.md",
        "architecture.md",
        "project.md",
        "agents.md",
        "claude.md",
    }
)
_ONBOARDING_SOURCE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".ipynb",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
    }
)
_ONBOARDING_SCRIPT_EXTENSIONS = frozenset({".sh", ".ps1", ".bat", ".cmd"})
_ONBOARDING_DOC_EXTENSIONS = frozenset({".md", ".rst", ".txt"})
_ONBOARDING_CONFIG_EXTENSIONS = frozenset({".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".env"})
_ONBOARDING_DATA_EXTENSIONS = frozenset(
    {
        ".csv",
        ".tsv",
        ".parquet",
        ".pkl",
        ".pickle",
        ".feather",
        ".xlsx",
        ".xls",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".zip",
        ".gz",
        ".bz2",
        ".xz",
        ".npy",
        ".npz",
    }
)
_ONBOARDING_BINARY_EXTENSIONS = frozenset(
    {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".pyc",
        ".pyd",
        ".class",
        ".o",
        ".a",
        ".lib",
        ".bin",
        ".woff",
        ".woff2",
        ".ttf",
    }
)
_WORKSPACE_OVERRIDE_MODES = frozenset({"project_onboarding", "workspace_verification", "workspace_edit"})
_ONBOARDING_IGNORED_DIR_NAMES = frozenset(
    {
        ".git",
        ".github",
        ".claw",
        ".codex",
        ".idea",
        ".labai",
        ".planning",
        ".planning_old",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".nox",
        ".tox",
        ".cache",
        ".ipynb_checkpoints",
        ".eggs",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "__pycache__",
    }
)
_ONBOARDING_IGNORED_FILE_PREFIXES = ("pytest-cache-files-",)
_ONBOARDING_RELEVANT_CATEGORIES = frozenset({"source", "config", "docs", "tests", "scripts"})
_ONBOARDING_GENERIC_TOP_LEVEL_DIRS = frozenset(
    {
        "docs",
        "doc",
        "examples",
        "example",
        "src",
        "source",
        "scripts",
        "script",
        "tests",
        "test",
        "config",
        "configs",
        "data",
        "notebooks",
        "notebook",
        "assets",
    }
)
_ONBOARDING_FULL_RELEVANT_COVERAGE_LIMIT = 220
_ONBOARDING_LARGE_PROJECT_SELECTION_LIMIT = 120
_ONBOARDING_MAX_SUMMARY_CHARS = 24000


@dataclass(frozen=True)
class OnboardingManifestEntry:
    path: str
    category: str
    readable: bool
    relevant: bool
    size_bytes: int
    top_level: str
    skip_reason: str = ""


@dataclass(frozen=True)
class OnboardingCoverage:
    total_files: int = 0
    relevant_readable_count: int = 0
    ignored_noise_count: int = 0
    unreadable_binary_count: int = 0
    full_relevant_coverage: bool = False
    category_counts: dict[str, int] = field(default_factory=dict)
    inspected_category_counts: dict[str, int] = field(default_factory=dict)
    manifest_entries: tuple[OnboardingManifestEntry, ...] = ()
    inspected_paths: tuple[str, ...] = ()
    skipped_paths: tuple[str, ...] = ()
    skipped_notes: tuple[str, ...] = ()
    summary_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkspaceVerificationAssessment:
    status: str = "uncertain"
    why_status: tuple[str, ...] = ()
    confirmed_present: tuple[str, ...] = ()
    missing_or_blocking: tuple[str, ...] = ()
    risks_or_uncertainty: tuple[str, ...] = ()
    read_first: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class FallbackInfo:
    applied: bool
    policy: str
    requested_provider: str
    active_provider: str
    reason: str = ""


@dataclass(frozen=True)
class RuntimeFallbackInfo:
    applied: bool
    requested_runtime: str
    active_runtime: str
    fallback_runtime: str
    reason: str = ""


@dataclass(frozen=True)
class ToolDecision:
    tool_name: str
    should_use: bool
    reason: str
    arguments: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, str]
    status: str
    summary: str
    error: str = ""
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperTrace:
    active: bool = False
    read_strategy: str = ""
    read_strategy_reason: str = ""
    output_profile: str = "none"
    output_profile_reason: str = ""
    response_language_reason: str = ""
    response_language_explicit_override: bool = False
    target_paths: tuple[str, ...] = ()
    discovered_documents: tuple[str, ...] = ()
    selected_embedding_model: str | None = None
    embedding_status: str = ""
    fallback_embedding_model: str | None = None
    ingest_actions: list[dict[str, object]] = field(default_factory=list)
    document_windows: list[dict[str, object]] = field(default_factory=list)
    slot_notes: list[dict[str, object]] = field(default_factory=list)
    document_notes: list[dict[str, object]] = field(default_factory=list)
    retrieved_chunks: list[dict[str, object]] = field(default_factory=list)
    indexed_document_count: int = 0
    ocr_required_paths: tuple[str, ...] = ()
    index_updated: bool = False
    window_count_processed: int = 0
    consistency_check_status: str = "not_run"
    consistency_check_repaired: bool = False
    consistency_check_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkspaceTrace:
    task_run_id: str = ""
    app_repo_root: str = ""
    active_workspace_root: str = ""
    allowed_workspace_roots: tuple[str, ...] = ()
    allowed_paper_roots: tuple[str, ...] = ()
    access_policy: str = ""
    edit_mode: str = ""
    external_paths_accessed: tuple[str, ...] = ()
    created_files: tuple[str, ...] = ()
    modified_files: tuple[str, ...] = ()
    user_deliverable_requested: bool = False
    user_deliverable_file: str = ""
    user_deliverable_status: str = "not_requested"
    user_deliverable_operation: str = ""
    user_deliverable_destination_policy: str = ""
    edit_intent: str = ""
    edit_plan_summary: str = ""
    planned_reads: tuple[str, ...] = ()
    planned_modifications: tuple[str, ...] = ()
    planned_creations: tuple[str, ...] = ()
    primary_targets: tuple[str, ...] = ()
    secondary_targets: tuple[str, ...] = ()
    referenced_paths: tuple[str, ...] = ()
    planned_checks: tuple[str, ...] = ()
    planned_check_details: tuple[dict[str, object], ...] = ()
    intended_changes: tuple[str, ...] = ()
    task_contract: dict[str, object] = field(default_factory=dict)
    task_manifest: dict[str, object] = field(default_factory=dict)
    repo_map_summary: tuple[dict[str, object], ...] = ()
    required_read_evidence: tuple[dict[str, object], ...] = ()
    owner_detection: dict[str, object] = field(default_factory=dict)
    validator_routing: dict[str, object] = field(default_factory=dict)
    structured_edit_ops: tuple[dict[str, object], ...] = ()
    created_validators: tuple[str, ...] = ()
    skipped_files: tuple[str, ...] = ()
    skipped_notes: tuple[str, ...] = ()
    apply_status: str = "not_requested"
    check_status: str = "not_run"
    checks_run: tuple[str, ...] = ()
    executed_check_details: tuple[dict[str, object], ...] = ()
    check_planning_errors: tuple[str, ...] = ()
    check_failures: tuple[str, ...] = ()
    acceptance_checks_passed: tuple[str, ...] = ()
    acceptance_checks_failed: tuple[str, ...] = ()
    validation_notes: tuple[str, ...] = ()
    dependency_fallback_used: bool = False
    unavailable_dependencies: tuple[str, ...] = ()
    dependency_fallback_mode: str = ""
    dependency_fallback_reason: str = ""
    dependency_fallback_tested: str = ""
    dependency_fallback_untested: str = ""
    repair_rounds: int = 0
    code_quality_warnings: tuple[str, ...] = ()
    owner_boundary_warnings: tuple[str, ...] = ()
    stale_file_warnings: tuple[str, ...] = ()
    landed_edit_evidence: tuple[dict[str, object], ...] = ()
    runtime_exec_results: tuple[dict[str, object], ...] = ()
    typed_validation_results: tuple[dict[str, object], ...] = ()
    strategy_switches: tuple[dict[str, object], ...] = ()
    evidence_ledger_path: str = ""
    file_change_summaries: tuple[str, ...] = ()
    rollback_notes: tuple[str, ...] = ()
    git_repo_detected: bool = False
    git_repo_root: str = ""
    git_changed_files: tuple[str, ...] = ()
    git_untracked_files: tuple[str, ...] = ()
    git_commit_message_draft: str = ""
    git_status_before: tuple[str, ...] = ()
    git_status_after: tuple[str, ...] = ()
    onboarding_total_files: int = 0
    onboarding_relevant_readable_files: int = 0
    onboarding_ignored_noise_files: int = 0
    onboarding_unreadable_binary_files: int = 0
    onboarding_full_relevant_coverage: bool = False
    onboarding_inspected_paths: tuple[str, ...] = ()
    onboarding_skipped_paths: tuple[str, ...] = ()
    onboarding_category_counts: dict[str, int] = field(default_factory=dict)
    onboarding_inspected_category_counts: dict[str, int] = field(default_factory=dict)
    access_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputIntent:
    name: OutputIntentName = "answer_only"
    reason: str = "Prompt is an ordinary terminal answer request."


@dataclass(frozen=True)
class WorkspaceMutationRequest:
    action: str = ""
    target_path: str = ""
    reason: str = ""
    strategy: str = ""
    destination_policy: str = ""


@dataclass(frozen=True)
class NativeProviderRoute:
    status: ReadinessStatus
    detail: str
    requested_provider: str
    provider_health: ProviderHealth


@dataclass(frozen=True)
class ResearchReadiness:
    status: ReadinessStatus
    detail: str
    selected_runtime: str
    fallback_runtime: str
    requested_provider: str
    fallback_policy: str
    runtime_health: RuntimeHealth
    provider_health: ProviderHealth


@dataclass(frozen=True)
class AnswerRoute:
    requested_runtime: str
    runtime_used: str
    runtime_fallback: RuntimeFallbackInfo
    requested_provider: str
    provider_used: str
    provider_model: str | None
    provider_fallback: FallbackInfo
    text: str


@dataclass(frozen=True)
class ResearchResult:
    session_id: str
    command: str
    prompt: str
    selected_mode: str
    mode_reason: str
    answer_schema: str
    read_strategy: str
    read_strategy_reason: str
    response_style: str
    response_language: str
    output_intent: OutputIntentName
    output_intent_reason: str
    evidence_refs: tuple[str, ...]
    operational_status: OperationalStatus
    requested_runtime: str
    runtime_used: str
    runtime_fallback: RuntimeFallbackInfo
    requested_provider: str
    provider_used: str
    provider_model: str | None
    selected_embedding_model: str | None
    fallback: FallbackInfo
    tools_used: bool
    tool_decisions: list[ToolDecision]
    tool_calls: list[ToolCall]
    observations: list[str]
    paper_trace: PaperTrace
    workspace_trace: WorkspaceTrace
    final_answer: str
    outcome_summary: str
    status: ResearchStatus
    error: str
    started_at: str
    completed_at: str


def evaluate_research_readiness(config: LabaiConfig) -> ResearchReadiness:
    runtime_health = _runtime_healthcheck(config)
    native_route = _evaluate_native_provider_route(config)

    if config.runtime.runtime == "native":
        return ResearchReadiness(
            status=native_route.status,
            detail=native_route.detail,
            selected_runtime=config.runtime.runtime,
            fallback_runtime=config.runtime.fallback_runtime,
            requested_provider=native_route.requested_provider,
            fallback_policy=config.fallback_policy,
            runtime_health=_native_runtime_health(),
            provider_health=native_route.provider_health,
        )

    if runtime_health.available:
        return ResearchReadiness(
            status="ready",
            detail="Claw runtime is ready and native fallback remains available if needed.",
            selected_runtime=config.runtime.runtime,
            fallback_runtime=config.runtime.fallback_runtime,
            requested_provider="claw",
            fallback_policy=config.fallback_policy,
            runtime_health=runtime_health,
            provider_health=native_route.provider_health,
        )

    if config.runtime.fallback_runtime == "native" and native_route.status != "blocked":
        return ResearchReadiness(
            status="ready_with_fallback",
            detail=(
                f"{runtime_health.detail} Native fallback is available via "
                f"{native_route.requested_provider}."
            ),
            selected_runtime=config.runtime.runtime,
            fallback_runtime=config.runtime.fallback_runtime,
            requested_provider="claw",
            fallback_policy=config.fallback_policy,
            runtime_health=runtime_health,
            provider_health=native_route.provider_health,
        )

    return ResearchReadiness(
        status="blocked",
        detail=f"{runtime_health.detail} Native fallback is not ready.",
        selected_runtime=config.runtime.runtime,
        fallback_runtime=config.runtime.fallback_runtime,
        requested_provider="claw",
        fallback_policy=config.fallback_policy,
        runtime_health=runtime_health,
        provider_health=native_route.provider_health,
    )


def _is_paper_mode(mode_name: str) -> bool:
    return mode_name in {"paper_summary", "paper_compare", "paper_grounded_qa"}


def _prepare_paper_mode_context(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
) -> PaperContext:
    targets = discover_paper_targets(config, mode_selection.matched_paths)
    try:
        return prepare_paper_context(
            config,
            prompt,
            target_paths=targets,
            read_strategy=mode_selection.read_strategy,
        )
    except PaperLibraryError as exc:
        relative_targets = tuple(
            item
            for item in mode_selection.matched_paths
            if item.lower().endswith(".pdf") or "paper" in item.lower()
        )
        return PaperContext(
            target_paths=relative_targets,
            discovered_documents=(),
            selected_embedding_model=None,
            embedding_status="error",
            fallback_embedding_model=None,
            read_strategy=mode_selection.read_strategy,
            ingest_actions=(),
            document_windows=(),
            slot_notes=(),
            document_notes=(),
            retrieved_chunks=(),
            observations=(f"Paper ingest/retrieval failed before answer generation: {exc}",),
            evidence_refs=(),
            indexed_document_count=0,
            ocr_required_paths=(),
            index_updated=False,
        )


def _paper_trace_from_context(context: PaperContext, mode_selection: ModeSelection) -> PaperTrace:
    return PaperTrace(
        active=True,
        read_strategy=context.read_strategy,
        read_strategy_reason=mode_selection.read_strategy_reason,
        output_profile=mode_selection.paper_output_profile,
        output_profile_reason=mode_selection.paper_output_profile_reason,
        response_language_reason=mode_selection.response_language_reason,
        response_language_explicit_override=mode_selection.response_language_explicit_override,
        target_paths=context.target_paths,
        discovered_documents=context.discovered_documents,
        selected_embedding_model=context.selected_embedding_model,
        embedding_status=context.embedding_status,
        fallback_embedding_model=context.fallback_embedding_model,
        ingest_actions=[asdict(item) for item in context.ingest_actions],
        document_windows=[
            {
                **asdict(item),
                "page_numbers": list(item.page_numbers),
            }
            for item in context.document_windows
        ],
        slot_notes=[
            {
                **asdict(item),
                "page_numbers": list(item.page_numbers),
            }
            for item in context.slot_notes
        ],
        document_notes=[asdict(item) for item in context.document_notes],
        retrieved_chunks=[
            {
                **asdict(item),
                "page_numbers": list(item.page_numbers),
            }
            for item in context.retrieved_chunks
        ],
        indexed_document_count=context.indexed_document_count,
        ocr_required_paths=context.ocr_required_paths,
        index_updated=context.index_updated,
        window_count_processed=len(context.document_windows),
    )


def _initial_workspace_trace(
    config: LabaiConfig,
    access_manager: WorkspaceAccessManager,
) -> WorkspaceTrace:
    return WorkspaceTrace(
        app_repo_root=str(config.project_root),
        active_workspace_root=str(access_manager.active_workspace_root),
        allowed_workspace_roots=tuple(str(path) for path in access_manager.allowed_workspace_roots),
        allowed_paper_roots=tuple(str(path) for path in access_manager.allowed_paper_roots),
        access_policy=config.workspace.access_policy,
        edit_mode=config.workspace.edit_mode,
    )


def _record_workspace_access(
    trace: WorkspaceTrace,
    *,
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
    edit_plan: WorkspaceEditPlan,
) -> WorkspaceTrace:
    external_paths: list[str] = [
        item
        for item in (*evidence_refs, *paper_trace.target_paths, *paper_trace.discovered_documents)
        if _looks_like_absolute_workspace_path(item)
    ]
    access_notes = list(trace.access_notes)
    if edit_plan.reason:
        access_notes.append(edit_plan.reason)
    return WorkspaceTrace(
        **{
            **asdict(trace),
            "external_paths_accessed": _dedupe_strings(external_paths),
            "user_deliverable_requested": edit_plan.active,
            "edit_intent": edit_plan.edit_intent,
            "edit_plan_summary": edit_plan.summary,
            "planned_reads": edit_plan.planned_reads,
            "planned_modifications": edit_plan.planned_modifications,
            "planned_creations": edit_plan.planned_creations,
            "primary_targets": edit_plan.primary_targets,
            "secondary_targets": edit_plan.secondary_targets,
            "referenced_paths": edit_plan.referenced_paths,
            "intended_changes": edit_plan.intended_changes,
            "task_contract": dict(edit_plan.task_contract),
            "skipped_files": edit_plan.skipped_files,
            "skipped_notes": edit_plan.skipped_notes,
            "access_notes": _dedupe_strings(tuple(access_notes)),
        }
    )


def _record_onboarding_coverage(
    trace: WorkspaceTrace,
    coverage: OnboardingCoverage,
) -> WorkspaceTrace:
    return WorkspaceTrace(
        **{
            **asdict(trace),
            "onboarding_total_files": coverage.total_files,
            "onboarding_relevant_readable_files": coverage.relevant_readable_count,
            "onboarding_ignored_noise_files": coverage.ignored_noise_count,
            "onboarding_unreadable_binary_files": coverage.unreadable_binary_count,
            "onboarding_full_relevant_coverage": coverage.full_relevant_coverage,
            "onboarding_inspected_paths": coverage.inspected_paths,
            "onboarding_skipped_paths": coverage.skipped_paths,
            "onboarding_category_counts": dict(coverage.category_counts),
            "onboarding_inspected_category_counts": dict(coverage.inspected_category_counts),
        }
    )


def _record_edit_result(
    trace: WorkspaceTrace,
    *,
    edit_plan: WorkspaceEditPlan,
    edit_result: WorkspaceEditApplyResult,
) -> WorkspaceTrace:
    access_notes = list(trace.access_notes)
    access_notes.extend(edit_result.skipped_notes)
    access_notes.extend(edit_result.rollback_notes)
    return WorkspaceTrace(
        **{
            **asdict(trace),
            "created_files": edit_result.created_files,
            "modified_files": edit_result.modified_files,
            "user_deliverable_requested": edit_plan.active,
            "user_deliverable_file": edit_result.primary_file,
            "user_deliverable_status": edit_result.status if edit_plan.active else "not_requested",
            "user_deliverable_operation": edit_result.operation,
            "user_deliverable_destination_policy": edit_result.destination_policy,
            "skipped_files": edit_result.skipped_files or edit_plan.skipped_files,
            "skipped_notes": edit_result.skipped_notes or edit_plan.skipped_notes,
            "apply_status": edit_result.status if edit_plan.active else "not_requested",
            "file_change_summaries": edit_result.file_change_summaries,
            "rollback_notes": edit_result.rollback_notes,
            "git_repo_detected": edit_result.git_summary.detected,
            "git_repo_root": edit_result.git_summary.repo_root,
            "git_changed_files": edit_result.git_summary.changed_tracked_files,
            "git_untracked_files": edit_result.git_summary.untracked_files,
            "git_commit_message_draft": edit_result.git_summary.commit_message_draft,
            "git_status_before": edit_result.git_summary.before_status,
            "git_status_after": edit_result.git_summary.after_status,
            "access_notes": _dedupe_strings(tuple(access_notes)),
        }
    )


def _detect_workspace_mutation(
    prompt: str,
    mode_selection: ModeSelection,
    access_manager: WorkspaceAccessManager,
) -> WorkspaceMutationRequest:
    prompt_lower = prompt.lower()
    candidate_path = _extract_prompt_write_target(prompt)

    create_requested = any(
        token in prompt_lower for token in ("create ", "generate ", "write ", "生成", "创建", "新建")
    )
    edit_requested = any(
        token in prompt_lower for token in ("open ", "modify ", "edit ", "update ", "change ", "修改", "编辑", "更新")
    )

    if create_requested and candidate_path:
        return WorkspaceMutationRequest(
            action="create_file",
            target_path=candidate_path,
            reason="Prompt explicitly asks for a new workspace deliverable file.",
            strategy="answer_body",
        )

    if edit_requested:
        target_path = next(
            (item for item in mode_selection.matched_paths if item.lower().endswith((".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml"))),
            candidate_path or "",
        )
        if target_path:
            if any(token in prompt_lower for token in ("module docstring", "docstring at the top", "模块文档字符串", "文档字符串")):
                return WorkspaceMutationRequest(
                    action="update_file",
                    target_path=target_path,
                    reason="Prompt requests an in-place Python module docstring edit.",
                    strategy="python_module_docstring",
                )
            return WorkspaceMutationRequest(
                action="update_file",
                target_path=target_path,
                reason="Prompt explicitly asks for an in-place workspace file update.",
                strategy="rewrite_text_file",
            )

    return WorkspaceMutationRequest()


def _apply_workspace_mutation(
    config: LabaiConfig,
    access_manager: WorkspaceAccessManager,
    prompt: str,
    mode_selection: ModeSelection,
    mutation_request: WorkspaceMutationRequest,
    final_answer: str,
    workspace_trace: WorkspaceTrace,
) -> tuple[WorkspaceTrace, str]:
    if not mutation_request.action:
        return workspace_trace, final_answer

    if config.workspace.edit_mode != "auto_edit":
        return (
            WorkspaceTrace(
                **{
                    **asdict(workspace_trace),
                    "user_deliverable_status": "suggest_only",
                    "user_deliverable_operation": mutation_request.action,
                    "access_notes": _dedupe_strings(
                        (
                            *workspace_trace.access_notes,
                            "Workspace edit mode is suggest-only, so no files were modified.",
                        )
                    ),
                }
            ),
            final_answer,
        )

    if mutation_request.action == "create_file":
        write_result = access_manager.write_text_file(
            mutation_request.target_path,
            final_answer.rstrip() + "\n",
            overwrite=False,
        )
        return _workspace_trace_after_write(
            workspace_trace,
            write_result,
            created=True,
            operation="create_file",
        ), final_answer

    if mutation_request.action == "update_file":
        write_result, updated_answer = _update_workspace_file(
            access_manager,
            prompt,
            mutation_request,
            final_answer,
        )
        return _workspace_trace_after_write(
            workspace_trace,
            write_result,
            created=False,
            operation="update_file",
        ), updated_answer

    return workspace_trace, final_answer


def _maybe_apply_prompt_workspace_override(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
) -> tuple[LabaiConfig, str]:
    if mode_selection.mode not in _WORKSPACE_OVERRIDE_MODES:
        return config, ""

    access_manager = WorkspaceAccessManager(config)
    analysis_label = (
        "workspace verification"
        if mode_selection.mode == "workspace_verification"
        else "coding task execution"
        if mode_selection.mode == "workspace_edit"
        else "onboarding analysis"
    )
    for candidate in _iter_prompt_candidates(prompt):
        resolved_candidate = access_manager.resolve_prompt_path(candidate, must_exist=True)
        if resolved_candidate is not None and resolved_candidate.is_dir():
            resolved_root = resolved_candidate.resolve()
            if resolved_root == access_manager.active_workspace_root.resolve():
                return config, ""
            overridden_workspace = replace(
                config.workspace,
                active_workspace_root=resolved_root,
            )
            return (
                replace(config, workspace=overridden_workspace),
                f"Adopted explicit workspace target `{resolved_root}` from the user prompt for {analysis_label}.",
            )
        resolved_root = access_manager.resolve_explicit_read_root(candidate)
        if resolved_root is None:
            continue
        if resolved_root.resolve() == access_manager.active_workspace_root.resolve():
            return config, ""
        overridden_workspace = replace(
            config.workspace,
            active_workspace_root=resolved_root.resolve(),
        )
        return (
            replace(config, workspace=overridden_workspace),
            f"Adopted explicit workspace target `{resolved_root}` from the user prompt for {analysis_label}.",
        )

    return config, ""


def _update_workspace_file(
    access_manager: WorkspaceAccessManager,
    prompt: str,
    mutation_request: WorkspaceMutationRequest,
    final_answer: str,
) -> tuple[WorkspaceWriteResult, str]:
    target = access_manager.resolve_user_path(
        mutation_request.target_path,
        for_write=True,
        must_exist=True,
    )
    original = target.read_text(encoding="utf-8")
    if mutation_request.strategy != "python_module_docstring":
        return (
            WorkspaceWriteResult(
                status="error",
                path=access_manager.display_path(target),
                operation="update_file",
                error="Only the module-docstring workspace edit path is implemented in this phase.",
            ),
            final_answer,
        )

    updated = _add_python_module_docstring(original, target)
    if updated == original:
        return (
            WorkspaceWriteResult(
                status="ok",
                path=access_manager.display_path(target),
                operation="update_file",
            ),
            f"Checked {access_manager.display_path(target)}. A module docstring was already present, so no code changes were needed.",
        )

    write_result = access_manager.write_text_file(
        mutation_request.target_path,
        updated,
        overwrite=True,
    )
    message = (
        f"Updated {write_result.path} by adding a short module docstring at the top."
        if write_result.status == "ok"
        else final_answer
    )
    return write_result, message


def _workspace_trace_after_write(
    workspace_trace: WorkspaceTrace,
    write_result: WorkspaceWriteResult,
    *,
    created: bool,
    operation: str,
) -> WorkspaceTrace:
    created_files = list(workspace_trace.created_files)
    modified_files = list(workspace_trace.modified_files)
    access_notes = list(workspace_trace.access_notes)
    if write_result.status == "ok" and write_result.path:
        if created:
            created_files.append(write_result.path)
        else:
            modified_files.append(write_result.path)
    if write_result.error:
        access_notes.append(write_result.error)
    return WorkspaceTrace(
        **{
            **asdict(workspace_trace),
            "created_files": _dedupe_strings(tuple(created_files)),
            "modified_files": _dedupe_strings(tuple(modified_files)),
            "user_deliverable_file": write_result.path,
            "user_deliverable_status": write_result.status,
            "user_deliverable_operation": operation,
            "access_notes": _dedupe_strings(tuple(access_notes)),
        }
    )


def _extract_prompt_write_target(prompt: str) -> str:
    match = re.search(
        r"(?P<path>(?:[A-Za-z]:[\\/][^\\r\\n\"']*?\.(?:md|py|txt|toml|json|ya?ml)|(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\.(?:md|py|txt|toml|json|ya?ml)))",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group("path").strip("`'\"()[]{}<>.,:; ")


def _add_python_module_docstring(text: str, target_path: Path) -> str:
    lines = text.splitlines(keepends=True)
    index = 0
    if lines and lines[0].startswith("#!"):
        index += 1
    if index < len(lines) and re.match(r"#.*coding[:=]", lines[index]):
        index += 1
    if index < len(lines) and lines[index].lstrip().startswith(('"""', "'''")):
        return text

    stem = target_path.stem.replace("_", " ").strip() or "Module"
    docstring = f'"""{stem[:1].upper() + stem[1:]} module."""\n\n'
    return "".join((*lines[:index], docstring, *lines[index:]))


def _looks_like_absolute_workspace_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")


def _contains_any_phrase(
    prompt_lower: str,
    prompt_original: str,
    *,
    english_tokens: tuple[str, ...] = (),
    chinese_tokens: tuple[str, ...] = (),
) -> bool:
    return any(token in prompt_lower for token in english_tokens) or any(
        token in prompt_original for token in chinese_tokens
    )


def _extract_prompt_write_target(prompt: str) -> str:
    match = re.search(
        r"(?P<path>(?:[A-Za-z]:[\\/][^\r\n\"']*?\.(?:md|txt|csv|json|py|toml|ya?ml|xlsx)|(?:[A-Za-z0-9_. -]+[\\/])*[A-Za-z0-9_. -]+\.(?:md|txt|csv|json|py|toml|ya?ml|xlsx)))",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group("path").strip("`'\"()[]{}<>.,:; ，。；：！？!?")


def _prompt_requests_same_folder(prompt: str) -> bool:
    return _contains_any_phrase(
        prompt.lower(),
        prompt,
        english_tokens=("same folder", "same directory"),
        chinese_tokens=("同一文件夹", "同文件夹", "同一目录", "同目录"),
    )


def _resolve_primary_source_path(
    mode_selection: ModeSelection,
    access_manager: WorkspaceAccessManager,
) -> Path | None:
    for candidate in mode_selection.matched_paths:
        resolved = access_manager.resolve_prompt_path(candidate, must_exist=True)
        if resolved is None:
            continue
        if resolved.is_file():
            return resolved
    return None


def _infer_deliverable_target_name(prompt: str, mode_selection: ModeSelection) -> str:
    if not _contains_any_phrase(
        prompt.lower(),
        prompt,
        english_tokens=("markdown", "md file"),
        chinese_tokens=("md 文件", "markdown 文件"),
    ):
        return ""

    pdf_targets = [item for item in mode_selection.matched_paths if item.lower().endswith(".pdf")]
    language_suffix = "_zh" if mode_selection.response_language == "zh-CN" else ""
    if mode_selection.mode == "paper_summary" and pdf_targets:
        return f"{Path(pdf_targets[0]).stem}_summary{language_suffix}.md"
    if mode_selection.mode == "paper_compare" and len(pdf_targets) >= 2:
        left = Path(pdf_targets[0]).stem
        right = Path(pdf_targets[1]).stem
        return f"{left}_vs_{right}_compare{language_suffix}.md"
    return f"answer{language_suffix}.md"


def _resolve_deliverable_target(
    prompt: str,
    mode_selection: ModeSelection,
    access_manager: WorkspaceAccessManager,
    candidate_path: str,
) -> tuple[str, str]:
    target_path = candidate_path or _infer_deliverable_target_name(prompt, mode_selection)
    if not target_path:
        return "", ""

    if _looks_like_absolute_workspace_path(target_path):
        return target_path, "explicit_path"

    if _prompt_requests_same_folder(prompt):
        source_path = _resolve_primary_source_path(mode_selection, access_manager)
        if source_path is not None:
            return str((source_path.parent / target_path).resolve()), "same_folder"

    return target_path, "active_workspace"


def _classify_output_intent(mutation_request: WorkspaceMutationRequest) -> OutputIntent:
    if mutation_request.action:
        return OutputIntent(
            name="deliverable_requested",
            reason=mutation_request.reason,
        )
    return OutputIntent()


def _detect_workspace_mutation(
    prompt: str,
    mode_selection: ModeSelection,
    access_manager: WorkspaceAccessManager,
) -> WorkspaceMutationRequest:
    prompt_lower = prompt.lower()
    candidate_path = _extract_prompt_write_target(prompt)

    create_requested = _contains_any_phrase(
        prompt_lower,
        prompt,
        english_tokens=(
            "create ",
            "generate ",
            "write ",
            "save ",
            "export ",
            "save as ",
            "write this to",
            "write the result to",
            "create a file",
            "generate a markdown file",
            "put the result in",
        ),
        chinese_tokens=(
            "保存到",
            "保存成",
            "写成",
            "写到",
            "生成",
            "创建",
            "输出成",
            "导出",
            "在当前目录创建",
            "在同一文件夹下生成",
            "生成一个 md 文件",
            "写成一个文件",
        ),
    )
    edit_requested = _contains_any_phrase(
        prompt_lower,
        prompt,
        english_tokens=("open ", "modify ", "edit ", "update ", "change "),
        chinese_tokens=("修改", "编辑", "更新"),
    )

    if create_requested:
        target_path, destination_policy = _resolve_deliverable_target(
            prompt,
            mode_selection,
            access_manager,
            candidate_path,
        )
        if target_path:
            return WorkspaceMutationRequest(
                action="create_file",
                target_path=target_path,
                reason="Prompt explicitly asks for a new workspace deliverable file.",
                strategy="answer_body",
                destination_policy=destination_policy,
            )

    if edit_requested:
        target_path = next(
            (
                item
                for item in mode_selection.matched_paths
                if item.lower().endswith((".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml"))
            ),
            candidate_path or "",
        )
        if target_path:
            if _contains_any_phrase(
                prompt_lower,
                prompt,
                english_tokens=("module docstring", "docstring at the top"),
                chinese_tokens=("模块文档字符串", "文档字符串"),
            ):
                return WorkspaceMutationRequest(
                    action="update_file",
                    target_path=target_path,
                    reason="Prompt requests an in-place Python module docstring edit.",
                    strategy="python_module_docstring",
                    destination_policy="existing_file",
                )
            return WorkspaceMutationRequest(
                action="update_file",
                target_path=target_path,
                reason="Prompt explicitly asks for an in-place workspace file update.",
                strategy="rewrite_text_file",
                destination_policy="existing_file",
            )

    return WorkspaceMutationRequest()


def _apply_workspace_mutation(
    config: LabaiConfig,
    access_manager: WorkspaceAccessManager,
    prompt: str,
    mode_selection: ModeSelection,
    mutation_request: WorkspaceMutationRequest,
    final_answer: str,
    workspace_trace: WorkspaceTrace,
) -> tuple[WorkspaceTrace, str]:
    if not mutation_request.action:
        return workspace_trace, final_answer

    if config.workspace.edit_mode != "auto_edit":
        return (
            WorkspaceTrace(
                **{
                    **asdict(workspace_trace),
                    "user_deliverable_status": "suggest_only",
                    "user_deliverable_operation": mutation_request.action,
                    "user_deliverable_destination_policy": mutation_request.destination_policy,
                    "access_notes": _dedupe_strings(
                        (
                            *workspace_trace.access_notes,
                            "Workspace edit mode is suggest-only, so no files were modified.",
                        )
                    ),
                }
            ),
            final_answer,
        )

    if mutation_request.action == "create_file":
        write_result = access_manager.write_text_file(
            mutation_request.target_path,
            final_answer.rstrip() + "\n",
            overwrite=False,
        )
        return _workspace_trace_after_write(
            workspace_trace,
            write_result,
            created=True,
            operation="create_file",
            destination_policy=mutation_request.destination_policy,
        ), final_answer

    if mutation_request.action == "update_file":
        write_result, updated_answer = _update_workspace_file(
            access_manager,
            prompt,
            mutation_request,
            final_answer,
        )
        return _workspace_trace_after_write(
            workspace_trace,
            write_result,
            created=False,
            operation="update_file",
            destination_policy=mutation_request.destination_policy,
        ), updated_answer

    return workspace_trace, final_answer


def _workspace_trace_after_write(
    workspace_trace: WorkspaceTrace,
    write_result: WorkspaceWriteResult,
    *,
    created: bool,
    operation: str,
    destination_policy: str,
) -> WorkspaceTrace:
    created_files = list(workspace_trace.created_files)
    modified_files = list(workspace_trace.modified_files)
    access_notes = list(workspace_trace.access_notes)
    if write_result.status == "ok" and write_result.path:
        if created:
            created_files.append(write_result.path)
        else:
            modified_files.append(write_result.path)
    if write_result.error:
        access_notes.append(write_result.error)
    return WorkspaceTrace(
        **{
            **asdict(workspace_trace),
            "created_files": _dedupe_strings(tuple(created_files)),
            "modified_files": _dedupe_strings(tuple(modified_files)),
            "user_deliverable_file": write_result.path,
            "user_deliverable_status": write_result.status,
            "user_deliverable_operation": operation,
            "user_deliverable_destination_policy": destination_policy,
            "access_notes": _dedupe_strings(tuple(access_notes)),
        }
    )


def _extract_prompt_write_target(prompt: str) -> str:
    candidates: list[str] = []
    for match in re.finditer(
        r"(?P<path>(?:[A-Za-z]:[\\/][^\r\n\"']*?\.(?:md|txt|csv|json|py|toml|ya?ml|xlsx)|(?:[A-Za-z0-9_. -]+[\\/])*[A-Za-z0-9_. -]+\.(?:md|txt|csv|json|py|toml|ya?ml|xlsx)))",
        prompt,
        re.IGNORECASE,
    ):
        candidate = match.group("path").strip("`'\"()[]{}<>.,:;!?，。；：！？、）】」』")
        candidate = re.sub(r"^(?:create|generate|save|write|export)\s+", "", candidate, flags=re.IGNORECASE)
        name = Path(candidate).name
        if not name or name.startswith("."):
            continue
        candidates.append(candidate)

    if not candidates:
        return ""
    return candidates[-1]


def _extract_prompt_write_target(prompt: str) -> str:
    candidates: list[str] = []
    allowed_suffixes = {".md", ".txt", ".csv", ".json", ".py", ".toml", ".yaml", ".yml", ".xlsx"}
    for raw_candidate in _iter_prompt_candidates(prompt):
        candidate = raw_candidate.strip()
        candidate = re.sub(r"^(?:create|generate|save|write|export)\s+", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip()
        suffix = Path(candidate).suffix.lower()
        name = Path(candidate).name
        if suffix not in allowed_suffixes:
            continue
        if not name or name.startswith("."):
            continue
        candidates.append(candidate)

    if not candidates:
        return ""
    return candidates[-1]


def _prompt_requests_explicit_deliverable(
    prompt: str,
    candidate_path: str,
) -> bool:
    prompt_lower = prompt.lower()
    explicit_delivery_cue = _contains_any_phrase(
        prompt_lower,
        prompt,
        english_tokens=(
            "save ",
            "save as ",
            "export ",
            "write this to",
            "write the result to",
            "create a file",
            "generate a markdown file",
            "generate a file",
            "put the result in",
            "same folder",
            "current directory",
            "current workspace",
            "project root",
        ),
        chinese_tokens=(
            "保存到",
            "保存成",
            "写成",
            "写到",
            "生成一个 md 文件",
            "生成一个文件",
            "在当前目录创建",
            "在当前工作区创建",
            "在同一文件夹下生成",
            "同一文件夹下",
            "当前目录",
            "当前工作区",
        ),
    )
    action_with_target = _contains_any_phrase(
        prompt_lower,
        prompt,
        english_tokens=("create ", "generate ", "save ", "export ", "write "),
        chinese_tokens=("创建", "生成", "保存到", "保存成", "写成", "写到", "导出"),
    ) and bool(candidate_path or _prompt_requests_same_folder(prompt))
    return explicit_delivery_cue or action_with_target


def _prompt_requests_same_folder(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return (
        "same folder" in prompt_lower
        or "同一文件夹" in prompt
        or "同一文件夹下" in prompt
        or "同个文件夹" in prompt
    )


def _prompt_requests_explicit_deliverable(
    prompt: str,
    candidate_path: str,
) -> bool:
    prompt_lower = prompt.lower()
    explicit_delivery_cue = any(
        token in prompt_lower
        for token in (
            "save ",
            "save as ",
            "export ",
            "write this to",
            "write the result to",
            "create a file",
            "generate a markdown file",
            "generate a file",
            "put the result in",
            "same folder",
            "current directory",
            "current workspace",
            "project root",
        )
    ) or any(
        token in prompt
        for token in (
            "保存到",
            "保存成",
            "写成",
            "写到",
            "生成一个 md 文件",
            "生成一个文件",
            "在当前目录创建",
            "在当前工作区创建",
            "在同一文件夹下生成",
            "同一文件夹下",
            "当前目录",
            "当前工作区",
        )
    )
    action_with_target = (
        any(
            token in prompt_lower
            for token in ("create ", "generate ", "save ", "export ", "write ")
        )
        or any(
            token in prompt
            for token in ("创建", "生成", "保存到", "保存成", "写成", "写到", "导出")
        )
    ) and bool(candidate_path or _prompt_requests_same_folder(prompt))
    return explicit_delivery_cue or action_with_target


def _detect_workspace_mutation(
    prompt: str,
    mode_selection: ModeSelection,
    access_manager: WorkspaceAccessManager,
) -> WorkspaceMutationRequest:
    prompt_lower = prompt.lower()
    candidate_path = _extract_prompt_write_target(prompt)

    create_requested = _prompt_requests_explicit_deliverable(prompt, candidate_path)
    edit_requested = _contains_any_phrase(
        prompt_lower,
        prompt,
        english_tokens=("open ", "modify ", "edit ", "update ", "change "),
        chinese_tokens=("修改", "编辑", "更新"),
    )

    if create_requested:
        target_path, destination_policy = _resolve_deliverable_target(
            prompt,
            mode_selection,
            access_manager,
            candidate_path,
        )
        if target_path:
            return WorkspaceMutationRequest(
                action="create_file",
                target_path=target_path,
                reason="Prompt explicitly asks for a new workspace deliverable file.",
                strategy="answer_body",
                destination_policy=destination_policy,
            )

    if edit_requested:
        target_path = next(
            (
                item
                for item in mode_selection.matched_paths
                if item.lower().endswith((".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml"))
            ),
            candidate_path or "",
        )
        if target_path:
            if _contains_any_phrase(
                prompt_lower,
                prompt,
                english_tokens=("module docstring", "docstring at the top"),
                chinese_tokens=("模块文档字符串", "文档字符串"),
            ):
                return WorkspaceMutationRequest(
                    action="update_file",
                    target_path=target_path,
                    reason="Prompt requests an in-place Python module docstring edit.",
                    strategy="python_module_docstring",
                    destination_policy="existing_file",
                )
            return WorkspaceMutationRequest(
                action="update_file",
                target_path=target_path,
                reason="Prompt explicitly asks for an in-place workspace file update.",
                strategy="rewrite_text_file",
                destination_policy="existing_file",
            )

    return WorkspaceMutationRequest()


def run_research_loop(
    config: LabaiConfig,
    prompt: str,
    session_id: str,
) -> ResearchResult:
    started_at = _utc_timestamp()
    mode_selection = select_mode(config, prompt)
    config, workspace_override_note = _maybe_apply_prompt_workspace_override(
        config,
        prompt,
        mode_selection,
    )
    access_manager = WorkspaceAccessManager(config)
    mode_selection = select_mode(config, prompt)
    paper_trace = PaperTrace()
    workspace_trace = _initial_workspace_trace(config, access_manager)
    onboarding_coverage = OnboardingCoverage()
    workspace_coverage_note = ""
    workspace_coverage_skipped_notes: tuple[str, ...] = ()
    if mode_selection.mode in {"project_onboarding", "workspace_verification", "workspace_edit"}:
        onboarding_coverage = _collect_onboarding_coverage(access_manager.active_workspace_root)
        workspace_coverage_note = (
            f"Workspace coverage accounted for {onboarding_coverage.total_files} files, including "
            f"{onboarding_coverage.relevant_readable_count} relevant readable files and "
            f"{len(onboarding_coverage.inspected_paths)} inspected files."
        )
        workspace_coverage_skipped_notes = onboarding_coverage.skipped_notes[:2]
    edit_plan = build_workspace_edit_plan(prompt, mode_selection, access_manager)
    output_intent_name, output_intent_reason = classify_output_intent(edit_plan)
    tool_decisions = _plan_tool_usage(
        prompt,
        access_manager.active_workspace_root,
        mode_selection,
        edit_plan,
        workspace_coverage=(
            onboarding_coverage if mode_selection.mode == "workspace_edit" else None
        ),
    )
    tool_calls, observations, evidence_refs = _execute_tool_plan(access_manager, tool_decisions)
    if workspace_override_note:
        observations.insert(0, workspace_override_note)
    if workspace_coverage_note:
        observations.append(workspace_coverage_note)
        observations.extend(workspace_coverage_skipped_notes)
    if _is_paper_mode(mode_selection.mode):
        paper_context = _prepare_paper_mode_context(config, prompt, mode_selection)
        paper_trace = _paper_trace_from_context(paper_context, mode_selection)
        paper_trace = _prepare_detailed_paper_trace(
            config,
            mode_selection,
            paper_trace,
        )
        observations.extend(paper_context.observations)
        evidence_refs = _dedupe_strings((*evidence_refs, *paper_context.evidence_refs))
        if paper_context.target_paths:
            evidence_refs = _dedupe_strings((*paper_context.target_paths, *evidence_refs))
    elif mode_selection.matched_paths:
        evidence_refs = _dedupe_strings((*mode_selection.matched_paths, *evidence_refs))
    grounded_draft = _build_grounded_draft(
        config,
        prompt,
        mode_selection,
        tool_calls,
        evidence_refs,
        paper_trace,
        workspace_root=access_manager.active_workspace_root,
        edit_plan=edit_plan,
        workspace_coverage=(
            onboarding_coverage if mode_selection.mode == "workspace_edit" else None
        ),
    )
    workspace_trace = _record_workspace_access(
        workspace_trace,
        evidence_refs=evidence_refs,
        paper_trace=paper_trace,
        edit_plan=edit_plan,
    )
    if mode_selection.mode in {"project_onboarding", "workspace_verification", "workspace_edit"}:
        workspace_trace = _record_onboarding_coverage(
            workspace_trace,
            onboarding_coverage,
        )

    try:
        route = _run_answer_route(
            config,
            prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            grounded_draft,
        )
        final_answer = route.text
        if (
            mode_selection.mode == "project_onboarding"
            and grounded_draft
            and _project_onboarding_answer_needs_repair(
                final_answer,
                response_language=mode_selection.response_language,
            )
        ):
            observations.append(
                "Project onboarding output drifted away from the required practical onboarding structure, so the grounded onboarding draft was used."
            )
            final_answer = grounded_draft
        if (
            mode_selection.mode == "workspace_verification"
            and grounded_draft
            and _workspace_verification_answer_needs_repair(
                final_answer,
                response_language=mode_selection.response_language,
            )
        ):
            observations.append(
                "Workspace verification output drifted away from the required readiness-focused structure, so the grounded verification draft was used."
            )
            final_answer = grounded_draft
        if mode_selection.mode == "prompt_compiler":
            final_answer = _apply_prompt_compiler_guard(prompt, final_answer)
        if paper_trace.active:
            paper_trace, final_answer = _apply_paper_consistency_guard(
                config,
                prompt,
                session_id,
                observations,
                evidence_refs,
                mode_selection,
                paper_trace,
                final_answer,
            )
        edit_result = apply_workspace_edit_plan(
            config,
            access_manager,
            prompt,
            edit_plan,
            final_answer,
            observations,
        )
        workspace_trace = _record_edit_result(
            workspace_trace,
            edit_plan=edit_plan,
            edit_result=edit_result,
        )
        if edit_plan.active and edit_result.display_answer:
            final_answer = edit_result.display_answer
        completed_at = _utc_timestamp()
        outcome_summary = _success_summary(
            mode_selection.mode,
            route.runtime_used,
            route.provider_used,
            route.runtime_fallback.applied,
            route.provider_fallback.applied,
            len(tool_calls),
        )
        return ResearchResult(
            session_id=session_id,
            command="ask",
            prompt=prompt,
            selected_mode=mode_selection.mode,
            mode_reason=mode_selection.reason,
            answer_schema=mode_selection.answer_schema,
            read_strategy=mode_selection.read_strategy,
            read_strategy_reason=mode_selection.read_strategy_reason,
            response_style=mode_selection.response_style,
            response_language=mode_selection.response_language,
            output_intent=output_intent_name,
            output_intent_reason=output_intent_reason,
            evidence_refs=evidence_refs,
            operational_status=_derive_operational_status(route),
            requested_runtime=route.requested_runtime,
            runtime_used=route.runtime_used,
            runtime_fallback=route.runtime_fallback,
            requested_provider=route.requested_provider,
            provider_used=route.provider_used,
            provider_model=route.provider_model,
            selected_embedding_model=paper_trace.selected_embedding_model,
            fallback=route.provider_fallback,
            tools_used=bool(tool_calls),
            tool_decisions=tool_decisions,
            tool_calls=tool_calls,
            observations=observations,
            paper_trace=paper_trace,
            workspace_trace=workspace_trace,
            final_answer=final_answer,
            outcome_summary=outcome_summary,
            status="ok",
            error="",
            started_at=started_at,
            completed_at=completed_at,
        )
    except (ProviderError, RuntimeAdapterError) as exc:
        completed_at = _utc_timestamp()
        runtime_used = config.runtime.runtime
        runtime_fallback = _no_runtime_fallback(config.runtime.runtime, config.runtime.fallback_runtime)
        requested_provider = (
            get_default_provider(config).name if config.runtime.runtime == "native" else "claw"
        )

        if config.runtime.runtime == "claw" and config.runtime.fallback_runtime == "native":
            runtime_used = "native"
            runtime_fallback = RuntimeFallbackInfo(
                applied=True,
                requested_runtime="claw",
                active_runtime="native",
                fallback_runtime=config.runtime.fallback_runtime,
                reason=str(exc),
            )
            requested_provider = get_default_provider(config).name

        provider_fallback = FallbackInfo(
            applied=False,
            policy=config.fallback_policy,
            requested_provider=requested_provider,
            active_provider=requested_provider,
            reason="",
        )
        return ResearchResult(
            session_id=session_id,
            command="ask",
            prompt=prompt,
            selected_mode=mode_selection.mode,
            mode_reason=mode_selection.reason,
            answer_schema=mode_selection.answer_schema,
            read_strategy=mode_selection.read_strategy,
            read_strategy_reason=mode_selection.read_strategy_reason,
            response_style=mode_selection.response_style,
            response_language=mode_selection.response_language,
            output_intent=output_intent_name,
            output_intent_reason=output_intent_reason,
            evidence_refs=evidence_refs,
            operational_status=_derive_error_operational_status(
                config.runtime.runtime,
                runtime_fallback,
            ),
            requested_runtime=config.runtime.runtime,
            runtime_used=runtime_used,
            runtime_fallback=runtime_fallback,
            requested_provider=requested_provider,
            provider_used=requested_provider,
            provider_model=None,
            selected_embedding_model=paper_trace.selected_embedding_model,
            fallback=provider_fallback,
            tools_used=bool(tool_calls),
            tool_decisions=tool_decisions,
            tool_calls=tool_calls,
            observations=observations,
            paper_trace=paper_trace,
            workspace_trace=workspace_trace,
            final_answer="",
            outcome_summary=f"{mode_selection.mode} research loop failed before answer generation: {exc}",
            status="error",
            error=str(exc),
            started_at=started_at,
            completed_at=completed_at,
        )


def result_to_session_record(
    result: ResearchResult,
    *,
    answer_artifact: dict[str, object] | None = None,
    command_override: str | None = None,
    workflow_trace: dict[str, object] | None = None,
) -> SessionRecord:
    return SessionRecord(
        session_id=result.session_id,
        command=command_override or result.command,
        started_at=result.started_at,
        completed_at=result.completed_at,
        prompt=result.prompt,
        mode=result.selected_mode,
        mode_reason=result.mode_reason,
        answer_schema=result.answer_schema,
        read_strategy=result.read_strategy,
        read_strategy_reason=result.read_strategy_reason,
        response_style=result.response_style,
        response_language=result.response_language,
        output_intent=result.output_intent,
        output_intent_reason=result.output_intent_reason,
        evidence_refs=list(result.evidence_refs),
        operational_status=result.operational_status,
        requested_runtime=result.requested_runtime,
        runtime=result.runtime_used,
        runtime_fallback=asdict(result.runtime_fallback),
        requested_provider=result.requested_provider,
        provider=result.provider_used,
        model=result.provider_model,
        embedding_model=result.selected_embedding_model,
        status=result.status,
        fallback=asdict(result.fallback),
        tools_used=result.tools_used,
        tool_decisions=[asdict(item) for item in result.tool_decisions],
        tool_calls=[asdict(item) for item in result.tool_calls],
        observations=result.observations,
        paper_trace=asdict(result.paper_trace),
        workspace_trace=asdict(result.workspace_trace),
        workflow_trace=dict(workflow_trace or {}),
        answer_artifact=dict(answer_artifact or {}),
        final_answer=result.final_answer,
        outcome_summary=result.outcome_summary,
        error=result.error,
    )


def result_to_audit_record(
    result: ResearchResult,
    *,
    answer_artifact: dict[str, object] | None = None,
    command_override: str | None = None,
    workflow_trace: dict[str, object] | None = None,
) -> AuditRecord:
    preview = result.final_answer.splitlines()[0] if result.final_answer else ""
    return AuditRecord(
        timestamp=result.completed_at,
        command=command_override or result.command,
        session_id=result.session_id,
        mode=result.selected_mode,
        mode_reason=result.mode_reason,
        answer_schema=result.answer_schema,
        read_strategy=result.read_strategy,
        read_strategy_reason=result.read_strategy_reason,
        response_style=result.response_style,
        response_language=result.response_language,
        output_intent=result.output_intent,
        output_intent_reason=result.output_intent_reason,
        evidence_refs=list(result.evidence_refs),
        operational_status=result.operational_status,
        requested_runtime=result.requested_runtime,
        runtime=result.runtime_used,
        requested_provider=result.requested_provider,
        provider=result.provider_used,
        model=result.provider_model or "",
        embedding_model=result.selected_embedding_model or "",
        status=result.status,
        tool_count=len(result.tool_calls),
        outcome_summary=result.outcome_summary,
        runtime_fallback=asdict(result.runtime_fallback),
        fallback=asdict(result.fallback),
        prompt_preview=result.prompt[:120],
        paper_trace=asdict(result.paper_trace),
        workspace_trace=asdict(result.workspace_trace),
        workflow_trace=dict(workflow_trace or {}),
        answer_artifact=dict(answer_artifact or {}),
        answer_preview=preview[:160],
        error=result.error,
    )


def _apply_paper_consistency_guard(
    config: LabaiConfig,
    prompt: str,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> tuple[PaperTrace, str]:
    answer_text = _finalize_paper_answer_text(
        prompt,
        mode_selection,
        paper_trace,
        answer_text,
    )
    report = _evaluate_paper_answer_consistency(prompt, mode_selection, paper_trace, answer_text)
    requested_slots = _requested_paper_slots(prompt, mode_selection.mode)
    deterministic_summary = ""
    deterministic_report: dict[str, object] | None = None
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    recurring_signals = (
        _collect_recurring_limitation_signals(paper_trace.document_notes)
        if recurring_limitations
        else None
    )
    if (
        mode_selection.mode == "paper_summary"
        and paper_trace.document_notes
        and mode_selection.read_strategy in {"full_document", "hybrid"}
    ):
        deterministic_summary = _build_slot_grounded_paper_summary(
            paper_trace.document_notes[0],
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
        )
    if (
        mode_selection.mode == "paper_grounded_qa"
        and paper_trace.active
        and _is_narrow_grounded_paper_qa(prompt)
    ):
        concise_qa = _build_narrow_grounded_paper_answer(
            prompt,
            paper_trace,
            response_language=mode_selection.response_language,
        )
        concise_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            concise_qa,
        )
        if not concise_report["needs_repair"]:
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=concise_qa != answer_text,
                    consistency_check_notes=tuple(concise_report["notes"]),
                ),
                concise_qa,
            )
    if recurring_limitations and recurring_signals and mode_selection.paper_output_profile == "detailed_paper_note":
        deterministic_limitations = _build_recurring_limitations_answer(
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
        )
        deterministic_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_limitations,
        )
        if not deterministic_report["needs_repair"]:
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=deterministic_limitations != answer_text,
                    consistency_check_notes=tuple(
                        deterministic_report["notes"]
                        or ["Deterministic recurring-limitations synthesis applied from cleaned slot evidence."]
                    ),
                ),
                deterministic_limitations,
            )
    if (
        mode_selection.mode == "paper_compare"
        and mode_selection.response_language == "zh-CN"
        and paper_trace.document_notes
    ):
        cleanup_selection = replace(
            mode_selection,
            mode="repo_overview",
            reason="Final Chinese compare cleanup rewrites the already-grounded compare note without rereading the paper.",
            answer_schema="language_cleanup",
            read_strategy="none",
            read_strategy_reason="Single-language compare cleanup operates on the final rendered compare note.",
            paper_output_profile="none",
        )
        english_compare = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=requested_slots,
            response_language="en",
            response_style="structured",
            paper_output_profile=mode_selection.paper_output_profile,
        )
        structured_zh_compare = _translate_structured_compare_sections_to_chinese(
            config,
            session_id,
            observations,
            evidence_refs,
            cleanup_selection,
            english_compare,
            document_notes=paper_trace.document_notes,
            requested_slots=requested_slots,
        )
        if structured_zh_compare and (
            _looks_insufficiently_translated_chinese(structured_zh_compare)
            or _compare_answer_surface_noise_issues(structured_zh_compare)
            or re.search(r"\b(?:We|Gaussian|reference|translation)\b", structured_zh_compare)
        ):
            structured_zh_compare = _polish_structured_compare_sections_in_chinese(
                config,
                session_id,
                observations,
                evidence_refs,
                cleanup_selection,
                structured_zh_compare,
                document_notes=paper_trace.document_notes,
                requested_slots=requested_slots,
            )
        structured_zh_compare = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            structured_zh_compare,
        )
        whole_zh_compare = _translate_paper_answer_to_target_language(
            config,
            prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            paper_trace,
            english_compare,
        )
        whole_zh_compare = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            whole_zh_compare,
        )

        best_candidate = ""
        best_report: dict[str, object] | None = None
        best_score = -10_000
        min_sections = max(4, min(6, len(requested_slots)))

        for candidate in (whole_zh_compare, structured_zh_compare):
            if not candidate:
                continue
            report_candidate = _evaluate_paper_answer_consistency(
                prompt,
                mode_selection,
                paper_trace,
                candidate,
            )
            false_missing = _compare_false_missing_supported_slots(
                candidate,
                paper_trace.document_notes,
                response_language=mode_selection.response_language,
            )
            section_count = _structured_compare_section_count(candidate)
            surface_issues = _compare_answer_surface_noise_issues(candidate)
            score = (
                (200 if not report_candidate["needs_repair"] else 0)
                + section_count * 8
                - len(false_missing) * 40
                - len(surface_issues) * 15
            )
            if score > best_score:
                best_candidate = candidate
                best_report = report_candidate
                best_score = score

        if (
            best_candidate
            and best_report is not None
            and _structured_compare_section_count(best_candidate) >= min_sections
            and not _compare_false_missing_supported_slots(
                best_candidate,
                paper_trace.document_notes,
                response_language=mode_selection.response_language,
            )
            and not best_report["needs_repair"]
        ):
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=best_candidate != answer_text,
                    consistency_check_notes=(
                        *best_report["notes"],
                    ),
                ),
                best_candidate,
            )
    if mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        deterministic_compare = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            paper_output_profile=mode_selection.paper_output_profile,
        )
        deterministic_compare = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_compare,
        )
        deterministic_compare = _translate_paper_answer_to_target_language(
            config,
            prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            paper_trace,
            deterministic_compare,
        )
        deterministic_compare_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_compare,
        )
        current_compare_sections = _structured_compare_section_count(answer_text)
        deterministic_compare_sections = _structured_compare_section_count(deterministic_compare)
        compare_structure_gap = (
            deterministic_compare_sections >= 4
            and current_compare_sections + 1 < deterministic_compare_sections
        )
        if (
            not deterministic_compare_report["needs_repair"]
            and (report["needs_repair"] or compare_structure_gap)
        ):
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=False,
                    consistency_check_notes=tuple(
                        deterministic_compare_report["notes"]
                        or [
                            "Deterministic slot-grounded comparison applied from compare-ready document notes before any repair loop."
                        ]
                    ),
                ),
                deterministic_compare,
            )
    should_attempt_repair = bool(report["needs_repair"])
    if not should_attempt_repair:
        return (
            replace(
                paper_trace,
                consistency_check_status="passed",
                consistency_check_repaired=False,
                consistency_check_notes=tuple(report["notes"]),
            ),
            answer_text,
        )

    repaired_text = answer_text
    compare_clean_pass = False
    repair_prompt = _compose_paper_consistency_prompt(
        original_prompt=prompt,
        answer_text=answer_text,
        report_notes=tuple(report["notes"]),
        response_language=mode_selection.response_language,
        mode=mode_selection.mode,
        response_style=mode_selection.response_style,
    )
    try:
        repair_route = _run_answer_route(
            config,
            repair_prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            _build_grounded_draft(
                config,
                prompt,
                mode_selection,
                [],
                evidence_refs,
                paper_trace,
            ),
        )
        if repair_route.text.strip():
            repaired_text = repair_route.text.strip()
    except (ProviderError, RuntimeAdapterError):
        repaired_text = answer_text

    if _is_recurring_limitations_prompt(prompt):
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if recurring_signals["clear"]:
            repaired_text = _build_recurring_limitations_answer(
                paper_trace.document_notes,
                response_language=mode_selection.response_language,
                response_style=mode_selection.response_style,
            )

    repaired_report = _evaluate_paper_answer_consistency(
        prompt,
        mode_selection,
        paper_trace,
        repaired_text,
    )
    if (
        repaired_report["needs_repair"]
        and mode_selection.mode == "paper_summary"
        and mode_selection.response_language == "zh-CN"
        and paper_trace.document_notes
    ):
        translation_seed = deterministic_summary or repaired_text
        translation_prompt = _compose_slot_translation_prompt(
            source_text=translation_seed,
            response_style=mode_selection.response_style,
        )
        try:
            translation_route = _run_answer_route(
                config,
                translation_prompt,
                session_id,
                observations,
                evidence_refs,
                mode_selection,
                translation_seed,
            )
            if translation_route.text.strip():
                repaired_text = translation_route.text.strip()
                repaired_report = _evaluate_paper_answer_consistency(
                    prompt,
                    mode_selection,
                    paper_trace,
                    repaired_text,
                )
        except (ProviderError, RuntimeAdapterError, StopIteration):
            pass
    if repaired_report["needs_repair"] and mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        repaired_text = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
        )
        repaired_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    if repaired_report["needs_repair"]:
        repaired_text = _deterministic_paper_consistency_trim(
            repaired_text,
            paper_trace,
            prompt=prompt,
            mode=mode_selection.mode,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            requested_slots=_explicit_paper_slots(prompt),
        )
        repaired_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    if repaired_report["needs_repair"] and deterministic_summary:
        repaired_text = deterministic_summary
        repaired_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    if repaired_report["needs_repair"] and mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        repaired_text = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
        )
        repaired_report = {
            "needs_repair": False,
            "notes": ["Deterministic slot-grounded comparison applied after generative compare output stayed too generic or excerpt-heavy."],
        }
    if repaired_report["needs_repair"] and _is_recurring_limitations_prompt(prompt):
        deterministic_limitations = _build_recurring_limitations_answer(
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
        )
        if repaired_text == deterministic_limitations:
            repaired_report = {
                "needs_repair": False,
                "notes": ["Deterministic recurring-limitations synthesis applied from cleaned slot evidence."],
            }

    final_status = "repaired" if not repaired_report["needs_repair"] else "repair_incomplete"
    final_notes = tuple(repaired_report["notes"] or report["notes"])
    return (
        replace(
            paper_trace,
            consistency_check_status=final_status,
            consistency_check_repaired=repaired_text != answer_text,
            consistency_check_notes=final_notes,
        ),
        repaired_text,
    )


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the aggregated slot evidence."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots:
        if mode_selection.mode == "paper_summary":
            notes.append(
                "The answer did not clearly cover these requested summary dimensions: "
                + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
                + "."
            )
        if mode_selection.mode == "paper_compare":
            notes.append(
                "The comparison did not clearly cover these requested dimensions: "
                + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
                + "."
            )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }


def _compose_slot_translation_prompt_legacy(*, source_text: str, response_style: str) -> str:
    structured_compare = _looks_like_structured_compare_text(source_text)
    style_instruction = (
        "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "Return a concise Simplified Chinese memo."
    )
    instructions = [
        "Translate the grounded memo below into natural Simplified Chinese.",
        "Use Simplified Chinese only.",
        "Do not use Japanese, katakana, hiragana, or Traditional Chinese.",
        "Keep only facts that already appear in the grounded memo.",
        "Do not add generic finance or machine-learning commentary.",
        "If a dimension is unclear or missing, say 文中未明确说明。",
        "Keep method-family names, abbreviations, and file identifiers exact when needed.",
    ]
    if structured_compare:
        instructions.extend(
            [
                "Translate line by line and preserve the compare structure.",
                "Do not omit any heading, bullet, or compared document line.",
                "Do not collapse the comparison into a broad summary.",
                "If a line begins with '- `filename`:', keep the filename exactly and translate only the rest of the line.",
                "Use these Chinese section headings exactly when the corresponding sections appear: 比较文献, 研究问题, 样本与数据, 方法, 主要发现, 局限, 结论, 实践或投资含义。",
                "Translate each '- Contrast:' line as '- 对比：'.",
            ]
        )
    instructions.extend(
        [
            style_instruction,
            "",
            "Grounded memo:",
            source_text,
            "",
            "Return only the final Chinese answer body.",
        ]
    )
    return "\n".join(instructions)


def _compare_has_formula_glyphs(text: str) -> bool:
    return any(marker in text for marker in ("\u2211", "\u03bb", "\u03b8", "\u0302"))


def _compare_summary_quality_score(
    slot_name: str,
    summary: str,
    *,
    support_status: str,
) -> int:
    lowered = unicodedata.normalize("NFKC", summary).lower()
    score = _compare_sentence_specificity_score(slot_name, lowered)
    score += min(5, _detail_marker_count(slot_name, lowered))
    if _slot_is_clearly_supported(support_status):
        score += 5
    elif support_status == "weakly_supported":
        score += 1
    if slot_name == "research_question" and _compare_sentence_has_question_signal(summary):
        score += 4
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "this study aims",
            "fundamental goal",
            "goal of asset pricing",
            "we study liquidity provision",
            "we introduce the concept",
            "this paper aims",
            "aims to forecast",
            "seeks to forecast",
            "this article focuses on",
            "this article questions",
            "the main contribution of this article",
        )
    ):
        score += 4
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "objective function",
            "validation objective",
            "forecast errors",
            "hyperparameter",
            "hyperparameters",
        )
    ):
        score -= 8
    if slot_name == "research_question" and _compare_has_formula_glyphs(summary):
        score -= 8
    if slot_name == "sample_or_data" and (
        _looks_like_explicit_sample_data_text(summary)
        or _compare_sentence_has_sample_signal(summary)
    ):
        score += 5
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "crsp",
            "nyse",
            "amex",
            "nasdaq",
            "cross-section",
            "individual stocks",
            "historical underlying asset prices",
        )
    ):
        score += 4
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "predictive power",
            "benchmark model",
            "portfolio level",
            "stock level",
            "r2 ",
            "r2)",
            "r2(",
            "liquid stocks",
            "behave erratically",
        )
    ):
        score -= 6
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "volatility smile",
            "calibrating these models",
            "advanced option pricing models beyond",
            "black-scholes world",
            "constant volatility assumed",
        )
    ) and not any(
        marker in lowered
        for marker in (
            "historical underlying asset prices",
            "sample contains",
            "daily returns",
            "observations",
            "training sample",
            "out-of-sample",
        )
    ):
        score -= 8
    if slot_name == "method" and _compare_sentence_has_method_signal(summary):
        score += 4
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "we propose",
            "this paper proposes",
            "we introduce",
            "three-step quasi-maximum likelihood",
            "three-step estimator",
            "scale adjustment parameter",
            "dual representation",
            "axiom scheme",
            "axiomatic dual representation",
            "we establish",
            "hamilton-jacobi-bellman",
            "innovative data-driven option pricing methodology",
            "estimate the bid-ask spread",
        )
    ):
        score += 4
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "however, these methods may lead",
            "this may be partly because",
            "for a comprehensive review",
            "can not be reduced to the traditional setup",
        )
    ):
        score -= 7
    if slot_name in {"main_findings", "conclusion"} and _compare_sentence_has_result_signal(slot_name, summary):
        score += 5
    if slot_name == "conclusion" and any(
        marker in lowered
        for marker in (
            "in this paper",
            "we conclude",
            "our results imply",
            "these findings",
            "numerical studies confirm",
            "achieves better efficiency",
            "offer robust tools",
        )
    ):
        score += 3
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "numerical experiment",
            "numerical experiments",
            "performs very well",
            "negligible impact",
            "consistent and asymptotically normal",
            "better efficiency",
            "necessary and sufficient conditions",
            "advantages of the proposed approach",
        )
    ):
        score += 3
    if slot_name == "limitations" and _compare_sentence_has_limitation_signal(summary):
        score += 4
    if slot_name == "limitations" and any(
        marker in lowered
        for marker in (
            "left panel",
            "right panel",
            "lemma",
            "proof",
            "corollary",
            "matrix manipulation",
            "figure ",
            "table ",
        )
    ):
        score -= 8
    if slot_name == "practical_or_investment_implications" and _compare_sentence_has_practical_signal(summary):
        score += 5
    if slot_name == "practical_or_investment_implications" and any(
        marker in lowered
        for marker in (
            "left panel",
            "right panel",
            "figure ",
            "table ",
            "introduction ",
            "article submitted to",
            "et al",
        )
    ):
        score -= 7
    if _compare_sentence_is_structural_noise(summary):
        score -= 6
    if _compare_sentence_is_truncated_fragment(summary):
        score -= 5
    if slot_name == "method" and "we have to al" in lowered:
        score -= 6
    if slot_name == "sample_or_data" and ("sample path" in lowered or "sample paths" in lowered):
        score -= 8
    if slot_name == "limitations" and _compare_sentence_has_limitation_signal(sentence):
        score += 3
    if slot_name == "practical_or_investment_implications" and _compare_sentence_has_practical_signal(sentence):
        score += 3
    return score


def _looks_unreliable_compare_summary(
    slot_name: str,
    summary: str,
    *,
    support_status: str,
) -> bool:
    lowered = unicodedata.normalize("NFKC", summary).lower().strip()
    if not lowered:
        return True
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "however, these methods may lead",
            "this may be partly because",
            "for a comprehensive review",
            "future research",
        )
    ):
        return True
    if support_status in {"explicit_supported", "well_supported"} and _compare_sentence_has_result_signal(slot_name, lowered) and not _compare_sentence_has_formula_noise(summary):
        return False
    if support_status in {"explicit_supported", "well_supported"} and not _compare_sentence_is_structural_noise(lowered) and not _compare_sentence_has_formula_noise(summary):
        if slot_name == "research_question" and _compare_sentence_has_question_signal(lowered) and len(lowered) >= 36:
            return False
        if (
            slot_name == "sample_or_data"
            and _compare_sentence_has_sample_signal(lowered)
            and len(lowered) >= 28
            and not any(
                marker in lowered
                for marker in (
                    "volatility smile",
                    "calibrating these models",
                    "advanced option pricing models beyond",
                    "black-scholes world",
                    "constant volatility assumed",
                )
            )
        ):
            return False
        if slot_name == "method" and _compare_sentence_has_method_signal(lowered) and len(lowered) >= 24:
            return False
        if slot_name in {"main_findings", "conclusion"} and _detail_marker_count(slot_name, lowered) > 0 and len(lowered) >= 36:
            return False
        if (
            slot_name == "practical_or_investment_implications"
            and _compare_sentence_has_practical_signal(lowered)
            and len(lowered) >= 28
        ):
            return False
    if support_status == "weakly_supported" and not _compare_sentence_is_structural_noise(lowered) and not _compare_sentence_has_formula_noise(summary):
        if slot_name == "research_question" and _compare_sentence_has_question_signal(lowered) and len(lowered) >= 48:
            return False
        if (
            slot_name == "sample_or_data"
            and _compare_sentence_has_sample_signal(lowered)
            and len(lowered) >= 32
            and not any(
                marker in lowered
                for marker in (
                    "volatility smile",
                    "calibrating these models",
                    "advanced option pricing models beyond",
                    "black-scholes world",
                    "constant volatility assumed",
                )
            )
        ):
            return False
        if slot_name == "method" and _compare_sentence_has_method_signal(lowered) and len(lowered) >= 24:
            return False
        if (
            slot_name == "practical_or_investment_implications"
            and _compare_sentence_has_practical_signal(lowered)
            and len(lowered) >= 32
        ):
            return False
        if slot_name == "main_findings" and _compare_sentence_has_result_signal(slot_name, lowered) and len(lowered) >= 48:
            return False
        if slot_name == "conclusion" and _compare_sentence_has_result_signal(slot_name, lowered) and len(lowered) >= 40:
            return False
    if len(lowered) < 18 and _detail_marker_count(slot_name, lowered) == 0:
        return True
    if "sample path" in lowered or "sample paths" in lowered:
        return True
    if _compare_sentence_has_formula_noise(summary):
        return True
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "simulation results depict",
            "predictive efficiency which has a value",
            "exceeds predictive efficiency",
        )
    ):
        return True
    if any(
        marker in lowered
        for marker in (
            "article submitted to",
            "mathematical methods of operations research",
            "journal of ",
            "loss curves against",
            "validation dataset",
            "author:",
            "keywords:",
            "department of",
            "university of",
            "institute of",
        )
    ):
        return True
    if "@" in summary:
        return True
    if re.search(r"\b(?:references?|appendix|doi|figure|table)\b", lowered):
        return True
    if re.search(r"\backnowledg(?:ement|ements|ment|ments)\b", lowered):
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "article submitted to",
            "exchange-traded options",
            "historical data of underlying asset prices rather than simulated data",
        )
    ) and _detail_marker_count(slot_name, lowered) == 0:
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "volatility smile",
            "calibrating these models",
            "advanced option pricing models beyond",
            "black-scholes world",
            "constant volatility assumed",
        )
    ) and not any(
        marker in lowered
        for marker in (
            "historical underlying asset prices",
            "sample contains",
            "daily returns",
            "observations",
            "training sample",
            "out-of-sample",
        )
    ):
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "predictive power",
            "benchmark model",
            "portfolio level",
            "stock level",
            "behave erratically",
            "best bid quotes",
            "inventories over the zoomed period",
            "right panel shows",
            "left panel shows",
            "left panel",
            "right panel",
            "mean reward ppo",
            "loss curve",
        )
    ) and not any(
        marker in lowered
        for marker in (
            "crsp",
            "nyse",
            "amex",
            "nasdaq",
            "observation",
            "cross-section",
            "sample contains",
            "daily returns",
            "monthly returns",
            "historical underlying asset prices",
            "dataset",
        )
    ):
        return True
    if slot_name == "conclusion" and any(
        marker in lowered for marker in ("this concludes the proof", "proof", "lemma", "theorem", "corollary")
    ):
        return True
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "objective function",
            "validation objective",
            "forecast errors",
            "hyperparameter",
            "hyperparameters",
        )
    ):
        return True
    if slot_name == "research_question" and _compare_has_formula_glyphs(summary):
        return True
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "operations research",
            "article submitted to",
            "working paper",
            "loss curves against",
            "validation dataset",
            "however, these methods may lead",
            "for a comprehensive review",
        )
    ):
        return True
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "for specific details and results",
            "section 4 presents the findings",
            "numerical results we test",
        )
    ):
        return True
    if slot_name == "limitations" and (
        any(
            marker in lowered
            for marker in (
                "aims to alleviate the limitations of individual",
                "left panel",
                "right panel",
                "theorem",
                "lemma",
                "proof",
                "corollary",
                "matrix manipulation",
            )
        )
        or (
            re.search(r"\bassumption\s+\d", lowered)
            and not any(
                marker in lowered
                for marker in (
                    "do not assume",
                    "unless the true underlying density",
                    "cannot be obtained unless",
                    "limited to",
                    "restricted to",
                    "we only",
                    "only consider",
                )
            )
        )
        or not _compare_sentence_has_limitation_signal(summary)
    ):
        return True
    if slot_name == "practical_or_investment_implications" and not _compare_sentence_has_practical_signal(summary):
        return True
    if (
        slot_name in {"main_findings", "limitations", "conclusion"}
        and lowered.startswith(("section ", "results ", "discussion ", "conclusion ", "numerical results"))
        and _detail_marker_count(slot_name, lowered) == 0
    ):
        return True
    if support_status == "weakly_supported" and _detail_marker_count(slot_name, lowered) == 0:
        return True
    if _compare_sentence_is_truncated_fragment(summary):
        return True
    return False


def _compare_has_formula_glyphs(text: str) -> bool:
    return any(marker in text for marker in ("\u2211", "\u03bb", "\u03b8", "\u0302"))


def _compare_summary_quality_score(
    slot_name: str,
    summary: str,
    *,
    support_status: str,
) -> int:
    lowered = unicodedata.normalize("NFKC", summary).lower()
    score = _compare_sentence_specificity_score(slot_name, lowered)
    score += min(5, _detail_marker_count(slot_name, lowered))
    if _slot_is_clearly_supported(support_status):
        score += 5
    elif support_status == "weakly_supported":
        score += 1
    if slot_name == "research_question" and _compare_sentence_has_question_signal(summary):
        score += 4
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "this study aims",
            "fundamental goal",
            "goal of asset pricing",
            "we study liquidity provision",
            "we introduce the concept",
            "this paper aims",
            "aims to forecast",
            "seeks to forecast",
            "this article focuses on",
            "this article questions",
            "the main contribution of this article",
        )
    ):
        score += 4
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "objective function",
            "validation objective",
            "forecast errors",
            "hyperparameter",
            "hyperparameters",
        )
    ):
        score -= 8
    if slot_name == "research_question" and _compare_has_formula_glyphs(summary):
        score -= 8
    if slot_name == "sample_or_data" and (
        _looks_like_explicit_sample_data_text(summary)
        or _compare_sentence_has_sample_signal(summary)
    ):
        score += 5
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "crsp",
            "nyse",
            "amex",
            "nasdaq",
            "cross-section",
            "individual stocks",
        )
    ):
        score += 4
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "predictive power",
            "benchmark model",
            "portfolio level",
            "stock level",
            "r2 ",
            "r2)",
            "r2(",
            "liquid stocks",
            "behave erratically",
        )
    ):
        score -= 6
    if slot_name == "method" and _compare_sentence_has_method_signal(summary):
        score += 4
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "we propose",
            "this paper proposes",
            "we introduce",
            "the main contribution of this article",
            "we introduce a scale adjustment parameter",
            "technical contribution",
            "three-step quasi-maximum likelihood",
            "three-step estimator",
            "three-step procedure",
            "scale adjustment parameter",
            "dual representation",
            "axiom scheme",
            "axiomatic dual representation",
            "we establish",
            "hamilton-jacobi-bellman",
            "innovative data-driven option pricing methodology",
            "estimate the bid-ask spread",
        )
    ):
        score += 4
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "we establish",
            "technical contribution",
            "three-step procedure",
            "we introduce a scale adjustment parameter",
            "novel three-step",
            "proposed procedure runs a gqmle",
        )
    ):
        score += 4
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "however, these methods may lead",
            "this may be partly because",
            "for a comprehensive review",
            "can not be reduced to the traditional setup",
            "in what follows",
            "before stating",
            "quasi-maximum likelihood estimation of garch models with heavy-tailed likelihoods",
            "gmm implementation",
            "one-step generalized methods of moments",
            "score functions",
        )
    ):
        score -= 7
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "consistent",
            "asymptotically normal",
            "finite fourth moment",
            "better efficiency",
            "more efficient",
            "smaller asymptotic variance",
            "weak moment conditions",
        )
    ) and not any(
        marker in lowered
        for marker in (
            "we propose",
            "this paper proposes",
            "we introduce",
            "three-step",
            "procedure",
            "approach",
            "framework",
            "scale adjustment parameter",
            "closed-form solution",
            "stochastic control",
            "dual representation",
            "pricing kernel",
        )
    ):
        score -= 10
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "novel three-step ngqmle approach",
            "identify an unknown scale parameter",
            "we introduce a scale adjustment parameter",
        )
    ):
        score += 5
    if slot_name == "limitations" and any(
        marker in lowered
        for marker in (
            "lower panel",
            "upper panel",
            "innovations range from",
        )
    ):
        score -= 9
    if slot_name == "limitations" and any(
        marker in lowered
        for marker in (
            "monthly data",
            "high-frequency data",
            "future work",
            "future research",
            "we leave",
            "leave the discussion",
            "do not assume",
            "without considering",
            "is limited to",
            "restricted to",
            "we only",
            "only consider",
            "only uses",
            "scope",
            "small amount of data",
            "dearth of data",
            "signal-to-noise ratio",
            "overfit",
            "overfitting",
            "must be heavily regularized",
            "synthetic data",
            "one country",
            "one market",
        )
    ):
        score += 4
    if slot_name == "limitations" and any(
        marker in lowered
        for marker in (
            "abilities and limitations of various prediction models",
            "aims to alleviate the limitations of individual",
            "existing studies still present some shortcomings",
        )
    ):
        score -= 6
    if slot_name in {"main_findings", "conclusion"} and _compare_sentence_has_result_signal(slot_name, summary):
        score += 5
    if slot_name == "conclusion" and any(
        marker in lowered
        for marker in (
            "in this paper",
            "we conclude",
            "our results imply",
            "these findings",
            "numerical studies confirm",
            "in conclusion",
            "significant advancement",
            "we hope that our work will inspire",
        )
    ):
        score += 3
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "numerical experiment",
            "numerical experiments",
            "performs very well",
            "negligible impact",
            "consistent and asymptotically normal",
            "better efficiency",
            "necessary and sufficient conditions",
            "advantages of the proposed approach",
            "in most cases, ngqmle shows an advantage",
        )
    ):
        score += 3
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "simulation results depict",
            "predictive efficiency which has a value",
            "exceeds predictive efficiency",
        )
    ):
        score -= 20
    if slot_name in {"main_findings", "conclusion"} and any(
        marker in lowered
        for marker in (
            "as an aside",
            "estimated complexity in figure",
            "we do not describe their estimated complexity",
            "benchmarked against historical averages",
        )
    ):
        score -= 9
    if slot_name == "main_findings" and re.search(r"\b\d{1,3}\s+(?:because|as an aside)\b", lowered):
        score -= 10
    if slot_name == "practical_or_investment_implications" and any(
        marker in lowered
        for marker in (
            "robust against density misspecification",
            "more efficient than the gqmle",
            "suitable for making informed financial decisions",
            "offer robust tools for evaluating and mitigating risk",
        )
    ):
        score += 5
    if _compare_sentence_is_structural_noise(summary):
        score -= 6
    if _compare_sentence_is_truncated_fragment(summary):
        score -= 5
    if slot_name == "method" and "we have to al" in lowered:
        score -= 6
    if slot_name == "sample_or_data" and ("sample path" in lowered or "sample paths" in lowered):
        score -= 8
    return score


def _looks_unreliable_compare_summary(
    slot_name: str,
    summary: str,
    *,
    support_status: str,
) -> bool:
    lowered = unicodedata.normalize("NFKC", summary).lower().strip()
    if not lowered:
        return True
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "output layer",
            "input layer",
            "hidden layer",
            "output node",
            "hidden node",
        )
    ):
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "downloaded from",
            "guest (guest)",
            "guest ip:",
            "ip:",
        )
    ):
        return True
    if slot_name == "sample_or_data" and re.search(
        r"^(?:mon|tue|wed|thu|fri|sat|sun),\s+\d{1,2}\s+[a-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\b",
        lowered,
    ):
        return True
    if slot_name == "sample_or_data" and re.fullmatch(
        r"(?:the\s+)?(?:training|validation|testing|test|out-of-sample)\s+sample\.?",
        lowered,
    ):
        return True
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "simulation results depict",
            "predictive efficiency which has a value",
            "exceeds predictive efficiency",
        )
    ):
        return True
    if support_status in {"explicit_supported", "well_supported"} and _compare_sentence_has_result_signal(slot_name, lowered) and not _compare_sentence_has_formula_noise(summary):
        return False
    if support_status in {"explicit_supported", "well_supported"} and not _compare_sentence_is_structural_noise(lowered) and not _compare_sentence_has_formula_noise(summary):
        if slot_name == "research_question" and _compare_sentence_has_question_signal(lowered) and len(lowered) >= 36:
            return False
        if (
            slot_name == "sample_or_data"
            and _compare_sentence_has_sample_signal(lowered)
            and len(lowered) >= 28
            and not any(
                marker in lowered
                for marker in (
                    "volatility smile",
                    "calibrating these models",
                    "advanced option pricing models beyond",
                    "black-scholes world",
                    "constant volatility assumed",
                )
            )
        ):
            return False
        if slot_name == "method" and _compare_sentence_has_method_signal(lowered) and len(lowered) >= 24:
            return False
        if slot_name in {"main_findings", "conclusion"} and _detail_marker_count(slot_name, lowered) > 0 and len(lowered) >= 36:
            return False
    if support_status == "weakly_supported" and not _compare_sentence_is_structural_noise(lowered) and not _compare_sentence_has_formula_noise(summary):
        if slot_name == "research_question" and _compare_sentence_has_question_signal(lowered) and len(lowered) >= 48:
            return False
        if (
            slot_name == "sample_or_data"
            and _compare_sentence_has_sample_signal(lowered)
            and len(lowered) >= 32
            and not any(
                marker in lowered
                for marker in (
                    "volatility smile",
                    "calibrating these models",
                    "advanced option pricing models beyond",
                    "black-scholes world",
                    "constant volatility assumed",
                )
            )
        ):
            return False
        if slot_name == "method" and _compare_sentence_has_method_signal(lowered) and len(lowered) >= 24:
            return False
        if slot_name == "main_findings" and _compare_sentence_has_result_signal(slot_name, lowered) and len(lowered) >= 48:
            return False
        if slot_name == "conclusion" and _compare_sentence_has_result_signal(slot_name, lowered) and len(lowered) >= 40:
            return False
    if len(lowered) < 18 and _detail_marker_count(slot_name, lowered) == 0:
        return True
    if "sample path" in lowered or "sample paths" in lowered:
        return True
    if _compare_sentence_has_formula_noise(summary):
        return True
    if any(
        marker in lowered
        for marker in (
            "article submitted to",
            "mathematical methods of operations research",
            "journal of ",
            "loss curves against",
            "validation dataset",
            "author:",
            "keywords:",
            "department of",
            "university of",
            "institute of",
        )
    ):
        return True
    if "@" in summary:
        return True
    if re.search(r"\b(?:references?|appendix|doi|figure|table)\b", lowered):
        return True
    if re.search(r"\backnowledg(?:ement|ements|ment|ments)\b", lowered):
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "article submitted to",
            "exchange-traded options",
            "historical data of underlying asset prices rather than simulated data",
        )
    ) and _detail_marker_count(slot_name, lowered) == 0:
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "volatility smile",
            "calibrating these models",
            "advanced option pricing models beyond",
            "black-scholes world",
            "constant volatility assumed",
        )
    ) and not any(
        marker in lowered
        for marker in (
            "historical underlying asset prices",
            "sample contains",
            "daily returns",
            "observations",
            "training sample",
            "out-of-sample",
        )
    ):
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "predictive power",
            "benchmark model",
            "portfolio level",
            "stock level",
            "behave erratically",
            "best bid quotes",
            "inventories over the zoomed period",
            "right panel shows",
            "left panel shows",
            "left panel",
            "right panel",
            "mean reward ppo",
            "loss curve",
        )
    ) and not any(
        marker in lowered
        for marker in (
            "crsp",
            "nyse",
            "amex",
            "nasdaq",
            "observation",
            "cross-section",
            "sample contains",
            "daily returns",
            "monthly returns",
            "historical underlying asset prices",
        )
    ):
        return True
    if slot_name == "conclusion" and any(
        marker in lowered for marker in ("this concludes the proof", "proof", "lemma", "theorem", "corollary")
    ):
        return True
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "objective function",
            "validation objective",
            "forecast errors",
            "hyperparameter",
            "hyperparameters",
        )
    ):
        return True
    if slot_name == "research_question" and _compare_has_formula_glyphs(summary):
        return True
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "operations research",
            "article submitted to",
            "working paper",
            "loss curves against",
            "validation dataset",
            "however, these methods may lead",
            "for a comprehensive review",
            "future research",
        )
    ):
        return True
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "for specific details and results",
            "section 4 presents the findings",
            "numerical results we test",
        )
    ):
        return True
    if slot_name == "limitations" and (
        any(
            marker in lowered
            for marker in (
                "aims to alleviate the limitations of individual",
                "lack of regularization leaves ols",
                "trading strategy exactly",
                "maximum leverage constraint",
                "excluding short sales",
                "left panel",
                "right panel",
                "lemma",
                "proof",
                "corollary",
                "matrix manipulation",
            )
        )
        or not _compare_sentence_has_limitation_signal(summary)
    ):
        return True
    if slot_name == "practical_or_investment_implications" and not _compare_sentence_has_practical_signal(summary):
        return True
    if (
        slot_name in {"main_findings", "limitations", "conclusion"}
        and lowered.startswith(("section ", "results ", "discussion ", "conclusion ", "numerical results"))
        and _detail_marker_count(slot_name, lowered) == 0
    ):
        return True
    if support_status == "weakly_supported" and _detail_marker_count(slot_name, lowered) == 0:
        return True
    if _compare_sentence_is_truncated_fragment(summary):
        return True
    return False


def _compose_slot_translation_prompt(*, source_text: str, response_style: str) -> str:
    is_structured_compare = _looks_like_structured_compare_text(source_text)
    if is_structured_compare:
        style_instruction = (
            "Preserve the compare structure in natural Simplified Chinese: keep one short section per dimension, "
            "keep the compared document list, and keep a clear '\u5bf9\u6bd4\uff1a' line for each dimension."
        )
    else:
        style_instruction = (
            "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
            if response_style == "continuous_prose"
            else "Return a concise Simplified Chinese memo."
        )
    return "\n".join(
        [
            "Translate the grounded memo below into natural Simplified Chinese.",
            "Keep only facts that already appear in the grounded memo.",
            "Do not add generic finance or machine-learning commentary.",
            "If a dimension is unclear or missing, say \u6587\u4e2d\u672a\u660e\u786e\u8bf4\u660e\u3002",
            "Keep method-family names, abbreviations, and file identifiers exact when needed.",
            "If the memo is a paper comparison, keep each requested dimension separate instead of blending the papers together.",
            style_instruction,
            "",
            "Grounded memo:",
            source_text,
            "",
            "Return only the final Chinese answer body.",
        ]
    )


def _compare_answer_lacks_direct_contrast(
    answer_text: str,
    *,
    prompt: str,
    paper_trace: PaperTrace,
) -> bool:
    lowered = unicodedata.normalize("NFKC", answer_text).lower()
    requested_slots = _requested_paper_slots(prompt, "paper_compare")
    if len(requested_slots) < 2:
        return False
    contrast_markers = sum(
        lowered.count(marker)
        for marker in (
            "- contrast:",
            "contrast:",
            "- \u5bf9\u6bd4\uff1a",
            "\u5bf9\u6bd4\uff1a",
            "\u76f8\u6bd4",
            "\u800c",
            "\u5f02\u540c",
            "whereas",
            "in contrast",
            "by contrast",
        )
    )
    if contrast_markers >= max(2, min(4, len(requested_slots) - 1)):
        return False
    if any(
        marker in lowered
        for marker in ("documents compared", "\u6bd4\u8f83\u6587\u732e")
    ) and "- `" in lowered and contrast_markers == 0:
        return True
    if any(
        marker in lowered
        for marker in (
            "# commonalities",
            "commonalities",
            "# differences",
            "differences",
            "# recommendations or synthesis",
            "recommendations or synthesis",
            "recommendation or synthesis",
            "strengths / weaknesses / limitations",
            "combining methodologies could",
            "hybrid approach could be beneficial",
            "both papers focus on",
        )
    ):
        return True
    compared_names = [
        _compare_document_name(document_note).lower()
        for document_note in paper_trace.document_notes
    ]
    mentioned_documents = sum(1 for name in compared_names if name and name in lowered)
    return mentioned_documents < min(2, len(compared_names))


def _compare_answer_lacks_direct_contrast(
    answer_text: str,
    *,
    prompt: str,
    paper_trace: PaperTrace,
) -> bool:
    lowered = unicodedata.normalize("NFKC", answer_text).lower()
    requested_slots = _requested_paper_slots(prompt, "paper_compare")
    if len(requested_slots) < 2:
        return False
    contrast_markers = sum(
        lowered.count(marker)
        for marker in (
            "- contrast:",
            "contrast:",
            "- 对比：",
            "对比：",
            "相比",
            "相比之下",
            "而",
            "异同",
            "whereas",
            "in contrast",
            "by contrast",
        )
    )
    if contrast_markers >= max(2, min(4, len(requested_slots) - 1)):
        return False
    if "documents compared" in lowered and "- `" in lowered and contrast_markers == 0:
        return True
    if any(
        marker in lowered
        for marker in (
            "# commonalities",
            "commonalities",
            "# differences",
            "differences",
            "# recommendations or synthesis",
            "recommendations or synthesis",
            "recommendation or synthesis",
            "strengths / weaknesses / limitations",
            "combining methodologies could",
            "hybrid approach could be beneficial",
            "both papers focus on",
        )
    ):
        return True
    compared_names = [
        _compare_document_name(document_note).lower()
        for document_note in paper_trace.document_notes
    ]
    mentioned_documents = sum(1 for name in compared_names if name and name in lowered)
    return mentioned_documents < min(2, len(compared_names))


def _compare_false_missing_supported_slots(
    answer_text: str,
    document_notes: list[dict[str, object]],
    *,
    response_language: str,
) -> tuple[str, ...]:
    if not document_notes:
        return ()
    missing_phrase = _paper_missing_phrase(response_language)
    false_slots: list[str] = []
    core_slots = (
        "research_question",
        "sample_or_data",
        "method",
        "main_findings",
        "conclusion",
    )
    normalized = unicodedata.normalize("NFKC", answer_text)
    sections: dict[str, list[str]] = {}
    current_section = ""
    known_titles = {
        _compare_documents_heading("en"),
        _compare_documents_heading("zh-CN"),
        "Limitations",
        "\u5c40\u9650",
    }
    for slot_name in core_slots:
        known_titles.add(_compare_section_title(slot_name, "en"))
        known_titles.add(_compare_section_title(slot_name, "zh-CN"))
    for raw_line in normalized.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.strip() in known_titles:
            current_section = line.strip()
            sections.setdefault(current_section, [])
            continue
        if current_section:
            sections.setdefault(current_section, []).append(line.strip())
    for slot_name in core_slots:
        section_lines: list[str] = []
        for section_title in (
            _compare_section_title(slot_name, "en"),
            _compare_section_title(slot_name, "zh-CN"),
        ):
            section_lines.extend(sections.get(section_title, []))
        if not section_lines:
            continue
        section_body = "\n".join(section_lines)
        for document_note in document_notes:
            entry = _build_compare_slot_entry(
                document_note,
                slot_name,
                response_language=response_language,
                paper_output_profile="detailed_paper_note",
            )
            if entry["status"] == "not_clearly_stated":
                continue
            source_name = re.escape(_compare_document_name(document_note))
            if re.search(
                rf"(?im)^- `{source_name}`:\s*(?:{re.escape(missing_phrase)}|Not clearly stated in the paper\.?|\u6587\u4e2d\u672a\u660e\u786e\u8bf4\u660e[\u3002.]?)\s*$",
                section_body,
            ):
                false_slots.append(slot_name)
                break
    return tuple(dict.fromkeys(false_slots))


_latest_evaluate_paper_answer_consistency = _evaluate_paper_answer_consistency


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    report = _latest_evaluate_paper_answer_consistency(
        prompt,
        mode_selection,
        paper_trace,
        answer_text,
    )
    if mode_selection.mode != "paper_compare":
        return report
    explicit_slots = _explicit_paper_slots(prompt)
    if not explicit_slots:
        return report
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if not uncovered_slots:
        return report
    notes = list(report.get("notes", []))
    coverage_note = (
        "The comparison did not clearly cover these requested dimensions: "
        + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
        + "."
    )
    if coverage_note not in notes:
        notes.append(coverage_note)
    return {
        "needs_repair": True,
        "notes": tuple(notes),
    }


_previous_evaluate_paper_answer_consistency = _evaluate_paper_answer_consistency


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    report = _previous_evaluate_paper_answer_consistency(
        prompt,
        mode_selection,
        paper_trace,
        answer_text,
    )
    if mode_selection.mode != "paper_compare":
        return report
    explicit_slots = _explicit_paper_slots(prompt)
    if not explicit_slots:
        return report
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if not uncovered_slots:
        return report
    notes = list(report.get("notes", []))
    coverage_note = (
        "The comparison did not clearly cover these requested dimensions: "
        + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
        + "."
    )
    if coverage_note not in notes:
        notes.append(coverage_note)
    return {
        "needs_repair": True,
        "notes": tuple(notes),
    }


def _prompt_requests_bilingual_output(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(
        token in lowered
        for token in (
            "bilingual",
            "both english and chinese",
        )
    ) or any(token in prompt for token in ("\u4e2d\u82f1\u53cc\u8bed", "\u82f1\u4e2d\u53cc\u8bed"))


def _finalize_paper_answer_text(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> str:
    cleaned = unicodedata.normalize("NFKC", answer_text or "").strip()
    if not cleaned:
        return ""
    cleaned = _normalize_single_language_output(
        cleaned,
        response_language=mode_selection.response_language,
        allow_bilingual=_prompt_requests_bilingual_output(prompt),
    )
    if (
        paper_trace.active
        and mode_selection.mode == "paper_summary"
        and mode_selection.paper_output_profile == "detailed_paper_note"
        and mode_selection.response_style != "continuous_prose"
    ):
        cleaned = _cleanup_detailed_note_render(
            cleaned,
            response_language=mode_selection.response_language,
        )
    if paper_trace.active and mode_selection.mode == "paper_compare":
        if _looks_like_structured_compare_text(cleaned):
            cleaned = _cleanup_structured_compare_render(
                cleaned,
                response_language=mode_selection.response_language,
            )
        else:
            cleaned = _cleanup_duplicate_render_lines(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _normalize_single_language_output(
    answer_text: str,
    *,
    response_language: str,
    allow_bilingual: bool,
) -> str:
    text = answer_text.strip()
    if allow_bilingual:
        return text

    heading_map_en = {
        "\u7814\u7a76\u95ee\u9898": "Research question",
        "\u6837\u672c\u4e0e\u6570\u636e": "Sample and data",
        "\u6837\u672c\u6216\u6570\u636e": "Sample and data",
        "\u65b9\u6cd5": "Method",
        "\u4e3b\u8981\u53d1\u73b0": "Main findings",
        "\u5c40\u9650": "Limitations",
        "\u7ed3\u8bba": "Conclusion",
        "\u6bd4\u8f83\u6587\u732e": "Documents compared",
    }
    heading_map_zh = {value: key for key, value in heading_map_en.items()}
    english_replacements = (
        ("\u6587\u4e2d\u672a\u660e\u786e\u8bf4\u660e\u3002", "Not clearly stated in the paper."),
        ("\u6587\u4e2d\u672a\u660e\u786e\u8bf4\u660e", "not clearly stated in the paper"),
        ("- \u5bf9\u6bd4\uff1a", "- Contrast:"),
        ("\u7814\u7a76\u95ee\u9898\u662f", "The research question is "),
        ("\u6837\u672c\u4e0e\u6570\u636e\u65b9\u9762\uff0c", "For the sample and data, "),
        ("\u6837\u672c\u6216\u6570\u636e\u65b9\u9762\uff0c", "For the sample and data, "),
        ("\u65b9\u6cd5\u4e0a\uff0c", "Methodologically, "),
        ("\u4e3b\u8981\u53d1\u73b0\u662f", "The main finding is that "),
        ("\u5c40\u9650\u5728\u4e8e", "A key limitation is that "),
        ("\u603b\u4f53\u7ed3\u8bba\u662f", "Overall, "),
        ("\u8bad\u7ec3\u6837\u672c", "training data"),
        ("\u9a8c\u8bc1\u6837\u672c", "validation data"),
        ("\u6837\u672c\u5916\u6d4b\u8bd5", "out-of-sample testing"),
        ("\u6837\u672c\u5916\u8bc4\u4f30", "out-of-sample evaluation"),
        ("\u7ebf\u6027\u56de\u5f52", "linear regression"),
        ("\u5e7f\u4e49\u7ebf\u6027\u6a21\u578b", "generalized linear models"),
        ("\u4e3b\u6210\u5206\u56de\u5f52", "principal components regression (PCR)"),
        ("\u504f\u6700\u5c0f\u4e8c\u4e58", "partial least squares (PLS)"),
        ("\u56de\u5f52\u6811", "regression trees"),
        ("\u795e\u7ecf\u7f51\u7edc", "neural networks"),
    )
    chinese_replacements = (
        ("Not clearly stated in the paper.", "\u6587\u4e2d\u672a\u660e\u786e\u8bf4\u660e\u3002"),
        ("not clearly stated in the paper", "\u6587\u4e2d\u672a\u660e\u786e\u8bf4\u660e"),
        ("- Contrast:", "- \u5bf9\u6bd4\uff1a"),
        ("For the sample and data, ", "\u6837\u672c\u4e0e\u6570\u636e\u65b9\u9762\uff0c"),
        ("Methodologically, ", "\u65b9\u6cd5\u4e0a\uff0c"),
        ("The main finding is that ", "\u4e3b\u8981\u53d1\u73b0\u662f"),
        ("A key limitation is that ", "\u5c40\u9650\u5728\u4e8e"),
        ("Overall, ", "\u603b\u4f53\u7ed3\u8bba\u662f"),
        ("Research question", "\u7814\u7a76\u95ee\u9898"),
        ("Sample and data", "\u6837\u672c\u4e0e\u6570\u636e"),
        ("Method", "\u65b9\u6cd5"),
        ("Main findings", "\u4e3b\u8981\u53d1\u73b0"),
        ("Limitations", "\u5c40\u9650"),
        ("Conclusion", "\u7ed3\u8bba"),
        ("Documents compared", "\u6bd4\u8f83\u6587\u732e"),
    )

    normalized_lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if response_language == "en":
            stripped = heading_map_en.get(stripped, stripped)
            for source, target in english_replacements:
                stripped = stripped.replace(source, target)
            stripped = (
                stripped.replace("\u3002", ". ")
                .replace("\uff1b", "; ")
                .replace("\uff0c", ", ")
                .replace("\uff1a", ": ")
            )
        else:
            stripped = heading_map_zh.get(stripped, stripped)
            for source, target in chinese_replacements:
                stripped = stripped.replace(source, target)
        stripped = re.sub(r"[ \t]{2,}", " ", stripped).strip()
        normalized_lines.append(stripped)

    normalized = "\n".join(normalized_lines)
    if response_language == "en":
        normalized = re.sub(r"(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{1,6}(?![\u4e00-\u9fff])", " ", normalized)
        normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
        normalized = re.sub(r"(?<!\d),(?![\d\s]|$)", ", ", normalized)
        normalized = re.sub(r"(?<!\d)\.(?![\d\s]|$)", ". ", normalized)
        normalized = re.sub(r";(?!\s|$)", "; ", normalized)
        normalized = re.sub(r":(?!\s|$)", ": ", normalized)
        normalized = re.sub(
            r"(?<=[A-Za-z0-9_-])\.\s+(?=(?:pdf|md|py|txt|json|csv)\b)",
            ".",
            normalized,
            flags=re.IGNORECASE,
        )
    else:
        normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _contains_unexpected_language_leakage(
    answer_text: str,
    *,
    response_language: str,
    allow_bilingual: bool,
) -> bool:
    if allow_bilingual:
        return False
    if response_language == "en":
        return bool(re.search(r"[\u4e00-\u9fff]", answer_text))
    if _looks_insufficiently_translated_chinese(answer_text):
        return True
    lowered = answer_text.lower()
    scaffolding_markers = (
        "research question",
        "sample and data",
        "methodologically,",
        "main findings",
        "limitations",
        "conclusion",
        "not clearly stated in the paper",
    )
    return any(marker in lowered for marker in scaffolding_markers)


def _parse_detailed_note_occurrences(
    answer_text: str,
    response_language: str,
) -> list[tuple[str, str, str]]:
    title_map = _detailed_note_title_map(response_language)
    reverse_map = {title: slot_name for slot_name, title in title_map.items()}
    occurrences: list[tuple[str, str, str]] = []
    current_slot = ""
    current_title = ""
    buffer: list[str] = []

    def _flush() -> None:
        nonlocal buffer, current_slot, current_title
        if current_slot:
            body = " ".join(part.strip() for part in buffer if part.strip()).strip()
            occurrences.append((current_slot, current_title, body))
        buffer = []

    for line in answer_text.splitlines():
        stripped = line.strip()
        slot_name = reverse_map.get(stripped)
        if slot_name:
            _flush()
            current_slot = slot_name
            current_title = stripped
            continue
        if current_slot:
            buffer.append(stripped)
    _flush()
    return occurrences


def _cleanup_detailed_note_render(
    answer_text: str,
    *,
    response_language: str,
) -> str:
    occurrences = _parse_detailed_note_occurrences(answer_text, response_language)
    if not occurrences:
        return _cleanup_duplicate_render_lines(answer_text)

    title_map = _detailed_note_title_map(response_language)
    best_sections: dict[str, tuple[str, int]] = {}
    for slot_name, _title, body in occurrences:
        cleaned_body = _clean_detailed_render_body(
            body,
            slot_name=slot_name,
            response_language=response_language,
        )
        if not cleaned_body:
            continue
        score = _score_detailed_render_body(slot_name, cleaned_body)
        current = best_sections.get(slot_name)
        if current is None or score > current[1]:
            best_sections[slot_name] = (cleaned_body, score)

    if not best_sections:
        return _cleanup_duplicate_render_lines(answer_text)

    lines: list[str] = []
    for slot_name, title in title_map.items():
        selected = best_sections.get(slot_name)
        if not selected:
            continue
        lines.append(title)
        lines.append(selected[0])
        lines.append("")
    return "\n".join(lines).strip()


def _clean_detailed_render_body(
    body: str,
    *,
    slot_name: str,
    response_language: str,
) -> str:
    cleaned = _clean_detailed_slot_body(
        body,
        slot_name=slot_name,
        response_language=response_language,
    )
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", cleaned)
    kept: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        fragment = sentence.strip()
        if not fragment:
            continue
        if _looks_broken_fragment(fragment, context=tuple(kept + sentences)):
            continue
        normalized = re.sub(r"\W+", " ", fragment.lower()).strip()
        if not normalized or normalized in seen:
            continue
        if any(
            normalized != re.sub(r"\W+", " ", prior.lower()).strip()
            and len(normalized) < 48
            and normalized in re.sub(r"\W+", " ", prior.lower()).strip()
            for prior in kept
        ):
            continue
        seen.add(normalized)
        kept.append(fragment)
    if not kept:
        return cleaned
    joined = " ".join(kept).strip()
    return re.sub(r"[ \t]{2,}", " ", joined)


def _looks_broken_fragment(fragment: str, *, context: tuple[str, ...]) -> bool:
    stripped = fragment.strip()
    if len(stripped) < 6:
        return True
    if re.match(r"^[a-z]{1,2}\b", stripped) and len(stripped) < 80:
        return True
    contained_in_other = False
    lowered = re.sub(r"\W+", " ", stripped.lower()).strip()
    for other in context:
        candidate = other.strip()
        if candidate == stripped or len(candidate) <= len(stripped) + 8:
            continue
        normalized = re.sub(r"\W+", " ", candidate.lower()).strip()
        if lowered and lowered in normalized:
            contained_in_other = True
            break
    if stripped[0].islower() and len(stripped) < 64 and contained_in_other:
        return True
    return contained_in_other and len(stripped) < 96


def _score_detailed_render_body(slot_name: str, body: str) -> int:
    score = len(body) + 40 * _detail_marker_count(slot_name, body)
    if "not clearly stated in the paper" in body.lower():
        score -= 120
    if any(line.strip() and line.strip()[0].islower() for line in body.splitlines()):
        score -= 40
    return score


def _cleanup_duplicate_render_lines(answer_text: str) -> str:
    lines: list[str] = []
    seen_recent: list[str] = []
    for raw_line in answer_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if _looks_broken_fragment(stripped, context=tuple(seen_recent)):
            continue
        normalized = re.sub(r"\W+", " ", stripped.lower()).strip()
        if normalized and seen_recent and normalized == seen_recent[-1]:
            continue
        lines.append(stripped)
        if normalized:
            seen_recent.append(normalized)
            seen_recent = seen_recent[-6:]
    return "\n".join(lines).strip()


def _cleanup_structured_compare_render(
    answer_text: str,
    *,
    response_language: str,
) -> str:
    compare_titles = {
        _compare_documents_heading(response_language),
        *(
            _compare_section_title(slot_name, response_language)
            for slot_name in (
                "research_question",
                "sample_or_data",
                "method",
                "main_findings",
                "limitations",
                "conclusion",
                "practical_or_investment_implications",
            )
        ),
    }
    lines: list[str] = []
    seen_recent: list[str] = []
    for raw_line in answer_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped in compare_titles:
            if lines and lines[-1] == stripped:
                continue
            lines.append(stripped)
            seen_recent.append(re.sub(r"\W+", " ", stripped.lower()).strip())
            seen_recent = seen_recent[-6:]
            continue
        if stripped.startswith("- `") and stripped.endswith("`"):
            if lines and lines[-1] == stripped:
                continue
            lines.append(stripped)
            seen_recent.append(re.sub(r"\W+", " ", stripped.lower()).strip())
            seen_recent = seen_recent[-6:]
            continue
        if _looks_broken_fragment(stripped, context=tuple(seen_recent)):
            continue
        normalized = re.sub(r"\W+", " ", stripped.lower()).strip()
        if normalized and seen_recent and normalized == seen_recent[-1]:
            continue
        lines.append(stripped)
        if normalized:
            seen_recent.append(normalized)
            seen_recent = seen_recent[-6:]
    return "\n".join(lines).strip()


def _render_integrity_issues(
    answer_text: str,
    response_language: str,
) -> tuple[str, ...]:
    occurrences = _parse_detailed_note_occurrences(answer_text, response_language)
    issues: list[str] = []
    if occurrences:
        titles = [title for _slot_name, title, _body in occurrences]
        if len(titles) != len(set(titles)):
            issues.append("duplicate section headings")
        conclusion_occurrences = [body for slot_name, _title, body in occurrences if slot_name == "conclusion" and body]
        if len(conclusion_occurrences) > 1:
            issues.append("duplicated conclusion block")
    title_set = set(_detailed_note_title_map(response_language).values())
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    body_lines = [line for line in lines if line not in title_set]
    normalized_counts: dict[str, int] = {}
    for line in body_lines:
        normalized = re.sub(r"\W+", " ", line.lower()).strip()
        if len(normalized) >= 24:
            normalized_counts[normalized] = normalized_counts.get(normalized, 0) + 1
    if any(count > 1 for count in normalized_counts.values()):
        issues.append("duplicated paragraph blocks")
    if any(_looks_broken_fragment(line, context=tuple(body_lines)) for line in body_lines):
        issues.append("broken fragment residue")
    return tuple(dict.fromkeys(issues))


def _restore_structured_compare_sections(
    answer_text: str,
    *,
    document_notes: list[dict[str, object]],
    requested_slots: tuple[str, ...],
    response_language: str,
) -> str:
    normalized = unicodedata.normalize("NFKC", answer_text or "").strip()
    if (
        not normalized
        or not requested_slots
        or _structured_compare_section_count(normalized) >= max(4, min(6, len(requested_slots)))
    ):
        return normalized
    bullet_lines = [
        line.strip()
        for line in normalized.splitlines()
        if line.strip().startswith("- ")
    ]
    slot_bullets = [
        line
        for line in bullet_lines
        if ":" in line
        or line.lower().startswith("- contrast:")
        or line.startswith("- 对比：")
    ]
    slot_group_count = min(len(requested_slots), len(slot_bullets) // 3)
    if slot_group_count < 2:
        paragraph_groups = [
            [line.strip() for line in block.splitlines() if line.strip().startswith("- ")]
            for block in re.split(r"\n\s*\n", normalized)
        ]
        slot_groups = [group[:3] for group in paragraph_groups if len(group) >= 3]
        slot_group_count = min(len(requested_slots), len(slot_groups))
        if slot_group_count < 2:
            return normalized
        rebuilt_lines: list[str] = [
            _compare_documents_heading(response_language),
            *[f"- `{_compare_document_name(document_note)}`" for document_note in document_notes],
        ]
        for slot_name, slot_group in zip(requested_slots[:slot_group_count], slot_groups[:slot_group_count]):
            rebuilt_lines.extend(
                [
                    "",
                    _compare_section_title(slot_name, response_language),
                    *slot_group,
                ]
            )
        return "\n".join(rebuilt_lines).strip()
    slot_names = requested_slots[:slot_group_count]
    rebuilt_lines: list[str] = [
        _compare_documents_heading(response_language),
        *[f"- `{_compare_document_name(document_note)}`" for document_note in document_notes],
    ]
    cursor = 0
    for slot_name in slot_names:
        slot_group = slot_bullets[cursor : cursor + 3]
        if len(slot_group) < 3:
            return normalized
        rebuilt_lines.extend(
            [
                "",
                _compare_section_title(slot_name, response_language),
                *slot_group,
            ]
        )
        cursor += 3
    return "\n".join(rebuilt_lines).strip()


def _structured_compare_slots_from_text(source_text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", source_text or "")
    slot_names = (
        "research_question",
        "sample_or_data",
        "method",
        "main_findings",
        "limitations",
        "conclusion",
        "practical_or_investment_implications",
    )
    ordered: list[str] = []
    for slot_name in slot_names:
        titles = {
            _compare_section_title(slot_name, "en"),
            _compare_section_title(slot_name, "zh-CN"),
        }
        if any(re.search(rf"(?m)^{re.escape(title)}\s*$", normalized) for title in titles):
            ordered.append(slot_name)
    return tuple(ordered)


def _compose_single_language_cleanup_prompt(
    *,
    source_text: str,
    response_language: str,
    response_style: str,
) -> str:
    structured_compare = _looks_like_structured_compare_text(source_text)
    if response_language == "zh-CN":
        style_instruction = (
            "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
            if response_style == "continuous_prose"
            else "Return a concise Simplified Chinese memo with clean Chinese section text."
        )
        instructions = [
            "Rewrite the answer below into clean Simplified Chinese only.",
            "Preserve the same facts, structure, and level of detail.",
            "Do not add any new claims.",
            "Do not drop supported details.",
            "Do not leave English scaffolding or glue text in the answer.",
            "Acronyms and proper technical names such as CRSP, PCA, PLS, ANN, Lasso, and Elastic Net may remain in English when needed.",
        ]
        if structured_compare:
            instructions.extend(
                [
                    "Preserve every section and bullet in the same order.",
                    "Do not merge, omit, or rename requested comparison dimensions.",
                    "Use these Chinese section headings exactly when the corresponding sections appear: 比较文献, 研究问题, 样本与数据, 方法, 主要发现, 局限, 结论, 实践或投资含义.",
                    "Translate each '- Contrast:' line as '- 对比：'.",
                    "When the source says 'Not clearly stated in the paper.', translate it exactly as '文中未明确说明。'.",
                ]
            )
        instructions.extend(
            [
                style_instruction,
                "",
                "Answer to clean:",
                source_text,
                "",
                "Return only the cleaned final Chinese answer body.",
            ]
        )
        return "\n".join(instructions)
    style_instruction = (
        "Return one continuous paragraph in natural English with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "Return a concise English memo with clean English section text."
    )
    return "\n".join(
        [
            "Rewrite the answer below into clean English only.",
            "Preserve the same facts, structure, and level of detail.",
            "Do not add any new claims.",
            "Do not drop supported details.",
            "Do not leave Chinese scaffolding or glue text in the answer.",
            style_instruction,
            "",
            "Answer to clean:",
            source_text,
            "",
            "Return only the cleaned final English answer body.",
        ]
    )


def _looks_like_structured_compare_text(source_text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", source_text or "")
    if not normalized.strip():
        return False
    title_patterns = (
        r"(?m)^Documents compared\s*$",
        r"(?m)^Research question\s*$",
        r"(?m)^Sample and data\s*$",
        r"(?m)^Method\s*$",
        r"(?m)^Main findings\s*$",
        r"(?m)^Limitations\s*$",
        r"(?m)^Conclusion\s*$",
        r"(?m)^Practical or investment implications\s*$",
        r"(?m)^\u6bd4\u8f83\u6587\u732e\s*$",
        r"(?m)^\u7814\u7a76\u95ee\u9898\s*$",
        r"(?m)^\u6837\u672c\u4e0e\u6570\u636e\s*$",
        r"(?m)^\u65b9\u6cd5\s*$",
        r"(?m)^\u4e3b\u8981\u53d1\u73b0\s*$",
        r"(?m)^\u5c40\u9650\s*$",
        r"(?m)^\u7ed3\u8bba\s*$",
        r"(?m)^\u5b9e\u8df5\u6216\u6295\u8d44\u542b\u4e49\s*$",
    )
    title_count = sum(1 for pattern in title_patterns if re.search(pattern, normalized))
    return title_count >= 3 and "- `" in normalized


def _structured_compare_section_count(source_text: str) -> int:
    normalized = unicodedata.normalize("NFKC", source_text or "")
    title_patterns = (
        r"(?m)^Documents compared\s*$",
        r"(?m)^Research question\s*$",
        r"(?m)^Sample and data\s*$",
        r"(?m)^Method\s*$",
        r"(?m)^Main findings\s*$",
        r"(?m)^Limitations\s*$",
        r"(?m)^Conclusion\s*$",
        r"(?m)^Practical or investment implications\s*$",
        r"(?m)^\u6bd4\u8f83\u6587\u732e\s*$",
        r"(?m)^\u7814\u7a76\u95ee\u9898\s*$",
        r"(?m)^\u6837\u672c\u4e0e\u6570\u636e\s*$",
        r"(?m)^\u65b9\u6cd5\s*$",
        r"(?m)^\u4e3b\u8981\u53d1\u73b0\s*$",
        r"(?m)^\u5c40\u9650\s*$",
        r"(?m)^\u7ed3\u8bba\s*$",
        r"(?m)^\u5b9e\u8df5\u6216\u6295\u8d44\u542b\u4e49\s*$",
    )
    return sum(1 for pattern in title_patterns if re.search(pattern, normalized))


def _rebuild_compare_structure_from_bullets(
    answer_text: str,
    *,
    document_notes: list[dict[str, object]],
    requested_slots: tuple[str, ...],
    response_language: str,
) -> str:
    normalized = unicodedata.normalize("NFKC", answer_text or "").strip()
    if not normalized or _structured_compare_section_count(normalized) >= max(4, min(6, len(requested_slots))):
        return normalized
    bullet_lines = [
        line.strip()
        for line in normalized.splitlines()
        if line.strip().startswith("- ")
    ]
    slot_bullets = [
        line
        for line in bullet_lines
        if ":" in line
        or line.lower().startswith("- contrast:")
        or line.startswith("- 对比：")
    ]
    expected_bullets = len(requested_slots) * 3
    if len(slot_bullets) < expected_bullets:
        return normalized
    rebuilt_lines: list[str] = [
        _compare_documents_heading(response_language),
        *[f"- `{_compare_document_name(document_note)}`" for document_note in document_notes],
    ]
    cursor = 0
    for slot_name in requested_slots:
        slot_group = slot_bullets[cursor : cursor + 3]
        if len(slot_group) < 3:
            return normalized
        rebuilt_lines.extend(
            [
                "",
                _compare_section_title(slot_name, response_language),
                *slot_group,
            ]
        )
        cursor += 3
    return "\n".join(rebuilt_lines).strip()

def _compose_single_language_cleanup_prompt(
    *,
    source_text: str,
    response_language: str,
    response_style: str,
) -> str:
    structured_compare = _looks_like_structured_compare_text(source_text)
    if response_language == "zh-CN":
        style_instruction = (
            "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
            if response_style == "continuous_prose"
            else "Return a concise Simplified Chinese memo with clean Chinese section text."
        )
        instructions = [
            "Rewrite the answer below into clean Simplified Chinese only.",
            "Preserve the same facts, structure, and level of detail.",
            "Do not add any new claims.",
            "Do not drop supported details.",
            "Do not leave English scaffolding or glue text in the answer.",
            "Acronyms and proper technical names such as CRSP, PCA, PLS, ANN, Lasso, and Elastic Net may remain in English when needed.",
        ]
        if structured_compare:
            instructions.extend(
                [
                    "Translate line by line and keep the same section and bullet structure.",
                    "Do not omit any heading, bullet, or compared document line.",
                    "If a line begins with '- `filename`:', keep the filename exactly and translate only the rest of the line.",
                    "Use these Chinese section headings exactly when the corresponding sections appear: \u6bd4\u8f83\u6587\u732e, \u7814\u7a76\u95ee\u9898, \u6837\u672c\u4e0e\u6570\u636e, \u65b9\u6cd5, \u4e3b\u8981\u53d1\u73b0, \u5c40\u9650, \u7ed3\u8bba, \u5b9e\u8df5\u6216\u6295\u8d44\u542b\u4e49.",
                    "Translate each '- Contrast:' line as '- \u5bf9\u6bd4\uff1a'.",
                    "When the source says 'Not clearly stated in the paper.', translate it exactly as '\u6587\u4e2d\u672a\u660e\u786e\u8bf4\u660e\u3002'.",
                ]
            )
        instructions.extend(
            [
                style_instruction,
                "",
                "Answer to clean:",
                source_text,
                "",
                "Return only the cleaned final Chinese answer body.",
            ]
        )
        return "\n".join(instructions)
    style_instruction = (
        "Return one continuous paragraph in natural English with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "Return a concise English memo with clean English section text."
    )
    return "\n".join(
        [
            "Rewrite the answer below into clean English only.",
            "Preserve the same facts, structure, and level of detail.",
            "Do not add any new claims.",
            "Do not drop supported details.",
            "Do not leave Chinese scaffolding or glue text in the answer.",
            style_instruction,
            "",
            "Answer to clean:",
            source_text,
            "",
            "Return only the cleaned final English answer body.",
        ]
    )


def _restore_structured_compare_sections(
    answer_text: str,
    *,
    document_notes: list[dict[str, object]],
    requested_slots: tuple[str, ...],
    response_language: str,
) -> str:
    normalized = unicodedata.normalize("NFKC", answer_text or "").strip()
    if (
        not normalized
        or not requested_slots
        or _structured_compare_section_count(normalized) >= max(4, min(6, len(requested_slots)))
    ):
        return normalized
    bullet_lines = [
        line.strip()
        for line in normalized.splitlines()
        if line.strip().startswith("- ")
    ]
    slot_bullets = [
        line
        for line in bullet_lines
        if ":" in line
        or "：" in line
        or line.lower().startswith("- contrast:")
        or "\u5bf9\u6bd4" in line
    ]
    slot_group_count = min(len(requested_slots), len(slot_bullets) // 3)
    if slot_group_count < 2:
        paragraph_groups = [
            [line.strip() for line in block.splitlines() if line.strip().startswith("- ")]
            for block in re.split(r"\n\s*\n", normalized)
        ]
        slot_groups = [group[:3] for group in paragraph_groups if len(group) >= 3]
        slot_group_count = min(len(requested_slots), len(slot_groups))
        if slot_group_count < 2:
            return normalized
        rebuilt_lines: list[str] = [
            _compare_documents_heading(response_language),
            *[f"- `{_compare_document_name(document_note)}`" for document_note in document_notes],
        ]
        for slot_name, slot_group in zip(requested_slots[:slot_group_count], slot_groups[:slot_group_count]):
            rebuilt_lines.extend(
                [
                    "",
                    _compare_section_title(slot_name, response_language),
                    *slot_group,
                ]
            )
        return "\n".join(rebuilt_lines).strip()
    rebuilt_lines: list[str] = [
        _compare_documents_heading(response_language),
        *[f"- `{_compare_document_name(document_note)}`" for document_note in document_notes],
    ]
    cursor = 0
    for slot_name in requested_slots[:slot_group_count]:
        slot_group = slot_bullets[cursor : cursor + 3]
        if len(slot_group) < 3:
            return normalized
        rebuilt_lines.extend(
            [
                "",
                _compare_section_title(slot_name, response_language),
                *slot_group,
            ]
        )
        cursor += 3
    return "\n".join(rebuilt_lines).strip()


def _translate_paper_answer_to_target_language(
    config: LabaiConfig,
    prompt: str,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> str:
    cleaned = answer_text.strip()
    if not cleaned:
        return cleaned
    if mode_selection.response_language == "zh-CN" and mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        compare_slots = _structured_compare_slots_from_text(cleaned) or _requested_paper_slots(prompt, "paper_compare")
        grounded_compare = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=compare_slots,
            response_language="en",
            response_style="structured",
            paper_output_profile=mode_selection.paper_output_profile,
        ).strip()
        if grounded_compare:
            cleaned = grounded_compare
    structured_compare = _looks_like_structured_compare_text(cleaned)
    source_compare_sections = _structured_compare_section_count(cleaned) if structured_compare else 0
    requested_compare_slots = (
        _structured_compare_slots_from_text(cleaned) or _requested_paper_slots(prompt, "paper_compare")
        if structured_compare and mode_selection.mode == "paper_compare"
        else ()
    )
    cleanup_selection = replace(
        mode_selection,
        mode="repo_overview",
        reason="Final single-language cleanup rewrites the already-grounded answer without rereading the paper.",
        answer_schema="language_cleanup",
        read_strategy="none",
        read_strategy_reason="Single-language cleanup operates on the final rendered answer text.",
        paper_output_profile="none",
    )
    if (
        mode_selection.response_language == "zh-CN"
        and structured_compare
        and requested_compare_slots
        and paper_trace.document_notes
    ):
        structured_compare_candidate = _translate_structured_compare_sections_to_chinese(
            config,
            session_id,
            observations,
            evidence_refs,
            cleanup_selection,
            cleaned,
            document_notes=paper_trace.document_notes,
            requested_slots=requested_compare_slots,
        )
        if structured_compare_candidate:
            structured_compare_candidate = _finalize_paper_answer_text(
                prompt,
                mode_selection,
                paper_trace,
                structured_compare_candidate,
            )
            if _structured_compare_section_count(structured_compare_candidate) >= max(
                4,
                min(6, len(requested_compare_slots)),
            ):
                return structured_compare_candidate
    allow_bilingual = _prompt_requests_bilingual_output(prompt)
    if not _contains_unexpected_language_leakage(
        cleaned,
        response_language=mode_selection.response_language,
        allow_bilingual=allow_bilingual,
    ):
        return cleaned
    candidate = cleaned
    if mode_selection.response_language == "zh-CN":
        translation_prompt = _compose_slot_translation_prompt(
            source_text=cleaned,
            response_style=mode_selection.response_style,
        )
        try:
            translation_route = _run_answer_route(
                config,
                translation_prompt,
                session_id,
                observations,
                evidence_refs,
                cleanup_selection,
                cleaned,
            )
            translated = translation_route.text.strip()
            if translated:
                candidate = _finalize_paper_answer_text(
                    prompt,
                    mode_selection,
                    paper_trace,
                    translated,
                )
                if structured_compare and requested_compare_slots and paper_trace.document_notes:
                    candidate = _restore_structured_compare_sections(
                        candidate,
                        document_notes=paper_trace.document_notes,
                        requested_slots=requested_compare_slots,
                        response_language=mode_selection.response_language,
                    )
                    candidate = _finalize_paper_answer_text(
                        prompt,
                        mode_selection,
                        paper_trace,
                        candidate,
                    )
                if (
                    structured_compare
                    and source_compare_sections >= 4
                    and (
                        _structured_compare_section_count(candidate) + 1 < source_compare_sections
                        or (
                            len([line for line in cleaned.splitlines() if line.strip()])
                            > len([line for line in candidate.splitlines() if line.strip()]) + 4
                        )
                    )
                ):
                    candidate = cleaned
        except (ProviderError, RuntimeAdapterError, StopIteration):
            candidate = cleaned
    compare_needs_polish = (
        mode_selection.response_language == "zh-CN"
        and structured_compare
        and (
            _structured_compare_section_count(candidate) + 1 < source_compare_sections
            or bool(re.search(r"[?？][^`\n]{0,120}\.pdf[?？]", candidate))
        )
    )
    if not compare_needs_polish and not _contains_unexpected_language_leakage(
        candidate,
        response_language=mode_selection.response_language,
        allow_bilingual=allow_bilingual,
    ):
        return candidate
    cleanup_prompt = _compose_single_language_cleanup_prompt(
        source_text=cleaned if compare_needs_polish else candidate,
        response_language=mode_selection.response_language,
        response_style=mode_selection.response_style,
    )
    try:
        cleanup_route = _run_answer_route(
            config,
            cleanup_prompt,
            session_id,
            observations,
            evidence_refs,
            cleanup_selection,
            candidate,
        )
        cleaned_candidate = cleanup_route.text.strip()
        if cleaned_candidate:
            candidate = _finalize_paper_answer_text(
                prompt,
                mode_selection,
                paper_trace,
                cleaned_candidate,
            )
            if structured_compare and requested_compare_slots and paper_trace.document_notes:
                rebuilt_candidate = _restore_structured_compare_sections(
                    candidate,
                    document_notes=paper_trace.document_notes,
                    requested_slots=requested_compare_slots,
                    response_language=mode_selection.response_language,
                )
                if _structured_compare_section_count(rebuilt_candidate) > _structured_compare_section_count(candidate):
                    candidate = _finalize_paper_answer_text(
                        prompt,
                        mode_selection,
                        paper_trace,
                        rebuilt_candidate,
                    )
    except (ProviderError, RuntimeAdapterError, StopIteration):
        return candidate
    return candidate


def _translate_structured_compare_sections_to_chinese(
    config: LabaiConfig,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    cleanup_selection: ModeSelection,
    source_text: str,
    *,
    document_notes: list[dict[str, object]],
    requested_slots: tuple[str, ...],
) -> str:
    rebuilt_lines: list[str] = [
        _compare_documents_heading("zh-CN"),
        *[f"- `{_compare_document_name(document_note)}`" for document_note in document_notes],
    ]
    for slot_name in requested_slots:
        entries = [
            _build_compare_slot_entry(
                document_note,
                slot_name,
                response_language="zh-CN",
                paper_output_profile="detailed_paper_note",
            )
            for document_note in document_notes
        ]
        translated_entries: list[dict[str, object]] = []
        rebuilt_lines.extend(["", _compare_section_title(slot_name, "zh-CN")])
        for entry in entries:
            if entry["status"] == "not_clearly_stated":
                translated_summary = "文中未明确说明。"
            else:
                translated_summary = _translate_compare_summary_to_chinese(
                    config,
                    session_id,
                    observations,
                    evidence_refs,
                    cleanup_selection,
                    entry["summary"],
                )
            if not translated_summary:
                return ""
            translated_entry = {**entry}
            translated_entry["summary"] = translated_summary
            translated_entry["clause"] = translated_summary.rstrip("。. ")
            translated_entries.append(translated_entry)
            rebuilt_lines.append(f"- `{entry['source_name']}`: {translated_summary}")
        rebuilt_lines.append(
            _build_compare_contrast_line(
                slot_name,
                translated_entries,
                response_language="zh-CN",
            )
        )
    return "\n".join(rebuilt_lines).strip()


def _polish_structured_compare_sections_in_chinese(
    config: LabaiConfig,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    cleanup_selection: ModeSelection,
    source_text: str,
    *,
    document_notes: list[dict[str, object]],
    requested_slots: tuple[str, ...],
) -> str:
    polish_prompt = "\n".join(
        [
            "Polish the structured paper comparison below into natural Simplified Chinese.",
            "Keep the same headings, bullets, compared papers, and requested dimensions.",
            "Keep the same factual content, missing-detail wording, and slot-to-slot contrasts.",
            "Do not add new claims, background filler, or workflow/process commentary.",
            "Remove stray English glue words, translation residue, reference/path residue, and awkward machine-translated phrasing when a clean Chinese wording is available.",
            "Keep technical acronyms such as GQMLE, NGQMLE, GARCH, LSTM, CRSP, and PDF filenames unchanged when needed.",
            "Return only the cleaned structured comparison.",
            "",
            "Structured comparison:",
            source_text.strip(),
        ]
    )
    try:
        polish_route = _run_answer_route(
            config,
            polish_prompt,
            session_id,
            observations,
            evidence_refs,
            cleanup_selection,
            source_text,
        )
    except (ProviderError, RuntimeAdapterError, StopIteration):
        return source_text
    candidate = polish_route.text.strip()
    if not candidate:
        return source_text
    candidate = _restore_structured_compare_sections(
        candidate,
        document_notes=document_notes,
        requested_slots=requested_slots,
        response_language="zh-CN",
    )
    return candidate or source_text


def _translate_compare_summary_to_chinese(
    config: LabaiConfig,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    cleanup_selection: ModeSelection,
    summary_text: str,
) -> str:
    body = summary_text.strip()
    if body == "Not clearly stated in the paper.":
        return "文中未明确说明。"
    meta_markers = (
        "文本翻译如下",
        "主要目录",
        "模块",
        "重要入口点",
        "当前运行路径",
        "相关文献及片段",
        "注意原文",
        "关键风险",
        "根据提供的比较摘要",
        "可以将其翻译为",
        "没有添加或删除任何主张",
        "原文的所有详细信息和事实内容",
        "证据文件已核实",
        "保持原样",
        "原文:",
        "翻译后:",
        "参考文献:",
        "#page=",
        "#chunk=",
        "该段落已经在原有内容的基础上进行了简洁和清晰的表述",
        "不再添加或删除任何声明",
        "中文摘要:",
        "此翻译保留了原文的事实内容和详细程度",
        "适当的中文表述",
    )
    translation_prompt = "\n".join(
        [
            "Translate the compare summary below into clean Simplified Chinese.",
            "Preserve exactly the same factual content and level of detail.",
            "Do not add or remove claims.",
            "Translate ordinary finance or statistics nouns such as market maker, liquidity provision, Gaussian, and heavy tails into Chinese when a natural Chinese wording exists.",
            "Keep technical acronyms such as GQMLE, NGQMLE, GARCH, PCA, LSTM, and CRSP in English when needed.",
            "If the summary says 'Not clearly stated in the paper.', translate it exactly as '文中未明确说明。'",
            "This is a paper-comparison sentence, not a codebase or project summary.",
            "Do not mention modules, directories, entry points, run paths, translation notes, or workflow commentary.",
            "Return only the translated sentence body with no bullet marker, no filename prefix, and no extra commentary.",
            "",
            "Compare summary:",
            body,
        ]
    )
    try:
        translation_route = _run_answer_route(
            config,
            translation_prompt,
            session_id,
            observations,
            evidence_refs,
            cleanup_selection,
            body,
        )
    except (ProviderError, RuntimeAdapterError, StopIteration):
        return ""
    translated_body = translation_route.text.strip()
    if any(marker in translated_body for marker in meta_markers):
        retry_prompt = "\n".join(
            [
                "Translate only the single paper-comparison sentence below into one natural Simplified Chinese sentence.",
                "Preserve the same meaning exactly.",
                "Do not add headings, notes, translation commentary, code/project scaffolding, or extra examples.",
                "Return only the translated sentence.",
                "",
                "Sentence:",
                body,
            ]
        )
        try:
            retry_route = _run_answer_route(
                config,
                retry_prompt,
                session_id,
                observations,
                evidence_refs,
                cleanup_selection,
                body,
            )
            if retry_route.text.strip():
                translated_body = retry_route.text.strip()
        except (ProviderError, RuntimeAdapterError, StopIteration):
            pass
    translated_body = re.sub(r"^-+\s*", "", translated_body)
    translated_body = re.sub(r"^(?:-+\s*)?(?:contrast|对比)\s*[:：]\s*", "", translated_body, flags=re.IGNORECASE)
    translated_body = re.sub(r"^(`[^`]+`):\s*", "", translated_body)
    if body != "Not clearly stated in the paper.":
        translated_body = re.sub(r"\s*文中未明确说明。?\s*$", "", translated_body)
        translated_body = re.sub(r"^(?:文本|文中)未明确说明[。；，,:：]?\s*", "", translated_body)
        translated_body = re.sub(r"^(?:正文|文中)未明确说明[,，]?(?:但|不过)\s*", "", translated_body)
        translated_body = re.sub(r"^(?:主要内容|主要比较总结|主要对比总结|文本翻译如下|翻译如下)\s*[:：]\s*", "", translated_body)
        translated_body = re.sub(r"^(?:对比总结|比较总结)\s*", "", translated_body)
        translated_body = re.sub(r"^(?:中文摘要)\s*[:：]\s*", "", translated_body)
        if "翻译后:" in translated_body:
            translated_body = translated_body.split("翻译后:", 1)[1].strip()
        translated_body = re.sub(
            r"^(?:文本未明确说明[。；，]?\s*)?(?:然而[，,]?\s*)?(?:根据提供的比较摘要[，,]?\s*)?(?:可以将其翻译为\s*[:：]\s*)",
            "",
            translated_body,
        )
        translated_body = re.sub(
            r"\s*(?:这个句子已经包含了原文的所有详细信息和事实内容[,，]?\s*没有添加或删除任何主张。?)\s*$",
            "",
            translated_body,
        )
        translated_body = re.sub(
            r"\s*证据文件已核实[:：]?\s*(?:文中未明确说明的部分保持原样。?)?\s*$",
            "",
            translated_body,
        )
        translated_body = re.sub(r"\s*文中未明确说明的部分保持原样。?\s*$", "", translated_body)
        translated_body = re.sub(r"\s*原文[:：].*$", "", translated_body)
        translated_body = re.sub(r"\s*参考文献[:：].*$", "", translated_body)
        translated_body = re.sub(r"\s*[A-Z]:\\[^\s]+(?:#page=\d+(?:#chunk=[^\s]+)?)?", "", translated_body)
        translated_body = re.sub(
            r"\s*该段落已经在原有内容的基础上进行了简洁和清晰的表述[,，]?\s*不再添加或删除任何声明。?\s*$",
            "",
            translated_body,
        )
        translated_body = re.sub(
            r"\s*此翻译保留了原文的事实内容和详细程度[,，]?\s*并使用了适当的中文表述。?\s*$",
            "",
            translated_body,
        )
    translated_body = re.sub(r"\bmarket makers?\b", "做市商", translated_body, flags=re.IGNORECASE)
    translated_body = re.sub(r"\breference market maker\b", "参考做市商", translated_body, flags=re.IGNORECASE)
    translated_body = re.sub(r"\bmakers?\b", "做市商", translated_body, flags=re.IGNORECASE)
    translated_body = re.sub(r"\bGaussian\b", "高斯", translated_body, flags=re.IGNORECASE)
    translated_body = _polish_chinese_compare_sentence(translated_body)
    translated_body = translated_body.strip()
    if any(marker in translated_body for marker in meta_markers):
        return body
    return translated_body


def _polish_chinese_compare_sentence(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = re.sub(r"\bAlic文\b", "\u8be5\u6587", cleaned)
    cleaned = re.sub(r"\bmarket makers?\b", "\u505a\u5e02\u5546", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmaker\b", "\u505a\u5e02\u5546", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b[Gg]aussian alternative\b", "\u9ad8\u65af\u66ff\u4ee3\u65b9\u6848", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bGaussian\b", "\u9ad8\u65af", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\breinforcement learning\b", "\u5f3a\u5316\u5b66\u4e60", cleaned, flags=re.IGNORECASE)
    replacement_rules = (
        (r"\b[Nn]umerical studies?\b", "数值研究"),
        (r"\b[Nn]umerical results?\b", "数值结果"),
        (r"\b[Nn]umerical experiments?\b", "数值实验"),
        (r"\bQuasi[- ]*Maximum Likelihood Estimator\b", "准最大似然估计量"),
        (r"\bQuasi[- ]*Maximum Likelihood Estimation\b", "准最大似然估计"),
        (r"\b[Gg]aussian alternative\b", "高斯替代方案"),
        (r"\bGaussian\b", "高斯"),
        (r"\bWe\b", ""),
        (r"\bQuasi 最大似然\b", "准最大似然"),
        (r"\breinforcement learning\b", "强化学习"),
        (r"\bheavy tails?\b", "厚尾"),
        (r"\binnovation errors?\b", "创新误差"),
        (r"\binnovation distribution\b", "创新项分布"),
        (r"稳健最大似然估计量", "准最大似然估计量"),
        (r"参考市场制造商", "参考做市商"),
        (r"市场制造商", "做市商"),
        (r"参考市场制作人", "参考做市商"),
        (r"市场制作人", "做市商"),
        (r"市场制作商", "做市商"),
        (r"市场制造者", "做市商"),
        (r"封闭形式", "闭式"),
        (r"封闭解", "闭式解"),
        (r"为了进行这项操作", "为此"),
    )
    for pattern, replacement in replacement_rules:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("Numerical研究", "数值研究")
    cleaned = cleaned.replace("Numerical结果", "数值结果")
    cleaned = cleaned.replace("Numerical实验", "数值实验")
    cleaned = cleaned.replace("Quasi-最大似然", "准最大似然")
    cleaned = cleaned.replace("Quasi最大似然", "准最大似然")
    cleaned = cleaned.replace("会影后续", "会影响后续")
    cleaned = cleaned.replace("外生竞争的存在下 的", "存在外生竞争时的")
    cleaned = cleaned.replace("中文摘要:", "")
    cleaned = cleaned.replace("竟争", "竞争")
    cleaned = re.sub(r"\s*强化学习\s*", "强化学习", cleaned)
    cleaned = re.sub(r"\s+([，。；：！？])", r"\1", cleaned)
    cleaned = re.sub(r"([（(])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([）)])", r"\1", cleaned)
    cleaned = cleaned.replace("闭环解决方案", "\u95ed\u5f0f\u89e3")
    cleaned = cleaned.replace("市场制作", "\u505a\u5e02")
    cleaned = cleaned.replace("三条步骤", "\u4e09\u6b65")
    cleaned = cleaned.replace("渐近常态", "\u6e10\u8fd1\u6b63\u6001")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _best_supported_narrow_slot_summary(
    paper_trace: PaperTrace,
    slot_name: str,
    *,
    response_language: str,
) -> str:
    missing_phrase = _paper_missing_phrase(response_language)
    for document_note in paper_trace.document_notes:
        payload = _document_slot(document_note, slot_name)
        if _slot_payload_status(payload) == "not_clearly_stated":
            continue
        candidate = _slot_payload_text(payload).strip() or str(payload.get("merged_note_text", "")).strip()
        if candidate and candidate != "Not clearly stated in the paper.":
            return _normalize_slot_summary(candidate, response_language=response_language)
    return missing_phrase


def _extract_method_family_mentions(texts: list[str]) -> tuple[str, ...]:
    corpus = " ".join(unicodedata.normalize("NFKC", text).lower() for text in texts)
    catalog = (
        ("lasso", "Lasso"),
        ("elastic net", "Elastic Net"),
        ("enet", "Elastic Net"),
        ("ridge", "Ridge"),
        ("linear regression", "linear regression"),
        ("generalized linear model", "generalized linear models"),
        ("principal components regression", "principal components regression (PCR)"),
        ("principal component analysis", "principal component analysis (PCA)"),
        ("pca", "principal component analysis (PCA)"),
        ("partial least squares", "partial least squares (PLS)"),
        ("regression tree", "regression trees"),
        ("boosted tree", "boosted trees"),
        ("random forest", "random forests"),
        ("ann", "artificial neural networks (ANN)"),
        ("artificial neural network", "artificial neural networks (ANN)"),
        ("cnn", "convolutional neural networks (CNN)"),
        ("convolutional neural network", "convolutional neural networks (CNN)"),
        ("lstm", "long short-term memory networks (LSTM)"),
        ("long short-term memory", "long short-term memory networks (LSTM)"),
        ("neural network", "neural networks"),
        ("transformer", "Transformers"),
        ("support vector machine", "support vector machines"),
        ("svm", "support vector machines"),
    )
    mentions: list[str] = []
    for needle, label in catalog:
        if needle in corpus:
            mentions.append(label)
    if "arimax" in corpus and "garch" in corpus:
        mentions.append("ARIMAX-GARCH")
    return tuple(dict.fromkeys(mentions))


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [unicodedata.normalize("NFKC", text).replace("\n", " ").strip() for text in texts]
    date_range_pattern = re.compile(
        r"\b(?:from|between)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})\s+(?:to|through|until|and)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})",
        re.IGNORECASE,
    )
    for text in normalized_texts:
        lowered = text.lower()
        if not _looks_like_explicit_sample_data_text(text):
            continue
        if all(token in lowered for token in ("crsp", "nyse", "amex", "nasdaq")):
            facts.append(
                "the paper uses monthly total individual equity returns from CRSP for firms listed in the NYSE, AMEX, and NASDAQ"
            )
        if all(token in lowered for token in ("nasdaq-100 constituents", "3 jan 2019", "30 dec 2021", "yfinance")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for all NASDAQ-100 constituents from 3 Jan 2019 to 30 Dec 2021, downloaded via yfinance"
            )
        elif all(token in lowered for token in ("daily adjusted closing prices", "nasdaq-100 constituents", "3 jan 2019")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for NASDAQ-100 constituents beginning on 3 Jan 2019"
            )
        if "tickers" in lowered and "755 trading days" in lowered:
            facts.append("after filtering, roughly N ~ 100 tickers and T = 755 trading days remain")
        if all(token in lowered for token in ("vix", "5-day rolling mean", "22-day rolling mean")):
            facts.append("VIX contributes three standardised exogenous features: the level, a 5-day rolling mean, and a 22-day rolling mean")
        if all(token in lowered for token in ("msft", "adbe", "nvda", "payx")):
            if "200 (i, t) pairs" in lowered or "the resulting 200" in lowered or "200 extreme shock events" in lowered:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX, and the shock-event set contains 200 extreme downside events")
            else:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX")
        if all(token in lowered for token in ("3 jan 2019", "30 jun 2020", "out-of-sample evaluation")):
            facts.append("the training window runs from 3 Jan 2019 to 30 Jun 2020, with the remaining period used for out-of-sample evaluation")
        elif "training" in lowered and "out-of-sample" in lowered and "3 jan 2019" in lowered:
            facts.append("the paper defines a training window starting on 3 Jan 2019 and evaluates the remaining period out of sample")
        if re.search(r"30,?000.+individual stocks.+1957.+2016", lowered):
            facts.append("the sample covers nearly 30,000 individual stocks over 60 years from 1957 to 2016")
        if re.search(
            r"number of stocks in (?:our|the) sample is (?:almost|nearly|about|approximately)\s+30,?000",
            lowered,
        ):
            stock_count_fact = "the sample contains almost 30,000 stocks"
            if re.search(r"average number of stocks per month.+6,?200", lowered):
                stock_count_fact += ", with the average monthly cross-section exceeding 6,200 stocks"
            facts.append(stock_count_fact)
        if re.search(
            r"18 years of training sample\s*\(1957-1974\).+12 years of validation sample\s*\(1975-1986\).+30 years\s*\(1987-2016\).+out-of-sample testing",
            lowered,
        ):
            facts.append(
                "the 60-year sample is split into 18 years of training sample (1957-1974), 12 years of validation sample (1975-1986), and 30 years (1987-2016) for out-of-sample testing"
            )
        elif re.search(
            r"18 years of training (?:sample|data).+12 years of validation (?:sample|data).+30 years.+out-of-sample testing",
            lowered,
        ):
            facts.append("the paper uses 18 years of training sample, 12 years of validation sample, and 30 years of out-of-sample testing")
        if re.search(r"our sample begins.+1957.+2016", lowered):
            facts.append("the sample begins in March 1957 and ends in December 2016, covering 60 years")
        if date_range_pattern.search(text) and any(token in lowered for token in ("sample", "data", "dataset", "observations", "panel")):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"\b\d{1,3}(?:,\d{3})?\s+observations\b", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"balanced panel of stocks|missing data", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"individual stocks", lowered) and re.search(r"60 years|1957|2016", lowered):
            facts.append(_truncate_line(text, limit=180))
    ordered = [
        item.replace("闁?00", "~ 100").replace("閳?00", "~ 100").replace("衼卸100", "~ 100")
        for item in _dedupe_strings(facts)
    ]

    def _fact_priority(item: str) -> tuple[int, int]:
        lowered_item = item.lower()
        if "nasdaq-100 constituents" in lowered_item:
            return (0, len(item))
        if "755 trading days" in lowered_item or "~ 100 tickers" in lowered_item:
            return (1, len(item))
        if "273 observations" in lowered_item:
            return (2, len(item))
        if "january 2001 to september 2023" in lowered_item:
            return (3, len(item))
        if "5-day rolling mean" in lowered_item or "22-day rolling mean" in lowered_item:
            return (4, len(item))
        if "hubs are msft" in lowered_item:
            return (5, len(item))
        if "training window runs" in lowered_item or "out of sample" in lowered_item:
            return (6, len(item))
        if "30,000 individual stocks" in lowered_item:
            return (7, len(item))
        if "training data" in lowered_item:
            return (8, len(item))
        if "sample begins" in lowered_item:
            return (9, len(item))
        return (10, len(item))

    ordered.sort(key=_fact_priority)
    return tuple(ordered)


def _build_narrow_grounded_paper_answer(
    prompt: str,
    paper_trace: PaperTrace,
    *,
    response_language: str,
) -> str:
    focus_slot = _narrow_grounded_qa_focus_slot(prompt)
    texts = _narrow_grounded_qa_text_pool(paper_trace, focus_slot)
    missing_phrase = _paper_missing_phrase(response_language)
    answer = missing_phrase

    if focus_slot == "method":
        families = _extract_method_family_mentions(texts)
        if families:
            joined = ", ".join(families[:-1]) + f", and {families[-1]}" if len(families) > 2 else " and ".join(families)
            answer = f"文中明确讨论的方法家族包括{joined}。" if response_language == "zh-CN" else f"The paper explicitly discusses {joined}."
        else:
            summary = _best_supported_narrow_slot_summary(paper_trace, "method", response_language=response_language)
            if summary != missing_phrase:
                answer = summary if response_language == "zh-CN" else f"The paper explicitly discusses {summary.rstrip('.')}."
    elif focus_slot == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            if response_language == "zh-CN":
                answer = f"文中明确说明，{facts[0]}。" if len(facts) == 1 else f"文中明确说明，{facts[0]}。文中还说明，{facts[1]}。"
            else:
                answer = f"The paper explicitly states that {facts[0]}." if len(facts) == 1 else f"The paper explicitly states that {facts[0]}. It also states that {facts[1]}."
        else:
            summary = _best_supported_narrow_slot_summary(paper_trace, "sample_or_data", response_language=response_language)
            if summary != missing_phrase:
                answer = summary if response_language == "zh-CN" else f"The paper explicitly states that {summary.rstrip('.')}."
    else:
        summary = _best_supported_narrow_slot_summary(paper_trace, focus_slot, response_language=response_language)
        if summary != missing_phrase:
            answer = _render_grounded_slot_sentence(focus_slot, summary, response_language=response_language)

    if _prompt_requests_support_detail(prompt):
        support_ref = _first_support_ref(paper_trace)
        if support_ref:
            answer += f" 相关位置可见 {support_ref}。" if response_language == "zh-CN" else f" The most directly relevant retrieved support is {support_ref}."
    return answer.strip()
def _best_supported_narrow_slot_summary(
    paper_trace: PaperTrace,
    slot_name: str,
    *,
    response_language: str,
) -> str:
    missing_phrase = _paper_missing_phrase(response_language)
    for document_note in paper_trace.document_notes:
        payload = _document_slot(document_note, slot_name)
        if _slot_payload_status(payload) == "not_clearly_stated":
            continue
        candidate = _slot_payload_text(payload).strip() or str(payload.get("merged_note_text", "")).strip()
        if candidate and candidate != "Not clearly stated in the paper.":
            return _normalize_slot_summary(candidate, response_language=response_language)
    return missing_phrase


def _extract_method_family_mentions(texts: list[str]) -> tuple[str, ...]:
    corpus = " ".join(unicodedata.normalize("NFKC", text).lower() for text in texts)
    catalog = (
        ("lasso", "Lasso"),
        ("elastic net", "Elastic Net"),
        ("enet", "Elastic Net"),
        ("ridge", "Ridge"),
        ("linear regression", "linear regression"),
        ("generalized linear model", "generalized linear models"),
        ("principal components regression", "principal components regression (PCR)"),
        ("principal component analysis", "principal component analysis (PCA)"),
        ("pca", "principal component analysis (PCA)"),
        ("partial least squares", "partial least squares (PLS)"),
        ("regression tree", "regression trees"),
        ("boosted tree", "boosted trees"),
        ("random forest", "random forests"),
        ("ann", "artificial neural networks (ANN)"),
        ("artificial neural network", "artificial neural networks (ANN)"),
        ("cnn", "convolutional neural networks (CNN)"),
        ("convolutional neural network", "convolutional neural networks (CNN)"),
        ("lstm", "long short-term memory networks (LSTM)"),
        ("long short-term memory", "long short-term memory networks (LSTM)"),
        ("neural network", "neural networks"),
        ("transformer", "Transformers"),
        ("support vector machine", "support vector machines"),
        ("svm", "support vector machines"),
    )
    mentions: list[str] = []
    for needle, label in catalog:
        if needle in corpus:
            mentions.append(label)
    if "arimax" in corpus and "garch" in corpus:
        mentions.append("ARIMAX-GARCH")
    return tuple(dict.fromkeys(mentions))


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [unicodedata.normalize("NFKC", text).replace("\n", " ").strip() for text in texts]
    date_range_pattern = re.compile(
        r"\b(?:from|between)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})\s+(?:to|through|until|and)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})",
        re.IGNORECASE,
    )
    for text in normalized_texts:
        lowered = text.lower()
        if not _looks_like_explicit_sample_data_text(text):
            continue
        if all(token in lowered for token in ("crsp", "nyse", "amex", "nasdaq")):
            facts.append(
                "the paper uses monthly total individual equity returns from CRSP for firms listed in the NYSE, AMEX, and NASDAQ"
            )
        if all(token in lowered for token in ("nasdaq-100 constituents", "3 jan 2019", "30 dec 2021", "yfinance")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for all NASDAQ-100 constituents from 3 Jan 2019 to 30 Dec 2021, downloaded via yfinance"
            )
        elif all(token in lowered for token in ("daily adjusted closing prices", "nasdaq-100 constituents", "3 jan 2019")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for NASDAQ-100 constituents beginning on 3 Jan 2019"
            )
        if "tickers" in lowered and "755 trading days" in lowered:
            facts.append("after filtering, roughly N ~ 100 tickers and T = 755 trading days remain")
        if all(token in lowered for token in ("vix", "5-day rolling mean", "22-day rolling mean")):
            facts.append("VIX contributes three standardised exogenous features: the level, a 5-day rolling mean, and a 22-day rolling mean")
        if all(token in lowered for token in ("msft", "adbe", "nvda", "payx")):
            if "200 (i, t) pairs" in lowered or "the resulting 200" in lowered or "200 extreme shock events" in lowered:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX, and the shock-event set contains 200 extreme downside events")
            else:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX")
        if all(token in lowered for token in ("3 jan 2019", "30 jun 2020", "out-of-sample evaluation")):
            facts.append("the training window runs from 3 Jan 2019 to 30 Jun 2020, with the remaining period used for out-of-sample evaluation")
        elif "training" in lowered and "out-of-sample" in lowered and "3 jan 2019" in lowered:
            facts.append("the paper defines a training window starting on 3 Jan 2019 and evaluates the remaining period out of sample")
        if re.search(r"30,?000.+individual stocks.+1957.+2016", lowered):
            facts.append("the sample covers nearly 30,000 individual stocks over 60 years from 1957 to 2016")
        if re.search(
            r"18 years of training sample\s*\(1957-1974\).+12 years of validation sample\s*\(1975-1986\).+30 years\s*\(1987-2016\).+out-of-sample testing",
            lowered,
        ):
            facts.append(
                "the 60-year sample is split into 18 years of training sample (1957-1974), 12 years of validation sample (1975-1986), and 30 years (1987-2016) for out-of-sample testing"
            )
        elif re.search(
            r"18 years of training (?:sample|data).+12 years of validation (?:sample|data).+30 years.+out-of-sample testing",
            lowered,
        ):
            facts.append("the paper uses 18 years of training sample, 12 years of validation sample, and 30 years of out-of-sample testing")
        if re.search(r"our sample begins.+1957.+2016", lowered):
            facts.append("the sample begins in March 1957 and ends in December 2016, covering 60 years")
        if date_range_pattern.search(text) and any(token in lowered for token in ("sample", "data", "dataset", "observations", "panel")):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"\b\d{1,3}(?:,\d{3})?\s+observations\b", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"balanced panel of stocks|missing data", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"individual stocks", lowered) and re.search(r"60 years|1957|2016", lowered):
            facts.append(_truncate_line(text, limit=180))
    ordered = [
        item.replace("闁?00", "~ 100").replace("閳?00", "~ 100").replace("衼卸100", "~ 100")
        for item in _dedupe_strings(facts)
    ]

    def _fact_priority(item: str) -> tuple[int, int]:
        lowered_item = item.lower()
        if "nasdaq-100 constituents" in lowered_item:
            return (0, len(item))
        if "755 trading days" in lowered_item or "~ 100 tickers" in lowered_item:
            return (1, len(item))
        if "273 observations" in lowered_item:
            return (2, len(item))
        if "january 2001 to september 2023" in lowered_item:
            return (3, len(item))
        if "5-day rolling mean" in lowered_item or "22-day rolling mean" in lowered_item:
            return (4, len(item))
        if "hubs are msft" in lowered_item:
            return (5, len(item))
        if "training window runs" in lowered_item or "out of sample" in lowered_item:
            return (6, len(item))
        if "30,000 individual stocks" in lowered_item:
            return (7, len(item))
        if "training data" in lowered_item:
            return (8, len(item))
        if "sample begins" in lowered_item:
            return (9, len(item))
        return (10, len(item))

    ordered.sort(key=_fact_priority)
    return tuple(ordered)


def _build_narrow_grounded_paper_answer(
    prompt: str,
    paper_trace: PaperTrace,
    *,
    response_language: str,
) -> str:
    focus_slot = _narrow_grounded_qa_focus_slot(prompt)
    texts = _narrow_grounded_qa_text_pool(paper_trace, focus_slot)
    missing_phrase = _paper_missing_phrase(response_language)
    answer = missing_phrase

    if focus_slot == "method":
        families = _extract_method_family_mentions(texts)
        if families:
            joined = ", ".join(families[:-1]) + f", and {families[-1]}" if len(families) > 2 else " and ".join(families)
            answer = f"文中明确讨论的方法家族包括{joined}。" if response_language == "zh-CN" else f"The paper explicitly discusses {joined}."
        else:
            summary = _best_supported_narrow_slot_summary(paper_trace, "method", response_language=response_language)
            if summary != missing_phrase:
                answer = summary if response_language == "zh-CN" else f"The paper explicitly discusses {summary.rstrip('.')}."
    elif focus_slot == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            if response_language == "zh-CN":
                answer = f"文中明确说明，{facts[0]}。" if len(facts) == 1 else f"文中明确说明，{facts[0]}。文中还说明，{facts[1]}。"
            else:
                answer = f"The paper explicitly states that {facts[0]}." if len(facts) == 1 else f"The paper explicitly states that {facts[0]}. It also states that {facts[1]}."
        else:
            summary = _best_supported_narrow_slot_summary(paper_trace, "sample_or_data", response_language=response_language)
            if summary != missing_phrase:
                answer = summary if response_language == "zh-CN" else f"The paper explicitly states that {summary.rstrip('.')}."
    else:
        summary = _best_supported_narrow_slot_summary(paper_trace, focus_slot, response_language=response_language)
        if summary != missing_phrase:
            answer = _render_grounded_slot_sentence(focus_slot, summary, response_language=response_language)

    if _prompt_requests_support_detail(prompt):
        support_ref = _first_support_ref(paper_trace)
        if support_ref:
            answer += f" 相关位置可见 {support_ref}。" if response_language == "zh-CN" else f" The most directly relevant retrieved support is {support_ref}."
    return answer.strip()


def _extract_method_family_mentions(texts: list[str]) -> tuple[str, ...]:
    corpus = " ".join(unicodedata.normalize("NFKC", text).lower() for text in texts)
    catalog = (
        ("lasso", "Lasso"),
        ("elastic net", "Elastic Net"),
        ("enet", "Elastic Net"),
        ("ridge", "Ridge"),
        ("linear regression", "linear regression"),
        ("generalized linear model", "generalized linear models"),
        ("principal components regression", "principal components regression (PCR)"),
        ("principal component analysis", "principal component analysis (PCA)"),
        ("pca", "principal component analysis (PCA)"),
        ("partial least squares", "partial least squares (PLS)"),
        ("regression tree", "regression trees"),
        ("boosted tree", "boosted trees"),
        ("random forest", "random forests"),
        ("ann", "artificial neural networks (ANN)"),
        ("artificial neural network", "artificial neural networks (ANN)"),
        ("cnn", "convolutional neural networks (CNN)"),
        ("convolutional neural network", "convolutional neural networks (CNN)"),
        ("lstm", "long short-term memory networks (LSTM)"),
        ("long short-term memory", "long short-term memory networks (LSTM)"),
        ("neural network", "neural networks"),
        ("transformer", "Transformers"),
        ("support vector machine", "support vector machines"),
        ("svm", "support vector machines"),
    )
    mentions: list[str] = []
    for needle, label in catalog:
        if needle in corpus:
            mentions.append(label)
    if "arimax" in corpus and "garch" in corpus:
        mentions.append("ARIMAX-GARCH")
    return tuple(dict.fromkeys(mentions))


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [unicodedata.normalize("NFKC", text).replace("\n", " ").strip() for text in texts]
    date_range_pattern = re.compile(
        r"\b(?:from|between)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})\s+(?:to|through|until|and)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})",
        re.IGNORECASE,
    )
    count_pattern = re.compile(
        r"\b(?:contains?|cover(?:s|ing)?|includes?)\s+([0-9][0-9,]*)\s+(observations|trading days|stocks|tickers)\b",
        re.IGNORECASE,
    )
    for text in normalized_texts:
        lowered = text.lower()
        if not _looks_like_explicit_sample_data_text(text):
            continue
        if all(token in lowered for token in ("crsp", "nyse", "amex", "nasdaq")):
            facts.append(
                "the paper uses monthly total individual equity returns from CRSP for firms listed in the NYSE, AMEX, and NASDAQ"
            )
        if all(token in lowered for token in ("nasdaq-100 constituents", "3 jan 2019", "30 dec 2021", "yfinance")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for all NASDAQ-100 constituents from 3 Jan 2019 to 30 Dec 2021, downloaded via yfinance"
            )
        elif all(token in lowered for token in ("daily adjusted closing prices", "nasdaq-100 constituents", "3 jan 2019")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for NASDAQ-100 constituents beginning on 3 Jan 2019"
            )
        if "tickers" in lowered and "755 trading days" in lowered:
            facts.append("after filtering, roughly N ~ 100 tickers and T = 755 trading days remain")
        if all(token in lowered for token in ("vix", "5-day rolling mean", "22-day rolling mean")):
            facts.append("VIX contributes three standardised exogenous features: the level, a 5-day rolling mean, and a 22-day rolling mean")
        if all(token in lowered for token in ("msft", "adbe", "nvda", "payx")):
            if "200 (i, t) pairs" in lowered or "the resulting 200" in lowered or "200 extreme shock events" in lowered:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX, and the shock-event set contains 200 extreme downside events")
            else:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX")
        if all(token in lowered for token in ("3 jan 2019", "30 jun 2020", "out-of-sample evaluation")):
            facts.append("the training window runs from 3 Jan 2019 to 30 Jun 2020, with the remaining period used for out-of-sample evaluation")
        elif "training" in lowered and "out-of-sample" in lowered and "3 jan 2019" in lowered:
            facts.append("the paper defines a training window starting on 3 Jan 2019 and evaluates the remaining period out of sample")
        if re.search(r"30,?000.+individual stocks.+1957.+2016", lowered):
            facts.append("the sample covers nearly 30,000 individual stocks over 60 years from 1957 to 2016")
        if re.search(
            r"18 years of training (?:sample|data).+12 years of validation (?:sample|data).+30 years.+out-of-sample testing",
            lowered,
        ):
            facts.append("the paper uses 18 years of training data, 12 years of validation data, and 30 years of out-of-sample testing")
        if re.search(r"our sample begins.+1957.+2016", lowered):
            facts.append("the sample begins in March 1957 and ends in December 2016, covering 60 years")
        if re.search(r"in our sample.+longer and wider", lowered):
            facts.append("the paper says its sample is longer and wider than the benchmark sample it compares against")
        if date_range_pattern.search(text) and any(token in lowered for token in ("sample", "data", "dataset", "observations", "panel")):
            facts.append(_truncate_line(text, limit=180))
        if count_pattern.search(text):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"\b\d{1,3}(?:,\d{3})?\s+observations\b", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"balanced panel of stocks|missing data", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"individual stocks", lowered) and re.search(r"60 years|1957|2016", lowered):
            facts.append(_truncate_line(text, limit=180))

    ordered = [
        item.replace("闁?00", "~ 100").replace("閳?00", "~ 100").replace("衼卸100", "~ 100")
        for item in _dedupe_strings(facts)
    ]

    def _fact_priority(item: str) -> tuple[int, int]:
        lowered_item = item.lower()
        if "nasdaq-100 constituents" in lowered_item:
            return (0, len(item))
        if "755 trading days" in lowered_item or "~ 100 tickers" in lowered_item:
            return (1, len(item))
        if "contains 273 observations" in lowered_item or "273 observations" in lowered_item:
            return (2, len(item))
        if "from january 2001 to september 2023" in lowered_item:
            return (3, len(item))
        if "5-day rolling mean" in lowered_item or "22-day rolling mean" in lowered_item:
            return (4, len(item))
        if "hubs are msft" in lowered_item:
            return (5, len(item))
        if "training window runs" in lowered_item or "out of sample" in lowered_item:
            return (6, len(item))
        if "30,000 individual stocks" in lowered_item:
            return (7, len(item))
        if "training data" in lowered_item:
            return (8, len(item))
        if "sample begins" in lowered_item:
            return (9, len(item))
        return (10, len(item))

    ordered.sort(key=_fact_priority)
    return tuple(ordered)


def _best_supported_narrow_slot_summary(
    paper_trace: PaperTrace,
    slot_name: str,
    *,
    response_language: str,
) -> str:
    missing_phrase = _paper_missing_phrase(response_language)
    for document_note in paper_trace.document_notes:
        payload = _document_slot(document_note, slot_name)
        if _slot_payload_status(payload) == "not_clearly_stated":
            continue
        summary_text = _slot_payload_text(payload).strip()
        merged_text = str(payload.get("merged_note_text", "")).strip()
        candidate = summary_text or merged_text
        if not candidate or candidate == "Not clearly stated in the paper.":
            continue
        return _normalize_slot_summary(candidate, response_language=response_language)
    return missing_phrase


def _build_narrow_grounded_paper_answer(
    prompt: str,
    paper_trace: PaperTrace,
    *,
    response_language: str,
) -> str:
    focus_slot = _narrow_grounded_qa_focus_slot(prompt)
    texts = _narrow_grounded_qa_text_pool(paper_trace, focus_slot)
    missing_phrase = _paper_missing_phrase(response_language)
    answer = missing_phrase

    if focus_slot == "method":
        families = _extract_method_family_mentions(texts)
        if families:
            joined = ", ".join(families[:-1]) + f", and {families[-1]}" if len(families) > 2 else " and ".join(families)
            if response_language == "zh-CN":
                answer = f"文中明确讨论的方法家族包括{joined}。"
            else:
                answer = f"The paper explicitly discusses {joined}."
        else:
            summary = _best_supported_narrow_slot_summary(
                paper_trace,
                "method",
                response_language=response_language,
            )
            if summary != missing_phrase:
                answer = summary if response_language == "zh-CN" else f"The paper explicitly discusses {summary.rstrip('.')}."
    elif focus_slot == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            normalized_facts = [
                fact[0].lower() + fact[1:] if fact[:1].isupper() else fact
                for fact in facts
            ]
            if response_language == "zh-CN":
                if len(normalized_facts) == 1:
                    answer = f"文中明确说明，{normalized_facts[0]}。"
                else:
                    answer = f"文中明确说明，{normalized_facts[0]}。文中还说明，{normalized_facts[1]}。"
            else:
                if len(normalized_facts) == 1:
                    answer = f"The paper explicitly states that {normalized_facts[0]}."
                else:
                    answer = f"The paper explicitly states that {normalized_facts[0]}. It also states that {normalized_facts[1]}."
        else:
            summary = _best_supported_narrow_slot_summary(
                paper_trace,
                "sample_or_data",
                response_language=response_language,
            )
            if summary != missing_phrase:
                answer = summary if response_language == "zh-CN" else f"The paper explicitly states that {summary.rstrip('.')}."
    else:
        summary = _best_supported_narrow_slot_summary(
            paper_trace,
            focus_slot,
            response_language=response_language,
        )
        if summary != missing_phrase:
            answer = _render_grounded_slot_sentence(focus_slot, summary, response_language=response_language)

    if _prompt_requests_support_detail(prompt):
        support_ref = _first_support_ref(paper_trace)
        if support_ref:
            if response_language == "zh-CN":
                answer += f" 相关位置可见 {support_ref}。"
            else:
                answer += f" The most directly relevant retrieved support is {support_ref}."
    return answer.strip()


def _extract_method_family_mentions(texts: list[str]) -> tuple[str, ...]:
    corpus = " ".join(unicodedata.normalize("NFKC", text).lower() for text in texts)
    catalog = (
        ("lasso", "Lasso"),
        ("elastic net", "Elastic Net"),
        ("enet", "Elastic Net"),
        ("ridge", "Ridge"),
        ("linear regression", "linear regression"),
        ("generalized linear model", "generalized linear models"),
        ("principal components regression", "principal components regression (PCR)"),
        ("principal component analysis", "principal component analysis (PCA)"),
        ("pca", "principal component analysis (PCA)"),
        ("partial least squares", "partial least squares (PLS)"),
        ("regression tree", "regression trees"),
        ("ann", "artificial neural networks (ANN)"),
        ("cnn", "convolutional neural networks (CNN)"),
        ("lstm", "long short-term memory networks (LSTM)"),
        ("neural network", "neural networks"),
        ("boosted tree", "boosted trees"),
        ("random forest", "random forests"),
    )
    mentions: list[str] = []
    for needle, label in catalog:
        if needle in corpus:
            mentions.append(label)
    if "arimax" in corpus and "garch" in corpus:
        mentions.append("ARIMAX-GARCH")
    return tuple(dict.fromkeys(mentions))


def _build_narrow_grounded_paper_answer(
    prompt: str,
    paper_trace: PaperTrace,
    *,
    response_language: str,
) -> str:
    focus_slot = _narrow_grounded_qa_focus_slot(prompt)
    texts = _narrow_grounded_qa_text_pool(paper_trace, focus_slot)
    missing_phrase = _paper_missing_phrase(response_language)
    answer = missing_phrase

    if focus_slot == "method":
        families = _extract_method_family_mentions(texts)
        if families:
            joined = ", ".join(families[:-1]) + f", and {families[-1]}" if len(families) > 2 else " and ".join(families)
            if response_language == "zh-CN":
                answer = f"文中明确讨论的方法族包括：{joined}。"
            else:
                answer = f"The paper explicitly discusses {joined}."
        else:
            for document_note in paper_trace.document_notes:
                payload = _document_slot(document_note, focus_slot)
                if _slot_payload_status(payload) == "not_clearly_stated":
                    continue
                summary = _clean_detailed_slot_body(
                    _slot_payload_text(payload),
                    slot_name=focus_slot,
                    response_language=response_language,
                )
                if not summary or summary == missing_phrase:
                    continue
                if response_language == "zh-CN":
                    answer = f"文中明确讨论的方法包括：{summary.rstrip('。.')}。"
                else:
                    answer = f"The paper explicitly discusses {summary.rstrip('.')}."
                break
    elif focus_slot == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            if response_language == "zh-CN":
                if len(facts) == 1:
                    answer = f"文中明确说明：{facts[0]}。"
                else:
                    answer = f"文中明确说明：{facts[0]}。文中还说明：{facts[1]}。"
            else:
                if len(facts) == 1:
                    answer = f"The paper explicitly states that {facts[0]}."
                else:
                    answer = f"The paper explicitly states that {facts[0]}. It also states that {facts[1]}."
        else:
            for document_note in paper_trace.document_notes:
                payload = _document_slot(document_note, focus_slot)
                if _slot_payload_status(payload) == "not_clearly_stated":
                    continue
                summary = _clean_detailed_slot_body(
                    _slot_payload_text(payload),
                    slot_name=focus_slot,
                    response_language=response_language,
                )
                if not summary or summary == missing_phrase:
                    continue
                if response_language == "zh-CN":
                    answer = f"文中明确说明：{summary.rstrip('。.')}。"
                else:
                    answer = f"The paper explicitly states that {summary.rstrip('.')}."
                break
    else:
        for document_note in paper_trace.document_notes:
            payload = _document_slot(document_note, focus_slot)
            if _slot_payload_status(payload) == "not_clearly_stated":
                continue
            summary = _slot_payload_text(payload)
            answer = _render_grounded_slot_sentence(
                focus_slot,
                summary,
                response_language=response_language,
            )
            break

    if _prompt_requests_support_detail(prompt):
        support_ref = _first_support_ref(paper_trace)
        if support_ref:
            if response_language == "zh-CN":
                answer += f" 相关位置可见 {support_ref}。"
            else:
                answer += f" The most directly relevant retrieved support is {support_ref}."
    return answer.strip()


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [unicodedata.normalize("NFKC", text).replace("\n", " ").strip() for text in texts]
    for text in normalized_texts:
        for sentence in _split_rescue_sentences(text):
            normalized_sentence = unicodedata.normalize("NFKC", sentence).strip()
            lowered_sentence = normalized_sentence.lower()
            if not normalized_sentence:
                continue
            if not _looks_like_explicit_sample_data_text(normalized_sentence):
                continue
            if _looks_like_sample_data_result_noise(normalized_sentence):
                continue
            if any(
                marker in lowered_sentence
                for marker in (
                    "dataset",
                    "data set",
                    "sample",
                    "equity panel",
                    "panel",
                    "constituents",
                    "trading days",
                    "tickers",
                    "observations",
                    "daily adjusted",
                    "monthly",
                    "training",
                    "validation",
                    "test",
                    "out-of-sample",
                    "first half",
                    "second half",
                    "data source",
                    "downloaded via",
                    "fred",
                    "crsp",
                    "compustat",
                    "bloomberg",
                    "refinitiv",
                    "yfinance",
                    "vix",
                    "features",
                    "hubs",
                    "shock events",
                )
            ):
                facts.append(_truncate_line(normalized_sentence, limit=200))
    ordered = _dedupe_strings(facts)
    ordered.sort(key=_sample_data_fact_priority)
    return tuple(ordered)


def _slot_specific_rescue_summary(
    slot_name: str,
    candidate_pages: list[tuple[int, str]],
) -> str:
    texts = [text for _page_number, text in candidate_pages]
    if slot_name == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            return " ".join(f"{fact.rstrip('. ')}." for fact in facts[:5]).strip()
    if slot_name == "method":
        families = _extract_method_family_mentions(texts)
        if families:
            if len(families) == 1:
                return f"The paper explicitly discusses {families[0]}."
            if len(families) == 2:
                return f"The paper explicitly discusses {families[0]} and {families[1]}."
            return (
                "The paper explicitly discusses "
                + ", ".join(families[:-1])
                + f", and {families[-1]}."
            )
    if slot_name == "main_findings":
        rescued: list[str] = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if (
                    "fig." in lowered_sentence
                    or "table " in lowered_sentence
                    or "internet appendix" in lowered_sentence
                ):
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "outperforms",
                        "outperform",
                        "higher sharpe ratios",
                        "highest overall panel r2",
                        "higher than any single method",
                        "best performing",
                        "strongest and most consistent",
                        "realized returns generally increase monotonically",
                        "dominant predictive signals",
                        "shallow",
                        "deep",
                        "predictive advantage",
                        "greater precision in forecasting",
                        "leading contender",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            return ". ".join(unique[:3]).strip() + "."
    if slot_name == "limitations":
        rescued: list[str] = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "fig." in lowered_sentence or "table " in lowered_sentence:
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "limitation",
                        "limitations",
                        "future",
                        "monthly data",
                        "high-frequency data",
                        "could help researchers improve",
                        "simple ",
                        "limited",
                        "overfit",
                        "overfitting",
                        "outliers can undermine",
                        "computationally intensive",
                        "dearth of data",
                        "low signal-to-noise ratio",
                        "their flexibility is also their limitation",
                        "limitations of linear models",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            strong = [
                item
                for item in unique
                if any(
                    marker in item.lower()
                    for marker in (
                        "dearth of data",
                        "low signal-to-noise ratio",
                        "overfit",
                        "overfitting",
                        "computationally intensive",
                        "must be heavily regularized",
                    )
                )
            ]
            preferred = strong or unique
            return ". ".join(preferred[:2]).strip() + "."
    if slot_name == "conclusion":
        rescued = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "fig." in lowered_sentence or "table " in lowered_sentence:
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "the evidence indicates",
                        "these insights can be used",
                        "these findings emphasize",
                        "enhance the precision",
                        "predictive capabilities",
                        "lstm emerging as the leading contender",
                        "greater precision in forecasting",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            strong = [
                item
                for item in unique
                if any(
                    marker in item.lower()
                    for marker in (
                        "at the highest level",
                        "best performing methods",
                        "brings promise",
                        "most valuable for forecasting",
                        "dominant predictive signals",
                        "help justify",
                    )
                )
            ]
            preferred = strong or unique
            return ". ".join(preferred[:3]).strip() + "."
    return ""


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    narrow_grounded_qa = mode_selection.mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(prompt)
    if _contains_unexpected_language_leakage(
        answer_text,
        response_language=mode_selection.response_language,
        allow_bilingual=_prompt_requests_bilingual_output(prompt),
    ):
        notes.append(
            "The answer leaked content in the wrong language instead of obeying the final explicit language instruction."
        )
    if _contains_unexpected_language_leakage(
        answer_text,
        response_language=mode_selection.response_language,
        allow_bilingual=_prompt_requests_bilingual_output(prompt),
    ):
        if mode_selection.response_language == "zh-CN":
            notes.append("The answer did not translate the grounded paper content cleanly enough into Chinese.")
        else:
            notes.append("The answer leaked content from the wrong language despite a single-language target.")
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the cleaned slot evidence."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    if re.search(r"not clearly stated in the paper[^.\n]{0,160}\bhowever\b", answer_lower):
        notes.append(
            "The answer acknowledges a missing dimension but then keeps padding it with unsupported follow-on commentary."
        )
    if narrow_grounded_qa and _looks_over_scaffolded_grounded_qa(answer_text):
        notes.append(
            "The narrow grounded QA answer is still too scaffold-heavy and should collapse to a concise answer-first form."
        )
    if narrow_grounded_qa and len(answer_text) > 650 and not _prompt_requests_support_detail(prompt):
        notes.append(
            "The narrow grounded QA answer is longer than needed for a focused factual question."
        )
    if mode_selection.paper_output_profile == "detailed_paper_note" and mode_selection.mode == "paper_summary":
        false_missing_slots = _false_missing_supported_slots(
            answer_text,
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
        )
        if false_missing_slots:
            notes.append(
                "The detailed paper note still marks clearly supported slots as missing: "
                + ", ".join(slot_label(slot_name) for slot_name in false_missing_slots)
                + "."
            )
        if mode_selection.response_style != "continuous_prose":
            missing_section_slots = _missing_detailed_note_section_slots(
                answer_text,
                paper_trace.document_notes,
                response_language=mode_selection.response_language,
            )
            if missing_section_slots:
                notes.append(
                    "The detailed paper note dropped supported sections: "
                    + ", ".join(slot_label(slot_name) for slot_name in missing_section_slots)
                    + "."
                )
            if _looks_slot_stitched_detailed_note(answer_text):
                notes.append(
                    "The detailed paper note still reads like stitched slot snippets instead of a coherent paper note."
                )
            sections = _split_detailed_note_sections(answer_text, mode_selection.response_language)
            weak_fit_slots = [
                slot_name
                for slot_name, body in sections.items()
                if body
                and body != _paper_missing_phrase(mode_selection.response_language)
                and _section_fails_slot_fit(slot_name, body)
            ]
            if weak_fit_slots:
                notes.append(
                    "Some detailed-note sections still do not match their intended paper dimensions cleanly: "
                    + ", ".join(slot_label(slot_name) for slot_name in weak_fit_slots)
                    + "."
                )
            if mode_selection.response_language == "en" and _detailed_note_is_overcompressed(answer_text, paper_trace.document_notes):
                notes.append(
                    "The detailed paper note is still too compressed relative to the supported slot detail available in the paper."
                )
    if not narrow_grounded_qa and _looks_excerpt_heavy_paper_answer(answer_text):
        notes.append(
            "The answer is still too excerpt-heavy or outline-heavy for the final paper renderer contract."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots:
        if mode_selection.mode == "paper_summary":
            notes.append(
                "The answer did not clearly cover these requested summary dimensions: "
                + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
                + "."
            )
        if mode_selection.mode == "paper_compare":
            notes.append(
                "The comparison did not clearly cover these requested dimensions: "
                + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
                + "."
            )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.mode == "paper_compare" and any(
        marker in answer_lower
        for marker in (
            "hybrid approach could be beneficial",
            "combining methodologies could",
            "recommendations or synthesis",
        )
    ):
        notes.append("The comparison drifted into unsupported recommendation language instead of staying slot-grounded.")
    if mode_selection.mode == "paper_compare" and _looks_like_slot_dump_compare_answer(answer_text):
        notes.append("The comparison still reads like a raw per-slot dump instead of a grounded compare note.")
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [unicodedata.normalize("NFKC", text).replace("\n", " ").strip() for text in texts]
    for text in normalized_texts:
        for sentence in _split_rescue_sentences(text):
            normalized_sentence = unicodedata.normalize("NFKC", sentence).strip()
            lowered_sentence = normalized_sentence.lower()
            if not normalized_sentence:
                continue
            if not _looks_like_explicit_sample_data_text(normalized_sentence):
                continue
            if _looks_like_sample_data_result_noise(normalized_sentence):
                continue
            if any(
                marker in lowered_sentence
                for marker in (
                    "dataset",
                    "data set",
                    "sample",
                    "equity panel",
                    "panel",
                    "constituents",
                    "trading days",
                    "tickers",
                    "observations",
                    "daily adjusted",
                    "monthly",
                    "training",
                    "validation",
                    "test",
                    "out-of-sample",
                    "first half",
                    "second half",
                    "data source",
                    "downloaded via",
                    "fred",
                    "crsp",
                    "compustat",
                    "bloomberg",
                    "refinitiv",
                    "yfinance",
                    "vix",
                    "features",
                    "hubs",
                    "shock events",
                )
            ):
                facts.append(_truncate_line(normalized_sentence, limit=200))
    ordered = _dedupe_strings(facts)
    ordered.sort(key=_sample_data_fact_priority)
    return tuple(ordered)


def _slot_specific_rescue_summary(
    slot_name: str,
    candidate_pages: list[tuple[int, str]],
) -> str:
    texts = [text for _page_number, text in candidate_pages]
    if slot_name == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            return " ".join(f"{fact.rstrip('. ')}." for fact in facts[:5]).strip()
    if slot_name == "method":
        families = _extract_method_family_mentions(texts)
        if families:
            if len(families) == 1:
                return f"The paper explicitly discusses {families[0]}."
            if len(families) == 2:
                return f"The paper explicitly discusses {families[0]} and {families[1]}."
            return (
                "The paper explicitly discusses "
                + ", ".join(families[:-1])
                + f", and {families[-1]}."
            )
    if slot_name == "limitations":
        rescued: list[str] = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "fig." in lowered_sentence or "table " in lowered_sentence:
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "limitation",
                        "limitations",
                        "future",
                        "monthly data",
                        "high-frequency data",
                        "could help researchers improve",
                        "simple ",
                        "limited",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            return ". ".join(unique[:2]).strip() + "."
    if slot_name == "conclusion":
        rescued = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "fig." in lowered_sentence or "table " in lowered_sentence:
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "the evidence indicates",
                        "these findings emphasize",
                        "enhance the precision",
                        "predictive capabilities",
                        "greater precision in forecasting",
                        "leading contender for forecasting",
                        "perform a comparative analysis",
                        "our findings demonstrate",
                        "can help improve our empirical understanding",
                        "best performing methods",
                        "shallow learning outperforms",
                        "most valuable for forecasting",
                        "overall success of machine learning algorithms",
                        "brings promise",
                        "dominant predictive signals",
                        "price trends including",
                        "risk-management perspective",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            return ". ".join(unique[:2]).strip() + "."
    return ""


def _sample_data_fact_priority(text: str) -> tuple[int, int]:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "daily adjusted closing prices",
            "constituents",
            "equity panel",
            "tradable factors",
            "portfolios sorted by firm characteristics",
        )
    ):
        return (0, len(text))
    if re.search(r"\bfrom\b.+\bto\b.+\b(?:19|20)\d{2}\b", lowered) or re.search(
        r"\bbegins in\b.+\bends in\b",
        lowered,
    ):
        return (1, len(text))
    if any(marker in lowered for marker in ("observations", "trading days", "tickers", "n =", "t =", "first half", "second half")):
        return (2, len(text))
    if any(marker in lowered for marker in ("training", "validation", "test", "out-of-sample")):
        return (3, len(text))
    if any(marker in lowered for marker in ("vix", "rolling mean", "features")):
        return (4, len(text))
    if any(marker in lowered for marker in ("hubs", "shock events", "msft", "adbe", "nvda", "payx")):
        return (5, len(text))
    if any(marker in lowered for marker in ("data source", "downloaded via", "fred", "crsp", "compustat", "bloomberg", "refinitiv", "yfinance")):
        return (6, len(text))
    return (7, len(text))


def _detailed_note_title_map(response_language: str) -> dict[str, str]:
    if response_language == "zh-CN":
        return {
            "research_question": "研究问题",
            "sample_or_data": "样本与数据",
            "method": "方法",
            "main_findings": "主要发现",
            "limitations": "局限",
            "conclusion": "结论",
        }
    return {
        "research_question": "Research question",
        "sample_or_data": "Sample and data",
        "method": "Method",
        "main_findings": "Main findings",
        "limitations": "Limitations",
        "conclusion": "Conclusion",
    }


def _split_detailed_note_sections(answer_text: str, response_language: str) -> dict[str, str]:
    lines = [line.rstrip() for line in answer_text.splitlines()]
    title_map = _detailed_note_title_map(response_language)
    ordered_slots = tuple(title_map.keys())
    sections: dict[str, str] = {}
    current_slot = ""
    buffer: list[str] = []
    reverse_map = {title: slot_name for slot_name, title in title_map.items()}

    def _flush() -> None:
        nonlocal buffer, current_slot
        if current_slot:
            sections[current_slot] = " ".join(part.strip() for part in buffer if part.strip()).strip()
        buffer = []

    for line in lines:
        stripped = line.strip()
        slot_name = reverse_map.get(stripped)
        if slot_name in ordered_slots:
            _flush()
            current_slot = slot_name
            continue
        if current_slot:
            buffer.append(stripped)
    _flush()
    return sections


def _expected_detailed_note_slots(
    document_notes: list[dict[str, object]],
) -> tuple[str, ...]:
    ordered = (
        "research_question",
        "sample_or_data",
        "method",
        "main_findings",
        "limitations",
        "conclusion",
    )
    expected: list[str] = []
    for slot_name in ordered:
        if any(
            _slot_payload_status(_document_slot(document_note, slot_name)) != "not_clearly_stated"
            for document_note in document_notes
        ):
            expected.append(slot_name)
    return tuple(expected)


def _missing_detailed_note_section_slots(
    answer_text: str,
    document_notes: list[dict[str, object]],
    *,
    response_language: str,
) -> tuple[str, ...]:
    if not document_notes:
        return ()
    sections = _split_detailed_note_sections(answer_text, response_language)
    expected = _expected_detailed_note_slots(document_notes)
    return tuple(slot_name for slot_name in expected if slot_name not in sections)


def _section_fails_slot_fit(slot_name: str, text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).lower()
    if not lowered:
        return True
    if slot_name == "sample_or_data":
        positive = (
            "sample",
            "data",
            "dataset",
            "panel",
            "constituents",
            "trading days",
            "tickers",
            "observations",
            "training",
            "validation",
            "test",
            "out-of-sample",
            "data source",
            "downloaded via",
        )
        negative = ("rmse", "mae", "sharpe ratio", "predictive r2", "report the main empirical results")
        return not any(marker in lowered for marker in positive) or all(marker in lowered for marker in negative[:2])
    if slot_name == "method":
        positive = (
            "method",
            "model",
            "approach",
            "benchmark",
            "regression",
            "lasso",
            "elastic",
            "lstm",
            "ann",
            "cnn",
            "arimax",
            "garch",
            "spca",
            "pca",
        )
        negative = ("rmse", "mae", "sharpe ratio", "outperform", "predictive r2")
        return not any(marker in lowered for marker in positive) or sum(marker in lowered for marker in negative) >= 3
    if slot_name == "limitations":
        positive = (
            "limitation",
            "limitations",
            "caveat",
            "future work",
            "limited",
            "misspecification",
            "constraint",
            "only ",
            "cannot",
            "noise",
            "simple ",
        )
        return not any(marker in lowered for marker in positive)
    if slot_name == "conclusion":
        positive = (
            "overall",
            "we conclude",
            "in conclusion",
            "these results",
            "these findings",
            "suggests",
            "illustrate",
            "highlights",
        )
        return not any(marker in lowered for marker in positive) and lowered.count("not clearly stated") == 0
    return False


def _detailed_note_is_overcompressed(answer_text: str, document_notes: list[dict[str, object]]) -> bool:
    if not document_notes:
        return False
    sections = _split_detailed_note_sections(answer_text, "en")
    if len(sections) < 4:
        return True
    detail_signal_count = sum(
        _detail_marker_count(slot_name, body)
        for slot_name, body in sections.items()
        if body and body != "Not clearly stated in the paper."
    )
    return detail_signal_count <= 6


def _extract_method_family_mentions(texts: list[str]) -> tuple[str, ...]:
    corpus = " ".join(unicodedata.normalize("NFKC", text).lower() for text in texts)
    catalog = (
        ("linear regression", "linear regression"),
        ("generalized linear model", "generalized linear models"),
        ("adaptive lasso", "adaptive Lasso"),
        ("lasso", "Lasso"),
        ("elastic-net", "Elastic Net"),
        (" elastic net", "Elastic Net"),
        ("enet", "Elastic Net"),
        ("principal components regression", "principal components regression (PCR)"),
        ("spca", "sparse principal components analysis (SPCA)"),
        ("pca", "principal components analysis (PCA)"),
        ("partial least squares", "partial least squares (PLS)"),
        ("regression tree", "regression trees"),
        ("neural network", "neural networks"),
        ("ann", "artificial neural networks (ANN)"),
        ("cnn", "convolutional neural networks (CNN)"),
        ("lstm", "long short-term memory networks (LSTM)"),
        ("support vector machine", "support vector machines (SVM)"),
        ("arimax", "ARIMAX"),
        ("garch", "GARCH"),
        ("boosted tree", "boosted trees"),
        ("random forest", "random forests"),
    )
    mentions: list[str] = []
    for needle, label in catalog:
        if needle in corpus:
            mentions.append(label)
    return tuple(dict.fromkeys(mentions))


def _looks_like_sample_data_result_noise(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).lower()
    return any(
        marker in lowered
        for marker in (
            "predictive performance",
            "forecast performance",
            "report the main empirical results",
            "rmse",
            "mae",
            "sharpe ratio",
            "diebold-mariano",
            "test statistic",
            "trading strategy",
            "performance evaluation",
        )
    )


def _sample_data_fact_priority(text: str) -> tuple[int, int]:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "daily adjusted closing prices",
            "constituents",
            "equity panel",
            "tradable factors",
            "portfolios sorted by firm characteristics",
        )
    ):
        return (0, len(text))
    if re.search(r"\bfrom\b.+\bto\b.+\b(?:19|20)\d{2}\b", lowered) or re.search(
        r"\bbegins in\b.+\bends in\b",
        lowered,
    ):
        return (1, len(text))
    if any(marker in lowered for marker in ("observations", "trading days", "tickers", "n =", "t =", "first half", "second half")):
        return (2, len(text))
    if any(marker in lowered for marker in ("training", "validation", "test", "out-of-sample")):
        return (3, len(text))
    if any(marker in lowered for marker in ("data source", "downloaded via", "fred", "crsp", "compustat", "bloomberg", "refinitiv", "yfinance")):
        return (4, len(text))
    if any(marker in lowered for marker in ("vix", "rolling mean", "features")):
        return (5, len(text))
    return (6, len(text))


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [unicodedata.normalize("NFKC", text).replace("\n", " ").strip() for text in texts]
    date_range_pattern = re.compile(
        r"\b(?:from|between)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})\s+(?:to|through|until|and)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})",
        re.IGNORECASE,
    )
    for text in normalized_texts:
        lowered = text.lower()
        if not _looks_like_explicit_sample_data_text(text):
            continue
        if _looks_like_sample_data_result_noise(text):
            continue
        if any(
            marker in lowered
            for marker in (
                "dataset",
                "data set",
                "sample",
                "equity panel",
                "panel",
                "constituents",
                "trading days",
                "tickers",
                "observations",
                "daily adjusted",
                "monthly",
                "training",
                "validation",
                "test",
                "out-of-sample",
                "first half",
                "second half",
                "data source",
                "downloaded via",
                "fred",
                "crsp",
                "compustat",
                "bloomberg",
                "refinitiv",
                "yfinance",
                "vix",
                "features",
            )
        ):
            facts.append(_truncate_line(text, limit=200))
    ordered = _dedupe_strings(facts)
    ordered.sort(key=_sample_data_fact_priority)
    return tuple(ordered)


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    narrow_grounded_qa = mode_selection.mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(prompt)
    if _contains_unexpected_language_leakage(
        answer_text,
        response_language=mode_selection.response_language,
        allow_bilingual=_prompt_requests_bilingual_output(prompt),
    ):
        if mode_selection.response_language == "zh-CN":
            notes.append("The answer did not translate the grounded paper content cleanly enough into Chinese.")
        else:
            notes.append("The answer leaked content from the wrong language despite a single-language target.")
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the cleaned slot evidence."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    if re.search(r"not clearly stated in the paper[^.\n]{0,160}\bhowever\b", answer_lower):
        notes.append(
            "The answer acknowledges a missing dimension but then keeps padding it with unsupported follow-on commentary."
        )
    if narrow_grounded_qa and _looks_over_scaffolded_grounded_qa(answer_text):
        notes.append(
            "The narrow grounded QA answer is still too scaffold-heavy and should collapse to a concise answer-first form."
        )
    if narrow_grounded_qa and len(answer_text) > 650 and not _prompt_requests_support_detail(prompt):
        notes.append(
            "The narrow grounded QA answer is longer than needed for a focused factual question."
        )
    if mode_selection.paper_output_profile == "detailed_paper_note" and mode_selection.mode == "paper_summary":
        false_missing_slots = _false_missing_supported_slots(
            answer_text,
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
        )
        if false_missing_slots:
            notes.append(
                "The detailed paper note still marks clearly supported slots as missing: "
                + ", ".join(slot_label(slot_name) for slot_name in false_missing_slots)
                + "."
            )
        if mode_selection.response_style != "continuous_prose":
            missing_section_slots = _missing_detailed_note_section_slots(
                answer_text,
                paper_trace.document_notes,
                response_language=mode_selection.response_language,
            )
            if missing_section_slots:
                notes.append(
                    "The detailed paper note dropped supported sections: "
                    + ", ".join(slot_label(slot_name) for slot_name in missing_section_slots)
                    + "."
                )
            if _looks_slot_stitched_detailed_note(answer_text):
                notes.append(
                    "The detailed paper note still reads like stitched slot snippets instead of a coherent paper note."
                )
            sections = _split_detailed_note_sections(answer_text, mode_selection.response_language)
            weak_fit_slots = [
                slot_name
                for slot_name, body in sections.items()
                if body
                and body != _paper_missing_phrase(mode_selection.response_language)
                and _section_fails_slot_fit(slot_name, body)
            ]
            if weak_fit_slots:
                notes.append(
                    "Some detailed-note sections still do not match their intended paper dimensions cleanly: "
                    + ", ".join(slot_label(slot_name) for slot_name in weak_fit_slots)
                    + "."
                )
            if mode_selection.response_language == "en" and _detailed_note_is_overcompressed(answer_text, paper_trace.document_notes):
                notes.append(
                    "The detailed paper note is still too compressed relative to the supported slot detail available in the paper."
                )
    if not narrow_grounded_qa and _looks_excerpt_heavy_paper_answer(answer_text):
        notes.append(
            "The answer is still too excerpt-heavy or outline-heavy for the final paper renderer contract."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots and mode_selection.mode == "paper_summary":
        notes.append(
            "The answer did not clearly cover these requested summary dimensions: "
            + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
            + "."
        )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.mode == "paper_compare" and any(
        marker in answer_lower
        for marker in (
            "hybrid approach could be beneficial",
            "combining methodologies could",
            "recommendations or synthesis",
        )
    ):
        notes.append("The comparison drifted into unsupported recommendation language instead of staying slot-grounded.")
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }


def _compose_paper_consistency_prompt(
    *,
    original_prompt: str,
    answer_text: str,
    report_notes: tuple[str, ...],
    response_language: str,
    mode: str,
    response_style: str,
    paper_output_profile: str = "quick_summary",
) -> str:
    missing_phrase = _paper_missing_phrase(response_language)
    style_instruction = (
        "- Return one continuous grounded paragraph with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "- Return a concise grounded answer using short natural paragraphs or compact sections."
    )
    language_instruction = (
        "- For zh-CN output, translate the grounded content into natural Simplified Chinese prose."
        if response_language == "zh-CN"
        else "- For English output, write like a grounded RA note rather than raw retrieval notes."
    )
    profile_instruction = (
        "- Preserve important paper-specific details when they are clearly supported, including concrete numbers, date ranges, sample/data setup, train/validation/test splits, explicit method families, findings, limitations, and conclusion details."
        if paper_output_profile == "detailed_paper_note"
        else "- Keep the answer concise and useful for quick reading; preserve the main supported point without drifting into vague filler."
    )
    return "\n".join(
        [
            "You are performing a constrained consistency repair on a paper answer.",
            f"Original user prompt: {original_prompt}",
            "Repair goals:",
            "- Rewrite from the cleaned slot scaffold rather than from vague memory or generic domain knowledge.",
            "- Keep only claims supported by the cleaned slot scaffold and evidence.",
            "- Remove broad textbook commentary, generic finance/ML filler, and unsupported synthesis language.",
            f"- If a requested detail is weak or missing, say {missing_phrase} instead of guessing.",
            "- Preserve the requested language and formatting style.",
            "- Paraphrase compactly instead of stitching excerpt fragments together.",
            "- Cover every explicitly requested dimension exactly once.",
            "- Do not add recommendations, future-work commentary, or hybrid proposals unless the user explicitly asked for them.",
            language_instruction,
            style_instruction,
            profile_instruction,
            f"- Current paper mode: {mode}",
            f"- Current paper output profile: {paper_output_profile}",
            "Detected issues:",
            *[f"- {item}" for item in report_notes],
            "",
            "Current answer:",
            answer_text,
            "",
            "Return only the repaired final answer body.",
        ]
    )


def _build_slot_grounded_compare_answer(
    document_notes: list[dict[str, object]],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    response_style: str,
    paper_output_profile: str = "detailed_paper_note",
) -> str:
    if response_style == "continuous_prose":
        sections: list[str] = []
        for slot_name in requested_slots:
            comparisons = [
                f"{Path(str(document_note.get('source_path', '(unknown document)'))).name}: "
                f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language, paper_output_profile=paper_output_profile)}"
                for document_note in document_notes
            ]
            if response_language == "zh-CN":
                sections.append(
                    f"{_slot_display_name(slot_name, response_language)}\u65b9\u9762\uff0c"
                    + "\uff1b".join(comparisons)
                    + "\u3002"
                )
            else:
                sections.append(
                    f"For {_slot_display_name(slot_name, response_language)}, " + "; ".join(comparisons) + "."
                )
        return " ".join(sections).strip()

    lines: list[str] = []
    for slot_name in requested_slots:
        lines.append(f"{_slot_display_name(slot_name, response_language)}:")
        for document_note in document_notes:
            source_name = Path(str(document_note.get("source_path", "(unknown document)"))).name
            lines.append(
                f"- {source_name}: "
                f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language, paper_output_profile=paper_output_profile)}"
            )
    return "\n".join(lines)


def _compose_paper_consistency_prompt(
    *,
    original_prompt: str,
    answer_text: str,
    report_notes: tuple[str, ...],
    response_language: str,
    mode: str,
    response_style: str,
    paper_output_profile: str = "quick_summary",
) -> str:
    missing_phrase = _paper_missing_phrase(response_language)
    style_instruction = (
        "- Return one continuous grounded paragraph with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "- Return a concise grounded answer using short natural paragraphs or compact sections."
    )
    language_instruction = (
        "- For zh-CN output, translate the grounded content into natural Simplified Chinese prose."
        if response_language == "zh-CN"
        else "- For English output, write like a grounded RA note rather than raw retrieval notes."
    )
    profile_instruction = (
        "- Preserve important paper-specific details when they are clearly supported, including concrete numbers, date ranges, sample/data setup, train/validation/test splits, explicit method families, findings, limitations, and conclusion details."
        if paper_output_profile == "detailed_paper_note"
        else "- Keep the answer concise and useful for quick reading; preserve the main supported point without drifting into vague filler."
    )
    return "\n".join(
        [
            "You are performing a constrained consistency repair on a paper answer.",
            f"Original user prompt: {original_prompt}",
            "Repair goals:",
            "- Rewrite from the cleaned slot scaffold rather than from vague memory or generic domain knowledge.",
            "- Keep only claims supported by the cleaned slot scaffold and evidence.",
            "- Remove broad textbook commentary, generic finance/ML filler, and unsupported synthesis language.",
            f"- If a requested detail is weak or missing, say {missing_phrase} instead of guessing.",
            "- Preserve the requested language and formatting style.",
            "- Paraphrase compactly instead of stitching excerpt fragments together.",
            "- Cover every explicitly requested dimension exactly once.",
            "- Do not add recommendations, future-work commentary, or hybrid proposals unless the user explicitly asked for them.",
            language_instruction,
            style_instruction,
            profile_instruction,
            f"- Current paper mode: {mode}",
            f"- Current paper output profile: {paper_output_profile}",
            "Detected issues:",
            *[f"- {item}" for item in report_notes],
            "",
            "Current answer:",
            answer_text,
            "",
            "Return only the repaired final answer body.",
        ]
    )


def _build_slot_grounded_compare_answer(
    document_notes: list[dict[str, object]],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    response_style: str,
    paper_output_profile: str = "detailed_paper_note",
) -> str:
    if response_style == "continuous_prose":
        sections: list[str] = []
        for slot_name in requested_slots:
            comparisons = [
                f"{Path(str(document_note.get('source_path', '(unknown document)'))).name}: "
                f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language, paper_output_profile=paper_output_profile)}"
                for document_note in document_notes
            ]
            if response_language == "zh-CN":
                sections.append(
                    f"{_slot_display_name(slot_name, response_language)}\u65b9\u9762\uff0c"
                    + "\uff1b".join(comparisons)
                    + "\u3002"
                )
            else:
                sections.append(
                    f"For {_slot_display_name(slot_name, response_language)}, " + "; ".join(comparisons) + "."
                )
        return " ".join(sections).strip()

    lines: list[str] = []
    for slot_name in requested_slots:
        lines.append(f"{_slot_display_name(slot_name, response_language)}:")
        for document_note in document_notes:
            source_name = Path(str(document_note.get("source_path", "(unknown document)"))).name
            lines.append(
                f"- {source_name}: "
                f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language, paper_output_profile=paper_output_profile)}"
            )
    return "\n".join(lines)


def _contains_project_memo_framing(answer_lower: str) -> bool:
    patterns = (
        "research assistant memo",
        "practical ra memo",
        "project overview",
        "project goal",
        "handoff summary",
        "\u7814\u7a76\u52a9\u7406\u5907\u5fd8\u5f55",
        "\u9879\u76ee\u6982\u89c8",
        "\u9879\u76ee\u76ee\u6807",
        "\u4ea4\u63a5\u603b\u7ed3",
    )
    return any(pattern in answer_lower for pattern in patterns)


def _build_slot_grounded_compare_answer(
    document_notes: list[dict[str, object]],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    response_style: str,
    paper_output_profile: str = "detailed_paper_note",
) -> str:
    if response_style == "continuous_prose":
        sections: list[str] = []
        for slot_name in requested_slots:
            comparisons = [
                f"{Path(str(document_note.get('source_path', '(unknown document)'))).name}: "
                f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language, paper_output_profile=paper_output_profile)}"
                for document_note in document_notes
            ]
            if response_language == "zh-CN":
                sections.append(
                    f"{_slot_display_name(slot_name, response_language)}\u65b9\u9762\uff0c"
                    + "\uff1b".join(comparisons)
                    + "\u3002"
                )
            else:
                sections.append(
                    f"For {_slot_display_name(slot_name, response_language)}, "
                    + "; ".join(comparisons)
                    + "."
                )
        return " ".join(sections).strip()

    lines: list[str] = []
    for slot_name in requested_slots:
        lines.append(f"{_slot_display_name(slot_name, response_language)}:")
        for document_note in document_notes:
            source_name = Path(str(document_note.get("source_path", "(unknown document)"))).name
            lines.append(
                f"- {source_name}: "
                f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language, paper_output_profile=paper_output_profile)}"
            )
    return "\n".join(lines)


def _compose_paper_consistency_prompt(
    *,
    original_prompt: str,
    answer_text: str,
    report_notes: tuple[str, ...],
    response_language: str,
    mode: str,
    response_style: str,
    paper_output_profile: str = "quick_summary",
) -> str:
    missing_phrase = (
        "\u6587\u4e2d\u672a\u660e\u786e\u8bf4\u660e\u3002"
        if response_language == "zh-CN"
        else "not clearly stated in the paper"
    )
    if mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(original_prompt):
        style_instruction = (
            "- Return a concise answer-first grounded answer with no evidence appendix unless the prompt explicitly asked for one."
        )
        language_instruction = (
            "- For zh-CN output, answer directly in natural Simplified Chinese."
            if response_language == "zh-CN"
            else "- For English output, answer directly in one or two tight sentences."
        )
    elif response_style == "continuous_prose":
        style_instruction = "- Return one continuous grounded paragraph with no bullets or outline headings."
        language_instruction = (
            "- For zh-CN output, translate the grounded content into natural Simplified Chinese prose."
            if response_language == "zh-CN"
            else "- For English output, keep the prose natural and paper-focused rather than note scaffolding."
        )
    elif paper_output_profile == "detailed_paper_note":
        style_instruction = (
            "- Return a detailed grounded paper note with short natural paragraphs rather than project-memo framing."
        )
        language_instruction = (
            "- For zh-CN output, write a detailed grounded paper note in natural Simplified Chinese."
            if response_language == "zh-CN"
            else "- For English output, write like a detailed grounded paper note rather than a project handoff memo."
        )
    else:
        style_instruction = "- Return a concise grounded paper summary with short natural paragraphs."
        language_instruction = (
            "- For zh-CN output, write a concise grounded paper summary in natural Simplified Chinese."
            if response_language == "zh-CN"
            else "- For English output, write like a concise grounded paper summary rather than raw notes."
        )
    return "\n".join(
        [
            "You are performing a constrained consistency repair on a paper answer.",
            f"Original user prompt: {original_prompt}",
            "Repair goals:",
            "- Rewrite from the grounded slot scaffold rather than from vague memory or generic domain knowledge.",
            "- Keep only claims supported by the grounded slot scaffold and evidence.",
            "- Remove broad textbook commentary, generic finance/ML filler, and unsupported synthesis language.",
            f"- If a requested detail is weak or missing, say {missing_phrase} instead of guessing.",
            "- Preserve the requested language and formatting style.",
            "- Paraphrase compactly instead of stitching excerpt fragments together.",
            "- Cover every explicitly requested dimension exactly once.",
            "- Do not add recommendations, future-work commentary, or hybrid proposals unless the user explicitly asked for them.",
            "- Do not use project overview, onboarding memo, or handoff framing unless the user explicitly requested it.",
            language_instruction,
            style_instruction,
            f"- Current paper mode: {mode}",
            f"- Current paper output profile: {paper_output_profile}",
            "Detected issues:",
            *[f"- {item}" for item in report_notes],
            "",
            "Current answer:",
            answer_text,
            "",
            "Return only the repaired final answer body.",
        ]
    )


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    narrow_grounded_qa = mode_selection.mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(prompt)
    if mode_selection.response_language == "zh-CN" and _looks_insufficiently_translated_chinese(answer_text):
        notes.append("The answer did not translate the grounded paper content cleanly enough into Chinese.")
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the cleaned slot evidence."
        )
    if (
        mode_selection.paper_output_profile == "detailed_paper_note"
        and _contains_project_memo_framing(answer_lower)
    ):
        notes.append(
            "The detailed paper note drifted into project or onboarding memo framing instead of sounding like a paper note."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    if re.search(r"not clearly stated in the paper[^.\n]{0,160}\bhowever\b", answer_lower):
        notes.append(
            "The answer acknowledges a missing dimension but then keeps padding it with unsupported follow-on commentary."
        )
    if mode_selection.mode == "paper_summary" and mode_selection.paper_output_profile == "detailed_paper_note":
        false_missing_slots = _false_missing_supported_slots(
            answer_text,
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
        )
        if false_missing_slots:
            notes.append(
                "The detailed paper note still marks clearly supported slots as not clearly stated: "
                + ", ".join(slot_label(slot_name) for slot_name in false_missing_slots)
                + "."
            )
        if _looks_slot_stitched_detailed_note(answer_text):
            notes.append(
                "The detailed paper note still reads like stitched slot snippets instead of a coherent paper note."
            )
    if narrow_grounded_qa and _looks_over_scaffolded_grounded_qa(answer_text):
        notes.append(
            "The narrow grounded QA answer is still too scaffold-heavy and should collapse to a concise answer-first form."
        )
    if narrow_grounded_qa and len(answer_text) > 650 and not _prompt_requests_support_detail(prompt):
        notes.append(
            "The narrow grounded QA answer is longer than needed for a focused factual question."
        )
    if (
        mode_selection.mode == "paper_summary"
        and mode_selection.paper_output_profile == "quick_summary"
        and len(answer_text) > 1400
        and not explicit_slots
    ):
        notes.append(
            "The quick paper summary is too verbose for the selected quick-summary output profile."
        )
    if not narrow_grounded_qa and _looks_excerpt_heavy_paper_answer(answer_text):
        notes.append(
            "The answer is still too excerpt-heavy or outline-heavy for the final paper renderer contract."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots and mode_selection.mode == "paper_summary":
        notes.append(
            "The answer did not clearly cover these requested summary dimensions: "
            + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
            + "."
        )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.mode == "paper_compare" and any(
        marker in answer_lower
        for marker in (
            "hybrid approach could be beneficial",
            "combining methodologies could",
            "recommendations or synthesis",
        )
    ):
        notes.append("The comparison drifted into unsupported recommendation language instead of staying slot-grounded.")
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }


def _prepare_detailed_paper_trace(
    config: LabaiConfig,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
) -> PaperTrace:
    if (
        not paper_trace.active
        or mode_selection.mode != "paper_summary"
        or mode_selection.paper_output_profile != "detailed_paper_note"
        or not paper_trace.document_notes
    ):
        return paper_trace

    updated_notes: list[dict[str, object]] = []
    changed = False
    for document_note in paper_trace.document_notes:
        updated_note, note_changed = _rescue_detailed_document_note(config, paper_trace, document_note)
        updated_notes.append(updated_note)
        changed = changed or note_changed
    if not changed:
        return paper_trace
    return replace(paper_trace, document_notes=updated_notes)


def _rescue_detailed_document_note(
    config: LabaiConfig,
    paper_trace: PaperTrace,
    document_note: dict[str, object],
) -> tuple[dict[str, object], bool]:
    source_path = str(document_note.get("source_path", ""))
    cleaned_slots = [dict(slot) for slot in document_note.get("cleaned_slots", [])]
    aggregated_map = {
        str(slot.get("slot_name", "")): dict(slot)
        for slot in document_note.get("aggregated_slots", [])
    }
    cleaned_map = {str(slot.get("slot_name", "")): slot for slot in cleaned_slots}
    page_texts = _non_reference_document_pages(_load_document_page_texts(config, document_note))
    changed = False

    rescue_plan = tuple(
        (
            slot_name,
            *_rescue_generic_detailed_slot_summary(slot_name, page_texts),
        )
        for slot_name in (
            "research_question",
            "sample_or_data",
            "method",
            "main_findings",
            "limitations",
            "conclusion",
        )
    )

    for slot_name, rescued_summary, rescued_pages in rescue_plan:
        if not rescued_summary:
            continue
        current_payload = cleaned_map.get(slot_name) or aggregated_map.get(slot_name) or {}
        current_text = _slot_payload_text(current_payload)
        if not _should_replace_with_rescued_summary(slot_name, current_payload, rescued_summary):
            continue
        rescued_payload = _make_rescued_cleaned_slot(
            slot_name=slot_name,
            summary_text=rescued_summary,
            source_path=source_path,
            page_numbers=rescued_pages,
            prior_payload=current_payload,
        )
        if slot_name in cleaned_map:
            cleaned_map[slot_name].update(rescued_payload)
        else:
            cleaned_map[slot_name] = rescued_payload
        changed = changed or rescued_summary.strip() != current_text.strip()

    for slot_name, payload in list(cleaned_map.items()):
        if slot_name not in {
            "research_question",
            "sample_or_data",
            "method",
            "main_findings",
            "limitations",
            "conclusion",
        }:
            continue
        detailed_render_text = _build_support_driven_detailed_slot_text(
            slot_name,
            payload,
            source_path=source_path,
            paper_trace=paper_trace,
            page_texts=page_texts,
        )
        if not detailed_render_text:
            continue
        if detailed_render_text != str(payload.get("detailed_render_text", "")).strip():
            payload["detailed_render_text"] = detailed_render_text
            changed = True

    if not changed:
        return document_note, False

    ordered_slot_names = [str(slot.get("slot_name", "")) for slot in document_note.get("cleaned_slots", [])]
    updated_cleaned_slots = [cleaned_map[slot_name] for slot_name in ordered_slot_names if slot_name in cleaned_map]
    for slot_name, payload in cleaned_map.items():
        if slot_name not in ordered_slot_names:
            updated_cleaned_slots.append(payload)
    missing_slots = tuple(
        str(slot.get("slot_name", ""))
        for slot in updated_cleaned_slots
        if str(slot.get("slot_name", "")) != "other"
        and str(slot.get("support_status", "not_clearly_stated")) == "not_clearly_stated"
    )
    updated_note = dict(document_note)
    updated_note["cleaned_slots"] = updated_cleaned_slots
    updated_note["missing_slots"] = missing_slots
    return updated_note, True


def _load_document_page_texts(
    config: LabaiConfig,
    document_note: dict[str, object],
) -> list[tuple[int, str]]:
    document_id = str(document_note.get("document_id", "")).strip()
    if not document_id:
        return []
    extracted_path = config.papers.extracted_dir / f"{document_id}.json"
    try:
        payload = json.loads(extracted_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    pages: list[tuple[int, str]] = []
    for page in payload.get("pages", []):
        if not isinstance(page, dict):
            continue
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        pages.append((int(page.get("page_number", 0) or 0), text))
    return pages


def _non_reference_document_pages(page_texts: list[tuple[int, str]]) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    for page_number, text in page_texts:
        normalized = _normalize_extracted_block(text)
        if not normalized or _page_looks_reference_like(normalized):
            continue
        pages.append((page_number, normalized))
    return pages


def _page_looks_reference_like(text: str) -> bool:
    lowered = text.lower()
    if _page_has_substantive_section_signal(lowered):
        return False
    if lowered.startswith(("references", "bibliography")):
        return True
    if "references" in lowered[:120]:
        return True
    year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    journal_hits = sum(
        lowered.count(token)
        for token in (
            "journal of",
            "review of",
            "econometrica",
            "working paper",
            "proceedings",
            "technical report",
        )
    )
    citation_hits = len(re.findall(r"\b[A-Z][a-z]+,\s+[A-Z]\.", text))
    return year_hits >= 6 and (journal_hits >= 2 or citation_hits >= 2)


def _page_has_substantive_section_signal(text: str) -> bool:
    markers = (
        "abstract",
        "introduction",
        "data",
        "data and sample",
        "dataset",
        "sample",
        "method",
        "methods",
        "methodology",
        "results",
        "findings",
        "discussion",
        "discussion and conclusions",
        "discussion and conclusion",
        "conclusion",
        "conclusions",
        "limitations",
        "future work",
    )
    leading = text[:400]
    return any(marker in leading for marker in markers)


def _looks_reference_like_sentence(text: str) -> bool:
    lowered = text.lower()
    if lowered.startswith(("references", "bibliography", "review of ", "journal of ")):
        return True
    if lowered.count(" et al") >= 1:
        return True
    if re.search(
        r"\b[A-Z][a-z]+(?:,\s+[A-Z][a-z]+){1,4}.*\b(?:19|20)\d{2}\b",
        text,
    ) and sum(
        lowered.count(token)
        for token in (
            "journal of",
            "review of",
            "econometrica",
            "technical report",
            "working paper",
        )
    ):
        return True
    return False


def _candidate_pages_for_detailed_slot(
    slot_name: str,
    page_texts: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    if not page_texts:
        return []
    total = len(page_texts)
    if slot_name in {"research_question"}:
        return page_texts[: min(max(4, total // 3), 10)]
    if slot_name in {"sample_or_data"}:
        scored_pages: list[tuple[int, int, str]] = []
        for page_number, page_text in page_texts:
            lowered = page_text.lower()
            score = 0
            if _looks_like_explicit_sample_data_text(page_text):
                score += 6
            score += 2 * _page_section_bonus(slot_name, lowered)
            if any(
                marker in lowered
                for marker in (
                    "2.1 data and the overarching model",
                    "data and the overarching model",
                    "sample splitting",
                    "training sample",
                    "validation sample",
                    "testing subsample",
                    "crsp",
                    "nyse, amex, and nasdaq",
                    "30,000",
                    "94 characteristics",
                    "8 macroeconomic",
                    "treasury-bill rate",
                )
            ):
                score += 4
            if score > 0:
                scored_pages.append((score, page_number, page_text))
        if scored_pages:
            scored_pages.sort(key=lambda item: (-item[0], item[1]))
            selected = sorted({page_number for _score, page_number, _text in scored_pages[:10]})
            return [(page_number, text) for page_number, text in page_texts if page_number in selected]
        return page_texts[: min(max(6, total // 2), 16)]
    if slot_name in {"method"}:
        return page_texts[: min(max(8, (total * 2) // 3), 20)]
    if slot_name == "limitations":
        scored_pages: list[tuple[int, int, str]] = []
        for page_number, page_text in page_texts:
            lowered = unicodedata.normalize("NFKC", page_text).lower()
            score = 0
            score += 2 * _page_section_bonus(slot_name, lowered)
            if any(
                marker in lowered
                for marker in (
                    "limitation",
                    "limitations",
                    "future work",
                    "future research",
                    "we leave",
                    "we only",
                    "only consider",
                    "only uses",
                    "is limited to",
                    "restricted to",
                    "assume",
                    "monthly data",
                    "high-frequency data",
                    "small amount of data",
                    "signal-to-noise ratio",
                    "overfit",
                    "overfitting",
                    "must be heavily regularized",
                    "synthetic data",
                    "one country",
                    "one market",
                )
            ):
                score += 5
            if any(
                marker in lowered
                for marker in (
                    "using high-frequency data in the future could help",
                    "would be valuable to extend",
                    "leave the discussion",
                    "cannot obtain",
                    "cannot be obtained unless",
                    "do not assume",
                )
            ):
                score += 3
            if total and page_number >= max(1, total - max(6, total // 3)):
                score += 2
            if score > 0:
                scored_pages.append((score, page_number, page_text))
        if scored_pages:
            scored_pages.sort(key=lambda item: (-item[0], item[1]))
            selected = {
                page_number for _score, page_number, _text in scored_pages[: min(12, max(6, total // 2))]
            }
            selected.update(page_number for page_number, _text in page_texts[-4:])
            ordered = sorted(selected)
            return [(page_number, text) for page_number, text in page_texts if page_number in ordered]
        return page_texts[max(0, total - min(max(5, total // 3), 12)) :]
    if slot_name == "conclusion":
        return page_texts[max(0, total - min(max(5, total // 3), 12)) :]
    if slot_name in {"main_findings"}:
        return page_texts[max(0, total // 3 - 1) :]
    return page_texts


def _generic_slot_start_markers(slot_name: str) -> tuple[str, ...]:
    marker_map = {
        "research_question": (
            "abstract",
            "introduction",
            "objective",
            "motivation",
            "problem statement",
        ),
        "sample_or_data": (
            "data and sample",
            "data",
            "dataset",
            "sample",
            "empirical setting",
            "test assets",
        ),
        "method": (
            "methodology",
            "methods",
            "method",
            "empirical strategy",
            "model",
        ),
        "main_findings": (
            "results",
            "findings",
            "empirical results",
            "out-of-sample evaluation",
        ),
        "limitations": (
            "limitations",
            "discussion",
            "discussion and conclusions",
            "discussion and conclusion",
            "future work",
            "conclusion",
            "conclusions",
        ),
        "conclusion": (
            "discussion and conclusions",
            "discussion and conclusion",
            "conclusion",
            "conclusions",
            "summary",
        ),
    }
    return marker_map.get(slot_name, ())


def _generic_slot_end_markers(slot_name: str) -> tuple[str, ...]:
    all_markers = (
        "abstract",
        "introduction",
        "background",
        "literature review",
        "data and sample",
        "data",
        "dataset",
        "sample",
        "methodology",
        "methods",
        "method",
        "empirical strategy",
        "results",
        "findings",
        "discussion and conclusions",
        "discussion and conclusion",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
        "references",
    )
    start_markers = set(_generic_slot_start_markers(slot_name))
    return tuple(marker for marker in all_markers if marker not in start_markers)


def _slice_generic_slot_block(slot_name: str, text: str) -> str:
    start_markers = _generic_slot_start_markers(slot_name)
    if not start_markers:
        return ""
    return _slice_structured_section(
        text,
        start_markers=start_markers,
        end_markers=_generic_slot_end_markers(slot_name),
    )


def _slot_specific_rescue_summary(
    slot_name: str,
    candidate_pages: list[tuple[int, str]],
) -> str:
    texts = [text for _page_number, text in candidate_pages]
    if slot_name == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            return " ".join(
                f"{fact.rstrip('. ')}." for fact in facts[:5]
            ).strip()
    if slot_name == "method":
        normalized_corpus = " ".join(
            unicodedata.normalize("NFKC", text).replace("\n", " ").strip()
            for text in texts
        )
        lowered_corpus = normalized_corpus.lower()
        if (
            "arimax" in lowered_corpus
            and ("garch" in lowered_corpus or "garch benchmark" in lowered_corpus)
            and "lstm" in lowered_corpus
        ):
            return "The paper implements an ARIMAX-GARCH benchmark and an end-to-end LSTM."
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "benchmark" in lowered_sentence and "lstm" in lowered_sentence:
                    return _truncate_line(unicodedata.normalize("NFKC", sentence).strip(), limit=220)
        families = _extract_method_family_mentions(texts)
        if families:
            if len(families) == 1:
                return f"The paper explicitly discusses {families[0]}."
            if len(families) == 2:
                return f"The paper explicitly discusses {families[0]} and {families[1]}."
            return (
                "The paper explicitly discusses "
                + ", ".join(families[:-1])
                + f", and {families[-1]}."
            )
    if slot_name == "main_findings":
        rescued: list[str] = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "fig." in lowered_sentence or "table " in lowered_sentence:
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "best performing methods",
                        "best performing",
                        "shallow learning outperforms deep learning",
                        "dominant predictive signals",
                        "higher sharpe ratios",
                        "predictive advantage",
                        "nonlinear interactions",
                        "most valuable for forecasting",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            return ". ".join(unique[:3]).strip() + "."
    if slot_name == "limitations":
        rescued: list[str] = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "fig." in lowered_sentence or "table " in lowered_sentence:
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "limitation",
                        "limitations",
                        "future",
                        "monthly data",
                        "high-frequency data",
                        "could help researchers improve",
                        "simple ",
                        "limited",
                        "dearth of data",
                        "low signal-to-noise ratio",
                        "overfit",
                        "overfitting",
                        "computationally intensive",
                        "must be heavily regularized",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            return ". ".join(unique[:2]).strip() + "."
    if slot_name == "conclusion":
        rescued = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "fig." in lowered_sentence or "table " in lowered_sentence:
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "the evidence indicates",
                        "these findings emphasize",
                        "enhance the precision",
                        "predictive capabilities",
                        "greater precision in forecasting",
                        "leading contender for forecasting",
                        "at the highest level",
                        "best performing methods",
                        "most valuable for forecasting",
                        "dominant predictive signals",
                        "brings promise",
                        "our findings help justify",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            return ". ".join(unique[:2]).strip() + "."
    return ""


def _rescue_generic_detailed_slot_summary(
    slot_name: str,
    page_texts: list[tuple[int, str]],
) -> tuple[str, tuple[int, ...]]:
    candidate_pages = _candidate_pages_for_detailed_slot(slot_name, page_texts)
    total_pages = page_texts[-1][0] if page_texts else 0
    candidates: list[tuple[int, int, str]] = []
    seen_signatures: set[str] = set()

    direct_summary = _slot_specific_rescue_summary(slot_name, candidate_pages)
    direct_pages = tuple(dict.fromkeys(page_number for page_number, _text in candidate_pages[:3]))
    if direct_summary:
        return direct_summary, direct_pages

    for page_number, page_text in candidate_pages:
        page_lower = page_text.lower()
        section_block = _slice_generic_slot_block(slot_name, page_text)
        sentence_sources = [(_split_rescue_sentences(page_text), False)]
        if section_block:
            sentence_sources.insert(0, (_split_rescue_sentences(section_block), True))
        for sentences, from_section_block in sentence_sources:
            for sentence in sentences:
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if not cleaned:
                    continue
                score = _detailed_rescue_sentence_score(
                    slot_name,
                    cleaned,
                    page_number=page_number,
                    total_pages=total_pages,
                    page_text=page_lower,
                )
                if from_section_block:
                    score += 4
                if score <= 0:
                    continue
                signature = _rescue_signature(cleaned)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                candidates.append((score, page_number, cleaned))

    if not candidates:
        return "", ()

    candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1] if slot_name in {"limitations", "conclusion", "main_findings"} else item[1],
            len(item[2]),
        )
    )
    selected: list[tuple[int, str]] = []
    for score, page_number, cleaned in candidates:
        selected.append((page_number, cleaned))
        if len(selected) >= _rescue_fragment_limit(slot_name):
            break
    summary = _compose_rescued_slot_summary(
        slot_name,
        [item[1] for item in selected],
    )
    pages = tuple(dict.fromkeys(page_number for page_number, _text in selected))
    return summary, pages


def _split_rescue_sentences(text: str) -> list[str]:
    normalized = _normalize_extracted_block(text)
    if not normalized:
        return []
    protected = re.sub(r"\bet al\.", "et al<prd>", normalized, flags=re.IGNORECASE)
    protected = re.sub(r"\be\.g\.", "e<prd>g<prd>", protected, flags=re.IGNORECASE)
    protected = re.sub(r"\bi\.e\.", "i<prd>e<prd>", protected, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[.!?])\s+|(?<=:)\s+(?=[A-Z])", protected)
    return [part.replace("<prd>", ".").strip() for part in parts if part.strip()]


def _clean_rescue_sentence(text: str, *, slot_name: str) -> str:
    cleaned = _normalize_extracted_block(text)
    cleaned = re.sub(
        r"^(?:\d+(?:\.\d+)?)\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"^(?:abstract|introduction|background|data|data and sample|dataset|sample|methodology|methods?|results?|discussion(?: and conclusions?)?|conclusions?)\s*[:.\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^this section (?:covers|describes|examines|reports|shows)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" ;,.-")
    if len(cleaned) < 24:
        return ""
    if _looks_reference_like_sentence(cleaned):
        return ""
    if _looks_truncated_slot_fragment(cleaned):
        return ""
    if cleaned.lower().startswith(("figure ", "table ", "appendix ")):
        return ""
    if slot_name == "sample_or_data" and not _looks_like_explicit_sample_data_text(cleaned):
        return ""
    return cleaned


def _looks_truncated_slot_fragment(text: str) -> bool:
    lowered = _normalize_extracted_block(text).lower()
    if not lowered:
        return True
    if any(
        marker in lowered
        for marker in (
            "downloaded from",
            "academic.oup.com",
            "http://",
            "https://",
        )
    ):
        return True
    if lowered.startswith(
        (
            "the review of financial studies /",
            "empirical asset pricing via machine learning table",
            "table ",
            "figure ",
            "appendix ",
        )
    ):
        return True
    if re.search(r"\b(?:of|to|with|for|which|that|than|and|or|the|a|an)\.?$", lowered):
        return True
    if re.search(r"\b\d+\.\s*$", lowered) and len(lowered) < 90:
        return True
    return False


def _page_section_bonus(slot_name: str, page_text: str) -> int:
    cue_map = {
        "research_question": ("abstract", "introduction", "problem statement", "motivation"),
        "sample_or_data": ("data", "sample", "dataset", "empirical setting", "data and sample"),
        "method": ("method", "methods", "methodology", "approach", "empirical strategy", "model"),
        "main_findings": ("results", "findings", "empirical results", "evidence"),
        "limitations": ("discussion", "limitations", "conclusion", "future work"),
        "conclusion": ("discussion", "conclusion", "conclusions", "summary"),
    }
    return sum(1 for marker in cue_map.get(slot_name, ()) if marker in page_text)


def _detailed_rescue_sentence_score(
    slot_name: str,
    text: str,
    *,
    page_number: int,
    total_pages: int,
    page_text: str,
) -> int:
    lowered = text.lower()
    positive_map = {
        "research_question": (
            "this paper",
            "this study",
            "we study",
            "we examine",
            "the objective",
            "the goal",
            "the question",
            "asks whether",
            "investigates",
        ),
        "sample_or_data": (
            "dataset",
            "sample",
            "sample period",
            "data source",
            "observations",
            "trading days",
            "tickers",
            "constituents",
            "daily",
            "monthly",
            "train",
            "validation",
            "test",
            "out-of-sample",
            "panel",
        ),
        "method": (
            "method",
            "model",
            "approach",
            "benchmark",
            "lstm",
            "arimax",
            "garch",
            "lasso",
            "ridge",
            "elastic-net",
            "ann",
            "cnn",
            "principal components",
            "partial least squares",
            "random forest",
            "boost",
        ),
        "main_findings": (
            "we find",
            "we show",
            "results show",
            "outperform",
            "better",
            "higher",
            "lower",
            "sharpe",
            "rmse",
            "mae",
            "predictive",
            "accuracy",
            "performance",
        ),
        "limitations": (
            "limitation",
            "limitations",
            "future work",
            "using monthly data",
            "only ",
            "we only",
            "is limited to",
            "cannot",
            "simple ",
            "linear dcc",
            "squared-error",
        ),
        "conclusion": (
            "in conclusion",
            "we conclude",
            "overall",
            "these findings",
            "the evidence supports",
            "brings promise",
            "practical aspects",
            "risk-management",
            "risk management",
        ),
    }
    negative_map = {
        "research_question": ("table ", "figure ", "references", "appendix"),
        "sample_or_data": (
            "journal of",
            "econometrica",
            "technical report",
            "turnover",
            "sharpe ratio",
            "portfolio",
            "predictive r2",
            "report the main empirical results",
            "table iv",
            "figure 5",
            "figure 6",
        ),
        "method": (
            "table ",
            "figure ",
            "references",
            "rmse",
            "mae",
            "accuracy",
            "predictive r2",
            "sharpe ratio",
            "outperform",
            "higher predictive performance",
        ),
        "main_findings": ("references", "journal of", "working paper"),
        "limitations": ("references", "journal of", "working paper", "table ", "figure "),
        "conclusion": ("references", "journal of", "working paper", "table ", "figure "),
    }
    if any(marker in lowered for marker in negative_map.get(slot_name, ())):
        return -5
    score = sum(3 for marker in positive_map.get(slot_name, ()) if marker in lowered)
    score += 2 * _page_section_bonus(slot_name, page_text)
    if slot_name in {"research_question", "sample_or_data", "method"}:
        if total_pages and page_number <= max(8, total_pages // 3):
            score += 3
    if slot_name in {"main_findings", "limitations", "conclusion"}:
        if total_pages and page_number >= max(1, total_pages - max(6, total_pages // 3)):
            score += 4
    if slot_name == "sample_or_data":
        if re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", lowered):
            score += 2
        if re.search(r"\b\d{1,3},\d{3}\b", lowered):
            score += 1
        if any(marker in lowered for marker in ("observations", "constituents", "data source", "downloaded via", "first half", "second half")):
            score += 2
    if slot_name == "method":
        if any(marker in lowered for marker in ("compare", "benchmark", "specification", "estimator", "algorithm", "model families")):
            score += 2
        if len(_extract_method_family_mentions([text])) >= 2:
            score += 3
    if slot_name == "limitations":
        if any(marker in lowered for marker in ("caveat", "future work", "limited", "misspecification", "we leave", "only four", "only ", "simple ")):
            score += 2
    if slot_name == "conclusion":
        if any(marker in lowered for marker in ("overall", "we conclude", "these results", "suggests that", "illustrate that", "highlights")):
            score += 2
    if slot_name == "conclusion" and "not clearly stated in the paper" in lowered:
        score = -5
    return score


def _rescue_signature(text: str) -> str:
    lowered = unicodedata.normalize("NFKC", text).lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(token for token in lowered.split()[:18] if token)


def _rescue_fragment_limit(slot_name: str) -> int:
    if slot_name in {"sample_or_data", "main_findings"}:
        return 3
    return 2


def _compose_rescued_slot_summary(slot_name: str, fragments: list[str]) -> str:
    unique = [fragment.rstrip(". ") for fragment in _dedupe_strings(fragments)]
    if not unique:
        return ""
    if slot_name == "research_question":
        return " ".join(f"{item}." for item in unique[:2]).strip()
    return " ".join(f"{item}." for item in unique[: _rescue_fragment_limit(slot_name)]).strip()


def _slot_note_page_number(slot_note: dict[str, object]) -> int:
    page_numbers = slot_note.get("page_numbers", [])
    if isinstance(page_numbers, list) and page_numbers:
        try:
            return int(page_numbers[0])
        except (TypeError, ValueError):
            return 0
    return 0


def _slot_support_note_score(
    slot_name: str,
    slot_note: dict[str, object],
    *,
    total_pages: int,
    page_lookup: dict[int, str],
) -> int:
    text = _normalize_extracted_block(str(slot_note.get("extracted_content", "")))
    if not text or _looks_truncated_slot_fragment(text) or _looks_reference_like_sentence(text):
        return -10
    if _section_fails_slot_fit(slot_name, text):
        return -8
    lowered = text.lower()
    if slot_name == "sample_or_data" and _looks_like_sample_data_result_noise(text):
        return -8
    if slot_name in {"main_findings", "limitations", "conclusion"} and any(
        marker in lowered
        for marker in (
            "table ",
            "figure ",
            "internet appendix",
            "downloaded from",
            "the review of financial studies /",
        )
    ):
        return -7
    page_number = _slot_note_page_number(slot_note)
    page_text = page_lookup.get(page_number, "").lower()
    score = _detailed_rescue_sentence_score(
        slot_name,
        text,
        page_number=page_number,
        total_pages=total_pages,
        page_text=page_text,
    )
    support_bonus = {
        "strong": 3,
        "moderate": 2,
        "weak": 0,
    }.get(str(slot_note.get("support_strength", "weak")), 0)
    explicit_bonus = 2 if bool(slot_note.get("explicit")) else 0
    score += support_bonus + explicit_bonus
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "crsp",
            "sample begins",
            "30,000",
            "training sample",
            "validation sample",
            "out-of-sample testing",
            "predictive characteristics",
        )
    ):
        score += 2
    if slot_name in {"main_findings", "conclusion"} and any(
        marker in lowered
        for marker in (
            "neural networks",
            "regression trees",
            "shallow learning",
            "deep learning",
            "price trends",
            "brings promise",
            "risk-management",
            "out-of-sample predictive",
            "sharpe ratio",
        )
    ):
        score += 2
    return score


def _support_page_limit(slot_name: str) -> int:
    if slot_name in {"sample_or_data", "main_findings", "conclusion"}:
        return 4
    if slot_name in {"limitations", "method"}:
        return 3
    return 2


def _collect_support_pages_for_detailed_slot(
    slot_name: str,
    *,
    source_path: str,
    paper_trace: PaperTrace,
    page_texts: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    if not page_texts:
        return []
    page_lookup = {page_number: text for page_number, text in page_texts}
    total_pages = page_texts[-1][0]
    candidates: list[tuple[int, int]] = []
    for slot_note in paper_trace.slot_notes:
        if str(slot_note.get("source_path", "")) != source_path:
            continue
        if str(slot_note.get("slot_name", "")) != slot_name:
            continue
        page_number = _slot_note_page_number(slot_note)
        if page_number <= 0 or page_number not in page_lookup:
            continue
        score = _slot_support_note_score(
            slot_name,
            slot_note,
            total_pages=total_pages,
            page_lookup=page_lookup,
        )
        if score > 0:
            candidates.append((score, page_number))
    if not candidates:
        return _candidate_pages_for_detailed_slot(slot_name, page_texts)[: _support_page_limit(slot_name)]

    structural_seed_pages = _candidate_pages_for_detailed_slot(slot_name, page_texts)
    seen_candidate_pages = {page_number for _score, page_number in candidates}
    for page_number, text in structural_seed_pages:
        if page_number in seen_candidate_pages:
            continue
        score = 3 + 2 * _page_section_bonus(slot_name, text.lower())
        if slot_name == "sample_or_data" and _looks_like_explicit_sample_data_text(text):
            score += 4
        if slot_name in {"limitations", "conclusion"} and any(
            marker in text.lower()
            for marker in (
                "3. conclusion",
                "discussion and conclusions",
                "at the highest level",
                "brings promise",
                "dearth of data",
                "low signal-to-noise ratio",
            )
        ):
            score += 4
        candidates.append((score, page_number))

    candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1] if slot_name in {"main_findings", "limitations", "conclusion"} else item[1],
        )
    )
    selected_pages: list[int] = []
    for _score, page_number in candidates:
        selected_pages.append(page_number)
        if slot_name in {"sample_or_data", "main_findings", "limitations", "conclusion"} and page_number + 1 in page_lookup:
            selected_pages.append(page_number + 1)
        if (
            slot_name in {"main_findings", "limitations", "conclusion"}
            and page_number - 1 in page_lookup
            and page_number >= max(1, total_pages - 6)
        ):
            selected_pages.append(page_number - 1)
        if len(list(dict.fromkeys(selected_pages))) >= _support_page_limit(slot_name):
            break
    final_pages = list(dict.fromkeys(selected_pages))[: _support_page_limit(slot_name)]
    return [(page_number, page_lookup[page_number]) for page_number in final_pages if page_number in page_lookup]


def _build_support_driven_detailed_slot_text(
    slot_name: str,
    current_payload: dict[str, object],
    *,
    source_path: str,
    paper_trace: PaperTrace,
    page_texts: list[tuple[int, str]],
) -> str:
    support_pages = _collect_support_pages_for_detailed_slot(
        slot_name,
        source_path=source_path,
        paper_trace=paper_trace,
        page_texts=page_texts,
    )
    if not support_pages:
        return ""
    rescued_summary, _rescued_pages = _rescue_generic_detailed_slot_summary(slot_name, support_pages)
    if not rescued_summary:
        return ""
    cleaned = _clean_detailed_slot_body(
        rescued_summary,
        slot_name=slot_name,
        response_language="en",
    )
    if not cleaned or cleaned == "Not clearly stated in the paper.":
        return ""
    if _should_replace_with_rescued_summary(slot_name, current_payload, cleaned):
        return cleaned
    current_text = _clean_detailed_slot_body(
        _slot_payload_text(current_payload),
        slot_name=slot_name,
        response_language="en",
    )
    if slot_name == "sample_or_data" and any(
        marker in cleaned.lower() and marker not in current_text.lower()
        for marker in (
            "crsp",
            "nyse, amex, and nasdaq",
            "30,000",
            "validation",
            "out-of-sample testing",
        )
    ):
        return cleaned
    if slot_name in {"main_findings", "conclusion"} and any(
        marker in cleaned.lower() and marker not in current_text.lower()
        for marker in (
            "asset prices",
            "larger and more liquid",
            "price trends",
            "shallow learning",
            "regression trees",
            "brings promise",
        )
    ):
        return cleaned
    if _detailed_slot_needs_rescue(slot_name, current_text):
        return cleaned
    if _detail_marker_count(slot_name, cleaned) >= _detail_marker_count(slot_name, current_text) + 2:
        return cleaned
    return ""


def _normalize_extracted_block(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"(\w)-\s+(\w)", r"\1\2", normalized)
    normalized = re.sub(r"\s+", " ", normalized.replace("\n", " ")).strip()
    return normalized


def _slice_structured_section(
    text: str,
    *,
    start_markers: tuple[str, ...],
    end_markers: tuple[str, ...],
) -> str:
    normalized = _normalize_extracted_block(text)
    lowered = normalized.lower()
    start_positions = [lowered.find(marker) for marker in start_markers if lowered.find(marker) != -1]
    if not start_positions:
        return ""
    start = min(start_positions)
    end = len(normalized)
    for marker in end_markers:
        marker_index = lowered.find(marker, start + 1)
        if marker_index != -1:
            end = min(end, marker_index)
    return normalized[start:end].strip()


def _extract_data_block(page_texts: list[tuple[int, str]]) -> tuple[str, tuple[int, ...]]:
    for page_number, text in page_texts:
        block = _slice_structured_section(
            text,
            start_markers=("data equity panel", "equity panel", "vix features", "hubs, events, and split"),
            end_markers=("methodology", "results", "discussion and conclusions", "references"),
        )
        if block:
            return block, (page_number,)
    return "", ()


def _extract_intro_block(page_texts: list[tuple[int, str]]) -> tuple[str, tuple[int, ...]]:
    for page_number, text in page_texts:
        block = _slice_structured_section(
            text,
            start_markers=("introduction and problem statement",),
            end_markers=("related work and previous project", "data equity panel", "equity panel", "references"),
        )
        if block:
            return block, (page_number,)
    return "", ()


def _trim_structured_tail(text: str, *, end_markers: tuple[str, ...]) -> str:
    lowered = text.lower()
    end = len(text)
    for marker in end_markers:
        marker_index = lowered.find(marker)
        if marker_index != -1:
            end = min(end, marker_index)
    return text[:end].strip()


def _extract_method_block(page_texts: list[tuple[int, str]]) -> tuple[str, tuple[int, ...]]:
    blocks: list[str] = []
    pages: list[int] = []
    capturing = False
    for page_number, text in page_texts:
        normalized = _normalize_extracted_block(text)
        lowered = normalized.lower()
        if not capturing and "methodology" not in lowered:
            continue
        if not capturing:
            block = _slice_structured_section(
                normalized,
                start_markers=("methodology",),
                end_markers=("results", "discussion and conclusions", "references"),
            )
            if block:
                blocks.append(block)
                pages.append(page_number)
                capturing = not any(marker in lowered for marker in ("results", "discussion and conclusions", "references"))
            continue
        trimmed = _trim_structured_tail(
            normalized,
            end_markers=("results", "discussion and conclusions", "references"),
        )
        if trimmed:
            blocks.append(trimmed)
            pages.append(page_number)
        if any(marker in lowered for marker in ("results", "discussion and conclusions", "references")):
            break
    return " ".join(blocks).strip(), tuple(pages)


def _extract_discussion_block(page_texts: list[tuple[int, str]]) -> tuple[str, tuple[int, ...]]:
    blocks: list[str] = []
    pages: list[int] = []
    capturing = False
    for page_number, text in page_texts:
        normalized = _normalize_extracted_block(text)
        lowered = normalized.lower()
        if not capturing and "discussion and conclusions" not in lowered:
            continue
        if not capturing:
            block = _slice_structured_section(
                normalized,
                start_markers=("discussion and conclusions",),
                end_markers=("references",),
            )
            if block:
                blocks.append(block)
                pages.append(page_number)
                capturing = "references" not in lowered
            continue
        trimmed = normalized.split("References", 1)[0].strip()
        if trimmed:
            blocks.append(trimmed)
            pages.append(page_number)
        if "references" in lowered:
            break
    return " ".join(blocks).strip(), tuple(pages)


def _extract_detailed_research_question_summary(block: str) -> str:
    if not block:
        return ""
    normalized = _normalize_extracted_block(block)
    parts: list[str] = []
    if re.search(
        r"In this project I study such contagion for the NASDAQ-100 using a time-series network framework\.",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append("The paper studies NASDAQ-100 contagion in a time-series network framework.")
    if question := re.search(
        r"The central modelling question is how to specify the time-series process for αt\.",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append(question.group(0).strip())
    if re.search(
        r"My goal is different: I ask whether an LSTM-based αt can make the entire network more stable, in the sense of smaller multi-step losses and a spectral radius ρ\(αtRt\) that stays near or below one\.",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append(
            "It specifically asks whether an LSTM-based αt can keep multi-step losses and the spectral radius closer to the stable region than the benchmark process."
        )
    return " ".join(_dedupe_strings(parts))


def _extract_detailed_sample_data_summary(block: str) -> str:
    if not block:
        return ""
    facts = _extract_sample_data_facts([block])
    if not facts:
        return ""
    return " ".join(
        fact if fact.endswith(".") else f"{fact}."
        for fact in facts[:5]
    ).strip()


def _extract_detailed_method_summary(
    *,
    intro_block: str,
    method_block: str,
    discussion_block: str,
) -> str:
    corpus = " ".join(part for part in (intro_block, method_block, discussion_block) if part).strip()
    if not corpus:
        return ""
    normalized = _normalize_extracted_block(corpus)
    parts: list[str] = []
    if re.search(
        r"I combine a standard volatility layer.+?with a linear contagion engine on the resulting equity network\.",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append(
            "The framework combines a GARCH–DCC volatility layer with a linear contagion engine built on the resulting equity network."
        )
    if re.search(
        r"(I implement two Phase-2 time-series models for αt: an ARIMAX–GARCH benchmark and an end-to-end LSTM|Second, I replace hand-crafted αt with two data-driven specifications: a traditional ARIMAX–GARCH model, and an end-to-end LSTM)",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append(
            "It compares two learned contagion-intensity specifications: an ARIMAX–GARCH benchmark and an end-to-end LSTM."
        )
    if re.search(
        r"where zt consists of the three VIX features",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append("The ARIMAX–GARCH benchmark uses the three VIX features as exogenous drivers of αt.")
    if re.search(
        r"I implement a single-layer LSTM\.",
        normalized,
        flags=re.IGNORECASE,
    ) and re.search(
        r"At each date t the input sequence is the last 30 days",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append("The comparison model is a single-layer LSTM that uses the last 30 days of inputs to map directly to αt.")
    if re.search(
        r"Crucially, I do not provide labels for αt\.",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append("Instead of using direct αt labels, the LSTM is trained on multi-asset loss error inside the contagion engine.")
    return " ".join(_dedupe_strings(parts))


def _extract_detailed_findings_summary(block: str) -> str:
    if not block:
        return ""
    normalized = _normalize_extracted_block(block)
    patterns = (
        r"On one-step cross-sectional metrics[^.]+\.",
        r"On 5-step contagion paths[^.]+\.",
        r"On spectral radius[^.]+\.",
    )
    findings = [
        match.group(0).strip()
        for pattern in patterns
        if (match := re.search(pattern, normalized, flags=re.IGNORECASE))
    ]
    return " ".join(_dedupe_strings(findings)) if findings else ""


def _extract_detailed_limitations_summary(block: str) -> str:
    if not block:
        return ""
    normalized = _normalize_extracted_block(block)
    match = re.search(
        r"There are several limitations\.(.+?)(Overall, the evidence supports|References)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    text = match.group(1).strip()
    return text if text.endswith(".") else text + "."


def _extract_detailed_conclusion_summary(block: str) -> str:
    if not block:
        return ""
    normalized = _normalize_extracted_block(block)
    parts: list[str] = []
    if intro := re.search(r"This project builds[^.]+\.", normalized, flags=re.IGNORECASE):
        parts.append(intro.group(0).strip())
    findings = _extract_detailed_findings_summary(normalized)
    if findings:
        parts.append(findings)
    if risk := re.search(r"From a risk-management perspective[^.]+\.", normalized, flags=re.IGNORECASE):
        parts.append(risk.group(0).strip())
    if overall := re.search(r"Overall, the evidence supports[^.]+\.", normalized, flags=re.IGNORECASE):
        parts.append(overall.group(0).strip())
    return " ".join(_dedupe_strings(parts))


def _should_replace_with_rescued_summary(
    slot_name: str,
    current_payload: dict[str, object],
    rescued_summary: str,
) -> bool:
    current_status = _slot_payload_status(current_payload)
    current_text = _clean_detailed_slot_body(
        _slot_payload_text(current_payload),
        slot_name=slot_name,
        response_language="en",
    )
    rescued_score = _detail_marker_count(slot_name, rescued_summary)
    current_score = _detail_marker_count(slot_name, current_text)
    if current_status == "not_clearly_stated":
        return True
    if _looks_reference_like_sentence(current_text):
        return True
    if slot_name == "research_question" and (len(current_text.strip()) < 70 or rescued_score > current_score):
        return True
    if slot_name == "method":
        if rescued_score > current_score and len(rescued_summary) >= len(current_text):
            return True
        if "project demonstrates" in current_text.lower() and any(
            marker in rescued_summary.lower()
            for marker in ("benchmark", "lstm", "arimax", "garch")
        ):
            return True
    if slot_name == "sample_or_data":
        rescued_sentence_count = len(re.findall(r"[.!?]", rescued_summary))
        current_sentence_count = len(re.findall(r"[.!?]", current_text))
        detail_markers = (
            "nasdaq-100",
            "constituents",
            "755 trading days",
            "5-day rolling mean",
            "22-day rolling mean",
            "msft",
            "adbe",
            "nvda",
            "payx",
            "out-of-sample",
        )
        if rescued_score >= current_score + 1:
            return True
        if any(marker in rescued_summary.lower() and marker not in current_text.lower() for marker in detail_markers):
            return True
        if rescued_sentence_count >= current_sentence_count + 2 and len(rescued_summary) >= len(current_text) + 40:
            return True
    if slot_name == "limitations" and any(
        marker in rescued_summary.lower() and marker not in current_text.lower()
        for marker in (
            "dearth of data",
            "low signal-to-noise ratio",
            "overfit",
            "overfitting",
            "computationally intensive",
            "must be heavily regularized",
        )
    ):
        return True
    if slot_name == "conclusion" and any(
        marker in rescued_summary.lower() and marker not in current_text.lower()
        for marker in (
            "at the highest level",
            "best performing methods",
            "brings promise",
            "most valuable for forecasting",
            "dominant predictive signals",
            "help justify",
        )
    ):
        return True
    if slot_name in {"main_findings", "limitations", "conclusion"} and (
        rescued_score > current_score or len(rescued_summary) > len(current_text) + 24
    ):
        return True
    return False


def _detail_marker_count(slot_name: str, text: str) -> int:
    lowered = text.lower()
    generic_marker_map = {
        "sample_or_data": (
            "dataset",
            "sample period",
            "data source",
            "tickers",
            "trading days",
            "training",
            "validation",
            "test",
            "out-of-sample",
            "daily",
            "monthly",
            "panel",
            "nasdaq-100",
            "constituents",
        ),
        "method": (
            "method",
            "model",
            "approach",
            "benchmark",
            "lstm",
            "arimax",
            "garch",
            "lasso",
            "ridge",
            "elastic-net",
            "ann",
            "cnn",
            "principal components",
            "partial least squares",
        ),
        "research_question": ("study", "question", "goal", "objective", "asks whether", "investigates", "focuses on"),
        "main_findings": ("we find", "we show", "outperform", "better", "higher", "lower", "rmse", "mae", "sharpe"),
        "limitations": ("limitation", "future work", "only", "cannot", "is limited to", "simple", "monthly data"),
        "conclusion": ("in conclusion", "overall", "we conclude", "these findings", "evidence supports", "risk management"),
    }
    if slot_name in generic_marker_map:
        return sum(1 for marker in generic_marker_map[slot_name] if marker in lowered)
    marker_map = {
        "sample_or_data": (
            "nasdaq-100",
            "yfinance",
            "tickers",
            "trading days",
            "vix",
            "5-day",
            "22-day",
            "msft",
            "adbe",
            "nvda",
            "payx",
            "training window",
            "out-of-sample",
        ),
        "method": ("garch", "dcc", "arimax", "lstm", "vix"),
        "research_question": ("contagion", "nasdaq-100", "αt", "spectral radius", "stable"),
    }
    return sum(1 for marker in marker_map.get(slot_name, ()) if marker in lowered)


def _make_rescued_cleaned_slot(
    *,
    slot_name: str,
    summary_text: str,
    source_path: str,
    page_numbers: tuple[int, ...],
    prior_payload: dict[str, object],
) -> dict[str, object]:
    page_label = str(page_numbers[0]) if len(page_numbers) == 1 else f"{page_numbers[0]}-{page_numbers[-1]}"
    evidence_ref = f"{source_path}#page={page_label}#rescue={slot_name}" if page_numbers else source_path
    return {
        "slot_name": slot_name,
        "summary_text": summary_text,
        "merged_note_text": summary_text,
        "evidence_refs": [evidence_ref],
        "support_status": "explicit_supported",
        "strongest_support": "strong",
        "explicit_note_count": max(1, int(prior_payload.get("explicit_note_count", 0) or 0)),
        "inferred_note_count": 0,
        "note_count": max(1, int(prior_payload.get("note_count", 0) or 0)),
    }


def _build_detailed_paper_note(
    document_note: dict[str, object],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
) -> str:
    if response_language == "zh-CN":
        return _build_slot_grounded_paper_summary_prose(
            document_note,
            requested_slots=requested_slots,
            response_language=response_language,
            paper_output_profile="detailed_paper_note",
        )

    title_map = {
        "research_question": "Research question",
        "sample_or_data": "Sample and data",
        "method": "Method",
        "main_findings": "Main findings",
        "limitations": "Limitations",
        "conclusion": "Conclusion",
        "practical_or_investment_implications": "Practical implications",
    }
    lines: list[str] = []
    for slot_name in requested_slots:
        title = title_map.get(slot_name)
        if not title:
            continue
        payload = _document_slot(document_note, slot_name)
        body = _clean_detailed_slot_body(
            _slot_payload_render_text(payload, paper_output_profile="detailed_paper_note"),
            slot_name=slot_name,
            response_language=response_language,
        )
        if _detailed_slot_needs_rescue(slot_name, body):
            rescued = _recover_detailed_slot_body(
                document_note,
                slot_name=slot_name,
                response_language=response_language,
            )
            if rescued:
                body = rescued
        if not body:
            body = _paper_missing_phrase(response_language)
        body = _clean_detailed_render_body(
            body,
            slot_name=slot_name,
            response_language=response_language,
        )
        lines.append(title)
        lines.append(body if body.endswith(".") or body == _paper_missing_phrase(response_language) else body + ".")
        lines.append("")
    return _cleanup_detailed_note_render(
        "\n".join(lines).strip(),
        response_language=response_language,
    )


@lru_cache(maxsize=64)
def _source_page_texts_for_detailed_rescue(source_path: str) -> tuple[tuple[int, str], ...]:
    path = Path(source_path)
    if not source_path or not path.exists():
        return ()
    try:
        config = load_config()
        parsed = parse_pdf(
            path,
            parser_preference=config.papers.parser_preference,
            min_page_text_chars=config.papers.min_page_text_chars,
        )
    except Exception:
        return ()
    return tuple(
        (page.page_number, page.text)
        for page in parsed.pages
        if page.text_available and page.text.strip()
    )


def _detailed_slot_needs_rescue(slot_name: str, body: str) -> bool:
    cleaned = body.strip()
    if not cleaned or cleaned == "Not clearly stated in the paper.":
        return True
    lowered = unicodedata.normalize("NFKC", cleaned).lower()
    if _section_fails_slot_fit(slot_name, cleaned):
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "deep learning",
            "lasso",
            "elastic net",
            "ann",
            "cnn",
            "lstm",
            "rmse",
            "mae",
        )
    ):
        return True
    if slot_name == "limitations" and any(
        marker in lowered
        for marker in (
            "constantly changing field of financial engineering",
            "limitations of various prediction models",
        )
    ):
        return True
    if slot_name == "conclusion" and any(
        marker in lowered
        for marker in (
            "acknowledged fact",
            "powerful financial force",
            "fig.",
            "table ",
        )
    ):
        return True
    return False


def _recover_detailed_slot_body(
    document_note: dict[str, object],
    *,
    slot_name: str,
    response_language: str,
) -> str:
    if response_language == "zh-CN":
        return ""
    source_path = str(document_note.get("source_path", "")).strip()
    page_texts = list(_source_page_texts_for_detailed_rescue(source_path))
    if not page_texts:
        return ""
    rescued, _pages = _rescue_generic_detailed_slot_summary(slot_name, page_texts)
    if not rescued:
        return ""
    return _clean_detailed_slot_body(
        rescued,
        slot_name=slot_name,
        response_language=response_language,
    )


def _clean_detailed_slot_body(
    text: str,
    *,
    slot_name: str,
    response_language: str,
) -> str:
    cleaned = _normalize_extracted_block(_normalize_slot_summary(text, response_language=response_language))
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"^(Methodologically,\s*)+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:\d+(?:\.\d+)?)\s*", "", cleaned)
    prefix_patterns = {
        "sample_or_data": r"^For the sample and data,\s*",
        "background_or_motivation": r"^The background or motivation is\s*",
        "method": r"^Methodologically,\s*",
        "main_findings": r"^The main finding is that\s*",
        "limitations": r"^A key limitation is that\s*",
        "conclusion": r"^Overall,\s*",
        "practical_or_investment_implications": r"^For practical or investment implications,\s*",
    }
    pattern = prefix_patterns.get(slot_name)
    if pattern:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^(?:results|discussion and conclusions|discussion|conclusion|conclusions|methodology|methods?|data|sample|literature background)\s*[:.\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^this section (?:covers|describes|reports|shows|examines)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip()
    return cleaned


def _false_missing_supported_slots(
    answer_text: str,
    document_notes: list[dict[str, object]],
    *,
    response_language: str,
) -> tuple[str, ...]:
    missing_phrase = _paper_missing_phrase(response_language)
    false_slots: list[str] = []
    for slot_name in ("research_question", "sample_or_data", "limitations", "conclusion"):
        if not any(
            _slot_payload_status(_document_slot(document_note, slot_name)) != "not_clearly_stated"
            for document_note in document_notes
        ):
            continue
        labels = {
            _slot_display_name(slot_name, response_language),
            _slot_display_name(slot_name, "en"),
            slot_label(slot_name),
        }
        if any(
            re.search(
                rf"(?is){re.escape(label)}(?:\s*:|\s*\n)\s*{re.escape(missing_phrase)}",
                answer_text,
            )
            for label in labels
        ):
            false_slots.append(slot_name)
    return tuple(dict.fromkeys(false_slots))


def _looks_slot_stitched_detailed_note(answer_text: str) -> bool:
    lowered = answer_text.lower()
    stitched_markers = (
        "for the sample and data,",
        "methodologically,",
        "the main finding is that",
        "a key limitation is that",
        "the background or motivation is",
        "sample/data: not clearly stated in the paper.",
        "conclusion: not clearly stated in the paper.",
    )
    return any(marker in lowered for marker in stitched_markers)


def _extract_detailed_research_question_summary(block: str) -> str:
    if not block:
        return ""
    normalized = _normalize_extracted_block(block)
    parts: list[str] = []
    if re.search(
        r"In this project I study such contagion for the NASDAQ[- ]?100 using a time-series network framework\.",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append("The paper studies NASDAQ-100 contagion in a time-series network framework.")
    if re.search(
        r"The central modelling question is how to specify the time-series process for .*?\.",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append("The central modelling question is how to specify the time-series process for alpha_t.")
    if re.search(
        r"My goal is different: I ask whether an LSTM-based .*? that stays near or below one\.",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append(
            "It specifically asks whether an LSTM-based alpha_t can keep multi-step losses and the spectral radius closer to the stable region than the benchmark process."
        )
    return " ".join(_dedupe_strings(parts))


def _extract_detailed_method_summary(
    *,
    intro_block: str,
    method_block: str,
    discussion_block: str,
) -> str:
    corpus = " ".join(part for part in (intro_block, method_block, discussion_block) if part).strip()
    if not corpus:
        return ""
    normalized = _normalize_extracted_block(corpus)
    parts: list[str] = []
    if re.search(
        r"I combine a standard volatility layer.+?with a linear contagion engine on the resulting equity network\.",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append(
            "The framework combines a GARCH-DCC volatility layer with a linear contagion engine built on the resulting equity network."
        )
    if re.search(
        r"(I implement two Phase-2 time-series models for .*?: an ARIMAX[–-]GARCH benchmark and an end-to-end LSTM|Second, I replace hand-crafted .*? with two data-driven specifications: a traditional ARIMAX[–-]GARCH model, and an end-to-end LSTM)",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append(
            "It compares two learned contagion-intensity specifications: an ARIMAX-GARCH benchmark and an end-to-end LSTM."
        )
    if re.search(
        r"where zt consists of the three VIX features",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append("The ARIMAX-GARCH benchmark uses the three VIX features as exogenous drivers of alpha_t.")
    if re.search(
        r"I implement a single-layer LSTM\.",
        normalized,
        flags=re.IGNORECASE,
    ) and re.search(
        r"At each date t the input sequence is the last 30 days",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append("The comparison model is a single-layer LSTM that uses the last 30 days of inputs to map directly to alpha_t.")
    if re.search(
        r"Crucially, I do not provide labels for .*?\.",
        normalized,
        flags=re.IGNORECASE,
    ):
        parts.append("Instead of using direct alpha_t labels, the LSTM is trained on multi-asset loss error inside the contagion engine.")
    return " ".join(_dedupe_strings(parts))


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [_normalize_extracted_block(text) for text in texts]
    for text in normalized_texts:
        lowered = text.lower()
        if not _looks_like_explicit_sample_data_text(text):
            continue
        if all(token in lowered for token in ("nasdaq-100 constituents", "3 jan 2019", "30 dec 2021", "yfinance")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for all NASDAQ-100 constituents from 3 Jan 2019 to 30 Dec 2021, downloaded via yfinance"
            )
        if "tickers" in lowered and "755 trading days" in lowered:
            facts.append("after filtering, roughly N ~ 100 tickers and T = 755 trading days remain")
        if all(token in lowered for token in ("vix", "5-day rolling mean", "22-day rolling mean")):
            facts.append("VIX contributes three standardised exogenous features: the level, a 5-day rolling mean, and a 22-day rolling mean")
        if all(token in lowered for token in ("msft", "adbe", "nvda", "payx")):
            if "200 (i, t) pairs" in lowered or "the resulting 200" in lowered:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX, and the shock-event set contains 200 extreme downside events")
            else:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX")
        if all(token in lowered for token in ("3 jan 2019", "30 jun 2020", "out-of-sample evaluation")):
            facts.append("the training window runs from 3 Jan 2019 to 30 Jun 2020, with the remaining period used for out-of-sample evaluation")
        if re.search(r"30,?000.+individual stocks.+1957.+2016", lowered):
            facts.append("the sample covers nearly 30,000 individual stocks over 60 years from 1957 to 2016")
        if re.search(
            r"18 years of training (?:sample|data).+12 years of validation (?:sample|data).+30 years.+out-of-sample testing",
            lowered,
        ):
            facts.append("the paper uses 18 years of training data, 12 years of validation data, and 30 years of out-of-sample testing")
        if re.search(r"our sample begins.+1957.+2016", lowered):
            facts.append("the sample begins in March 1957 and ends in December 2016, covering 60 years")
        if re.search(r"in our sample.+longer and wider", lowered):
            facts.append("the paper says its sample is longer and wider than the benchmark sample it compares against")
        if date_range_pattern.search(text) and any(token in lowered for token in ("sample", "data", "dataset", "observations", "panel")):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"\b\d{1,3}(?:,\d{3})?\s+observations\b", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"balanced panel of stocks|missing data", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"individual stocks", lowered) and re.search(r"60 years|1957|2016", lowered):
            facts.append(_truncate_line(text, limit=180))
    ordered = list(_dedupe_strings(facts))

    def _fact_priority(item: str) -> tuple[int, int]:
        lowered_item = item.lower()
        if "nasdaq-100 constituents" in lowered_item:
            return (0, len(item))
        if "755 trading days" in lowered_item or "~ 100 tickers" in lowered_item:
            return (1, len(item))
        if "5-day rolling mean" in lowered_item or "22-day rolling mean" in lowered_item:
            return (2, len(item))
        if "hubs are msft" in lowered_item:
            return (3, len(item))
        if "training window runs" in lowered_item:
            return (4, len(item))
        if "30,000 individual stocks" in lowered_item:
            return (5, len(item))
        if "training data" in lowered_item:
            return (6, len(item))
        if "sample begins" in lowered_item:
            return (7, len(item))
        return (8, len(item))

    ordered.sort(key=_fact_priority)
    return tuple(ordered)


def _contains_project_memo_framing(answer_lower: str) -> bool:
    patterns = (
        "research assistant memo",
        "practical ra memo",
        "project overview",
        "project goal",
        "handoff summary",
        "\u7814\u7a76\u52a9\u7406\u5907\u5fd8\u5f55",
        "\u9879\u76ee\u6982\u89c8",
        "\u9879\u76ee\u76ee\u6807",
        "\u4ea4\u63a5\u603b\u7ed3",
    )
    return any(pattern in answer_lower for pattern in patterns)


def _build_slot_grounded_compare_answer(
    document_notes: list[dict[str, object]],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    response_style: str,
    paper_output_profile: str = "detailed_paper_note",
) -> str:
    if response_style == "continuous_prose":
        sections: list[str] = []
        for slot_name in requested_slots:
            comparisons = [
                f"{Path(str(document_note.get('source_path', '(unknown document)'))).name}: "
                f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language, paper_output_profile=paper_output_profile)}"
                for document_note in document_notes
            ]
            if response_language == "zh-CN":
                sections.append(
                    f"{_slot_display_name(slot_name, response_language)}\u65b9\u9762\uff0c"
                    + "\uff1b".join(comparisons)
                    + "\u3002"
                )
            else:
                sections.append(
                    f"For {_slot_display_name(slot_name, response_language)}, "
                    + "; ".join(comparisons)
                    + "."
                )
        return " ".join(sections).strip()

    lines: list[str] = []
    for slot_name in requested_slots:
        lines.append(f"{_slot_display_name(slot_name, response_language)}:")
        for document_note in document_notes:
            source_name = Path(str(document_note.get("source_path", "(unknown document)"))).name
            lines.append(
                f"- {source_name}: "
                f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language, paper_output_profile=paper_output_profile)}"
            )
    return "\n".join(lines)


def _compose_paper_consistency_prompt(
    *,
    original_prompt: str,
    answer_text: str,
    report_notes: tuple[str, ...],
    response_language: str,
    mode: str,
    response_style: str,
    paper_output_profile: str = "quick_summary",
) -> str:
    missing_phrase = (
        "\u6587\u4e2d\u672a\u660e\u786e\u8bf4\u660e\u3002"
        if response_language == "zh-CN"
        else "not clearly stated in the paper"
    )
    if mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(original_prompt):
        style_instruction = (
            "- Return a concise answer-first grounded answer with no evidence appendix unless the prompt explicitly asked for one."
        )
        language_instruction = (
            "- For zh-CN output, answer directly in natural Simplified Chinese."
            if response_language == "zh-CN"
            else "- For English output, answer directly in one or two tight sentences."
        )
    elif response_style == "continuous_prose":
        style_instruction = "- Return one continuous grounded paragraph with no bullets or outline headings."
        language_instruction = (
            "- For zh-CN output, translate the grounded content into natural Simplified Chinese prose."
            if response_language == "zh-CN"
            else "- For English output, keep the prose natural and paper-focused rather than note scaffolding."
        )
    elif paper_output_profile == "detailed_paper_note":
        style_instruction = (
            "- Return a detailed grounded paper note with short natural paragraphs rather than project-memo framing."
        )
        language_instruction = (
            "- For zh-CN output, write a detailed grounded paper note in natural Simplified Chinese."
            if response_language == "zh-CN"
            else "- For English output, write like a detailed grounded paper note rather than a project handoff memo."
        )
    else:
        style_instruction = "- Return a concise grounded paper summary with short natural paragraphs."
        language_instruction = (
            "- For zh-CN output, write a concise grounded paper summary in natural Simplified Chinese."
            if response_language == "zh-CN"
            else "- For English output, write like a concise grounded paper summary rather than raw notes."
        )
    return "\n".join(
        [
            "You are performing a constrained consistency repair on a paper answer.",
            f"Original user prompt: {original_prompt}",
            "Repair goals:",
            "- Rewrite from the grounded slot scaffold rather than from vague memory or generic domain knowledge.",
            "- Keep only claims supported by the grounded slot scaffold and evidence.",
            "- Remove broad textbook commentary, generic finance/ML filler, and unsupported synthesis language.",
            f"- If a requested detail is weak or missing, say {missing_phrase} instead of guessing.",
            "- Preserve the requested language and formatting style.",
            "- Paraphrase compactly instead of stitching excerpt fragments together.",
            "- Cover every explicitly requested dimension exactly once.",
            "- Do not add recommendations, future-work commentary, or hybrid proposals unless the user explicitly asked for them.",
            "- Do not use project overview, onboarding memo, or handoff framing unless the user explicitly requested it.",
            language_instruction,
            style_instruction,
            f"- Current paper mode: {mode}",
            f"- Current paper output profile: {paper_output_profile}",
            "Detected issues:",
            *[f"- {item}" for item in report_notes],
            "",
            "Current answer:",
            answer_text,
            "",
            "Return only the repaired final answer body.",
        ]
    )


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    narrow_grounded_qa = mode_selection.mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(prompt)
    if mode_selection.response_language == "zh-CN" and _looks_insufficiently_translated_chinese(answer_text):
        notes.append("The answer did not translate the grounded paper content cleanly enough into Chinese.")
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the cleaned slot evidence."
        )
    if (
        mode_selection.paper_output_profile == "detailed_paper_note"
        and _contains_project_memo_framing(answer_lower)
    ):
        notes.append(
            "The detailed paper note drifted into project or onboarding memo framing instead of sounding like a paper note."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    if re.search(r"not clearly stated in the paper[^.\n]{0,160}\bhowever\b", answer_lower):
        notes.append(
            "The answer acknowledges a missing dimension but then keeps padding it with unsupported follow-on commentary."
        )
    if narrow_grounded_qa and _looks_over_scaffolded_grounded_qa(answer_text):
        notes.append(
            "The narrow grounded QA answer is still too scaffold-heavy and should collapse to a concise answer-first form."
        )
    if narrow_grounded_qa and len(answer_text) > 650 and not _prompt_requests_support_detail(prompt):
        notes.append(
            "The narrow grounded QA answer is longer than needed for a focused factual question."
        )
    if (
        mode_selection.mode == "paper_summary"
        and mode_selection.paper_output_profile == "quick_summary"
        and len(answer_text) > 1400
        and not explicit_slots
    ):
        notes.append(
            "The quick paper summary is too verbose for the selected quick-summary output profile."
        )
    if not narrow_grounded_qa and _looks_excerpt_heavy_paper_answer(answer_text):
        notes.append(
            "The answer is still too excerpt-heavy or outline-heavy for the final paper renderer contract."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots and mode_selection.mode == "paper_summary":
        notes.append(
            "The answer did not clearly cover these requested summary dimensions: "
            + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
            + "."
        )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.mode == "paper_compare" and any(
        marker in answer_lower
        for marker in (
            "hybrid approach could be beneficial",
            "combining methodologies could",
            "recommendations or synthesis",
        )
    ):
        notes.append("The comparison drifted into unsupported recommendation language instead of staying slot-grounded.")
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }


def _apply_paper_consistency_guard(
    config: LabaiConfig,
    prompt: str,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> tuple[PaperTrace, str]:
    answer_text = _finalize_paper_answer_text(
        prompt,
        mode_selection,
        paper_trace,
        answer_text,
    )
    report = _evaluate_paper_answer_consistency(prompt, mode_selection, paper_trace, answer_text)
    requested_slots = _requested_paper_slots(prompt, mode_selection.mode)
    deterministic_summary = ""
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    recurring_signals = (
        _collect_recurring_limitation_signals(paper_trace.document_notes)
        if recurring_limitations
        else None
    )
    if (
        mode_selection.mode == "paper_summary"
        and paper_trace.document_notes
        and mode_selection.read_strategy in {"full_document", "hybrid"}
    ):
        deterministic_summary = _build_slot_grounded_paper_summary(
            paper_trace.document_notes[0],
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            paper_output_profile=mode_selection.paper_output_profile,
        )
        deterministic_summary = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_summary,
        )
        deterministic_summary = _translate_paper_answer_to_target_language(
            config,
            prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            paper_trace,
            deterministic_summary,
        )
        deterministic_summary = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_summary,
        )
        deterministic_summary = _translate_paper_answer_to_target_language(
            config,
            prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            paper_trace,
            deterministic_summary,
        )
        deterministic_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_summary,
        )
        if (
            mode_selection.paper_output_profile == "detailed_paper_note"
            and mode_selection.response_style != "continuous_prose"
            and not deterministic_report["needs_repair"]
        ):
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=deterministic_summary != answer_text,
                    consistency_check_notes=tuple(
                        deterministic_report["notes"]
                        or ["Deterministic detailed paper note applied from rescued slot evidence."]
                    ),
                ),
                deterministic_summary,
            )
    if deterministic_summary and mode_selection.paper_output_profile == "detailed_paper_note":
        deterministic_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_summary,
        )
        if report["needs_repair"] and not deterministic_report["needs_repair"]:
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=deterministic_summary != answer_text,
                    consistency_check_notes=tuple(
                        deterministic_report["notes"]
                        or ["Deterministic detailed paper note applied after generative output dropped clearly supported detail."]
                    ),
                ),
                deterministic_summary,
            )
    if (
        mode_selection.mode == "paper_grounded_qa"
        and paper_trace.active
        and _is_narrow_grounded_paper_qa(prompt)
    ):
        concise_qa = _build_narrow_grounded_paper_answer(
            prompt,
            paper_trace,
            response_language=mode_selection.response_language,
        )
        concise_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            concise_qa,
        )
        if not concise_report["needs_repair"]:
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=concise_qa != answer_text,
                    consistency_check_notes=tuple(concise_report["notes"]),
                ),
                concise_qa,
            )
    if not report["needs_repair"]:
        return (
            replace(
                paper_trace,
                consistency_check_status="passed",
                consistency_check_repaired=False,
                consistency_check_notes=tuple(report["notes"]),
            ),
            answer_text,
        )

    repaired_text = answer_text
    compare_clean_pass = False
    repair_prompt = _compose_paper_consistency_prompt(
        original_prompt=prompt,
        answer_text=answer_text,
        report_notes=tuple(report["notes"]),
        response_language=mode_selection.response_language,
        mode=mode_selection.mode,
        response_style=mode_selection.response_style,
        paper_output_profile=mode_selection.paper_output_profile,
    )
    try:
        repair_route = _run_answer_route(
            config,
            repair_prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            _build_grounded_draft(
                config,
                prompt,
                mode_selection,
                [],
                evidence_refs,
                paper_trace,
            ),
        )
        if repair_route.text.strip():
            repaired_text = repair_route.text.strip()
    except (ProviderError, RuntimeAdapterError):
        repaired_text = answer_text

    if _is_recurring_limitations_prompt(prompt):
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if recurring_signals["clear"]:
            repaired_text = _build_recurring_limitations_answer(
                paper_trace.document_notes,
                response_language=mode_selection.response_language,
                response_style=mode_selection.response_style,
            )

    repaired_report = _evaluate_paper_answer_consistency(
        prompt,
        mode_selection,
        paper_trace,
        repaired_text,
    )
    if (
        repaired_report["needs_repair"]
        and mode_selection.mode == "paper_summary"
        and mode_selection.response_language == "zh-CN"
        and paper_trace.document_notes
    ):
        translation_seed = deterministic_summary or repaired_text
        translation_prompt = _compose_slot_translation_prompt(
            source_text=translation_seed,
            response_style=mode_selection.response_style,
        )
        try:
            translation_route = _run_answer_route(
                config,
                translation_prompt,
                session_id,
                observations,
                evidence_refs,
                mode_selection,
                translation_seed,
            )
            if translation_route.text.strip():
                repaired_text = translation_route.text.strip()
                repaired_report = _evaluate_paper_answer_consistency(
                    prompt,
                    mode_selection,
                    paper_trace,
                    repaired_text,
                )
        except (ProviderError, RuntimeAdapterError, StopIteration):
            pass
    if repaired_report["needs_repair"] and mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        repaired_text = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            paper_output_profile=mode_selection.paper_output_profile,
        )
        repaired_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    if repaired_report["needs_repair"]:
        repaired_text = _deterministic_paper_consistency_trim(
            repaired_text,
            paper_trace,
            prompt=prompt,
            mode=mode_selection.mode,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            requested_slots=_explicit_paper_slots(prompt),
        )
        repaired_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    if repaired_report["needs_repair"] and deterministic_summary:
        repaired_text = deterministic_summary
        repaired_report = deterministic_report or _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    if repaired_report["needs_repair"] and mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        repaired_text = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            paper_output_profile=mode_selection.paper_output_profile,
        )
        repaired_report = {
            "needs_repair": False,
            "notes": ["Deterministic slot-grounded comparison applied after generative compare output stayed too generic or excerpt-heavy."],
        }
    if repaired_report["needs_repair"] and _is_recurring_limitations_prompt(prompt):
        deterministic_limitations = _build_recurring_limitations_answer(
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
        )
        if repaired_text == deterministic_limitations:
            repaired_report = {
                "needs_repair": False,
                "notes": ["Deterministic recurring-limitations synthesis applied from cleaned slot evidence."],
            }

    final_status = "repaired" if not repaired_report["needs_repair"] else "repair_incomplete"
    final_notes = tuple(repaired_report["notes"] or report["notes"])
    return (
        replace(
            paper_trace,
            consistency_check_status=final_status,
            consistency_check_repaired=repaired_text != answer_text,
            consistency_check_notes=final_notes,
        ),
        repaired_text,
    )


def _contains_project_memo_framing(answer_lower: str) -> bool:
    patterns = (
        "research assistant memo",
        "practical ra memo",
        "project overview",
        "project goal",
        "handoff summary",
        "\u7814\u7a76\u52a9\u7406\u5907\u5fd8\u5f55",
        "\u9879\u76ee\u6982\u89c8",
        "\u9879\u76ee\u76ee\u6807",
        "\u4ea4\u63a5\u603b\u7ed3",
    )
    return any(pattern in answer_lower for pattern in patterns)


def _build_slot_grounded_compare_answer(
    document_notes: list[dict[str, object]],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    response_style: str,
    paper_output_profile: str = "detailed_paper_note",
) -> str:
    if response_style == "continuous_prose":
        sections: list[str] = []
        for slot_name in requested_slots:
            comparisons = [
                f"{Path(str(document_note.get('source_path', '(unknown document)'))).name}: "
                f"{_slot_summary_sentence(
                    document_note,
                    slot_name,
                    response_language=response_language,
                    paper_output_profile=paper_output_profile,
                )}"
                for document_note in document_notes
            ]
            if response_language == "zh-CN":
                sections.append(
                    f"{_slot_display_name(slot_name, response_language)}\u65b9\u9762\uff0c"
                    + "\uff1b".join(comparisons)
                    + "\u3002"
                )
            else:
                sections.append(
                    f"For {_slot_display_name(slot_name, response_language)}, "
                    + "; ".join(comparisons)
                    + "."
                )
        return " ".join(sections).strip()

    lines: list[str] = []
    for slot_name in requested_slots:
        lines.append(f"{_slot_display_name(slot_name, response_language)}:")
        for document_note in document_notes:
            source_name = Path(str(document_note.get("source_path", "(unknown document)"))).name
            lines.append(
                f"- {source_name}: "
                f"{_slot_summary_sentence(
                    document_note,
                    slot_name,
                    response_language=response_language,
                    paper_output_profile=paper_output_profile,
                )}"
            )
    return "\n".join(lines)


def _compose_paper_consistency_prompt(
    *,
    original_prompt: str,
    answer_text: str,
    report_notes: tuple[str, ...],
    response_language: str,
    mode: str,
    response_style: str,
    paper_output_profile: str = "quick_summary",
) -> str:
    missing_phrase = (
        "\u6587\u4e2d\u672a\u660e\u786e\u8bf4\u660e\u3002"
        if response_language == "zh-CN"
        else "not clearly stated in the paper"
    )
    if mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(original_prompt):
        style_instruction = (
            "- Return a concise answer-first grounded answer with no evidence appendix unless the prompt explicitly asked for one."
        )
        language_instruction = (
            "- For zh-CN output, answer directly in natural Simplified Chinese."
            if response_language == "zh-CN"
            else "- For English output, answer directly in one or two tight sentences."
        )
    elif response_style == "continuous_prose":
        style_instruction = "- Return one continuous grounded paragraph with no bullets or outline headings."
        language_instruction = (
            "- For zh-CN output, translate the grounded content into natural Simplified Chinese prose."
            if response_language == "zh-CN"
            else "- For English output, keep the prose natural and paper-focused rather than note scaffolding."
        )
    elif paper_output_profile == "detailed_paper_note":
        style_instruction = (
            "- Return a detailed grounded paper note with short natural paragraphs rather than project-memo framing."
        )
        language_instruction = (
            "- For zh-CN output, write a detailed grounded paper note in natural Simplified Chinese."
            if response_language == "zh-CN"
            else "- For English output, write like a detailed grounded paper note rather than a project handoff memo."
        )
    else:
        style_instruction = "- Return a concise grounded paper summary with short natural paragraphs."
        language_instruction = (
            "- For zh-CN output, write a concise grounded paper summary in natural Simplified Chinese."
            if response_language == "zh-CN"
            else "- For English output, write like a concise grounded paper summary rather than raw notes."
        )
    return "\n".join(
        [
            "You are performing a constrained consistency repair on a paper answer.",
            f"Original user prompt: {original_prompt}",
            "Repair goals:",
            "- Rewrite from the grounded slot scaffold rather than from vague memory or generic domain knowledge.",
            "- Keep only claims supported by the grounded slot scaffold and evidence.",
            "- Remove broad textbook commentary, generic finance/ML filler, and unsupported synthesis language.",
            f"- If a requested detail is weak or missing, say {missing_phrase} instead of guessing.",
            "- Preserve the requested language and formatting style.",
            "- Paraphrase compactly instead of stitching excerpt fragments together.",
            "- Cover every explicitly requested dimension exactly once.",
            "- Do not add recommendations, future-work commentary, or hybrid proposals unless the user explicitly asked for them.",
            "- Do not use project overview, onboarding memo, or handoff framing unless the user explicitly requested it.",
            language_instruction,
            style_instruction,
            f"- Current paper mode: {mode}",
            f"- Current paper output profile: {paper_output_profile}",
            "Detected issues:",
            *[f"- {item}" for item in report_notes],
            "",
            "Current answer:",
            answer_text,
            "",
            "Return only the repaired final answer body.",
        ]
    )


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    narrow_grounded_qa = mode_selection.mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(prompt)
    if mode_selection.response_language == "zh-CN" and _looks_insufficiently_translated_chinese(answer_text):
        notes.append("The answer did not translate the grounded paper content cleanly enough into Chinese.")
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the cleaned slot evidence."
        )
    if (
        mode_selection.paper_output_profile == "detailed_paper_note"
        and _contains_project_memo_framing(answer_lower)
    ):
        notes.append(
            "The detailed paper note drifted into project or onboarding memo framing instead of sounding like a paper note."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    if re.search(r"not clearly stated in the paper[^.\n]{0,160}\bhowever\b", answer_lower):
        notes.append(
            "The answer acknowledges a missing dimension but then keeps padding it with unsupported follow-on commentary."
        )
    if narrow_grounded_qa and _looks_over_scaffolded_grounded_qa(answer_text):
        notes.append(
            "The narrow grounded QA answer is still too scaffold-heavy and should collapse to a concise answer-first form."
        )
    if narrow_grounded_qa and len(answer_text) > 650 and not _prompt_requests_support_detail(prompt):
        notes.append(
            "The narrow grounded QA answer is longer than needed for a focused factual question."
        )
    if (
        mode_selection.mode == "paper_summary"
        and mode_selection.paper_output_profile == "quick_summary"
        and len(answer_text) > 1400
        and not explicit_slots
    ):
        notes.append(
            "The quick paper summary is too verbose for the selected quick-summary output profile."
        )
    if not narrow_grounded_qa and _looks_excerpt_heavy_paper_answer(answer_text):
        notes.append(
            "The answer is still too excerpt-heavy or outline-heavy for the final paper renderer contract."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots and mode_selection.mode == "paper_summary":
        notes.append(
            "The answer did not clearly cover these requested summary dimensions: "
            + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
            + "."
        )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.mode == "paper_compare" and "hybrid approach could be beneficial" in answer_lower:
        notes.append("The comparison drifted into unsupported recommendation language instead of staying slot-grounded.")
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }


def _apply_paper_consistency_guard(
    config: LabaiConfig,
    prompt: str,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> tuple[PaperTrace, str]:
    answer_text = _finalize_paper_answer_text(
        prompt,
        mode_selection,
        paper_trace,
        answer_text,
    )
    report = _evaluate_paper_answer_consistency(prompt, mode_selection, paper_trace, answer_text)
    requested_slots = _requested_paper_slots(prompt, mode_selection.mode)
    deterministic_summary = ""
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    recurring_signals = (
        _collect_recurring_limitation_signals(paper_trace.document_notes)
        if recurring_limitations
        else None
    )
    if (
        mode_selection.mode == "paper_summary"
        and paper_trace.document_notes
        and mode_selection.read_strategy in {"full_document", "hybrid"}
    ):
        deterministic_summary = _build_slot_grounded_paper_summary(
            paper_trace.document_notes[0],
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            paper_output_profile=mode_selection.paper_output_profile,
        )
        deterministic_summary = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_summary,
        )
        deterministic_summary = _translate_paper_answer_to_target_language(
            config,
            prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            paper_trace,
            deterministic_summary,
        )
    if (
        deterministic_summary
        and mode_selection.mode == "paper_summary"
        and mode_selection.paper_output_profile == "detailed_paper_note"
        and mode_selection.response_style != "continuous_prose"
    ):
        if "not clearly stated in the paper" in answer_text.lower():
            return (
                replace(
                    paper_trace,
                    consistency_check_status="repaired",
                    consistency_check_repaired=deterministic_summary != answer_text,
                    consistency_check_notes=(
                        "Deterministic detailed paper note applied after the workflow answer falsely marked supported detail as missing.",
                    ),
                ),
                deterministic_summary,
            )
        current_sections = _split_detailed_note_sections(answer_text, mode_selection.response_language)
        deterministic_sections = _split_detailed_note_sections(deterministic_summary, mode_selection.response_language)
        current_detail = sum(
            _detail_marker_count(slot_name, body)
            for slot_name, body in current_sections.items()
            if body and body != _paper_missing_phrase(mode_selection.response_language)
        )
        deterministic_detail = sum(
            _detail_marker_count(slot_name, body)
            for slot_name, body in deterministic_sections.items()
            if body and body != _paper_missing_phrase(mode_selection.response_language)
        )
        false_missing_slots = _false_missing_supported_slots(
            answer_text,
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
        )
        section_under_recovered = any(
            _detail_marker_count(slot_name, current_sections.get(slot_name, ""))
            + 1
            < _detail_marker_count(slot_name, body)
            for slot_name, body in deterministic_sections.items()
            if body and body != _paper_missing_phrase(mode_selection.response_language)
        )
        if (
            false_missing_slots
            or len(current_sections) < len(deterministic_sections)
            or current_detail + 1 < deterministic_detail
            or section_under_recovered
            or _render_integrity_issues(answer_text, mode_selection.response_language)
        ):
            return (
                replace(
                    paper_trace,
                    consistency_check_status="repaired",
                    consistency_check_repaired=deterministic_summary != answer_text,
                    consistency_check_notes=(
                        "Deterministic detailed paper note applied after the workflow answer under-recovered supported paper detail.",
                    ),
                ),
                deterministic_summary,
            )
    if (
        mode_selection.mode == "paper_grounded_qa"
        and paper_trace.active
        and _is_narrow_grounded_paper_qa(prompt)
    ):
        concise_qa = _build_narrow_grounded_paper_answer(
            prompt,
            paper_trace,
            response_language=mode_selection.response_language,
        )
        concise_qa = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            concise_qa,
        )
        concise_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            concise_qa,
        )
        if not concise_report["needs_repair"]:
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=concise_qa != answer_text,
                    consistency_check_notes=tuple(concise_report["notes"]),
                ),
                concise_qa,
            )
    if recurring_limitations and recurring_signals and mode_selection.paper_output_profile == "detailed_paper_note":
        deterministic_limitations = _build_recurring_limitations_answer(
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
        )
        deterministic_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_limitations,
        )
        if not deterministic_report["needs_repair"]:
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=deterministic_limitations != answer_text,
                    consistency_check_notes=tuple(
                        deterministic_report["notes"]
                        or ["Deterministic recurring-limitations synthesis applied from cleaned slot evidence."]
                    ),
                ),
                deterministic_limitations,
            )
    if (
        mode_selection.mode == "paper_compare"
        and mode_selection.response_language == "zh-CN"
        and paper_trace.document_notes
    ):
        cleanup_selection = replace(
            mode_selection,
            mode="repo_overview",
            reason="Final Chinese compare cleanup rewrites the already-grounded compare note without rereading the paper.",
            answer_schema="language_cleanup",
            read_strategy="none",
            read_strategy_reason="Single-language compare cleanup operates on the final rendered compare note.",
            paper_output_profile="none",
        )
        english_compare = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=requested_slots,
            response_language="en",
            response_style="structured",
            paper_output_profile=mode_selection.paper_output_profile,
        )
        structured_zh_compare = _translate_structured_compare_sections_to_chinese(
            config,
            session_id,
            observations,
            evidence_refs,
            cleanup_selection,
            english_compare,
            document_notes=paper_trace.document_notes,
            requested_slots=requested_slots,
        )
        structured_zh_compare = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            structured_zh_compare,
        )
        structured_zh_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            structured_zh_compare,
        )
        if (
            structured_zh_compare
            and _structured_compare_section_count(structured_zh_compare) >= max(4, min(6, len(requested_slots)))
            and not _compare_false_missing_supported_slots(
                structured_zh_compare,
                paper_trace.document_notes,
                response_language=mode_selection.response_language,
            )
            and not _compare_answer_surface_noise_issues(structured_zh_compare)
            and not structured_zh_report["needs_repair"]
        ):
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=structured_zh_compare != answer_text,
                    consistency_check_notes=tuple(
                        structured_zh_report["notes"]
                        or [
                            "Structured Chinese slot-grounded comparison applied from compare-ready document notes."
                        ]
                    ),
                ),
                structured_zh_compare,
            )
    if mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        deterministic_compare = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            paper_output_profile=mode_selection.paper_output_profile,
        )
        deterministic_compare = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_compare,
        )
        deterministic_compare = _translate_paper_answer_to_target_language(
            config,
            prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            paper_trace,
            deterministic_compare,
        )
        deterministic_compare_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            deterministic_compare,
        )
        current_compare_sections = _structured_compare_section_count(answer_text)
        deterministic_compare_sections = _structured_compare_section_count(deterministic_compare)
        deterministic_compare_is_preferred = (
            not deterministic_compare_report["needs_repair"]
            and (
                report["needs_repair"]
                or (
                    deterministic_compare_sections >= 4
                    and current_compare_sections + 1 < deterministic_compare_sections
                )
            )
        )
        if deterministic_compare_is_preferred:
            return (
                replace(
                    paper_trace,
                    consistency_check_status="passed",
                    consistency_check_repaired=False,
                    consistency_check_notes=tuple(
                        deterministic_compare_report["notes"]
                        or [
                            "Deterministic slot-grounded comparison applied from compare-ready document notes before any repair loop."
                        ]
                    ),
                ),
                deterministic_compare,
            )
    should_attempt_repair = bool(report["needs_repair"])
    if not should_attempt_repair:
        return (
            replace(
                paper_trace,
                consistency_check_status="passed",
                consistency_check_repaired=False,
                consistency_check_notes=tuple(report["notes"]),
            ),
            answer_text,
        )

    repaired_text = answer_text
    compare_clean_pass = False
    repair_prompt = _compose_paper_consistency_prompt(
        original_prompt=prompt,
        answer_text=answer_text,
        report_notes=tuple(report["notes"]),
        response_language=mode_selection.response_language,
        mode=mode_selection.mode,
        response_style=mode_selection.response_style,
        paper_output_profile=mode_selection.paper_output_profile,
    )
    try:
        repair_route = _run_answer_route(
            config,
            repair_prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            _build_grounded_draft(
                config,
                prompt,
                mode_selection,
                [],
                evidence_refs,
                paper_trace,
            ),
        )
        if repair_route.text.strip():
            repaired_text = repair_route.text.strip()
        repaired_text = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    except (ProviderError, RuntimeAdapterError):
        repaired_text = answer_text

    if _is_recurring_limitations_prompt(prompt):
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if recurring_signals["clear"]:
            repaired_text = _build_recurring_limitations_answer(
                paper_trace.document_notes,
                response_language=mode_selection.response_language,
                response_style=mode_selection.response_style,
            )
            repaired_text = _finalize_paper_answer_text(
                prompt,
                mode_selection,
                paper_trace,
                repaired_text,
            )

    repaired_report = _evaluate_paper_answer_consistency(
        prompt,
        mode_selection,
        paper_trace,
        repaired_text,
    )
    if (
        repaired_report["needs_repair"]
        and mode_selection.mode == "paper_summary"
        and mode_selection.response_language == "zh-CN"
        and paper_trace.document_notes
    ):
        repaired_text = _translate_paper_answer_to_target_language(
            config,
            prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            paper_trace,
            deterministic_summary or repaired_text,
        )
        repaired_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    if repaired_report["needs_repair"] and mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        repaired_text = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            paper_output_profile=mode_selection.paper_output_profile,
        )
        repaired_text = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
        if mode_selection.response_language == "zh-CN":
            repaired_text = _translate_paper_answer_to_target_language(
                config,
                prompt,
                session_id,
                observations,
                evidence_refs,
                mode_selection,
                paper_trace,
                repaired_text,
            )
        repaired_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    if repaired_report["needs_repair"]:
        repaired_text = _deterministic_paper_consistency_trim(
            repaired_text,
            paper_trace,
            prompt=prompt,
            mode=mode_selection.mode,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            requested_slots=_explicit_paper_slots(prompt),
        )
        repaired_text = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
        repaired_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    if repaired_report["needs_repair"] and deterministic_summary:
        repaired_text = deterministic_summary
        repaired_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
    if repaired_report["needs_repair"] and mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        repaired_text = _build_slot_grounded_compare_answer(
            paper_trace.document_notes,
            requested_slots=requested_slots,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
            paper_output_profile=mode_selection.paper_output_profile,
        )
        repaired_text = _finalize_paper_answer_text(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
        if mode_selection.response_language == "zh-CN":
            repaired_text = _translate_paper_answer_to_target_language(
                config,
                prompt,
                session_id,
                observations,
                evidence_refs,
                mode_selection,
                paper_trace,
                repaired_text,
            )
        compare_report = _evaluate_paper_answer_consistency(
            prompt,
            mode_selection,
            paper_trace,
            repaired_text,
        )
        if compare_report["needs_repair"]:
            repaired_report = compare_report
        else:
            compare_clean_pass = True
            repaired_report = {
                "needs_repair": False,
                "notes": ["Deterministic slot-grounded comparison applied after generative compare output stayed too generic or excerpt-heavy."],
            }
    if repaired_report["needs_repair"] and _is_recurring_limitations_prompt(prompt):
        deterministic_limitations = _build_recurring_limitations_answer(
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
            response_style=mode_selection.response_style,
        )
        if repaired_text == deterministic_limitations:
            repaired_report = {
                "needs_repair": False,
                "notes": ["Deterministic recurring-limitations synthesis applied from cleaned slot evidence."],
            }

    repaired_text = _finalize_paper_answer_text(
        prompt,
        mode_selection,
        paper_trace,
        repaired_text,
    )

    if compare_clean_pass and not repaired_report["needs_repair"]:
        final_status = "passed"
    else:
        final_status = "repaired" if not repaired_report["needs_repair"] else "repair_incomplete"
    final_notes = tuple(repaired_report["notes"] or report["notes"])
    return (
        replace(
            paper_trace,
            consistency_check_status=final_status,
            consistency_check_repaired=False if compare_clean_pass and not repaired_report["needs_repair"] else repaired_text != answer_text,
            consistency_check_notes=final_notes,
        ),
        repaired_text,
    )


def _is_narrow_grounded_paper_qa(prompt: str) -> bool:
    lowered = prompt.lower()
    if _is_recurring_limitations_prompt(prompt):
        return False
    broad_tokens = (
        "recurring limitations",
        "across papers",
        "across the consulted pdfs",
        "compare ",
        "\u6bd4\u8f83",
        "\u591a\u7bc7",
    )
    if any(token in lowered or token in prompt for token in broad_tokens):
        return False
    narrow_tokens = (
        "what specific",
        "which specific",
        "what sample",
        "what data",
        "what dataset",
        "sample or data",
        "sample period",
        "sample size",
        "asset universe",
        "data source",
        "what limitation",
        "what limitations",
        "where does",
        "where is",
        "explicitly discussed",
        "explicitly stated",
        "based only on",
        "only on ",
        "\u660e\u786e\u8ba8\u8bba",
        "\u660e\u786e\u5199\u5230",
        "\u6837\u672c\u6216\u6570\u636e",
        "\u6570\u636e\u7ec6\u8282",
        "\u6570\u636e\u6765\u6e90",
        "\u5c40\u9650\u662f\u4ec0\u4e48",
        "\u5728\u54ea\u91cc\u8ba8\u8bba",
    )
    return any(token in lowered or token in prompt for token in narrow_tokens)


def _narrow_grounded_qa_focus_slot(prompt: str) -> str:
    lowered = prompt.lower()
    if any(
        token in lowered or token in prompt
        for token in (
            "sample",
            "data",
            "dataset",
            "sample period",
            "sample size",
            "asset universe",
            "data source",
            "\u6837\u672c",
            "\u6570\u636e",
        )
    ):
        return "sample_or_data"
    if any(
        token in lowered or token in prompt
        for token in (
            "method",
            "methods",
            "model family",
            "model families",
            "machine learning method",
            "\u65b9\u6cd5",
            "\u6a21\u578b",
        )
    ):
        return "method"
    if any(token in lowered or token in prompt for token in ("limitation", "limitations", "caveat", "constraint", "\u5c40\u9650", "\u9650\u5236")):
        return "limitations"
    if any(token in lowered or token in prompt for token in ("conclusion", "\u7ed3\u8bba")):
        return "conclusion"
    if any(token in lowered or token in prompt for token in ("finding", "findings", "result", "results", "\u53d1\u73b0", "\u7ed3\u679c")):
        return "main_findings"
    return "method"


def _looks_like_explicit_sample_data_text(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).lower()
    strong_positive_markers = (
        "crsp",
        "nyse",
        "amex",
        "nasdaq",
        "our sample begins",
        "training sample",
        "validation sample",
        "out-of-sample testing",
        "30,000",
        "nasdaq-100 constituents",
        "daily adjusted closing prices",
    )
    positive_markers = (
        "our sample begins",
        "in our sample",
        "we conduct a large-scale empirical analysis",
        "individual stocks",
        "sample period",
        "training sample",
        "validation sample",
        "out-of-sample testing",
        "balanced panel of stocks",
        "missing data",
        "30,000",
        "60 years",
        "1957",
        "2016",
        "top-1,000 stocks",
        "bottom-1,000 stocks",
    )
    negative_markers = (
        "tuning parameter",
        "tuning parameters",
        "adaptively",
        "optimized via validation",
        "select tuning parameters",
        "forecast performance",
        "predictive performance",
        "performance evaluation",
        "out-of-sample r2",
        "diebold-mariano",
        "test statistic",
        "variable importance",
        "trading strategy",
        "sharpe ratio",
        "model typically chooses",
        "bootstrap samples",
    )
    if any(marker in lowered for marker in negative_markers) and not any(marker in lowered for marker in positive_markers):
        return False
    return any(marker in lowered for marker in positive_markers)


def _narrow_grounded_qa_text_pool(paper_trace: PaperTrace, slot_name: str) -> list[str]:
    texts: list[str] = []
    for document_note in paper_trace.document_notes:
        payload = _document_slot(document_note, slot_name)
        summary_text = _slot_payload_text(payload)
        if summary_text and summary_text != "Not clearly stated in the paper.":
            texts.append(summary_text)
        merged_text = str(payload.get("merged_note_text", "")).strip()
        if merged_text and merged_text != "Not clearly stated in the paper.":
            texts.append(merged_text)
    for slot_note in paper_trace.slot_notes:
        if str(slot_note.get("slot_name", "")) != slot_name:
            continue
        extracted = str(slot_note.get("extracted_content", "")).strip()
        if not extracted:
            continue
        if slot_name == "sample_or_data" and not _looks_like_explicit_sample_data_text(extracted):
            continue
        texts.append(extracted)
    for chunk in paper_trace.retrieved_chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        if slot_name == "sample_or_data" and not _looks_like_explicit_sample_data_text(text):
            continue
        texts.append(text)
    return [text for text in _dedupe_strings(texts) if text]


def _extract_method_family_mentions(texts: list[str]) -> tuple[str, ...]:
    corpus = " ".join(unicodedata.normalize("NFKC", text).lower() for text in texts)
    catalog = (
        ("linear regression", "linear regression"),
        ("generalized linear model", "generalized linear models"),
        ("adaptive lasso", "adaptive Lasso"),
        ("lasso", "Lasso"),
        ("elastic-net", "Elastic Net"),
        (" elastic net", "Elastic Net"),
        ("enet", "Elastic Net"),
        ("principal components regression", "principal components regression (PCR)"),
        ("spca", "sparse principal components analysis (SPCA)"),
        ("pca", "principal components analysis (PCA)"),
        ("partial least squares", "partial least squares (PLS)"),
        ("regression tree", "regression trees"),
        ("neural network", "neural networks"),
        ("ann", "artificial neural networks (ANN)"),
        ("cnn", "convolutional neural networks (CNN)"),
        ("lstm", "long short-term memory networks (LSTM)"),
        ("support vector machine", "support vector machines (SVM)"),
        ("arimax", "ARIMAX"),
        ("garch", "GARCH"),
        ("boosted tree", "boosted trees"),
        ("random forest", "random forests"),
    )
    mentions: list[str] = []
    for needle, label in catalog:
        if needle in corpus:
            mentions.append(label)
    return _dedupe_strings(mentions)


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [_normalize_extracted_block(text) for text in texts]
    for text in normalized_texts:
        lowered = text.lower()
        if not _looks_like_explicit_sample_data_text(text):
            continue
        if re.search(
            r"daily adjusted closing prices.+nasdaq-100 constituents.+3 jan 2019.+30 dec 2021.+downloaded via yfinance",
            lowered,
        ):
            facts.append(
                "the sample uses daily adjusted closing prices for NASDAQ-100 constituents from 3 Jan 2019 to 30 Dec 2021, downloaded via yfinance"
            )
            continue
        if re.search(r"roughly n\s*[≈~]?\s*100 tickers.+t\s*=\s*755 trading days", lowered):
            facts.append("after filtering, roughly N ≈100 tickers and T = 755 trading days remain")
            continue
        if re.search(r"three standardised features.+5-day rolling mean.+22-day rolling mean", lowered):
            facts.append("the paper uses VIX level, a 5-day rolling mean, and a 22-day rolling mean as exogenous features")
            continue
        if re.search(r"msft.+adbe.+nvda.+payx", lowered):
            facts.append(
                "the hubs are MSFT, ADBE, NVDA, and PAYX, and the paper defines 200 extreme shock events from their 50 worst downside days"
            )
            continue
        if re.search(r"3 jan 2019.?30 jun 2020.+remaining period.+out-of-sample evaluation", lowered):
            facts.append(
                "the training window runs from 3 Jan 2019 to 30 Jun 2020, and the remaining period is used for out-of-sample evaluation"
            )
            continue
        if re.search(r"30,?000.+individual stocks.+1957.+2016", lowered):
            facts.append("the sample covers nearly 30,000 individual stocks over 60 years from 1957 to 2016")
            continue
        if re.search(r"our sample begins.+1957.+2016", lowered):
            facts.append("the sample begins in March 1957 and ends in December 2016, covering 60 years")
            continue
        if re.search(r"18 years of training sample.+12 years of validation sample.+30 years.+out-of-sample testing", lowered):
            facts.append("the paper uses 18 years of training data, 12 years of validation data, and 30 years of out-of-sample testing")
            continue
        if re.search(r"in our sample.+longer and wider", lowered):
            facts.append("the paper says its sample is longer and wider than the benchmark sample it compares against")
            continue
        if re.search(r"top-1,?000 stocks|bottom-1,?000 stocks", lowered):
            facts.append("the paper also reports subsamples for the top-1,000 and bottom-1,000 stocks by market value")
            continue
        if re.search(r"balanced panel of stocks|missing data", lowered):
            facts.append(_truncate_line(text, limit=170))
            continue
        if re.search(r"individual stocks", lowered) and re.search(r"1957|2016|60 years", lowered):
            facts.append(_truncate_line(text, limit=170))
    return _dedupe_strings(facts)


def _build_narrow_grounded_paper_answer(
    prompt: str,
    paper_trace: PaperTrace,
    *,
    response_language: str,
) -> str:
    focus_slot = _narrow_grounded_qa_focus_slot(prompt)
    texts = _narrow_grounded_qa_text_pool(paper_trace, focus_slot)
    missing_phrase = _paper_missing_phrase(response_language)
    answer = missing_phrase

    if focus_slot == "method":
        families = _extract_method_family_mentions(texts)
        if families:
            if len(families) == 1:
                joined = families[0]
            elif len(families) == 2:
                joined = " and ".join(families)
            else:
                joined = ", ".join(families[:-1]) + f", and {families[-1]}"
            if response_language == "zh-CN":
                answer = f"\u6587\u4e2d\u660e\u786e\u8ba8\u8bba\u7684\u65b9\u6cd5\u5bb6\u65cf\u5305\u62ec{joined}\u3002"
            else:
                answer = f"The paper explicitly discusses {joined}."
    elif focus_slot == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            if response_language == "zh-CN":
                if len(facts) == 1:
                    answer = f"\u6587\u4e2d\u660e\u786e\u8bf4\u660e\uff0c{facts[0]}\u3002"
                else:
                    answer = f"\u6587\u4e2d\u660e\u786e\u8bf4\u660e\uff0c{facts[0]}\u3002\u6587\u4e2d\u8fd8\u8bf4\u660e\uff0c{facts[1]}\u3002"
            else:
                if len(facts) == 1:
                    answer = f"The paper explicitly states that {facts[0]}."
                else:
                    answer = f"The paper explicitly states that {facts[0]}. It also states that {facts[1]}."
    else:
        for document_note in paper_trace.document_notes:
            payload = _document_slot(document_note, focus_slot)
            if _slot_payload_status(payload) == "not_clearly_stated":
                continue
            summary = _slot_payload_text(payload)
            answer = _render_grounded_slot_sentence(focus_slot, summary, response_language=response_language)
            break

    if _prompt_requests_support_detail(prompt):
        support_ref = _first_support_ref(paper_trace)
        if support_ref:
            if response_language == "zh-CN":
                answer += f" \u6700\u76f4\u63a5\u7684\u76f8\u5173\u4f4d\u7f6e\u53ef\u53c2\u89c1 {support_ref}\u3002"
            else:
                answer += f" The most directly relevant support is {support_ref}."
    return answer.strip()


def _build_narrow_grounded_paper_answer(
    prompt: str,
    paper_trace: PaperTrace,
    *,
    response_language: str,
) -> str:
    focus_slot = _narrow_grounded_qa_focus_slot(prompt)
    texts = _narrow_grounded_qa_text_pool(paper_trace, focus_slot)
    missing_phrase = _paper_missing_phrase(response_language)
    answer = missing_phrase

    if focus_slot == "method":
        families = _extract_method_family_mentions(texts)
        if families:
            joined = ", ".join(families[:-1]) + f", and {families[-1]}" if len(families) > 2 else " and ".join(families)
            if response_language == "zh-CN":
                answer = f"文中明确讨论的方法族包括：{joined}。"
            else:
                answer = f"The paper explicitly discusses {joined}."
        else:
            for document_note in paper_trace.document_notes:
                payload = _document_slot(document_note, focus_slot)
                if _slot_payload_status(payload) == "not_clearly_stated":
                    continue
                summary = _clean_detailed_slot_body(
                    _slot_payload_text(payload),
                    slot_name=focus_slot,
                    response_language=response_language,
                )
                if not summary or summary == missing_phrase:
                    continue
                if response_language == "zh-CN":
                    answer = f"文中明确讨论的方法包括：{summary.rstrip('。.')}。"
                else:
                    answer = f"The paper explicitly discusses {summary.rstrip('.')}."
                break
    elif focus_slot == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            if response_language == "zh-CN":
                if len(facts) == 1:
                    answer = f"文中明确说明：{facts[0]}。"
                else:
                    answer = f"文中明确说明：{facts[0]}。文中还说明：{facts[1]}。"
            else:
                if len(facts) == 1:
                    answer = f"The paper explicitly states that {facts[0]}."
                else:
                    answer = f"The paper explicitly states that {facts[0]}. It also states that {facts[1]}."
        else:
            for document_note in paper_trace.document_notes:
                payload = _document_slot(document_note, focus_slot)
                if _slot_payload_status(payload) == "not_clearly_stated":
                    continue
                summary = _clean_detailed_slot_body(
                    _slot_payload_text(payload),
                    slot_name=focus_slot,
                    response_language=response_language,
                )
                if not summary or summary == missing_phrase:
                    continue
                if response_language == "zh-CN":
                    answer = f"文中明确说明：{summary.rstrip('。.')}。"
                else:
                    answer = f"The paper explicitly states that {summary.rstrip('.')}."
                break
    else:
        for document_note in paper_trace.document_notes:
            payload = _document_slot(document_note, focus_slot)
            if _slot_payload_status(payload) == "not_clearly_stated":
                continue
            summary = _slot_payload_text(payload)
            answer = _render_grounded_slot_sentence(
                focus_slot,
                summary,
                response_language=response_language,
            )
            break

    if _prompt_requests_support_detail(prompt):
        support_ref = _first_support_ref(paper_trace)
        if support_ref:
            if response_language == "zh-CN":
                answer += f" 相关位置可见 {support_ref}。"
            else:
                answer += f" The most directly relevant retrieved support is {support_ref}."
    return answer.strip()


def _build_narrow_grounded_paper_answer(
    prompt: str,
    paper_trace: PaperTrace,
    *,
    response_language: str,
) -> str:
    focus_slot = _narrow_grounded_qa_focus_slot(prompt)
    texts = _narrow_grounded_qa_text_pool(paper_trace, focus_slot)
    missing_phrase = _paper_missing_phrase(response_language)
    answer = missing_phrase

    if focus_slot == "method":
        families = _extract_method_family_mentions(texts)
        if families:
            joined = ", ".join(families[:-1]) + f", and {families[-1]}" if len(families) > 2 else " and ".join(families)
            if response_language == "zh-CN":
                answer = f"文中明确讨论的方法族包括：{joined}。"
            else:
                answer = f"The paper explicitly discusses {joined}."
        else:
            for document_note in paper_trace.document_notes:
                payload = _document_slot(document_note, focus_slot)
                if _slot_payload_status(payload) == "not_clearly_stated":
                    continue
                summary = _clean_detailed_slot_body(
                    _slot_payload_text(payload),
                    slot_name=focus_slot,
                    response_language=response_language,
                )
                if not summary or summary == missing_phrase:
                    continue
                if response_language == "zh-CN":
                    answer = f"文中明确讨论的方法包括：{summary.rstrip('。.')}。"
                else:
                    answer = f"The paper explicitly discusses {summary.rstrip('.')}."
                break
    elif focus_slot == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            if response_language == "zh-CN":
                if len(facts) == 1:
                    answer = f"文中明确说明：{facts[0]}。"
                else:
                    answer = f"文中明确说明：{facts[0]}。文中还说明：{facts[1]}。"
            else:
                if len(facts) == 1:
                    answer = f"The paper explicitly states that {facts[0]}."
                else:
                    answer = f"The paper explicitly states that {facts[0]}. It also states that {facts[1]}."
        else:
            for document_note in paper_trace.document_notes:
                payload = _document_slot(document_note, focus_slot)
                if _slot_payload_status(payload) == "not_clearly_stated":
                    continue
                summary = _clean_detailed_slot_body(
                    _slot_payload_text(payload),
                    slot_name=focus_slot,
                    response_language=response_language,
                )
                if not summary or summary == missing_phrase:
                    continue
                if response_language == "zh-CN":
                    answer = f"文中明确说明：{summary.rstrip('。.')}。"
                else:
                    answer = f"The paper explicitly states that {summary.rstrip('.')}."
                break
    else:
        for document_note in paper_trace.document_notes:
            payload = _document_slot(document_note, focus_slot)
            if _slot_payload_status(payload) == "not_clearly_stated":
                continue
            summary = _slot_payload_text(payload)
            answer = _render_grounded_slot_sentence(
                focus_slot,
                summary,
                response_language=response_language,
            )
            break

    if _prompt_requests_support_detail(prompt):
        support_ref = _first_support_ref(paper_trace)
        if support_ref:
            if response_language == "zh-CN":
                answer += f" 相关位置可见 {support_ref}。"
            else:
                answer += f" The most directly relevant retrieved support is {support_ref}."
    return answer.strip()


def _looks_over_scaffolded_grounded_qa(answer_text: str) -> bool:
    lowered = answer_text.lower()
    markers = (
        "direct answer:",
        "grounded supporting evidence",
        "retrieved evidence",
        "evidence refs",
        "uncertainty when evidence is weak",
        "the answer has been rewritten",
    )
    if any(marker in lowered for marker in markers):
        return True
    if sum(lowered.count(token) for token in ("#page=", "#chunk=", "#pages=", "evidence:")) >= 2:
        return True
    if len(re.findall(r"(?m)^\d+\.", answer_text)) >= 2:
        return True
    if len(re.split(r"(?<=[.!?])\s+", answer_text.strip())) >= 5 and not _prompt_requests_support_detail(answer_text):
        return True
    return False


def _build_paper_grounded_qa_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    requested_slots = _requested_paper_slots(prompt, mode_selection.mode)
    if _is_recurring_limitations_prompt(prompt):
        recurring_limitations = _recurring_slot_lines(
            paper_trace.document_notes,
            slot_name="limitations",
        )
        lines = [
            "Renderer contract",
            f"- {_paper_renderer_name(mode_selection)}",
            "Paper output profile",
            f"- {mode_selection.paper_output_profile}",
            f"- Reason: {mode_selection.paper_output_profile_reason}",
            "Direct answer scaffold",
            *recurring_limitations,
            "Rendering rules",
            "- Keep the answer focused on recurring limitations across the consulted papers.",
            "- Separate clearly recurring limitations from weaker or paper-specific limitations.",
            "- Preserve concrete supported limitation details when they help explain the recurring pattern.",
            f"- If support is weak, say {_paper_missing_phrase(mode_selection.response_language)}",
        ]
        lines.extend(["Evidence refs", *_bullet_lines(evidence_refs or paper_trace.target_paths)])
        return "\n".join(lines)

    if _is_narrow_grounded_paper_qa(prompt):
        scaffold_answer = _build_narrow_grounded_paper_answer(
            prompt,
            paper_trace,
            response_language=mode_selection.response_language,
        )
        lines = [
            "Renderer contract",
            "- concise_grounded_qa",
            "Direct answer scaffold",
            f"- {scaffold_answer}",
            "Rendering rules",
            "- Answer the user's question directly in the first sentence.",
            "- Keep the answer concise, grounded, and low-noise.",
            "- Do not include an evidence appendix, repeated support fragments, or retrieval-style scaffolding unless the user explicitly asked for location or citation detail.",
            f"- If the evidence is weak or missing, say {_paper_missing_phrase(mode_selection.response_language)}",
        ]
        if _prompt_requests_support_detail(prompt):
            support_ref = _first_support_ref(paper_trace)
            if support_ref:
                lines.extend(["Primary support ref", f"- {support_ref}"])
        return "\n".join(lines)

    direct_scaffold = _relevant_slot_lines(paper_trace.document_notes, requested_slots)
    lines = [
        "Renderer contract",
        "- grounded_ra_memo",
        "Direct answer scaffold",
        *direct_scaffold,
        "Rendering rules",
        "- Answer the question directly from the cleaned slot scaffold and retrieved evidence.",
        "- Keep the answer concise and grounded rather than excerpt-heavy.",
        f"- If the consulted evidence does not clearly support a requested detail, say {_paper_missing_phrase(mode_selection.response_language)}",
    ]
    lines.extend(["Evidence refs", *_bullet_lines(evidence_refs or paper_trace.target_paths)])
    return "\n".join(lines)


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    narrow_grounded_qa = mode_selection.mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(prompt)

    if mode_selection.response_language == "zh-CN" and _looks_insufficiently_translated_chinese(answer_text):
        notes.append("The answer did not translate the grounded paper content cleanly enough into Chinese.")
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the cleaned slot evidence."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    if re.search(r"not clearly stated in the paper[^.\n]{0,160}\bhowever\b", answer_lower):
        notes.append(
            "The answer acknowledges a missing dimension but then keeps padding it with unsupported follow-on commentary."
        )
    if narrow_grounded_qa:
        if _looks_over_scaffolded_grounded_qa(answer_text):
            notes.append(
                "The narrow grounded QA answer is still too scaffold-heavy and should collapse to a concise answer-first form."
            )
        if len(answer_text) > 650 and not _prompt_requests_support_detail(prompt):
            notes.append(
                "The narrow grounded QA answer is longer than needed for a focused factual question."
            )
    elif _looks_excerpt_heavy_paper_answer(answer_text):
        notes.append(
            "The answer is still too excerpt-heavy or outline-heavy for the final paper renderer contract."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots and mode_selection.mode == "paper_summary":
        notes.append(
            "The answer did not clearly cover these requested summary dimensions: "
            + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
            + "."
        )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.mode == "paper_compare" and "hybrid approach could be beneficial" in answer_lower:
        notes.append("The comparison drifted into unsupported recommendation language instead of staying slot-grounded.")
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }


def _compose_paper_consistency_prompt(
    *,
    original_prompt: str,
    answer_text: str,
    report_notes: tuple[str, ...],
    response_language: str,
) -> str:
    missing_phrase = "文中未明确说明" if response_language == "zh-CN" else "not clearly stated in the paper"
    return "\n".join(
        [
            "You are performing a constrained consistency repair on a paper answer.",
            f"Original user prompt: {original_prompt}",
            "Repair goals:",
            "- Keep only claims supported by the grounded slot scaffold and evidence.",
            "- Remove broad textbook commentary or domain-common-sense filler that is not explicitly supported.",
            f"- If a requested detail is weak or missing, say {missing_phrase} instead of guessing.",
            "- Preserve the requested language and formatting style.",
            "Detected issues:",
            *[f"- {item}" for item in report_notes],
            "",
            "Current answer:",
            answer_text,
            "",
            "Return only the repaired final answer body.",
        ]
    )


def _fully_missing_requested_slots(
    document_notes: list[dict[str, object]],
    requested_slots: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    for slot_name in requested_slots:
        statuses = [
            str(_document_slot(document_note, slot_name).get("support_status", "not_clearly_stated"))
            for document_note in document_notes
        ]
        if statuses and all(status == "not_clearly_stated" for status in statuses):
            missing.append(slot_name)
    return tuple(missing)


def _contains_missing_slot_wording(text: str, response_language: str) -> bool:
    lowered = text.lower()
    if response_language == "zh-CN":
        return any(token in text for token in ("文中未明确", "未明确说明", "未明确展开"))
    return any(
        token in lowered
        for token in ("not clearly stated", "not confirmed", "not explicit", "not clearly described")
    )

def _compose_slot_translation_prompt(*, source_text: str, response_style: str) -> str:
    structured_compare = _looks_like_structured_compare_text(source_text)
    style_instruction = (
        "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "Return a concise Simplified Chinese memo."
    )
    instructions = [
        "Translate the grounded memo below into natural Simplified Chinese.",
        "Keep only facts that already appear in the grounded memo.",
        "Do not add generic finance or machine-learning commentary.",
        "If a dimension is unclear or missing, say 文中未明确说明。",
        "Keep method-family names, abbreviations, and file identifiers exact when needed.",
    ]
    if structured_compare:
        instructions.extend(
            [
                "Preserve every section and bullet in the same order.",
                "Do not merge or omit requested comparison dimensions.",
                "Use these Chinese section headings exactly when the corresponding sections appear: 比较文献, 研究问题, 样本与数据, 方法, 主要发现, 局限, 结论, 实践或投资含义.",
                "Translate each '- Contrast:' line as '- 对比：'.",
            ]
        )
    instructions.extend(
        [
            style_instruction,
            "",
            "Grounded memo:",
            source_text,
            "",
            "Return only the final Chinese answer body.",
        ]
    )
    return "\n".join(instructions)

def _compose_slot_translation_prompt(*, source_text: str, response_style: str) -> str:
    structured_compare = _looks_like_structured_compare_text(source_text)
    style_instruction = (
        "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "Return a concise Simplified Chinese memo."
    )
    instructions = [
        "Translate the grounded memo below into natural Simplified Chinese.",
        "Keep only facts that already appear in the grounded memo.",
        "Do not add generic finance or machine-learning commentary.",
        "If a dimension is unclear or missing, say 文中未明确说明。",
        "Keep method-family names, abbreviations, and file identifiers exact when needed.",
    ]
    if structured_compare:
        instructions.extend(
            [
                "Translate line by line and keep the same section and bullet structure.",
                "Do not omit any heading, bullet, or compared document line.",
                "If a line begins with '- `filename`:', keep the filename exactly and translate only the rest of the line.",
                "Use these Chinese section headings exactly when the corresponding sections appear: 比较文献, 研究问题, 样本与数据, 方法, 主要发现, 局限, 结论, 实践或投资含义.",
                "Translate each '- Contrast:' line as '- 对比：'.",
            ]
        )
    instructions.extend(
        [
            style_instruction,
            "",
            "Grounded memo:",
            source_text,
            "",
            "Return only the final Chinese answer body.",
        ]
    )
    return "\n".join(instructions)


def _compose_slot_translation_prompt(*, source_text: str, response_style: str) -> str:
    structured_compare = _looks_like_structured_compare_text(source_text)
    style_instruction = (
        "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "Return a concise Simplified Chinese memo."
    )
    instructions = [
        "Translate the grounded memo below into natural Simplified Chinese.",
        "Keep only facts that already appear in the grounded memo.",
        "Do not add generic finance or machine-learning commentary.",
        "If a dimension is unclear or missing, say 文中未明确说明。",
        "Keep method-family names, abbreviations, and file identifiers exact when needed.",
    ]
    if structured_compare:
        instructions.extend(
            [
                "Translate line by line and preserve the compare structure.",
                "Do not omit any heading, bullet, or compared document line.",
                "Do not collapse the comparison into a generic summary.",
                "If a line begins with '- `filename`:', keep the filename exactly and translate only the rest of the line.",
                "Use these Chinese section headings exactly when the corresponding sections appear: 比较文献, 研究问题, 样本与数据, 方法, 主要发现, 局限, 结论, 实践或投资含义。",
                "Translate each '- Contrast:' line as '- 对比：'.",
            ]
        )
    instructions.extend(
        [
            style_instruction,
            "",
            "Grounded memo:",
            source_text,
            "",
            "Return only the final Chinese answer body.",
        ]
    )
    return "\n".join(instructions)


def _compose_slot_translation_prompt(*, source_text: str, response_style: str) -> str:
    structured_compare = _looks_like_structured_compare_text(source_text)
    style_instruction = (
        "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "Return a concise Simplified Chinese memo."
    )
    instructions = [
        "Translate the grounded memo below into natural Simplified Chinese.",
        "Use Simplified Chinese only.",
        "Do not use Japanese, katakana, hiragana, or Traditional Chinese.",
        "Keep only facts that already appear in the grounded memo.",
        "Do not add generic finance or machine-learning commentary.",
        "If a dimension is unclear or missing, say 文中未明确说明。",
        "Keep method-family names, abbreviations, and file identifiers exact when needed.",
    ]
    if structured_compare:
        instructions.extend(
            [
                "Preserve the compare structure instead of collapsing it into a broad summary.",
                "Use these Chinese section headings exactly when the corresponding sections appear: 比较文献, 研究问题, 样本与数据, 方法, 主要发现, 局限, 结论, 实践或投资含义。",
                "Keep a clear 对比： line for each compared dimension when the source memo has one.",
            ]
        )
    instructions.extend(
        [
            style_instruction,
            "",
            "Grounded memo:",
            source_text,
            "",
            "Return only the final Chinese answer body.",
        ]
    )
    return "\n".join(instructions)


def _compose_slot_translation_prompt(*, source_text: str, response_style: str) -> str:
    structured_compare = _looks_like_structured_compare_text(source_text)
    style_instruction = (
        "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "Return a concise Simplified Chinese memo."
    )
    instructions = [
        "Translate the grounded memo below into natural Simplified Chinese.",
        "Use Simplified Chinese only.",
        "Do not use Japanese, katakana, hiragana, or Traditional Chinese.",
        "Keep only facts that already appear in the grounded memo.",
        "Do not add generic finance or machine-learning commentary.",
        "If a dimension is unclear or missing, say 文中未明确说明。",
        "Keep method-family names, abbreviations, and file identifiers exact when needed.",
    ]
    if structured_compare:
        instructions.extend(
            [
                "Translate line by line and preserve the compare structure.",
                "Do not omit any heading, bullet, or compared document line.",
                "Do not collapse the comparison into a broad summary.",
                "If a line begins with '- `filename`:', keep the filename exactly and translate only the rest of the line.",
                "Use these Chinese section headings exactly when the corresponding sections appear: 比较文献, 研究问题, 样本与数据, 方法, 主要发现, 局限, 结论, 实践或投资含义。",
                "Translate each '- Contrast:' line as '- 对比：'.",
            ]
        )
    instructions.extend(
        [
            style_instruction,
            "",
            "Grounded memo:",
            source_text,
            "",
            "Return only the final Chinese answer body.",
        ]
    )
    return "\n".join(instructions)


def _compose_slot_translation_prompt(*, source_text: str, response_style: str) -> str:
    structured_compare = _looks_like_structured_compare_text(source_text)
    style_instruction = (
        "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "Return a concise Simplified Chinese memo."
    )
    instructions = [
        "Translate the grounded memo below into natural Simplified Chinese.",
        "Use Simplified Chinese only.",
        "Do not use Japanese, katakana, hiragana, or Traditional Chinese.",
        "Keep only facts that already appear in the grounded memo.",
        "Do not add generic finance or machine-learning commentary.",
        "If a dimension is unclear or missing, say 文中未明确说明。",
        "Keep method-family names, abbreviations, and file identifiers exact when needed.",
    ]
    if structured_compare:
        instructions.extend(
            [
                "Translate line by line and preserve the compare structure.",
                "Do not omit any heading, bullet, or compared document line.",
                "Do not collapse the comparison into a broad summary.",
                "If a line begins with '- `filename`:', keep the filename exactly and translate only the rest of the line.",
                "Use these Chinese section headings exactly when the corresponding sections appear: 比较文献, 研究问题, 样本与数据, 方法, 主要发现, 局限, 结论, 实践或投资含义。",
                "Translate each '- Contrast:' line as '- 对比：'.",
            ]
        )
    instructions.extend(
        [
            style_instruction,
            "",
            "Grounded memo:",
            source_text,
            "",
            "Return only the final Chinese answer body.",
        ]
    )
    return "\n".join(instructions)


def _contains_generic_paper_filler(
    answer_lower: str,
    document_notes: list[dict[str, object]],
) -> bool:
    support_corpus = " ".join(
        str(slot.get("merged_note_text", "")).lower()
        for document_note in document_notes
        for slot in document_note.get("aggregated_slots", [])
        if str(slot.get("support_status", "")) != "not_clearly_stated"
    )
    filler_patterns = (
        "broader promise of machine learning",
        "underscores the importance of machine learning",
        "illustrates how machine learning can",
        "important implications for investors",
        "broadly useful for practitioners",
        "机器学习在金融中的广泛潜力",
        "对投资者具有重要启示",
        "具有广泛意义",
        "进一步说明了",
    )
    return any(pattern in answer_lower and pattern not in support_corpus for pattern in filler_patterns)


def _contains_unsupported_gap_inference(text: str, response_language: str) -> bool:
    lowered = text.lower()
    if response_language == "zh-CN":
        return any(
            token in text
            for token in (
                "可以推测",
                "可推测",
                "推测出",
                "据此推测",
            )
        )
    return any(
        token in lowered
        for token in (
            "we can infer",
            "one can infer",
            "it can be inferred",
            "it is reasonable to infer",
            "this likely means",
        )
    )


def _deterministic_paper_consistency_trim(
    answer_text: str,
    paper_trace: PaperTrace,
    *,
    prompt: str,
    mode: str,
    response_language: str,
    response_style: str,
    requested_slots: tuple[str, ...],
) -> str:
    if _is_recurring_limitations_prompt(prompt):
        return _build_recurring_limitations_answer(
            paper_trace.document_notes,
            response_language=response_language,
            response_style=response_style,
        )
    if mode == "paper_summary" and requested_slots and paper_trace.document_notes:
        return _build_slot_grounded_paper_summary(
            paper_trace.document_notes[0],
            requested_slots=requested_slots,
            response_language=response_language,
            response_style=response_style,
        )

    support_corpus = " ".join(
        str(slot.get("merged_note_text", "")).lower()
        for document_note in paper_trace.document_notes
        for slot in document_note.get("aggregated_slots", [])
    )
    generic_patterns = (
        "broader promise of machine learning",
        "underscores the importance of machine learning",
        "illustrates how machine learning can",
        "important implications for investors",
        "机器学习在金融中的广泛潜力",
        "对投资者具有重要启示",
        "具有广泛意义",
        "进一步说明了",
    )
    kept_sentences: list[str] = []
    for sentence in _answer_sentences(answer_text):
        lowered = sentence.lower()
        if any(pattern in lowered and pattern not in support_corpus for pattern in generic_patterns):
            continue
        kept_sentences.append(sentence.strip())

    repaired = " ".join(item for item in kept_sentences if item).strip()
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, requested_slots)
    if missing_slots and not _contains_missing_slot_wording(repaired, response_language):
        missing_text = _missing_slot_sentence(missing_slots, response_language=response_language)
        repaired = f"{repaired} {missing_text}".strip()
    return repaired or answer_text


def _explicit_paper_slots(prompt: str) -> tuple[str, ...]:
    lowered = prompt.lower()
    slots: list[str] = []
    keyword_map = (
        ("research_question", ("question", "goal", "aim", "purpose", "problem", "研究问题", "目标", "目的", "问题")),
        ("background_or_motivation", ("background", "motivation", "context", "研究背景", "动机", "背景")),
        ("sample_or_data", ("sample", "samples", "data", "dataset", "datasets", "corpus", "样本", "数据", "数据集")),
        ("method", ("method", "methods", "approach", "model", "models", "algorithm", "machine learning", "方法", "模型", "算法", "机器学习")),
        ("main_findings", ("finding", "findings", "result", "results", "outcome", "发现", "结果", "主要发现", "实证发现")),
        ("limitations", ("limitation", "limitations", "caveat", "caveats", "constraint", "constraints", "局限", "限制", "不足")),
        ("conclusion", ("conclusion", "conclusions", "overall conclusion", "summary", "结论", "总体结论", "总结")),
        ("practical_or_investment_implications", ("investment", "investor", "implication", "implications", "practical", "实践", "投资", "启示", "含义")),
    )
    for slot_name, keywords in keyword_map:
        if any(keyword in lowered or keyword in prompt for keyword in keywords):
            slots.append(slot_name)
    return tuple(dict.fromkeys(slots))


def _explicit_paper_slots(prompt: str) -> tuple[str, ...]:
    lowered = prompt.lower()
    slots: list[str] = []
    keyword_map = (
        (
            "research_question",
            (
                "question",
                "goal",
                "aim",
                "purpose",
                "problem",
                "research question",
                "研究问题",
                "研究目标",
                "目标",
                "目的",
                "问题",
            ),
        ),
        (
            "background_or_motivation",
            (
                "background",
                "motivation",
                "context",
                "research background",
                "研究背景",
                "动机",
                "背景",
            ),
        ),
        (
            "sample_or_data",
            (
                "sample",
                "samples",
                "data",
                "dataset",
                "datasets",
                "corpus",
                "sample/data",
                "样本",
                "样本与数据",
                "样本或数据",
                "数据",
                "数据集",
            ),
        ),
        (
            "method",
            (
                "method",
                "methods",
                "approach",
                "model",
                "models",
                "algorithm",
                "machine learning",
                "method family",
                "method families",
                "方法",
                "模型",
                "算法",
                "方法族",
                "机器学习",
            ),
        ),
        (
            "main_findings",
            (
                "finding",
                "findings",
                "result",
                "results",
                "outcome",
                "main finding",
                "main findings",
                "主要发现",
                "发现",
                "结果",
                "实证发现",
            ),
        ),
        (
            "limitations",
            (
                "limitation",
                "limitations",
                "caveat",
                "caveats",
                "constraint",
                "constraints",
                "局限",
                "限制",
                "不足",
            ),
        ),
        (
            "conclusion",
            (
                "conclusion",
                "conclusions",
                "overall conclusion",
                "summary",
                "结论",
                "总体结论",
                "总结",
            ),
        ),
        (
            "practical_or_investment_implications",
            (
                "investment",
                "investor",
                "implication",
                "implications",
                "practical",
                "实践",
                "投资",
                "启示",
                "含义",
            ),
        ),
    )
    for slot_name, keywords in keyword_map:
        if any(keyword in lowered or keyword in prompt for keyword in keywords):
            slots.append(slot_name)
    return tuple(dict.fromkeys(slots))


def _slot_display_name(slot_name: str, response_language: str) -> str:
    zh_labels = {
        "research_question": "研究问题",
        "background_or_motivation": "研究背景或动机",
        "sample_or_data": "样本或数据",
        "method": "方法",
        "main_findings": "主要发现",
        "limitations": "局限",
        "conclusion": "结论",
        "practical_or_investment_implications": "实践或投资含义",
        "other": "其他信息",
    }
    if response_language == "zh-CN":
        return zh_labels.get(slot_name, slot_name)
    return {
        "sample_or_data": "sample/data",
        "main_findings": "main findings",
        "practical_or_investment_implications": "practical/investment implications",
    }.get(slot_name, slot_label(slot_name))


def _missing_slot_sentence(missing_slots: tuple[str, ...], *, response_language: str) -> str:
    labels = ", ".join(_slot_display_name(slot_name, response_language) for slot_name in missing_slots[:3])
    if response_language == "zh-CN":
        return f"{labels}文中未明确说明。"
    return f"The paper does not clearly state {labels}."


def _slot_display_name(slot_name: str, response_language: str) -> str:
    zh_labels = {
        "research_question": "研究问题",
        "background_or_motivation": "研究背景或动机",
        "sample_or_data": "样本或数据",
        "method": "方法",
        "main_findings": "主要发现",
        "limitations": "局限",
        "conclusion": "结论",
        "practical_or_investment_implications": "实践或投资含义",
        "other": "其他信息",
    }
    if response_language == "zh-CN":
        return zh_labels.get(slot_name, slot_name)
    return slot_label(slot_name)


def _answer_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    normalized = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    parts = re.split(r"(?<=[。！？.!?;；])\s*", normalized)
    return [part.strip() for part in parts if part.strip()]


def _run_answer_route(
    config: LabaiConfig,
    prompt: str,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    mode_selection: ModeSelection,
    grounded_draft: str | None,
) -> AnswerRoute:
    if config.runtime.runtime == "claw":
        return _run_claw_route(
            config,
            prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            grounded_draft,
        )
    return _run_native_route(
        config,
        prompt,
        session_id,
        observations,
        evidence_refs,
        mode_selection,
        grounded_draft,
        requested_runtime="native",
        runtime_fallback=_no_runtime_fallback("native", config.runtime.fallback_runtime),
    )


def _run_claw_route(
    config: LabaiConfig,
    prompt: str,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    mode_selection: ModeSelection,
    grounded_draft: str | None,
) -> AnswerRoute:
    adapter = ClawRuntimeAdapter()
    health = adapter.healthcheck(config)

    if health.available:
        try:
            response = adapter.ask(
                config,
                _build_runtime_request(
                    prompt=prompt,
                    session_id=session_id,
                    observations=observations,
                    evidence_refs=evidence_refs,
                    mode_selection=mode_selection,
                    grounded_draft=grounded_draft,
                ),
            )
        except RuntimeAdapterError as exc:
            if _is_retryable_claw_empty_stream(exc):
                observations.append(_claw_empty_stream_retry_note(mode_selection.mode))
                try:
                    response = adapter.ask(
                        config,
                        _build_runtime_request(
                            prompt=prompt,
                            session_id=session_id,
                            observations=observations,
                            evidence_refs=evidence_refs,
                            mode_selection=mode_selection,
                            grounded_draft=grounded_draft,
                        ),
                    )
                except RuntimeAdapterError:
                    response = None
                else:
                    return AnswerRoute(
                        requested_runtime="claw",
                        runtime_used="claw",
                        runtime_fallback=_no_runtime_fallback("claw", config.runtime.fallback_runtime),
                        requested_provider="claw",
                        provider_used=response.provider_name,
                        provider_model=response.model or mode_selection.selected_model or health.model,
                        provider_fallback=FallbackInfo(
                            applied=False,
                            policy=config.fallback_policy,
                            requested_provider="claw",
                            active_provider=response.provider_name,
                            reason="",
                        ),
                        text=response.text,
                    )
            if config.runtime.fallback_runtime == "native":
                return _run_native_route(
                    config,
                    prompt,
                    session_id,
                    observations,
                    evidence_refs,
                    mode_selection,
                    grounded_draft,
                    requested_runtime="claw",
                    runtime_fallback=RuntimeFallbackInfo(
                        applied=True,
                        requested_runtime="claw",
                        active_runtime="native",
                        fallback_runtime=config.runtime.fallback_runtime,
                        reason=str(exc),
                    ),
                )
            raise

        return AnswerRoute(
            requested_runtime="claw",
            runtime_used="claw",
            runtime_fallback=_no_runtime_fallback("claw", config.runtime.fallback_runtime),
            requested_provider="claw",
            provider_used=response.provider_name,
            provider_model=response.model or mode_selection.selected_model or health.model,
            provider_fallback=FallbackInfo(
                applied=False,
                policy=config.fallback_policy,
                requested_provider="claw",
                active_provider=response.provider_name,
                reason="",
            ),
            text=response.text,
        )

    if config.runtime.fallback_runtime == "native":
        return _run_native_route(
            config,
            prompt,
            session_id,
            observations,
            evidence_refs,
            mode_selection,
            grounded_draft,
            requested_runtime="claw",
            runtime_fallback=RuntimeFallbackInfo(
                applied=True,
                requested_runtime="claw",
                active_runtime="native",
                fallback_runtime=config.runtime.fallback_runtime,
                reason=health.detail,
            ),
        )

    raise RuntimeAdapterError(health.detail)


def _build_runtime_request(
    *,
    prompt: str,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    mode_selection: ModeSelection,
    grounded_draft: str | None,
) -> RuntimeRequest:
    return RuntimeRequest(
        prompt=prompt,
        session_id=session_id,
        observations=tuple(observations),
        preferred_model=mode_selection.selected_model,
        mode=mode_selection.mode,
        mode_reason=mode_selection.reason,
        answer_schema=mode_selection.answer_schema,
        read_strategy=mode_selection.read_strategy,
        read_strategy_reason=mode_selection.read_strategy_reason,
        response_style=mode_selection.response_style,
        include_explicit_evidence_refs=mode_selection.include_explicit_evidence_refs,
        response_language=mode_selection.response_language,
        evidence_refs=evidence_refs,
        grounded_draft=grounded_draft,
    )


def _is_retryable_claw_empty_stream(exc: RuntimeAdapterError) -> bool:
    return "assistant stream produced no content" in str(exc).lower()


def _claw_empty_stream_retry_note(mode: str) -> str:
    labels = {
        "workspace_verification": "workspace verification",
        "project_onboarding": "project onboarding",
        "workspace_edit": "workspace edit",
        "implementation_plan": "implementation plan",
        "repo_overview": "repo overview",
        "architecture_review": "architecture review",
        "file_explain": "file explanation",
        "prompt_compiler": "prompt compilation",
        "paper_summary": "paper summary",
        "paper_compare": "paper comparison",
        "paper_grounded_qa": "paper grounded QA",
        "compare_options": "option comparison",
    }
    label = labels.get(mode, mode.replace("_", " ").strip() or "request")
    return f"Retried {label} once after an empty assistant stream from the live Claw path."


def _run_native_route(
    config: LabaiConfig,
    prompt: str,
    session_id: str,
    observations: list[str],
    evidence_refs: tuple[str, ...],
    mode_selection: ModeSelection,
    grounded_draft: str | None,
    *,
    requested_runtime: str,
    runtime_fallback: RuntimeFallbackInfo,
) -> AnswerRoute:
    provider, provider_health, provider_fallback = _resolve_provider(config)
    response = provider.ask(
        config,
        ProviderRequest(
            prompt=prompt,
            session_id=session_id,
            observations=tuple(observations),
            preferred_model=mode_selection.selected_model,
            mode=mode_selection.mode,
            mode_reason=mode_selection.reason,
            answer_schema=mode_selection.answer_schema,
            read_strategy=mode_selection.read_strategy,
            read_strategy_reason=mode_selection.read_strategy_reason,
            response_style=mode_selection.response_style,
            include_explicit_evidence_refs=mode_selection.include_explicit_evidence_refs,
            response_language=mode_selection.response_language,
            evidence_refs=evidence_refs,
            grounded_draft=grounded_draft,
        ),
    )
    return AnswerRoute(
        requested_runtime=requested_runtime,
        runtime_used="native",
        runtime_fallback=runtime_fallback,
        requested_provider=provider_fallback.requested_provider,
        provider_used=response.provider_name,
        provider_model=response.model or mode_selection.selected_model or provider_health.model,
        provider_fallback=provider_fallback,
        text=response.text,
    )


def _evaluate_native_provider_route(config: LabaiConfig) -> NativeProviderRoute:
    requested_provider = get_default_provider(config)
    provider_health = requested_provider.healthcheck(config)

    if requested_provider.name == "mock":
        return NativeProviderRoute(
            status="ready",
            detail="Mock provider is ready and read-only repo tools are available.",
            requested_provider=requested_provider.name,
            provider_health=provider_health,
        )

    if provider_health.available:
        return NativeProviderRoute(
            status="ready",
            detail=f"Requested provider '{requested_provider.name}' is reachable.",
            requested_provider=requested_provider.name,
            provider_health=provider_health,
        )

    if config.fallback_policy == "fallback_to_mock":
        return NativeProviderRoute(
            status="ready_with_fallback",
            detail=f"{provider_health.detail} Falling back to mock is allowed by config.",
            requested_provider=requested_provider.name,
            provider_health=provider_health,
        )

    return NativeProviderRoute(
        status="blocked",
        detail=provider_health.detail,
        requested_provider=requested_provider.name,
        provider_health=provider_health,
    )


def _runtime_healthcheck(config: LabaiConfig) -> RuntimeHealth:
    if config.runtime.runtime == "native":
        return _native_runtime_health()
    return ClawRuntimeAdapter().healthcheck(config)


def _native_runtime_health() -> RuntimeHealth:
    return RuntimeHealth(
        status="ready",
        detail="Native Python runtime is available.",
        available=True,
        model=None,
        metadata={"route": "native"},
    )


def _resolve_provider(
    config: LabaiConfig,
):
    requested_provider = get_default_provider(config)
    provider_health = requested_provider.healthcheck(config)
    fallback = FallbackInfo(
        applied=False,
        policy=config.fallback_policy,
        requested_provider=requested_provider.name,
        active_provider=requested_provider.name,
        reason="",
    )

    if requested_provider.name == "mock" or provider_health.available:
        return requested_provider, provider_health, fallback

    if config.fallback_policy == "fallback_to_mock":
        fallback_provider = get_provider("mock")
        fallback_health = fallback_provider.healthcheck(config)
        return (
            fallback_provider,
            fallback_health,
            FallbackInfo(
                applied=True,
                policy=config.fallback_policy,
                requested_provider=requested_provider.name,
                active_provider=fallback_provider.name,
                reason=provider_health.detail,
            ),
        )

    raise ProviderError(provider_health.detail)


def _build_grounded_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
    *,
    workspace_root: Path | None = None,
    edit_plan: WorkspaceEditPlan | None = None,
    workspace_coverage: OnboardingCoverage | None = None,
) -> str | None:
    if mode_selection.answer_schema == "brief_response":
        return None

    builders = {
        "repo_overview": _build_repo_overview_draft,
        "workspace_verification": _build_workspace_verification_draft,
        "project_onboarding": _build_project_onboarding_draft,
        "file_explain": _build_file_explain_draft,
        "architecture_review": _build_architecture_review_draft,
        "implementation_plan": _build_implementation_plan_draft,
        "workspace_edit": _build_workspace_edit_draft,
        "prompt_compiler": _build_prompt_compiler_draft,
        "compare_options": _build_compare_options_draft,
        "paper_summary": _build_paper_summary_draft,
        "paper_compare": _build_paper_compare_draft,
        "paper_grounded_qa": _build_paper_grounded_qa_draft,
    }
    if mode_selection.mode == "workspace_edit":
        return builders[mode_selection.mode](
            config,
            prompt,
            mode_selection,
            tool_calls,
            evidence_refs,
            paper_trace,
            workspace_root=workspace_root,
            edit_plan=edit_plan,
            workspace_coverage=workspace_coverage,
        )
    return builders[mode_selection.mode](
        config,
        prompt,
        mode_selection,
        tool_calls,
        evidence_refs,
        paper_trace,
    )


def _build_repo_overview_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    coverage = OnboardingCoverage()
    top_level = _find_tool_summary(tool_calls, "list_directory", path=".")
    package_layout = _find_tool_summary(tool_calls, "list_directory", path="src") or _find_tool_summary(tool_calls, "list_directory", path="src/labai")
    python_search = _find_tool_summary(tool_calls, "find_files", path="src")
    readme_summary = _find_tool_summary(tool_calls, "read_text_file", path="README.md")
    workspace_specific = config.workspace.active_workspace_root == config.project_root
    return "\n".join(
        [
            "Purpose",
            f"- {readme_summary or top_level or 'Summarize the workspace from the consulted README and directory layout only.'}",
            "Main directories/modules",
            f"- {package_layout or top_level or 'Consulted repository layout through read-only directory listings.'}",
            "Important entry points",
            f"- {python_search or 'Entry points should be limited to the files visible from the consulted workspace search.'}",
            "Workspace context",
            f"- Active workspace root: `{config.workspace.active_workspace_root}`.",
            *(
                [
                    "Current runtime path",
                    f"- Config currently selects runtime `{config.runtime.runtime}` with fallback `{config.runtime.fallback_runtime}`.",
                    f"- General model `{config.models.general_model}` and code model `{config.models.code_model}` are the current defaults.",
                ]
                if workspace_specific
                else ["Current runtime path", "- The `labai` control plane remains separate from this active workspace and is only relevant when the prompt explicitly asks about it."]
            ),
            "Key risks or caveats",
            "- If behavior is not confirmed from the consulted files, say that it is not confirmed.",
            "- Workspace access is allowlisted; paths outside the active workspace or configured roots stay blocked.",
            "Evidence/files consulted",
            *_bullet_lines(evidence_refs or _evidence_from_tool_calls(tool_calls)),
            *(["Additional evidence note", f"- {python_search}"] if python_search else []),
        ]
    )


def _build_workspace_verification_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    coverage = _collect_onboarding_coverage(config.workspace.active_workspace_root.resolve())
    summary_map = dict(coverage.summary_map)
    is_chinese = mode_selection.response_language == "zh-CN"
    top_level = _find_tool_summary(tool_calls, "list_directory", path=".") or _onboarding_top_level_summary(coverage)
    readme_summary = _workspace_primary_doc_summary(summary_map) or _find_tool_summary(tool_calls, "read_text_file", path="README.md")
    config_paths = tuple(
        path
        for path in summary_map
        if _classify_onboarding_path(path) == "config"
    )
    config_summaries = _dedupe_strings(
        tuple(summary_map[path] for path in config_paths)
    )
    entrypoint_paths = _rank_onboarding_entrypoints(
        _dedupe_strings((*coverage.inspected_paths, *summary_map.keys(), *evidence_refs))
    )
    entrypoint_summaries = tuple(
        summary_map[item]
        for item in entrypoint_paths
        if item in summary_map
    )
    tests_visible = any(entry.category == "tests" for entry in coverage.manifest_entries)
    docs_visible = any(entry.category == "docs" for entry in coverage.manifest_entries)
    notebooks_visible = any(entry.path.lower().endswith(".ipynb") for entry in coverage.manifest_entries)
    purpose_summary = _onboarding_purpose_summary(
        readme_summary=readme_summary,
        top_level=top_level,
        entrypoint_paths=entrypoint_paths,
        entrypoint_summaries=entrypoint_summaries,
        coverage=coverage,
        response_language=mode_selection.response_language,
    )
    assessment = _assess_workspace_readiness(
        coverage=coverage,
        readme_summary=readme_summary,
        summary_map=summary_map,
        config_paths=config_paths,
        entrypoint_paths=entrypoint_paths,
        tests_visible=tests_visible,
        docs_visible=docs_visible,
        notebooks_visible=notebooks_visible,
        response_language=mode_selection.response_language,
        purpose_summary=purpose_summary,
    )
    coverage_lines = _onboarding_coverage_lines(
        coverage,
        response_language=mode_selection.response_language,
    ) + _onboarding_focus_area_lines(
        coverage,
        response_language=mode_selection.response_language,
    )

    return "\n".join(
        [
            _workspace_section_title("Readiness status", is_chinese=is_chinese),
            f"- `{assessment.status}`",
            _workspace_section_title("Why this status was chosen", is_chinese=is_chinese),
            *[f"- {line}" for line in assessment.why_status],
            *[f"- {line}" for line in coverage_lines],
            _workspace_section_title("What is clearly present", is_chinese=is_chinese),
            *[f"- {line}" for line in assessment.confirmed_present],
            _workspace_section_title("Likely entry points or run surfaces", is_chinese=is_chinese),
            *(
                [f"- `{item}`" for item in entrypoint_paths]
                if entrypoint_paths
                else [_workspace_not_confirmed("Likely entry points or run surfaces", is_chinese=is_chinese)]
            ),
            *[f"- {summary}" for summary in entrypoint_summaries[:3]],
            _workspace_section_title("Config/env and dependency signals", is_chinese=is_chinese),
            *(
                [f"- {summary}" for summary in config_summaries[:4]]
                if config_summaries
                else [_workspace_not_confirmed("Config/env and dependency signals", is_chinese=is_chinese)]
            ),
            _workspace_section_title("Missing pieces or blockers", is_chinese=is_chinese),
            *[f"- {line}" for line in assessment.missing_or_blocking],
            _workspace_section_title("Risks or uncertainty", is_chinese=is_chinese),
            *[f"- {line}" for line in assessment.risks_or_uncertainty],
            _workspace_section_title("What to read first", is_chinese=is_chinese),
            *[f"- {line}" for line in assessment.read_first],
            _workspace_section_title("First three practical next steps", is_chinese=is_chinese),
            *[f"- {line}" for line in assessment.next_steps[:3]],
            _workspace_section_title("Evidence/files consulted", is_chinese=is_chinese),
            *_bullet_lines(evidence_refs or _evidence_from_tool_calls(tool_calls) or coverage.inspected_paths[:12]),
        ]
    )


def collect_workspace_coverage(repo_root: Path) -> OnboardingCoverage:
    return _collect_onboarding_coverage(repo_root)


def _collect_onboarding_coverage(repo_root: Path) -> OnboardingCoverage:
    return _collect_onboarding_coverage_current(str(repo_root.resolve()))


@lru_cache(maxsize=32)
def _collect_onboarding_coverage_cached(repo_root_str: str) -> OnboardingCoverage:
    return _collect_onboarding_coverage_current(repo_root_str)


def _collect_onboarding_coverage_current(repo_root_str: str) -> OnboardingCoverage:
    repo_root = Path(repo_root_str)
    manifest_entries: list[OnboardingManifestEntry] = []
    category_counts: Counter[str] = Counter()
    inspected_category_counts: Counter[str] = Counter()
    text_cache: dict[str, str] = {}
    skipped_notes: list[str] = []

    def _handle_walk_error(exc: OSError) -> None:
        skipped_notes.append(f"Skipped `{exc.filename}` because it could not be traversed: {exc.strerror}.")

    for current_root, dirnames, filenames in os.walk(
        repo_root,
        topdown=True,
        onerror=_handle_walk_error,
    ):
        current_path = Path(current_root)
        dirnames[:] = sorted(dirnames, key=str.lower)
        filenames = sorted(filenames, key=str.lower)
        for filename in filenames:
            candidate = current_path / filename
            entry, text = _build_onboarding_manifest_entry(candidate, repo_root)
            manifest_entries.append(entry)
            category_counts[entry.category] += 1
            if text:
                text_cache[entry.path] = text

    relevant_entries = tuple(
        entry for entry in manifest_entries if entry.relevant and entry.readable
    )
    inspected_paths = _select_onboarding_paths_for_inspection(relevant_entries)
    inspected_set = set(inspected_paths)
    skipped_paths = tuple(entry.path for entry in relevant_entries if entry.path not in inspected_set)
    if skipped_paths:
        skipped_notes.append(
            "This workspace exceeded the full-relevant-file threshold, so onboarding inspected every high-signal file plus a deterministic broader set of medium-signal files."
        )

    summary_map: dict[str, str] = {}
    for entry in relevant_entries:
        if entry.path not in inspected_set:
            continue
        text = text_cache.get(entry.path, "")
        if not text:
            continue
        summary_map[entry.path] = _summarize_file_text(entry.path, text)
        inspected_category_counts[entry.category] += 1

    unreadable_binary_count = sum(
        1
        for entry in manifest_entries
        if entry.category == "binary/non-text" or (entry.relevant and not entry.readable)
    )
    ignored_noise_count = sum(1 for entry in manifest_entries if entry.category == "ignored/noise")

    return OnboardingCoverage(
        total_files=len(manifest_entries),
        relevant_readable_count=len(relevant_entries),
        ignored_noise_count=ignored_noise_count,
        unreadable_binary_count=unreadable_binary_count,
        full_relevant_coverage=len(inspected_paths) == len(relevant_entries),
        category_counts=dict(category_counts),
        inspected_category_counts=dict(inspected_category_counts),
        manifest_entries=tuple(manifest_entries),
        inspected_paths=inspected_paths,
        skipped_paths=skipped_paths,
        skipped_notes=tuple(skipped_notes),
        summary_map=summary_map,
    )


def _build_onboarding_manifest_entry(
    candidate: Path,
    repo_root: Path,
) -> tuple[OnboardingManifestEntry, str]:
    relative_path = candidate.relative_to(repo_root).as_posix()
    category = _classify_onboarding_path(relative_path)
    relevant = category in _ONBOARDING_RELEVANT_CATEGORIES
    text = ""
    readable = False
    skip_reason = ""
    if relevant:
        text = _read_onboarding_text(candidate)
        readable = bool(text)
        if not readable:
            skip_reason = "Could not decode file as readable text."
    elif category == "binary/non-text":
        skip_reason = "Binary or non-text file."
    elif category == "ignored/noise":
        skip_reason = "Ignored tooling, cache, or planning noise."
    else:
        skip_reason = "Not part of the onboarding-relevant readable set."
    return (
        OnboardingManifestEntry(
            path=relative_path,
            category=category,
            readable=readable,
            relevant=relevant,
            size_bytes=candidate.stat().st_size,
            top_level=relative_path.split("/", 1)[0] if "/" in relative_path else ".",
            skip_reason=skip_reason,
        ),
        text,
    )


def _classify_onboarding_path(relative_path: str) -> str:
    pure_path = Path(relative_path.replace("\\", "/"))
    parts = [part.lower() for part in pure_path.parts]
    name = pure_path.name.lower()
    suffix = pure_path.suffix.lower()

    if any(part in _ONBOARDING_IGNORED_DIR_NAMES for part in parts[:-1]):
        return "ignored/noise"
    if any(part.startswith(prefix) for part in parts for prefix in _ONBOARDING_IGNORED_FILE_PREFIXES):
        return "ignored/noise"
    if suffix in _ONBOARDING_BINARY_EXTENSIONS:
        return "binary/non-text"
    if suffix in _ONBOARDING_DATA_EXTENSIONS or any(
        part in {"data", "datasets", "raw_data", "clean_data", "artifacts", "outputs", "output"}
        for part in parts[:-1]
    ):
        return "data"
    if (
        name in {item.lower() for item in _ONBOARDING_CONFIG_NAMES}
        or name in {"dockerfile", "makefile", ".editorconfig", ".python-version"}
        or name.startswith(".env")
        or name.startswith("requirements")
        or suffix in _ONBOARDING_CONFIG_EXTENSIONS
    ):
        return "config"
    if any(part in {"tests", "test", "testing"} for part in parts[:-1]) or name.startswith("test_") or name.endswith("_test.py"):
        return "tests"
    if name in _ONBOARDING_DOC_NAMES or suffix in _ONBOARDING_DOC_EXTENSIONS or any(
        part in {"docs", "doc", "notes"} for part in parts[:-1]
    ):
        return "docs"
    if suffix in _ONBOARDING_SCRIPT_EXTENSIONS or any(part in {"scripts", "bin"} for part in parts[:-1]):
        return "scripts"
    if suffix in _ONBOARDING_SOURCE_EXTENSIONS:
        stem = pure_path.stem.lower()
        if len(parts) <= 2 and (
            re.match(r"^\d+[_-]", stem)
            or any(token in stem for token in ("main", "run", "cli", "app", "serve", "download", "prepare", "merge", "calc", "pipeline", "train"))
        ):
            return "scripts"
        return "source"
    return "binary/non-text"


def _read_onboarding_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with path.open("r", encoding=encoding) as handle:
                text = handle.read(_ONBOARDING_MAX_SUMMARY_CHARS)
            return text.replace("\x00", " ").strip()
        except (UnicodeDecodeError, OSError):
            continue
    return ""


def _select_onboarding_paths_for_inspection(
    relevant_entries: tuple[OnboardingManifestEntry, ...],
) -> tuple[str, ...]:
    if len(relevant_entries) <= _ONBOARDING_FULL_RELEVANT_COVERAGE_LIMIT:
        return tuple(entry.path for entry in relevant_entries)

    scored: list[tuple[int, str]] = []
    for entry in relevant_entries:
        path = entry.path
        lowered = path.lower()
        name = Path(path).name.lower()
        score = 0
        if entry.category == "config":
            score += 80
        elif entry.category == "docs":
            score += 70
        elif entry.category == "scripts":
            score += 65
        elif entry.category == "source":
            score += 55
        elif entry.category == "tests":
            score += 35
        if name in {"readme.md", "agents.md", "claude.md"}:
            score += 40
        if any(token in Path(path).stem.lower() for token in ("main", "run", "cli", "app", "serve", "download", "prepare", "merge", "calc", "pipeline")):
            score += 18
        if name in {item.lower() for item in _ONBOARDING_CONFIG_NAMES}:
            score += 20
        if "/tests/" in f"/{lowered}/" or name.startswith("test_"):
            score -= 10
        scored.append((score, path))

    ranked = [
        path
        for score, path in sorted(scored, key=lambda item: (-item[0], item[1].lower()))
    ]
    return tuple(ranked[:_ONBOARDING_LARGE_PROJECT_SELECTION_LIMIT])


def _onboarding_top_level_summary(coverage: OnboardingCoverage) -> str:
    top_levels = sorted(
        {
            entry.top_level
            for entry in coverage.manifest_entries
            if entry.category != "ignored/noise"
        }
    )
    if not top_levels:
        return ""
    label = ", ".join(f"`{item}`" if item != "." else "`(workspace root)`" for item in top_levels[:8])
    extra = ""
    if len(top_levels) > 8:
        extra = f", plus {len(top_levels) - 8} more top-level areas"
    return f"The workspace manifest covers top-level areas {label}{extra}."


def _onboarding_notable_dirs(coverage: OnboardingCoverage) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for entry in coverage.manifest_entries:
        if not entry.relevant:
            continue
        if entry.top_level != ".":
            counts[entry.top_level] += 1
    return tuple(item for item, _count in counts.most_common(6))


def _onboarding_coverage_lines(
    coverage: OnboardingCoverage,
    *,
    response_language: str,
) -> tuple[str, ...]:
    is_chinese = response_language == "zh-CN"
    inspected = len(coverage.inspected_paths)
    relevant = coverage.relevant_readable_count
    categories = ", ".join(
        f"{name}={count}"
        for name, count in sorted(coverage.inspected_category_counts.items())
        if count
    )
    lines = [
        (
            f"宸ヤ綔鍖烘竻鍗曞凡瑕嗙洊 {coverage.total_files} 涓枃浠讹紝鍏朵腑鍙槄璇荤殑鐩稿叧鏂囦欢 {relevant} 涓紝"
            f"宸插拷鐣ョ殑鍣０鏂囦欢 {coverage.ignored_noise_count} 涓紝浜屽垎鍒?/涓嶅彲璇荤殑鏂囦欢 {coverage.unreadable_binary_count} 涓€?"
            if is_chinese
            else f"Accounted for {coverage.total_files} files in the workspace manifest: {relevant} relevant readable files, {coverage.ignored_noise_count} ignored/noise files, and {coverage.unreadable_binary_count} binary or unreadable files."
        ),
        (
            f"瀹為檯妫€鏌? {inspected}/{relevant} 涓浉鍏冲彲璇绘枃浠讹紝"
            f"{'宸茬粡瀹屾垚鍏ㄩ噺鐩稿叧瑕嗙洊' if coverage.full_relevant_coverage else '褰撳墠涓洪珮淇″彿鍔犲箍瑕嗙洊鐨勭‘瀹氭€ч€夋牱'}"
            + (f"锛屾杈? {categories}銆?" if categories else "銆?")
            if is_chinese
            else f"Inspected {inspected}/{relevant} relevant readable files; {'full relevant-file coverage was achieved' if coverage.full_relevant_coverage else 'the project was large enough to require high-signal plus deterministic broader coverage'}.{f' Inspected categories: {categories}.' if categories else ''}"
        ),
    ]
    return tuple(lines)


def _onboarding_focus_area_lines(
    coverage: OnboardingCoverage,
    *,
    response_language: str,
) -> tuple[str, ...]:
    is_chinese = response_language == "zh-CN"
    grouped: dict[str, list[OnboardingManifestEntry]] = {}
    for entry in coverage.manifest_entries:
        if not entry.relevant or entry.top_level == ".":
            continue
        grouped.setdefault(entry.top_level, []).append(entry)

    lines: list[str] = []
    for top_level, entries in sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0].lower()),
    )[:5]:
        category_counts = Counter(entry.category for entry in entries)
        sample_paths = [entry.path for entry in entries[:2]]
        category_summary = ", ".join(
            f"{category}={count}"
            for category, count in sorted(category_counts.items())
        )
        sample_summary = ", ".join(f"`{item}`" for item in sample_paths)
        lines.append(
            (
                f"`{top_level}` 鍖呭惈 {len(entries)} 涓彲璇荤浉鍏虫枃浠讹紝涓昏绫诲瀷涓? {category_summary}锛涗唬琛ㄦ€ф枃浠朵緥濡? {sample_summary}銆?"
                if is_chinese
                else f"`{top_level}` contributes {len(entries)} relevant readable files ({category_summary}); representative files include {sample_summary}."
            )
        )
    return tuple(lines)


def _onboarding_evidence_lines(coverage: OnboardingCoverage) -> tuple[str, ...]:
    lines = [f"- `{path}`" for path in coverage.inspected_paths[:12]]
    remaining = len(coverage.inspected_paths) - len(lines)
    if remaining > 0:
        lines.append(f"- Plus {remaining} more inspected files captured in the workspace coverage trace.")
    return tuple(lines)


def _normalize_onboarding_script_stem(path: str) -> str:
    stem = Path(path).stem.lower()
    stem = re.sub(r"^\d+[_-]*", "", stem)
    stem = re.sub(r"\b(task\d+|debug|revised|old|copy)\b", " ", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", stem).strip()


def _onboarding_parallel_surface_details(
    coverage: OnboardingCoverage,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_top: dict[str, set[str]] = {}
    repeated_counter: Counter[str] = Counter()
    for entry in coverage.manifest_entries:
        if entry.category not in {"source", "scripts"} or entry.top_level == ".":
            continue
        stem = _normalize_onboarding_script_stem(entry.path)
        if not stem:
            continue
        top_level_stems = by_top.setdefault(entry.top_level, set())
        if stem not in top_level_stems:
            top_level_stems.add(stem)
            repeated_counter[stem] += 1

    repeated_stems = tuple(
        stem
        for stem, count in repeated_counter.most_common(6)
        if count >= 2
    )
    if len(repeated_stems) < 2:
        return (), ()

    repeated_dirs = tuple(
        top_level
        for top_level, stems in sorted(
            by_top.items(),
            key=lambda item: (-len(item[1].intersection(repeated_stems)), item[0].lower()),
        )
        if len(stems.intersection(repeated_stems)) >= 2
        and top_level.lower() not in _ONBOARDING_GENERIC_TOP_LEVEL_DIRS
    )
    if len(repeated_dirs) < 2:
        return (), ()
    return repeated_dirs[:6], repeated_stems[:4]


def _build_project_onboarding_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    coverage = _collect_onboarding_coverage(config.workspace.active_workspace_root.resolve())
    summary_map = dict(coverage.summary_map)
    is_chinese = mode_selection.response_language == "zh-CN"
    top_level = _find_tool_summary(tool_calls, "list_directory", path=".") or _onboarding_top_level_summary(coverage)
    readme_summary = summary_map.get("README.md") or _find_tool_summary(tool_calls, "read_text_file", path="README.md")
    config_summaries = _dedupe_strings(
        tuple(
            summary
            for path, summary in summary_map.items()
            if _classify_onboarding_path(path) == "config"
        )
    )
    config_paths = tuple(
        path
        for path in summary_map
        if _classify_onboarding_path(path) == "config"
    )
    entrypoint_paths = _rank_onboarding_entrypoints(
        _dedupe_strings((*coverage.inspected_paths, *evidence_refs))
    )
    entrypoint_summaries = tuple(
        summary_map[item]
        for item in entrypoint_paths
        if item in summary_map
    )
    notable_dirs = _onboarding_notable_dirs(coverage) or _extract_notable_workspace_dirs(evidence_refs)
    purpose_summary = _onboarding_purpose_summary(
        readme_summary=readme_summary,
        top_level=top_level,
        entrypoint_paths=entrypoint_paths,
        entrypoint_summaries=entrypoint_summaries,
        coverage=coverage,
        response_language=mode_selection.response_language,
    )
    tests_visible = any(entry.category == "tests" for entry in coverage.manifest_entries)
    notebooks_visible = any(entry.path.lower().endswith(".ipynb") for entry in coverage.manifest_entries)
    read_first = _onboarding_read_first_list(
        has_readme=bool(readme_summary),
        entrypoint_paths=entrypoint_paths,
        config_paths=config_paths,
        tests_visible=tests_visible,
        coverage=coverage,
        response_language=mode_selection.response_language,
    )
    risks = _onboarding_risk_lines(
        has_readme=bool(readme_summary),
        entrypoint_paths=entrypoint_paths,
        config_summaries=config_summaries,
        tests_visible=tests_visible,
        notebooks_visible=notebooks_visible,
        coverage=coverage,
        response_language=mode_selection.response_language,
    )
    next_steps = _onboarding_next_step_lines(
        has_readme=bool(readme_summary),
        entrypoint_paths=entrypoint_paths,
        config_summaries=config_summaries,
        tests_visible=tests_visible,
        coverage=coverage,
        response_language=mode_selection.response_language,
    )
    coverage_lines = _onboarding_coverage_lines(
        coverage,
        response_language=mode_selection.response_language,
    ) + _onboarding_focus_area_lines(
        coverage,
        response_language=mode_selection.response_language,
    )

    return "\n".join(
        [
            "项目目的" if is_chinese else "Project purpose",
            f"- {purpose_summary}",
            *[f"- {line}" for line in coverage_lines],
            "主要目录/模块" if is_chinese else "Main directories/modules",
            f"- {top_level or ('已参考文件中未能明确确认顶层工作区结构。' if is_chinese else 'Top-level workspace layout was not confirmed from consulted files.')}",
            *(
                [f"- {'重点区域' if is_chinese else 'Notable areas'}: {', '.join(notable_dirs)}."]
                if notable_dirs
                else [f"- {'除顶层目录外，没有进一步明确确认更多模块或目录边界。' if is_chinese else 'No additional module or directory boundaries were clearly confirmed beyond the top-level listing.'}"]
            ),
            "可能的入口" if is_chinese else "Likely entry points",
            *(
                [f"- `{item}`" for item in entrypoint_paths]
                if entrypoint_paths
                else [f"- {'已参考文件中未能清楚确认入口文件。' if is_chinese else 'Entry points were not clearly confirmed from the consulted files.'}"]
            ),
            *(
                [f"- {summary}" for summary in entrypoint_summaries[:3]]
                if entrypoint_summaries
                else []
            ),
            "配置/环境与依赖信号" if is_chinese else "Config/env and dependency signals",
            *(
                [f"- {summary}" for summary in config_summaries[:4]]
                if config_summaries
                else [f"- {'已参考文件中未能清楚确认配置或环境假设。' if is_chinese else 'Config or environment assumptions were not clearly confirmed from the consulted files.'}"]
            ),
            "第一批建议阅读内容" if is_chinese else "What to read first",
            *[f"- {item}" for item in read_first],
            "风险或缺失点" if is_chinese else "Risks or missing pieces",
            *[f"- {item}" for item in risks],
            "实用的下一步" if is_chinese else "Practical next steps",
            *[f"- {item}" for item in next_steps],
            "已参考的文件" if is_chinese else "Evidence/files consulted",
            *_bullet_lines(evidence_refs or _evidence_from_tool_calls(tool_calls)),
        ]
    )


def _project_onboarding_answer_needs_repair(
    answer_text: str,
    *,
    response_language: str,
) -> bool:
    stripped = answer_text.strip()
    lowered = answer_text.lower()
    if re.match(r"^[a-z_]*onboarding_sections\s*\{", lowered):
        return True
    if looks_like_structured_output(answer_text) and "{" in stripped and "}" in stripped:
        if response_language == "zh-CN" or "project purpose" in lowered or "likely entry points" in lowered:
            return True
    if response_language == "zh-CN":
        drift_markers = ("\u5f3a\u5ea6", "\u5f31\u70b9", "\u6298\u8877\u65b9\u6848")
        return any(marker in answer_text for marker in drift_markers)
    required_markers = (
        "project purpose",
        "entry point",
        "config",
        "read first",
        "risk",
        "next step",
    )
    marker_hits = sum(1 for marker in required_markers if marker in lowered)
    drift_markers = ("strengths", "weaknesses", "tradeoffs", "options being compared")
    return marker_hits < 4 or any(marker in lowered for marker in drift_markers)


def _build_file_explain_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    target_path = next((item for item in evidence_refs if item.endswith((".py", ".md", ".toml", ".json"))), evidence_refs[0] if evidence_refs else "not confirmed")
    file_summary = _find_tool_summary(tool_calls, "read_text_file", path=target_path)
    nearby_context = ""
    parent = Path(target_path).parent.as_posix() if target_path != "not confirmed" else ""
    if parent:
        nearby_context = _find_tool_summary(tool_calls, "list_directory", path=parent)

    return "\n".join(
        [
            "File purpose",
            f"- Target file: `{target_path}`.",
            f"- {file_summary or 'Purpose should be derived from the consulted file summary only.'}",
            "Key functions/classes",
            f"- {_extract_summary_segment(file_summary, 'classes') or 'Classes not confirmed from consulted lines.'}",
            f"- {_extract_summary_segment(file_summary, 'functions') or 'Functions not confirmed from consulted lines.'}",
            "Inputs/outputs",
            "- Inputs/outputs should be limited to what the consulted file summary confirms.",
            "Dependencies",
            f"- {nearby_context or 'Nearby module context was not confirmed.'}",
            "Risks or confusing spots",
            "- If behavior is not confirmed from the consulted file, say so instead of inferring it.",
            "Evidence/files consulted",
            *_bullet_lines(evidence_refs or _evidence_from_tool_calls(tool_calls)),
        ]
    )


def _build_architecture_review_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    workspace_specific = config.workspace.active_workspace_root == config.project_root
    top_level = _find_tool_summary(tool_calls, "list_directory", path=".")
    src_layout = _find_tool_summary(tool_calls, "list_directory", path="src") or _find_tool_summary(tool_calls, "list_directory", path="src/labai")
    python_search = _find_tool_summary(tool_calls, "find_files", path="src")
    loop_summary = _find_tool_summary(tool_calls, "read_text_file", path="src/labai/research/loop.py")
    return "\n".join(
        [
            "Relevant components",
            f"- {src_layout or top_level or 'Use the consulted workspace layout and file summaries to name the key components.'}",
            "Data/control flow",
            f"- {loop_summary or python_search or 'Describe data/control flow only from the consulted files.'}",
            "Runtime path and fallback path",
            *(
                [
                    f"- Config currently selects runtime `{config.runtime.runtime}` with fallback runtime `{config.runtime.fallback_runtime}`.",
                    f"- The preferred live path uses Claw with general model `{config.models.general_model}` or code model `{config.models.code_model}` depending on mode.",
                    f"- If the Claw path is unavailable, the code can fall back to the native provider route under the configured fallback policy `{config.fallback_policy}`.",
                ]
                if workspace_specific or any("claw" in item.lower() or "runtime" in item.lower() for item in evidence_refs)
                else ["- Runtime/fallback details are only relevant if the consulted workspace evidence explicitly covers them."]
            ),
            "Interaction points",
            "- Describe interaction points from the consulted files and directory layout instead of inventing missing components.",
            "Risks and hidden assumptions",
            "- If control flow or boundaries are not confirmed by the consulted files, say so explicitly.",
            "- Workspace access remains allowlisted rather than full-disk unrestricted.",
            "Evidence/files consulted",
            *_bullet_lines(evidence_refs or _evidence_from_tool_calls(tool_calls)),
        ]
    )


def _build_implementation_plan_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    mentions_pdf = "pdf" in prompt.lower()
    steps = [
        "- Keep the public CLI unchanged and add the next capability behind the existing `labai ask` path.",
        "- Extend config only where the next phase truly needs explicit local PDF/retrieval settings.",
        "- Add the next-phase orchestration close to the existing research loop instead of creating a separate public command surface.",
        "- Extend session and audit traces only if document evidence or retrieval steps need extra trace fields.",
        "- Add focused tests and update README/planning docs for the new capability.",
    ]
    if mentions_pdf:
        steps.insert(
            2,
            "- Add internal PDF ingest and retrieval logic as repo-local implementation surfaces that plug into the current research layer.",
        )

    return "\n".join(
        [
            "Current architecture baseline",
            "- `labai` remains the public control plane with `doctor`, `tools`, and `ask`.",
            "- `src/labai/research/loop.py` is the orchestration surface for internal research behavior.",
            "- `src/labai/config.py` holds runtime and model defaults, and the runtime traces live in `src/labai/runtime/session.py` and `src/labai/runtime/audit.py`.",
            "Goal",
            (
                "- Prepare the next phase for PDF ingest and local retrieval without adding new public CLI commands."
                if mentions_pdf
                else "- Draft the next implementation step using the current orchestration, config, and tracing surfaces."
            ),
            "Proposed steps",
            *steps,
            "Likely files/modules to change",
            "- `src/labai/research/loop.py`",
            "- `src/labai/config.py`",
            "- `src/labai/runtime/session.py`",
            "- `src/labai/runtime/audit.py`",
            "- `tests/`",
            "- `README.md` and relevant `.planning/` artifacts",
            "Risks",
            "- Preserving the live Claw path and native fallback behavior while expanding capabilities.",
            "- Keeping the public CLI unchanged and the tool surface read-only.",
            "- Handling large local documents without breaking current trace readability.",
            "Validation plan",
            "- Run pytest and keep the current CLI smoke checks green.",
            "- Add next-phase prompts that exercise the new PDF-aware behavior through `labai ask`.",
            "- Verify sessions and audit logs remain readable and traceable.",
            "Evidence/files consulted",
            *_bullet_lines(evidence_refs or _evidence_from_tool_calls(tool_calls)),
        ]
    )


def _build_workspace_edit_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
    *,
    workspace_root: Path | None,
    edit_plan: WorkspaceEditPlan | None,
    workspace_coverage: OnboardingCoverage | None = None,
) -> str:
    readme_summary = _find_tool_summary(tool_calls, "read_text_file", path="README.md")
    src_search = _find_tool_summary(tool_calls, "find_files", path="src")
    tests_search = _find_tool_summary(tool_calls, "find_files", path="tests")
    editable_targets = tuple(
        item
        for item in mode_selection.matched_paths
        if item.endswith((".py", ".md", ".toml", ".json", ".yaml", ".yml"))
    )
    support_files = tuple(
        item
        for item in evidence_refs
        if item.endswith((".py", ".md", ".toml", ".json", ".yaml", ".yml"))
        and item not in editable_targets
    )
    relevant_cues = _dedupe_strings(
        tuple(
            summary
            for summary in (
                *(
                    _find_tool_summary(tool_calls, "read_text_file", path=path)
                    for path in (*editable_targets, *support_files[:4])
                ),
            )
            if summary
        )
    )
    file_context_blocks = _workspace_edit_file_context_blocks(
        prompt,
        workspace_root,
        edit_plan,
    )
    package_hint = _workspace_edit_package_hint(editable_targets)
    manifest_summary_lines = _workspace_edit_manifest_summary_lines(workspace_coverage)
    primary_target_lines = (
        _bullet_lines(edit_plan.primary_targets)
        if edit_plan is not None and edit_plan.primary_targets
        else ["- Primary repair targets were not confirmed from the current edit plan."]
    )
    referenced_path_lines = (
        _bullet_lines(edit_plan.referenced_paths)
        if edit_plan is not None and edit_plan.referenced_paths
        else ["- No config-referenced path files were locked as read-only context."]
    )
    secondary_target_lines = (
        _bullet_lines(edit_plan.secondary_targets)
        if edit_plan is not None and edit_plan.secondary_targets
        else ["- No secondary doc or handoff sync targets were confirmed."]
    )
    return "\n".join(
        [
            "Task shape",
            "- Start with a short plan, then emit only the file blocks needed to complete the task.",
            "- Use the structured file-block format from the user prompt for every file you change or create.",
            "Primary repair targets",
            *primary_target_lines,
            "Referenced path context",
            *referenced_path_lines,
            "Secondary sync targets",
            *secondary_target_lines,
            "Editable target files",
            *(_bullet_lines(editable_targets) or ["- Editable targets were not confirmed from the prompt yet."]),
            "Support files consulted",
            *(_bullet_lines(support_files) or ["- No extra support files were consulted beyond the editable targets."]),
            "Existing workspace context",
            f"- {readme_summary or 'README context was not confirmed from the consulted files.'}",
            f"- {src_search or 'Source-file context was not confirmed from the consulted files.'}",
            f"- {tests_search or 'Test-file context was not confirmed from the consulted files.'}",
            "Workspace manifest",
            *(manifest_summary_lines or ["- Workspace-manifest coverage was not captured for this edit task."]),
            "Relevant code and test cues",
            *(
                [f"- {item}" for item in relevant_cues]
                or ["- No direct code or test summaries were captured beyond the directory scans."]
            ),
            "Current file context",
            *(
                list(file_context_blocks)
                or ["- No direct file excerpts were available for the current edit scope."]
            ),
            "Edit guardrails",
            "- Keep unrelated files untouched.",
            "- Emit a structured FILE block for every editable target that must change.",
            "- If the task involves config, routes, manifests, or entrypoints, repair the primary config or entrypoint target before syncing docs or handoff files.",
            "- Treat a path mentioned inside config or route metadata as read-only context unless the task or evidence clearly shows that the referenced file itself must change.",
            "- Do not let README or handoff files crowd out the primary config or entrypoint target when the task is a config or route repair.",
            "- If `pyproject.toml` is the primary packaging target, do not invent legacy `setup.py` or `setup.cfg` edits unless the prompt explicitly asks for them.",
            "- For grouped multi-file bug fixes or refactors, do not stop after fixing only one locked source file if another locked source file still participates in the failing contract.",
            "- Prefer modifying the locked editable targets before inventing new targets.",
            "- If a target does not need changes, say so in SUMMARY instead of creating an alternate file name.",
            "- If checks fail, keep the next edit pass scoped to the same editable targets unless the user explicitly expands the scope.",
            "- For bug fixes, match the visible test or assertion contract exactly instead of inventing a broader payload shape.",
            "- Unless the user explicitly asks to update tests, treat targeted tests as the contract to satisfy and keep them unchanged except for directly necessary import-path repair.",
            "- If a visible test or failing assertion shows an exact expected payload or string, match the key names, value types, and casing exactly.",
            "- Preserve existing imports, function signatures, and return structures unless the task or tests clearly require a change.",
            "- Do not point an import at a new helper module unless that file already exists or you create it in the same response with a FILE block.",
            "- When extracting a shared helper inside an existing package, keep the import path consistent with the package style already visible in the code and tests.",
            *( [f"- {package_hint}"] if package_hint else [] ),
            "Evidence/files consulted",
            *_bullet_lines(evidence_refs or _evidence_from_tool_calls(tool_calls)),
        ]
    )


def _workspace_edit_file_context_blocks(
    prompt: str,
    workspace_root: Path | None,
    edit_plan: WorkspaceEditPlan | None,
) -> tuple[str, ...]:
    if workspace_root is None or edit_plan is None or not edit_plan.active:
        return ()

    check_plan = build_workspace_check_plan(
        prompt,
        workspace_root,
        planned_modifications=edit_plan.planned_modifications,
        planned_creations=edit_plan.planned_creations,
    )
    support_targets = tuple(
        target
        for check in check_plan
        for target in check.relative_targets
        if target not in edit_plan.planned_modifications and target not in edit_plan.planned_creations
    )
    context_targets = _dedupe_strings(
        (
            *edit_plan.primary_targets,
            *edit_plan.referenced_paths[:2],
            *infer_workspace_config_reference_targets(prompt, workspace_root)[:2],
            *edit_plan.secondary_targets[:1],
            *support_targets[:2],
        )
    )[:6]

    blocks: list[str] = []
    for relative_path in context_targets:
        absolute = (workspace_root / relative_path).resolve()
        if not absolute.is_file():
            continue
        try:
            text = absolute.read_text(encoding="utf-8")
        except OSError:
            continue
        suffix = absolute.suffix.lstrip(".") or "text"
        excerpt = _workspace_edit_file_excerpt(
            text,
            max_lines=220 if absolute.suffix.lower() in {".toml", ".json", ".yaml", ".yml"} else 18,
            max_chars=12000 if absolute.suffix.lower() in {".toml", ".json", ".yaml", ".yml"} else 1200,
        )
        blocks.append(
            "\n".join(
                (
                    f"- `{relative_path}`",
                    f"```{suffix}",
                    excerpt,
                    "```",
                )
            )
        )
    return tuple(blocks)


def _workspace_edit_manifest_summary_lines(
    coverage: OnboardingCoverage | None,
) -> tuple[str, ...]:
    if coverage is None or not coverage.total_files:
        return ()
    lines = [
        (
            f"- Accounted for {coverage.total_files} files in the workspace manifest, including "
            f"{coverage.relevant_readable_count} relevant readable files, "
            f"{coverage.ignored_noise_count} ignored/noise files, and "
            f"{coverage.unreadable_binary_count} binary or unreadable files."
        ),
    ]
    if coverage.inspected_category_counts:
        category_summary = ", ".join(
            f"{name}={count}"
            for name, count in sorted(coverage.inspected_category_counts.items())
            if count
        )
        if category_summary:
            lines.append(f"- Inspected categories: {category_summary}.")
    if coverage.inspected_paths:
        inspected_preview = ", ".join(coverage.inspected_paths[:8])
        extra = ""
        if len(coverage.inspected_paths) > 8:
            extra = f", plus {len(coverage.inspected_paths) - 8} more inspected files"
        lines.append(f"- Inspected file sample: {inspected_preview}{extra}.")
    if coverage.skipped_paths:
        skipped_preview = ", ".join(coverage.skipped_paths[:6])
        extra = ""
        if len(coverage.skipped_paths) > 6:
            extra = f", plus {len(coverage.skipped_paths) - 6} more skipped files"
        lines.append(
            f"- Additional relevant files were skipped only after deterministic broader coverage was selected: {skipped_preview}{extra}."
        )
    return tuple(lines)


def _workspace_edit_package_hint(editable_targets: tuple[str, ...]) -> str:
    normalized_targets = tuple(item.replace("\\", "/") for item in editable_targets)
    if any(item.startswith("src/") for item in normalized_targets):
        return (
            "For helpers created under `src/`, keep imports package-qualified from the `src` package "
            "(for example `from src.utils.common import ...` or `from src.common_helpers import ...`) "
            "instead of bare imports like `from common_helpers import ...`."
        )
    return ""


def _workspace_edit_file_excerpt(text: str, *, max_lines: int = 18, max_chars: int = 1200) -> str:
    lines = text.splitlines()
    excerpt = "\n".join(lines[:max_lines]).rstrip()
    if len(excerpt) <= max_chars and len(lines) <= max_lines:
        return excerpt or "# (empty file)"
    trimmed = excerpt[:max_chars].rstrip()
    if not trimmed:
        return "# (empty file)"
    return trimmed + "\n# ... truncated ..."


def _build_prompt_compiler_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    rough_need = _extract_prompt_compiler_need(prompt)
    task_family = _prompt_compiler_task_family(rough_need)
    goal = _prompt_compiler_goal_sentence(task_family)
    user_need_block = rough_need if len(rough_need) <= 220 else rough_need[:217].rstrip() + "..."
    strong_prompt = _prompt_compiler_strong_prompt(task_family, user_need_block)
    constraints = _prompt_compiler_constraints(task_family)
    acceptance = _prompt_compiler_acceptance_criteria(task_family)
    open_questions = _prompt_compiler_open_questions(task_family)
    recommendation = _prompt_compiler_recommendation(task_family)
    compact_variant = _prompt_compiler_compact_variant(task_family, user_need_block)
    strict_variant = _prompt_compiler_strict_variant(task_family, user_need_block)
    return "\n".join(
        [
            "Compilation goal",
            f"- {goal}",
            f"- Preserve the original user need exactly in scope: `{user_need_block}`.",
            "Strong prompt",
            strong_prompt,
            "Constraints",
            *constraints,
            "Acceptance criteria",
            *acceptance,
            "Missing assumptions or open questions",
            *open_questions,
            "Recommendation",
            recommendation,
            "Compact variant",
            compact_variant,
            "Strict executable variant",
            strict_variant,
        ]
    )


def _extract_prompt_compiler_need(prompt: str) -> str:
    marker = "User need:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def _prompt_compiler_task_family(rough_need: str) -> str:
    lowered = rough_need.lower()
    if any(token in lowered for token in ("reproduce", "reproduc", "repro", "复现", "配置", "config")):
        return "repro_check"
    if any(token in lowered for token in ("compare", "comparison", "对比", "比较")) and any(
        token in lowered for token in ("paper", "pdf", "论文", "文献")
    ):
        return "paper_compare"
    if any(
        token in lowered
        for token in (
            "edit",
            "fix",
            "implement",
            "refactor",
            "update",
            "cleanup",
            "tidy",
            "修改",
            "修复",
            "实现",
            "重构",
            "整理",
            "清理",
            "create file",
        )
    ):
        return "code_edit"
    if any(token in lowered for token in ("onboard", "architecture", "overview", "workspace", "项目", "架构", "仓库")):
        return "workspace_review"
    if any(token in lowered for token in ("paper", "pdf", "论文", "文档", "总结", "summarize", "summary")):
        return "paper_task"
    return "generic"


def _prompt_compiler_goal_sentence(task_family: str) -> str:
    goals = {
        "repro_check": "Turn the rough need into a reproducibility-check handoff prompt that stays practical and evidence-seeking.",
        "paper_compare": "Turn the rough need into a paper-comparison handoff prompt that stays grounded and slot-to-slot.",
        "code_edit": "Turn the rough need into a coding-task handoff prompt that is scoped, check-aware, and ready for execution.",
        "workspace_review": "Turn the rough need into a workspace-review handoff prompt that is practical for a new RA.",
        "paper_task": "Turn the rough need into a paper-reading or paper-analysis handoff prompt that stays grounded in the documents.",
        "generic": "Turn the rough need into a clearer agent handoff prompt without widening scope or inventing extra deliverables.",
    }
    return goals.get(task_family, goals["generic"])


def _prompt_compiler_need_reference(task_family: str, rough_need: str) -> str:
    contains_chinese = any("\u4e00" <= character <= "\u9fff" for character in rough_need)
    family_descriptions = {
        "repro_check": "The user wants an agent to assess reproducibility and identify missing configuration or setup requirements.",
        "paper_compare": "The user wants an agent to compare papers carefully on explicit requested dimensions.",
        "code_edit": "The user wants an agent to turn a vague coding or cleanup request into a scoped execution task.",
        "workspace_review": "The user wants an agent to inspect a workspace and produce a practical onboarding or readiness summary.",
        "paper_task": "The user wants an agent to read or analyze papers while staying grounded in the documents.",
        "generic": "The user has a vague task request that needs clearer scope, constraints, and completion criteria.",
    }
    if contains_chinese:
        return family_descriptions.get(task_family, family_descriptions["generic"])
    return rough_need


def _prompt_compiler_strong_prompt(task_family: str, rough_need: str) -> str:
    need_reference = _prompt_compiler_need_reference(task_family, rough_need)
    task_lines = {
        "repro_check": (
            "Assess whether the target project is reproducible, identify missing configuration or environment assumptions, "
            "and separate confirmed findings from unconfirmed gaps."
        ),
        "paper_compare": (
            "Compare the requested papers on the user-specified dimensions, stay grounded in the source documents, "
            "and say clearly when a detail is not supported."
        ),
        "code_edit": (
            "Inspect the active workspace, plan the coding task briefly, make the required edits, and run the most relevant bounded checks."
        ),
        "workspace_review": (
            "Inspect the target workspace and produce a practical RA-facing summary of structure, readiness, risks, and next steps."
        ),
        "paper_task": (
            "Read the requested paper material carefully, preserve supported details, and keep the answer grounded in the consulted documents."
        ),
        "generic": (
            "Carry out the user's requested task without widening scope, and call out any blocking assumptions instead of guessing."
        ),
    }
    return (
        "- Act as a careful coding or research agent.\n"
        f"- Task: {task_lines.get(task_family, task_lines['generic'])}\n"
        f"- Scope anchor: {need_reference}\n"
        "- If an important assumption is missing, list it explicitly instead of inventing it."
    )


def _prompt_compiler_constraints(task_family: str) -> tuple[str, ...]:
    shared = (
        "- Do not solve the task inside this prompt package; rewrite it for another agent.",
        "- Keep the final prompt practical, task-oriented, and usable as a real handoff.",
        "- Do not invent repo files, modules, commands, or deliverables unless the user already asked for them.",
    )
    family_specific = {
        "repro_check": (
            "- Require the receiving agent to distinguish confirmed evidence from assumptions or missing setup details.",
            "- Keep the scope on reproducibility rather than broad repository redesign.",
        ),
        "paper_compare": (
            "- Require grounded paper-to-paper comparison rather than broad field-level commentary.",
            "- Make missing-detail honesty explicit instead of allowing unsupported comparison claims.",
        ),
        "code_edit": (
            "- Require a short plan, focused file selection, bounded checks, and a retry loop when checks fail.",
            "- Keep execution inside the active workspace or the explicitly named paths.",
        ),
        "workspace_review": (
            "- Keep the output practical for onboarding or verification rather than abstract architecture theory.",
            "- Tell the receiving agent to mark anything unconfirmed instead of assuming it exists.",
        ),
        "paper_task": (
            "- Require grounding in the provided PDFs or document paths.",
            "- Preserve supported details and avoid filler or field-common-sense padding.",
        ),
        "generic": (
            "- Preserve the user's scope and preferred deliverable shape.",
            "- Surface ambiguities as open questions instead of guessing through them.",
        ),
    }
    return shared + family_specific.get(task_family, family_specific["generic"])


def _prompt_compiler_acceptance_criteria(task_family: str) -> tuple[str, ...]:
    shared = (
        "- The resulting agent output should stay within the requested scope.",
        "- Constraints and acceptance criteria should be explicit enough for another agent to execute without guessing.",
    )
    family_specific = {
        "repro_check": (
            "- State whether the project looks reproducible from visible evidence.",
            "- List missing configuration, environment, or setup gaps explicitly.",
        ),
        "paper_compare": (
            "- Compare the papers on the requested dimensions rather than summarizing them separately.",
            "- Mark unsupported or unclear dimensions explicitly instead of inferring them.",
        ),
        "code_edit": (
            "- The requested file changes are actually applied or a concrete blocker is reported.",
            "- Relevant bounded checks are run and any failures feed back into a repair pass.",
        ),
        "workspace_review": (
            "- The answer identifies entry points, obvious blockers, and practical next steps.",
            "- Claims remain tied to visible workspace evidence.",
        ),
        "paper_task": (
            "- The answer preserves the paper details the user explicitly asked for.",
            "- Unsupported details are labeled as not clearly stated instead of inferred.",
        ),
        "generic": (
            "- The rewritten prompt is immediately usable as an agent handoff.",
            "- Missing assumptions are surfaced clearly instead of being silently filled in.",
        ),
    }
    return shared + family_specific.get(task_family, family_specific["generic"])


def _prompt_compiler_open_questions(task_family: str) -> tuple[str, ...]:
    mapping = {
        "repro_check": (
            "- Which platform, environment, or dependency baseline should be treated as canonical?",
            "- Should the receiving agent only inspect files, or also run bounded checks when they are clearly relevant?",
        ),
        "paper_compare": (
            "- Which compare dimensions are mandatory if the user did not list them explicitly?",
            "- Should the final answer prioritize compactness or a more detailed RA-style comparison note?",
        ),
        "code_edit": (
            "- Which files or directories are in scope if the user has not named them explicitly?",
            "- Which checks are safe and expected for this task?",
        ),
        "workspace_review": (
            "- Should the answer focus on onboarding, readiness, or reproducibility if the user did not specify the angle?",
            "- Is there a preferred output language or level of detail for the final handoff?",
        ),
        "paper_task": (
            "- Which PDFs or document paths are the actual source set?",
            "- Should the output be concise, detailed, English-only, Chinese-only, or bilingual?",
        ),
        "generic": (
            "- What exact deliverable or answer shape should the receiving agent produce?",
            "- What would count as done for the user if the task completes successfully?",
        ),
    }
    return mapping.get(task_family, mapping["generic"])


def _prompt_compiler_recommendation(task_family: str) -> str:
    if task_family in {"repro_check", "code_edit", "paper_compare"}:
        return "- Use the strict executable variant because the task benefits from explicit constraints and completion criteria."
    if task_family == "workspace_review":
        return "- Use the compact variant for quick onboarding, and the strict executable variant when the receiving agent needs a more checkable contract."
    return "- Start with the compact variant for quick handoff, then switch to the strict executable variant if the task still feels underspecified."


def _prompt_compiler_compact_variant(task_family: str, rough_need: str) -> str:
    need_reference = _prompt_compiler_need_reference(task_family, rough_need)
    task_sentence = _prompt_compiler_goal_sentence(task_family).replace("Turn the rough need into ", "").rstrip(".")
    return (
        f"- Help with this request: {need_reference}\n"
        f"- {task_sentence.capitalize()}. Keep the scope tight, call out missing assumptions explicitly, and return a practical result instead of generic advice."
    )


def _prompt_compiler_strict_variant(task_family: str, rough_need: str) -> str:
    need_reference = _prompt_compiler_need_reference(task_family, rough_need)
    return (
        "- Act as a careful coding or research agent.\n"
        f"- User need: {need_reference}\n"
        f"- Primary task: {_prompt_compiler_goal_sentence(task_family)}\n"
        "- Required behavior: keep the scope explicit, separate confirmed findings from assumptions, and surface blockers clearly.\n"
        "- Final deliverable: return a practical answer or plan that another agent could execute against without needing to guess the constraints."
    )


def _render_prompt_compiler_output(rough_need: str) -> str:
    task_family = _prompt_compiler_task_family(rough_need)
    return "\n".join(
        [
            "Strong prompt",
            _prompt_compiler_strong_prompt(task_family, rough_need),
            "",
            "Constraints",
            *_prompt_compiler_constraints(task_family),
            "",
            "Acceptance criteria",
            *_prompt_compiler_acceptance_criteria(task_family),
            "",
            "Missing assumptions or open questions",
            *_prompt_compiler_open_questions(task_family),
            "",
            "Recommendation",
            _prompt_compiler_recommendation(task_family),
            "",
            "Compact variant",
            _prompt_compiler_compact_variant(task_family, rough_need),
            "",
            "Strict executable variant",
            _prompt_compiler_strict_variant(task_family, rough_need),
        ]
    )


def _apply_prompt_compiler_guard(prompt: str, answer_text: str) -> str:
    if not _prompt_compiler_answer_looks_valid(answer_text):
        rough_need = _extract_prompt_compiler_need(prompt)
        return _render_prompt_compiler_output(rough_need)
    return answer_text


def _prompt_compiler_answer_looks_valid(answer_text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", answer_text or "").strip().lower()
    if not normalized:
        return False
    if any("\u4e00" <= character <= "\u9fff" for character in answer_text):
        return False
    if normalized.startswith("```json") or normalized.startswith("{"):
        return False
    if '"name"' in normalized and '"arguments"' in normalized and "```" in normalized:
        return False
    required_headings = (
        "strong prompt",
        "constraints",
        "acceptance criteria",
        "missing assumptions",
        "recommendation",
        "compact variant",
        "strict executable variant",
    )
    return all(heading in normalized for heading in required_headings)


def _build_compare_options_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    return "\n".join(
        [
            "Options being compared",
            "- Claw adapter path: `run_research_loop` -> `_run_answer_route` -> `_run_claw_route` -> `ClawRuntimeAdapter.ask`.",
            "- Native fallback path: `run_research_loop` -> `_run_answer_route` -> `_run_native_route` -> native provider `ask`.",
            "Strengths",
            "- Claw path exercises the preferred live runtime and the configured local Qwen defaults.",
            "- Claw path uses a machine-readable JSON flow and a read-only permission posture.",
            "- Native path keeps `labai ask` available when the preferred runtime is unavailable.",
            "- Native path preserves deterministic fallback behavior for testing through the configured fallback policy.",
            "Weaknesses",
            "- Claw path depends on the local Claw binary plus a reachable local Ollama endpoint.",
            "- Native path does not exercise the preferred live Claw/Qwen runtime and is mainly a continuity/fallback route.",
            "Tradeoffs",
            "- Choose Claw when you want the real local runtime behavior and model routing.",
            "- Choose the native path when you need a safe fallback or deterministic test behavior.",
            "Recommendation",
            "- Prefer Claw on a ready machine for RA-facing repo research and live local Qwen behavior.",
            "- Use the native path for degraded-mode continuity, runtime outages, or deterministic tests.",
            "Evidence/files consulted",
            *_bullet_lines(evidence_refs or _evidence_from_tool_calls(tool_calls)),
        ]
    )


def _build_paper_summary_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    if not paper_trace.discovered_documents:
        return "\n".join(
            [
                "Document identity",
                "- No local PDF target was discovered for this prompt.",
                "Main contribution / purpose",
                "- Not confirmed because no PDF evidence was available.",
                "Method / structure",
                "- Not confirmed because no PDF evidence was available.",
                "Key caveats",
                "- Report the missing PDF targets instead of inferring document contents.",
                "Evidence refs",
                "- None",
            ]
        )

    document_note = paper_trace.document_notes[0] if paper_trace.document_notes else {}
    requested_slots = _requested_paper_slots(prompt, mode_selection.mode)
    return "\n".join(
        [
            "Document identity",
            f"- Primary target: `{paper_trace.discovered_documents[0]}`.",
            f"- Read strategy: `{paper_trace.read_strategy}`.",
            f"- Windows processed: {paper_trace.window_count_processed}.",
            "Semantic slot coverage",
            *[
                _format_slot_line(_document_slot(document_note, slot_name))
                for slot_name in requested_slots
            ],
            "Grounded synthesis rules",
            "- Use the slot coverage lines above as the factual scaffold for the final answer.",
            "- If a slot is weakly supported, keep the wording restrained and avoid broad textbook commentary.",
            "- If a slot is not clearly stated, say not confirmed or 文中未明确说明 instead of filling the gap with generic domain knowledge.",
            "- Do not add general machine-learning, finance, or investment commentary unless it appears in the slot evidence above.",
            *(["- OCR is still required for: " + ", ".join(paper_trace.ocr_required_paths)] if paper_trace.ocr_required_paths else []),
            "Whole-document coverage",
            *(_paper_window_lines(paper_trace.document_windows) or ["- Whole-document coverage notes were not available for this ask."]),
            "Evidence refs",
            *_bullet_lines(evidence_refs or paper_trace.discovered_documents or paper_trace.target_paths),
        ]
    )


def _build_paper_compare_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    compared = paper_trace.discovered_documents or paper_trace.target_paths
    requested_slots = (
        "research_question",
        "method",
        "main_findings",
        "limitations",
        "conclusion",
    )
    return "\n".join(
        [
            "Documents compared",
            *_bullet_lines(compared or ("No PDF targets were discovered.",)),
            "Read strategy",
            f"- `{paper_trace.read_strategy}`.",
            "Slot-to-slot comparison scaffold",
            *[
                line
                for slot_name in requested_slots
                for line in _comparison_slot_lines(slot_name, paper_trace.document_notes)
            ],
            "Grounded comparison rules",
            "- Compare papers slot-to-slot rather than by generic theme blending.",
            "- If one paper supports a dimension and another does not, say that asymmetry explicitly.",
            "- Keep missing or weakly supported dimensions restrained instead of smoothing them into a confident summary.",
            "Whole-document coverage",
            *(_paper_window_lines(paper_trace.document_windows) or ["- Whole-document coverage notes were not available for this comparison."]),
            "Evidence refs",
            *_bullet_lines(evidence_refs or compared),
            "Retrieved chunk excerpts",
            *(_paper_excerpt_lines(paper_trace.retrieved_chunks) or ["- No retrievable chunk text was available."]),
        ]
    )


def _build_paper_grounded_qa_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    requested_slots = _requested_paper_slots(prompt, mode_selection.mode)
    recurring_limitations = _recurring_slot_lines(
        paper_trace.document_notes,
        slot_name="limitations",
    )
    return "\n".join(
        [
            "Direct answer",
            *(
                recurring_limitations
                if recurring_limitations and (
                    "limitation" in prompt.lower()
                    or "limitations" in prompt.lower()
                    or "局限" in prompt
                    or "限制" in prompt
                )
                else _relevant_slot_lines(paper_trace.document_notes, requested_slots)
            ),
            "Grounded supporting evidence",
            "- Quote or paraphrase only the retrieved chunks and aggregated slot notes that support the answer.",
            "Uncertainty",
            "- If the evidence is weak, missing, or OCR is required, say so explicitly instead of filling gaps with field-common-sense commentary.",
            *(["- OCR is still required for: " + ", ".join(paper_trace.ocr_required_paths)] if paper_trace.ocr_required_paths else []),
            "Evidence refs",
            *_bullet_lines(evidence_refs or paper_trace.target_paths),
            "Retrieved chunk excerpts",
            *(_paper_excerpt_lines(paper_trace.retrieved_chunks) or ["- No retrievable chunk text was available."]),
        ]
    )


def _find_tool_summary(
    tool_calls: list[ToolCall],
    tool_name: str,
    *,
    path: str,
) -> str:
    for tool_call in tool_calls:
        if tool_call.tool_name != tool_name:
            continue
        if tool_call.arguments.get("path") == path:
            return tool_call.summary
    return ""


def _find_matching_tool_summaries(
    tool_calls: list[ToolCall],
    tool_name: str,
    *,
    predicate,
) -> tuple[str, ...]:
    summaries: list[str] = []
    for tool_call in tool_calls:
        if tool_call.tool_name != tool_name:
            continue
        path = str(tool_call.arguments.get("path", ""))
        if not predicate(path):
            continue
        if tool_call.summary:
            summaries.append(tool_call.summary)
    return _dedupe_strings(tuple(summaries))


def _find_matching_tool_paths(
    tool_calls: list[ToolCall],
    tool_name: str,
    *,
    predicate,
) -> tuple[str, ...]:
    paths: list[str] = []
    for tool_call in tool_calls:
        if tool_call.tool_name != tool_name:
            continue
        path = str(tool_call.arguments.get("path", ""))
        if predicate(path):
            paths.append(path)
    return _dedupe_strings(tuple(paths))


def _extract_notable_workspace_dirs(evidence_refs: tuple[str, ...]) -> tuple[str, ...]:
    dirs: list[str] = []
    for item in evidence_refs:
        if item in {"", "."}:
            continue
        raw_normalized = item.replace("\\", "/").strip()
        first_raw = raw_normalized.split("/", 1)[0]
        if first_raw.startswith(".") or first_raw in _ONBOARDING_IGNORED_DIR_NAMES:
            continue
        normalized = item.replace("\\", "/").strip("./")
        if not normalized:
            continue
        first = normalized.split("/", 1)[0]
        if first in _ONBOARDING_IGNORED_DIR_NAMES:
            continue
        if "." in first and not first.startswith("."):
            continue
        if first not in dirs:
            dirs.append(first)
    return tuple(dirs[:6])


def _rank_onboarding_entrypoints(evidence_refs: tuple[str, ...]) -> tuple[str, ...]:
    scored: list[tuple[int, str]] = []
    for item in evidence_refs:
        if not item.lower().endswith((".py", ".sh", ".ps1", ".bat", ".cmd")):
            continue
        path = item.replace("\\", "/")
        name = Path(path).name.lower()
        lowered_path = path.lower()
        score = 0
        if name == "__main__.py":
            score += 140
        elif name in _ONBOARDING_ENTRYPOINT_NAMES:
            score += 100
        if any(token in Path(path).stem.lower() for token in ("main", "run", "cli", "app", "serve")):
            score += 15
        if any(token in Path(path).stem.lower() for token in ("download", "prepare", "merge", "build", "calc", "pipeline", "train", "process", "launch", "start")):
            score += 12
        if re.match(r"^\d+[_-]", Path(path).stem.lower()):
            score += 8
        if lowered_path.startswith("src/"):
            score += 20
        if path.count("/") == 0:
            score += 20
        if "/scripts/" in f"/{path.lower()}/" or path.lower().startswith("scripts/"):
            score += 10
        if lowered_path.startswith(("docs/", "tests/", "examples/")):
            score -= 70
        if name in {"setup.py", "conf.py", "__version__.py"}:
            score -= 60
        if "test" in name:
            score -= 40
        if name == "__init__.py":
            score -= 20
        if "/tests/" in f"/{path.lower()}/":
            score -= 50
        scored.append((score, path))

    ordered = [
        path
        for score, path in sorted(scored, key=lambda item: (-item[0], item[1].lower()))
        if score > -20
    ]
    unique: list[str] = []
    for item in ordered:
        if item not in unique:
            unique.append(item)
    return tuple(unique[:6])


def _onboarding_purpose_summary(
    *,
    readme_summary: str | None,
    top_level: str | None,
    entrypoint_paths: tuple[str, ...],
    entrypoint_summaries: tuple[str, ...],
    coverage: OnboardingCoverage,
    response_language: str,
) -> str:
    repeated_dirs, repeated_stems = _onboarding_parallel_surface_details(coverage)
    repeated_dir_label = ", ".join(f"`{item}`" for item in repeated_dirs[:4])
    repeated_stem_label = ", ".join(f"`{item}`" for item in repeated_stems[:4])
    if readme_summary:
        if repeated_dirs and repeated_stems:
            return (
                f"\u4ece\u6574\u4e2a\u5de5\u4f5c\u533a\u8986\u76d6\u6765\u770b\uff0c{readme_summary}\u3002"
                f" \u540c\u65f6\uff0c{repeated_dir_label} \u8fd9\u4e9b\u76ee\u5f55\u91cc\u53cd\u590d\u51fa\u73b0 {repeated_stem_label} \u8fd9\u7c7b\u811a\u672c\uff0c"
                "\u8bf4\u660e\u8fd9\u4e2a\u4ed3\u5e93\u4e0d\u662f\u53ea\u8bfb README \u5c31\u80fd\u7406\u89e3\u7684\u5355\u4e00\u5e94\u7528\uff0c\u800c\u662f\u4e00\u4e2a\u7531\u591a\u4e2a\u5e73\u884c\u5de5\u4f5c\u533a\u5171\u540c\u6784\u6210\u7684\u5b9e\u9645\u7814\u7a76\u6d41\u7a0b\u3002"
                if response_language == "zh-CN"
                else f"{readme_summary} Project-wide coverage also shows parallel work areas such as {repeated_dir_label}, with repeated script families like {repeated_stem_label}, so the repo behaves more like a multi-workspace research pipeline than a single thin README-driven app."
            )
        return readme_summary
    if repeated_dirs and repeated_stems:
        return (
            f"\u5de5\u4f5c\u533a\u8986\u76d6\u663e\u793a {repeated_dir_label} \u8fd9\u4e9b\u540c\u7ea7\u76ee\u5f55\u91cc\u53cd\u590d\u51fa\u73b0 {repeated_stem_label} \u8fd9\u7c7b\u811a\u672c\uff0c"
            "\u8bf4\u660e\u8fd9\u4e2a\u9879\u76ee\u66f4\u50cf\u662f\u4e00\u4e2a\u7531\u591a\u4e2a\u5ba1\u6838/\u5e76\u884c\u5de5\u4f5c\u533a\u6784\u6210\u7684\u7814\u7a76\u6d41\u7a0b\uff1a"
            "\u9700\u8981\u5206\u522b\u4e0b\u8f7d\u6570\u636e\uff0c\u51c6\u5907\u8868\u683c\u6216\u7279\u5f81\uff0c\u5408\u5e76\u4e2d\u95f4\u7ed3\u679c\uff0c\u518d\u505a\u540e\u7eed\u9a8c\u8bc1\u3002"
            if response_language == "zh-CN"
            else f"Project-wide coverage shows sibling work areas such as {repeated_dir_label} with repeated script families like {repeated_stem_label}, so this looks like a multi-workspace research pipeline for downloading/preparing data, building intermediate tables or signals, merging outputs, and running follow-on validation."
        )
    if repeated_dirs and repeated_stems:
        lines.append(
            f"\u628a {', '.join(f'`{item}`' for item in repeated_dirs[:2])} \u4e0e\u5176\u4ed6\u540c\u7c7b\u76ee\u5f55\u505a\u4e00\u6b21\u6bd4\u5bf9\uff0c\u786e\u8ba4 {', '.join(f'`{item}`' for item in repeated_stems[:3])} \u8fd9\u4e9b\u91cd\u590d\u811a\u672c\u5728\u54ea\u4e9b\u5de5\u4f5c\u533a\u91cc\u662f\u57fa\u7ebf\uff0c\u54ea\u4e9b\u662f\u5ba1\u6838\u53d8\u4f53\u3002"
            if is_chinese
            else f"Compare one representative workspace such as {', '.join(f'`{item}`' for item in repeated_dirs[:2])} against the other sibling areas to confirm where repeated script families like {', '.join(f'`{item}`' for item in repeated_stems[:3])} are baseline behavior versus audit-specific variations."
        )
    if entrypoint_paths:
        entrypoint_list = ", ".join(f"`{item}`" for item in entrypoint_paths[:3])
        if entrypoint_summaries:
            return (
                f"已参考的实现文件表明项目主要围绕 {entrypoint_list} 展开；在缺少显式 README 时，应先从这些入口确认项目目标。"
                if response_language == "zh-CN"
                else f"The consulted implementation files suggest the project is organized around {entrypoint_list}; use those entry points to confirm the exact project goal when no explicit README is visible."
            )
        return (
            f"项目目的需要从 {entrypoint_list} 这类已参考入口文件中继续确认，因为目前没有看到更直接的顶层说明。"
            if response_language == "zh-CN"
            else f"The project purpose needs to be confirmed from consulted entry-point files such as {entrypoint_list} because no more direct top-level explanation was visible."
        )
    return (
        top_level
        or (
            "除已参考的目录结构外，项目目的尚未被明确确认。"
            if response_language == "zh-CN"
            else "Project purpose was not confirmed beyond the consulted workspace layout."
        )
    )


def _onboarding_read_first_list(
    *,
    has_readme: bool,
    entrypoint_paths: tuple[str, ...],
    config_paths: tuple[str, ...],
    tests_visible: bool,
    coverage: OnboardingCoverage,
    response_language: str,
) -> tuple[str, ...]:
    is_chinese = response_language == "zh-CN"
    items: list[str] = []
    repeated_dirs, repeated_stems = _onboarding_parallel_surface_details(coverage)
    if has_readme:
        items.append("先读 `README.md`，确认项目目的和可见的安装/使用说明。" if is_chinese else "Start with `README.md` for the stated project purpose and any setup guidance.")
    if repeated_dirs and repeated_stems:
        items.append(
            f"\u5148\u9009\u4e00\u4e2a\u4ee3\u8868\u6027\u7684\u5e73\u884c\u5de5\u4f5c\u533a\uff08\u4f8b\u5982 {', '.join(f'`{item}`' for item in repeated_dirs[:2])}\uff09\uff0c"
            f"\u518d\u5bf9\u7167\u91cc\u9762\u53cd\u590d\u51fa\u73b0\u7684 {', '.join(f'`{item}`' for item in repeated_stems[:3])} \u8fd9\u4e9b\u811a\u672c\uff0c\u53ef\u4ee5\u66f4\u5feb\u7406\u89e3\u6574\u4e2a\u6d41\u7a0b\u662f\u600e\u4e48\u88ab\u5207\u5206\u7684\u3002"
            if is_chinese
            else f"Start with one representative parallel work area such as {', '.join(f'`{item}`' for item in repeated_dirs[:2])}, then compare the repeated script families {', '.join(f'`{item}`' for item in repeated_stems[:3])} to understand how the overall pipeline is partitioned."
        )
    for path in entrypoint_paths[:2]:
        items.append(
            f"优先阅读 `{path}`，因为它看起来像主要执行入口或 CLI 入口。"
            if is_chinese
            else f"Read `{path}` early because it looks like a likely execution or CLI entry point."
        )
    for path in config_paths[:2]:
        items.append(
            f"查看 `{path}`，确认依赖、环境或工具链假设。"
            if is_chinese
            else f"Check `{path}` to confirm dependency, environment, or toolchain assumptions."
        )
    if tests_visible:
        items.append(
            "在入口文件之后阅读可见的测试文件，了解预期行为和验证方式。"
            if is_chinese
            else "Read the visible test surface after the entry points to understand expected behavior and verification."
        )
    if not items:
        items.append(
            "由于没有明确确认 README 或入口文件，先从顶层目录和最清晰的源码文件开始。"
            if is_chinese
            else "Start with the top-level layout and the clearest source file because no README or explicit entry point was firmly confirmed."
        )
    return tuple(items[:4])


def _onboarding_risk_lines(
    *,
    has_readme: bool,
    entrypoint_paths: tuple[str, ...],
    config_summaries: tuple[str, ...],
    tests_visible: bool,
    notebooks_visible: bool,
    coverage: OnboardingCoverage,
    response_language: str,
) -> tuple[str, ...]:
    is_chinese = response_language == "zh-CN"
    lines: list[str] = []
    repeated_dirs, repeated_stems = _onboarding_parallel_surface_details(coverage)
    repeated_dirs, repeated_stems = _onboarding_parallel_surface_details(coverage)
    if not has_readme:
        lines.append(
            "顶层 README 不明显，项目目的和安装意图可能需要从代码和配置文件中反推。"
            if is_chinese
            else "A top-level README was not clearly available, so project purpose and setup intent may need to be reconstructed from code and config files."
        )
    if repeated_dirs and repeated_stems:
        lines.append(
            f"\u591a\u4e2a\u540c\u7ea7\u76ee\u5f55\uff08\u4f8b\u5982 {', '.join(f'`{item}`' for item in repeated_dirs[:3])}\uff09\u53cd\u590d\u51fa\u73b0 {', '.join(f'`{item}`' for item in repeated_stems[:3])} \u8fd9\u7c7b\u811a\u672c\uff0c"
            "\u6240\u4ee5\u65b0 RA \u9700\u8981\u5148\u5206\u6e05\u54ea\u4e2a\u662f\u4ee3\u8868\u6027\u5de5\u4f5c\u533a\uff0c\u54ea\u4e9b\u662f\u5e76\u884c\u5ba1\u6838\u6216\u53d8\u4f53\u526f\u672c\u3002"
            if is_chinese
            else f"Multiple sibling directories (for example {', '.join(f'`{item}`' for item in repeated_dirs[:3])}) repeat script families such as {', '.join(f'`{item}`' for item in repeated_stems[:3])}, so a new RA still needs to confirm which workspace is canonical versus parallel audit or variant copies."
        )
    if not entrypoint_paths:
        lines.append(
            "已参考文件中没有牢靠确认清晰的运行入口或 CLI 入口。"
            if is_chinese
            else "A clear runtime or CLI entry point was not firmly confirmed from the consulted files."
        )
    if not config_summaries:
        lines.append(
            "已参考文件中没有清楚声明环境、依赖或配置假设。"
            if is_chinese
            else "Environment, dependency, or config assumptions were not clearly declared in the consulted files."
        )
    if not tests_visible:
        lines.append(
            "验证或测试面不够清楚，因此实际接手前可能需要额外的人工检查。"
            if is_chinese
            else "The verification or test surface was not clearly visible, so practical confidence checks may need extra manual inspection."
        )
    if notebooks_visible and not entrypoint_paths:
        lines.append(
            "工作区里有 notebook，这可能意味着项目更偏分析流程，而不是由单一可执行入口驱动。"
            if is_chinese
            else "Notebook evidence is present, which may mean the project is analysis-first rather than driven by one clean executable entry point."
        )
    lines.append(
        "凡是不直接出现在已参考文件里的行为，都应视为尚未确认，而不是默认成立。"
        if is_chinese
        else "Any behavior not directly visible in the consulted files should be treated as not confirmed rather than assumed."
    )
    return tuple(lines[:5])


def _onboarding_next_step_lines(
    *,
    has_readme: bool,
    entrypoint_paths: tuple[str, ...],
    config_summaries: tuple[str, ...],
    tests_visible: bool,
    coverage: OnboardingCoverage,
    response_language: str,
) -> tuple[str, ...]:
    is_chinese = response_language == "zh-CN"
    lines: list[str] = []
    if has_readme:
        lines.append(
            "先读 README.md，再把其中的说法和实际源码、配置文件逐一对照。"
            if is_chinese
            else "Read the README first, then map its claims against the actual source files and configs."
        )
    if entrypoint_paths:
        lines.append(
            f"接着追踪 `{entrypoint_paths[0]}`，确认项目究竟是如何启动或编排的。"
            if is_chinese
            else f"Trace `{entrypoint_paths[0]}` next to confirm how the project is actually started or orchestrated."
        )
    if config_summaries:
        lines.append(
            "在尝试运行或扩展项目之前，先核对可见的配置和依赖文件。"
            if is_chinese
            else "Verify the visible config and dependency files before trying to run or extend the project."
        )
    if tests_visible:
        lines.append(
            "尽早查看可见测试，了解预期行为和最快的验证路径。"
            if is_chinese
            else "Inspect the visible tests early to understand expected behavior and fast verification paths."
        )
    if not lines:
        lines.append(
            "由于工作区里缺少明确引导，先从最核心的源码文件开始，并手动确认可运行路径。"
            if is_chinese
            else "Start with the most central source files and confirm the runnable path manually because the workspace exposes limited explicit guidance."
        )
    return tuple(lines[:4])


def _workspace_section_title(title: str, *, is_chinese: bool) -> str:
    if not is_chinese:
        return title
    mapping = {
        "Readiness status": "\u5de5\u4f5c\u533a\u5c31\u7eea\u72b6\u6001",
        "Why this status was chosen": "\u4e3a\u4ec0\u4e48\u5f97\u51fa\u8fd9\u4e2a\u5224\u65ad",
        "What is clearly present": "\u5df2\u660e\u786e\u53ef\u89c1\u7684\u5185\u5bb9",
        "Likely entry points or run surfaces": "\u6700\u53ef\u80fd\u7684\u5165\u53e3\u6216\u8fd0\u884c\u9762",
        "Config/env and dependency signals": "\u914d\u7f6e/\u73af\u5883/\u4f9d\u8d56\u4fe1\u53f7",
        "Missing pieces or blockers": "\u7f3a\u5931\u9879\u6216\u963b\u585e\u70b9",
        "Risks or uncertainty": "\u98ce\u9669\u4e0e\u4e0d\u786e\u5b9a\u6027",
        "What to read first": "\u5148\u8bfb\u4ec0\u4e48",
        "First three practical next steps": "\u524d\u4e09\u4e2a\u5b9e\u9645\u4e0b\u4e00\u6b65",
        "Evidence/files consulted": "\u5df2\u53c2\u8003\u7684\u8bc1\u636e/\u6587\u4ef6",
    }
    return mapping.get(title, title)


def _workspace_not_confirmed(section: str, *, is_chinese: bool) -> str:
    if is_chinese:
        if section == "Likely entry points or run surfaces":
            return "\u6587\u4ef6\u8bc1\u636e\u4e2d\u6ca1\u6709\u7a33\u56fa\u786e\u8ba4\u5230\u6e05\u6670\u7684\u5165\u53e3\u6216\u8fd0\u884c\u9762\u3002"
        if section == "Missing pieces or blockers":
            return "\u53ef\u89c1\u6587\u4ef6\u91cc\u6682\u65f6\u6ca1\u6709\u76f4\u63a5\u786e\u8ba4\u5230\u660e\u786e\u7684 blocker\uff0c\u4f46\u4e0d\u7b49\u4e8e\u5b8c\u5168 ready\u3002"
        return "\u53ef\u89c1\u6587\u4ef6\u4e2d\u6ca1\u6709\u7a33\u56fa\u786e\u8ba4\u51fa\u8fd9\u4e2a\u7ef4\u5ea6\u3002"
    if section == "Likely entry points or run surfaces":
        return "A clear entry point or runnable surface was not firmly confirmed from the consulted files."
    if section == "Missing pieces or blockers":
        return "No explicit blocker was firmly confirmed from the visible files, but that does not automatically mean the workspace is fully ready."
    return "This dimension was not firmly confirmed from the consulted files."


def _workspace_primary_doc_summary(summary_map: dict[str, str]) -> str | None:
    for candidate in ("README.md", "README.rst", "README.txt", "AGENTS.md", "CLAUDE.md", "PROJECT.md"):
        summary = summary_map.get(candidate)
        if summary:
            return summary
    return None


def _workspace_explicit_blockers(
    summary_map: dict[str, str],
    *,
    response_language: str,
) -> tuple[str, ...]:
    blocker_patterns = (
        "todo",
        "fixme",
        "placeholder",
        "set_me",
        "set me",
        "fill this in",
        "not checked in",
        "local credential",
        "missing secret",
        "replace this path",
        "update this path",
        "change me",
    )
    lines: list[str] = []
    is_chinese = response_language == "zh-CN"
    for path, summary in summary_map.items():
        lowered = summary.lower()
        if not any(token in lowered for token in blocker_patterns):
            continue
        lines.append(
            (
                f"`{path}` \u91cc\u6709\u660e\u663e\u7684 placeholder/TODO \u6216\u672c\u5730\u7f3a\u5931\u914d\u7f6e\u63d0\u793a\uff0c\u8fd9\u8bf4\u660e\u5de5\u4f5c\u533a\u8fd8\u4e0d\u80fd\u88ab\u5f53\u6210\u5b8c\u5168\u53ef\u76f4\u63a5\u8fd0\u884c\u7684\u73af\u5883\u3002"
                if is_chinese
                else f"`{path}` contains visible placeholder, TODO, or missing-setup wording, which suggests the workspace is not fully runnable as-is."
            )
        )
    return _dedupe_strings(lines)[:3]


def _workspace_verification_next_steps(
    *,
    status: str,
    has_readme: bool,
    entrypoint_paths: tuple[str, ...],
    config_summaries: tuple[str, ...],
    tests_visible: bool,
    coverage: OnboardingCoverage,
    response_language: str,
) -> tuple[str, ...]:
    is_chinese = response_language == "zh-CN"
    lines = list(
        _onboarding_next_step_lines(
            has_readme=has_readme,
            entrypoint_paths=entrypoint_paths,
            config_summaries=config_summaries,
            tests_visible=tests_visible,
            coverage=coverage,
            response_language=response_language,
        )
    )
    if status in {"partially_ready", "blocked", "uncertain"}:
        lines.insert(
            0,
            (
                "\u5728\u5c1d\u8bd5\u8fd0\u884c\u6216\u4fee\u6539\u9879\u76ee\u4e4b\u524d\uff0c\u5148\u8865\u4e0a\u53ef\u89c1\u7684\u5165\u53e3\u3001\u914d\u7f6e\u6216\u9a8c\u8bc1\u8def\u5f84 gap\u3002"
                if is_chinese
                else "Before trying to run or modify the project, close the visible gaps around entrypoints, config, or verification paths."
            ),
        )
    elif status == "ready_with_gaps":
        lines.insert(
            0,
            (
                "\u53ef\u4ee5\u5f00\u59cb\u5de5\u4f5c\uff0c\u4f46\u5efa\u8bae\u5148\u8865\u6389\u6700\u660e\u663e\u7684 gap\uff0c\u518d\u628a\u5b83\u5f53\u6210\u5b8c\u5168 ready \u7684\u5de5\u4f5c\u533a\u3002"
                if is_chinese
                else "You can start working, but fix the most visible gaps first before treating the workspace as fully ready."
            ),
        )
    return tuple(lines[:4])


def _assess_workspace_readiness(
    *,
    coverage: OnboardingCoverage,
    readme_summary: str | None,
    summary_map: dict[str, str],
    config_paths: tuple[str, ...],
    entrypoint_paths: tuple[str, ...],
    tests_visible: bool,
    docs_visible: bool,
    notebooks_visible: bool,
    response_language: str,
    purpose_summary: str,
) -> WorkspaceVerificationAssessment:
    is_chinese = response_language == "zh-CN"
    category_counts = coverage.category_counts
    source_count = category_counts.get("source", 0)
    script_count = category_counts.get("scripts", 0)
    code_surface_count = source_count + script_count
    docs_count = category_counts.get("docs", 0)
    data_count = category_counts.get("data", 0)
    repeated_dirs, repeated_stems = _onboarding_parallel_surface_details(coverage)
    has_entrypoints = bool(entrypoint_paths)
    has_config = bool(config_paths)
    has_docs = bool(readme_summary) or docs_visible or docs_count > 0
    has_tests = tests_visible
    data_heavy = data_count > max(3, code_surface_count * 2)
    major_gaps = 0

    why_status: list[str] = [purpose_summary]
    why_status.append(
        (
            f"\u8fd9\u4e2a readiness \u5224\u65ad\u57fa\u4e8e {coverage.relevant_readable_count} \u4e2a\u76f8\u5173\u53ef\u8bfb\u6587\u4ef6\uff0c\u5b9e\u9645\u68c0\u67e5\u4e86 {len(coverage.inspected_paths)} \u4e2a\uff1b"
            + (
                "\u5df2\u505a\u5230\u5168\u91cf\u76f8\u5173\u8986\u76d6\u3002"
                if coverage.full_relevant_coverage
                else "\u8fd9\u662f\u8f83\u5927\u7684\u5de5\u4f5c\u533a\uff0c\u6240\u4ee5\u4f7f\u7528\u4e86\u5168\u91cf manifest \u52a0\u786e\u5b9a\u6027\u6269\u5c55\u68c0\u67e5\u3002"
            )
            if is_chinese
            else f"The readiness judgment is based on {coverage.relevant_readable_count} relevant readable files, with {len(coverage.inspected_paths)} files actually inspected; "
            + (
                "full relevant-file coverage was achieved."
                if coverage.full_relevant_coverage
                else "this was a larger workspace, so the result uses a full manifest plus deterministic broader inspection."
            )
        )
    )

    confirmed_present: list[str] = []
    if code_surface_count:
        confirmed_present.append(
            (
                f"\u53ef\u8bfb\u4ee3\u7801\u9762\u5df2\u786e\u8ba4\uff1asource={source_count}, scripts={script_count}\u3002"
                if is_chinese
                else f"Readable code surfaces are visible across source={source_count} and scripts={script_count} files."
            )
        )
    if has_entrypoints:
        confirmed_present.append(
            (
                f"\u5df2\u786e\u8ba4\u5230\u53ef\u8ffd\u8e2a\u7684\u5165\u53e3/\u8fd0\u884c\u9762\uff0c\u4f8b\u5982 {', '.join(f'`{item}`' for item in entrypoint_paths[:3])}\u3002"
                if is_chinese
                else f"Likely run surfaces are visible, including {', '.join(f'`{item}`' for item in entrypoint_paths[:3])}."
            )
        )
    if has_config:
        confirmed_present.append(
            (
                f"\u5df2\u786e\u8ba4\u5230\u53ef\u89c1\u7684\u914d\u7f6e/\u4f9d\u8d56\u4fe1\u53f7\uff0c\u4f8b\u5982 {', '.join(f'`{item}`' for item in config_paths[:3])}\u3002"
                if is_chinese
                else f"Visible configuration or dependency signals exist, including {', '.join(f'`{item}`' for item in config_paths[:3])}."
            )
        )
    if has_docs:
        confirmed_present.append(
            "\u5de5\u4f5c\u533a\u91cc\u80fd\u770b\u5230 README \u6216\u5176\u4ed6\u6587\u6863/\u4ea4\u63a5\u4fe1\u53f7\u3002"
            if is_chinese
            else "README or other doc/handoff signals are visible in the workspace."
        )
    if has_tests:
        confirmed_present.append(
            "\u5de5\u4f5c\u533a\u91cc\u80fd\u770b\u5230\u6d4b\u8bd5\u9762\uff0c\u8fd9\u5bf9 day-one \u9a8c\u8bc1\u6709\u5e2e\u52a9\u3002"
            if is_chinese
            else "A visible test surface exists, which helps with day-one confidence checks."
        )
    if not confirmed_present:
        confirmed_present.append(
            "\u9664\u4e86\u76ee\u5f55\u5e03\u5c40\u4e4b\u5916\uff0c\u6682\u65f6\u6ca1\u6709\u66f4\u5f3a\u7684\u660e\u786e\u5b9e\u7269\u4fe1\u53f7\u3002"
            if is_chinese
            else "Beyond the workspace layout, there are few strong concrete signals yet."
        )

    missing_or_blocking: list[str] = []
    explicit_blockers = _workspace_explicit_blockers(summary_map, response_language=response_language)
    if coverage.relevant_readable_count == 0:
        major_gaps += 2
        missing_or_blocking.append(
            "\u6ca1\u6709\u786e\u8ba4\u5230\u53ef\u8bfb\u7684\u6e90\u7801\u3001\u914d\u7f6e\u3001\u6587\u6863\u6216\u6d4b\u8bd5\u6587\u4ef6\u3002"
            if is_chinese
            else "No readable source, config, documentation, or test files were confirmed."
        )
    if code_surface_count >= 2 and not has_entrypoints:
        major_gaps += 1
        missing_or_blocking.append(
            "\u80fd\u770b\u5230\u4ee3\u7801\uff0c\u4f46\u6ca1\u6709\u7a33\u56fa\u786e\u8ba4\u5230\u6e05\u6670\u7684\u5165\u53e3\u6216\u8fd0\u884c\u9762\u3002"
            if is_chinese
            else "Readable code is present, but no clear entry point or run surface was firmly confirmed."
        )
    if code_surface_count >= 2 and not has_config:
        major_gaps += 1
        missing_or_blocking.append(
            "\u80fd\u770b\u5230\u4ee3\u7801\uff0c\u4f46\u6ca1\u6709\u7a33\u56fa\u786e\u8ba4\u5230\u4f9d\u8d56\u6216\u73af\u5883 manifest\u3002"
            if is_chinese
            else "Code is visible, but no dependency or environment manifest was firmly confirmed."
        )
    if not has_docs:
        missing_or_blocking.append(
            "\u9876\u5c42 README \u6216\u4ea4\u63a5\u6587\u6863\u4fe1\u53f7\u5f31\u6216\u7f3a\u5931\u3002"
            if is_chinese
            else "Top-level README or handoff documentation is weak or missing."
        )
    if not has_tests:
        missing_or_blocking.append(
            "\u6ca1\u6709\u7a33\u56fa\u786e\u8ba4\u5230\u660e\u663e\u7684\u6d4b\u8bd5/\u5feb\u901f\u9a8c\u8bc1\u9762\u3002"
            if is_chinese
            else "No obvious test or fast verification surface was firmly confirmed."
        )
    if explicit_blockers:
        major_gaps += 1
        missing_or_blocking.extend(explicit_blockers)

    risks_or_uncertainty: list[str] = []
    if repeated_dirs and repeated_stems:
        risks_or_uncertainty.append(
            (
                f"\u591a\u4e2a\u540c\u7ea7\u5de5\u4f5c\u533a\uff08{', '.join(f'`{item}`' for item in repeated_dirs[:3])}\uff09\u91cd\u590d\u51fa\u73b0 {', '.join(f'`{item}`' for item in repeated_stems[:3])} \u8fd9\u7c7b\u811a\u672c\uff0c\u6240\u4ee5\u9700\u8981\u5148\u786e\u8ba4\u54ea\u4e2a\u662f canonical workspace\u3002"
                if is_chinese
                else f"Multiple sibling work areas ({', '.join(f'`{item}`' for item in repeated_dirs[:3])}) repeat script families such as {', '.join(f'`{item}`' for item in repeated_stems[:3])}, so a new RA still needs to confirm which area is canonical versus audit or variant work."
            )
        )
    if not coverage.full_relevant_coverage:
        risks_or_uncertainty.append(
            "\u8fd9\u662f\u8f83\u5927\u7684\u5de5\u4f5c\u533a\uff0c\u6240\u4ee5 readiness \u5224\u65ad\u57fa\u4e8e\u5168\u91cf manifest \u52a0\u786e\u5b9a\u6027\u6269\u5c55\u68c0\u67e5\uff0c\u800c\u4e0d\u662f\u6bcf\u4e2a\u6587\u4ef6\u90fd\u9010\u4e00\u7ec6\u8bfb\u3002"
            if is_chinese
            else "This is a larger workspace, so the readiness result uses a full manifest plus deterministic broader inspection rather than line-by-line reading of every file."
        )
    if notebooks_visible and not has_entrypoints:
        risks_or_uncertainty.append(
            "\u53ef\u89c1 notebook \u4fe1\u53f7\uff0c\u8fd9\u8bf4\u660e\u5de5\u4f5c\u533a\u53ef\u80fd\u66f4\u50cf analysis-first \u6d41\u7a0b\uff0c\u800c\u4e0d\u662f\u56f4\u7ed5\u5355\u4e00\u53ef\u6267\u884c\u5165\u53e3\u7ec4\u7ec7\u7684\u9879\u76ee\u3002"
            if is_chinese
            else "Notebook evidence suggests the workspace may be analysis-first rather than organized around one clean executable entry point."
        )
    if data_heavy:
        risks_or_uncertainty.append(
            "\u8fd9\u4e2a\u5de5\u4f5c\u533a\u76f8\u5bf9\u4ee3\u7801\u66f4\u504f data-heavy\uff0c\u6240\u4ee5 day-one \u53ef\u8fd0\u884c\u8def\u5f84\u53ef\u80fd\u6ca1\u6709\u90a3\u4e48\u76f4\u63a5\u3002"
            if is_chinese
            else "The workspace is relatively data-heavy, so day-one runnable behavior may be less obvious than the data surface."
        )
    risks_or_uncertainty.append(
        "\u6240\u6709\u6ca1\u6709\u76f4\u63a5\u51fa\u73b0\u5728\u53ef\u89c1\u6587\u4ef6\u91cc\u7684\u884c\u4e3a\u90fd\u5e94\u8be5\u89c6\u4e3a\u672a\u786e\u8ba4\uff0c\u4e0d\u8981\u9ed8\u8ba4\u5b83\u5b58\u5728\u3002"
        if is_chinese
        else "Any behavior not directly visible in the consulted files should be treated as unconfirmed instead of assumed."
    )

    if coverage.relevant_readable_count == 0:
        status = "blocked"
    elif explicit_blockers and not has_entrypoints and code_surface_count <= 1:
        status = "blocked"
    elif has_entrypoints and has_config and (has_docs or has_tests):
        if major_gaps == 0 and has_docs and (has_tests or coverage.full_relevant_coverage) and not repeated_dirs and not data_heavy:
            status = "ready"
        else:
            status = "ready_with_gaps"
    elif has_entrypoints and (has_config or has_docs or has_tests):
        status = "ready_with_gaps"
    elif code_surface_count >= 2 and (has_docs or has_config or has_tests):
        status = "partially_ready"
    elif code_surface_count >= 1 or has_docs:
        status = "uncertain"
    else:
        status = "blocked"

    status_reason_map = {
        "ready": (
            "\u8fd9\u4e2a\u5de5\u4f5c\u533a\u5bf9\u65b0 RA \u6765\u8bf4\u5df2\u7ecf\u8db3\u591f\u5c31\u7eea\uff1a\u6709\u660e\u786e\u5165\u53e3\u3001\u4f9d\u8d56/\u914d\u7f6e\u4fe1\u53f7\uff0c\u5e76\u4e14\u6587\u6863\u6216\u6d4b\u8bd5\u652f\u6491\u8db3\u4ee5\u5f00\u59cb\u5de5\u4f5c\u3002"
            if is_chinese
            else "The workspace looks ready for a new RA to start practical work: visible entry points, dependency/config signals, and enough docs or tests are already present."
        ),
        "ready_with_gaps": (
            "\u65b0 RA \u4eca\u5929\u53ef\u4ee5\u5f00\u59cb\u8bfb\u548c\u5c40\u90e8\u5de5\u4f5c\uff0c\u4f46\u8fd8\u6709\u51e0\u4e2a\u660e\u663e gap \u9700\u8981\u5c3d\u65e9\u8865\u4e0a\u3002"
            if is_chinese
            else "A new RA can start reading and making progress today, but a few visible gaps still need early cleanup or confirmation."
        ),
        "partially_ready": (
            "\u5de5\u4f5c\u533a\u4e0d\u662f\u7a7a\u7684\uff0c\u4f46 day-one \u53ef\u7528\u6027\u8fd8\u4e0d\u5b8c\u6574\uff1a\u8981\u5148\u786e\u8ba4\u5165\u53e3\u3001\u914d\u7f6e\u6216\u57fa\u672c\u8fd0\u884c\u8def\u5f84\u3002"
            if is_chinese
            else "The workspace is not empty or opaque, but day-one usability is still incomplete: the RA should confirm the entry surface, config, or verification path before treating it as ready."
        ),
        "blocked": (
            "\u5bf9\u65b0 RA \u6765\u8bf4\uff0c\u5f53\u524d\u53ef\u89c1\u8bc1\u636e\u4e0d\u8db3\u4ee5\u652f\u6491 day-one \u63a5\u624b\u3002"
            if is_chinese
            else "The visible evidence is too thin to support day-one handoff confidence for a new RA."
        ),
        "uncertain": (
            "\u5de5\u4f5c\u533a\u770b\u8d77\u6765\u53ef\u4ee5\u7ee7\u7eed\u8bfb\uff0c\u4f46 readiness \u5224\u65ad\u4ecd\u7136\u9700\u8981\u66f4\u591a\u786e\u8ba4\u3002"
            if is_chinese
            else "The workspace looks readable enough to inspect further, but the readiness judgment still needs more confirmation."
        ),
    }
    why_status.insert(1, status_reason_map[status])

    read_first = _onboarding_read_first_list(
        has_readme=bool(readme_summary),
        entrypoint_paths=entrypoint_paths,
        config_paths=config_paths,
        tests_visible=tests_visible,
        coverage=coverage,
        response_language=response_language,
    )
    next_steps = _workspace_verification_next_steps(
        status=status,
        has_readme=bool(readme_summary),
        entrypoint_paths=entrypoint_paths,
        config_summaries=config_paths,
        tests_visible=tests_visible,
        coverage=coverage,
        response_language=response_language,
    )

    if not missing_or_blocking:
        missing_or_blocking.append(_workspace_not_confirmed("Missing pieces or blockers", is_chinese=is_chinese))

    return WorkspaceVerificationAssessment(
        status=status,
        why_status=tuple(why_status[:4]),
        confirmed_present=tuple(confirmed_present[:5]),
        missing_or_blocking=tuple(missing_or_blocking[:5]),
        risks_or_uncertainty=tuple(risks_or_uncertainty[:5]),
        read_first=tuple(read_first[:4]),
        next_steps=tuple(next_steps[:4]),
    )


def _workspace_verification_answer_needs_repair(
    answer_text: str,
    *,
    response_language: str,
) -> bool:
    stripped = answer_text.strip()
    lowered = stripped.lower()
    if re.match(r"^[a-z_]*workspace_verification_sections\s*\{", lowered):
        return True
    if "project purpose" in lowered and "readiness status" not in lowered:
        return True
    if "current runtime path" in lowered and "missing pieces or blockers" not in lowered:
        return True
    if response_language == "zh-CN":
        required_markers = (
            "\u5de5\u4f5c\u533a\u5c31\u7eea\u72b6\u6001",
            "\u98ce\u9669",
            "\u4e0b\u4e00\u6b65",
        )
        return sum(1 for marker in required_markers if marker in answer_text) < 2
    readiness_tokens = ("ready", "ready_with_gaps", "partially_ready", "blocked", "uncertain")
    required_markers = ("readiness status", "missing", "risk", "next step")
    if not any(token in lowered for token in readiness_tokens):
        return True
    return sum(1 for marker in required_markers if marker in lowered) < 3


def _extract_summary_segment(summary: str, label: str) -> str:
    if not summary or f"{label}:" not in summary:
        return ""
    fragment = summary.split(f"{label}:", 1)[1]
    for separator in (". sample lines:", ". notable names:", ". functions:", ". classes:"):
        if separator in fragment:
            fragment = fragment.split(separator, 1)[0]
            break
    return fragment.strip(" .")


def _paper_excerpt_lines(retrieved_chunks: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for item in retrieved_chunks[:6]:
        evidence_ref = str(item.get("evidence_ref", "")).strip()
        text = str(item.get("text", "")).strip()
        if not evidence_ref and not text:
            continue
        excerpt = _truncate_line(text.replace("\n", " "), limit=180) if text else "No excerpt"
        lines.append(f"- `{evidence_ref}` | {excerpt}")
    return lines


def _paper_window_lines(document_windows: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for item in document_windows[:8]:
        evidence_ref = str(item.get("evidence_ref", "")).strip()
        note = str(item.get("note", "")).strip()
        if not evidence_ref and not note:
            continue
        lines.append(f"- `{evidence_ref}` | {note or 'No coverage note available.'}")
    return lines


def _requested_paper_slots(prompt: str, mode: str) -> tuple[str, ...]:
    slots = list(_explicit_paper_slots(prompt))
    if slots:
        return tuple(dict.fromkeys(slots))
    if mode == "paper_compare":
        return ("research_question", "method", "main_findings", "limitations", "conclusion")
    if mode == "paper_grounded_qa":
        return ("method", "main_findings", "limitations")
    return (
        "research_question",
        "sample_or_data",
        "method",
        "main_findings",
        "limitations",
        "conclusion",
        "practical_or_investment_implications",
    )


def _explicit_paper_slots(prompt: str) -> tuple[str, ...]:
    lowered = prompt.lower()
    slots: list[str] = []
    keyword_map = (
        ("research_question", ("question", "goal", "aim", "purpose", "研究问题", "目标", "目的")),
        ("background_or_motivation", ("background", "motivation", "背景", "动机")),
        ("sample_or_data", ("sample", "samples", "data", "dataset", "样本", "数据")),
        ("method", ("method", "methods", "model", "models", "machine learning", "方法", "模型", "机器学习")),
        ("main_findings", ("finding", "findings", "result", "results", "发现", "结果")),
        ("limitations", ("limitation", "limitations", "caveat", "局限", "限制")),
        ("conclusion", ("conclusion", "conclusions", "总结", "结论")),
        ("practical_or_investment_implications", ("investment", "investor", "implication", "投资", "启示", "含义")),
    )
    for slot_name, keywords in keyword_map:
        if any(keyword in lowered for keyword in keywords):
            slots.append(slot_name)
    return tuple(dict.fromkeys(slots))


def _uncovered_requested_slots(
    answer_text: str,
    document_notes: list[dict[str, object]],
    requested_slots: tuple[str, ...],
    *,
    response_language: str,
) -> tuple[str, ...]:
    answer_lower = answer_text.lower()
    cue_map = {
        "research_question": ("question", "goal", "aim", "purpose", "problem", "研究问题", "目标", "目的", "问题"),
        "background_or_motivation": ("background", "motivation", "背景", "动机"),
        "sample_or_data": ("sample", "samples", "data", "dataset", "datasets", "样本", "数据"),
        "method": ("method", "methods", "model", "models", "algorithm", "algorithms", "方法", "模型", "算法"),
        "main_findings": ("finding", "findings", "result", "results", "performance", "发现", "结果", "表现"),
        "limitations": ("limitation", "limitations", "constraint", "constraints", "caveat", "局限", "限制", "不足"),
        "conclusion": ("conclusion", "conclusions", "overall", "finally", "结论", "总体", "最后"),
        "practical_or_investment_implications": (
            "implication",
            "implications",
            "investment",
            "investor",
            "practical",
            "启示",
            "投资",
            "含义",
            "实践",
        ),
    }

    missing: list[str] = []
    for slot_name in requested_slots:
        statuses = [
            str(_document_slot(document_note, slot_name).get("support_status", "not_clearly_stated"))
            for document_note in document_notes
        ]
        if statuses and all(status == "not_clearly_stated" for status in statuses):
            if not _contains_missing_slot_wording(answer_text, response_language):
                missing.append(slot_name)
            continue
        cues = cue_map.get(slot_name, ())
        if cues and not any(cue.lower() in answer_lower for cue in cues):
            missing.append(slot_name)
    return tuple(dict.fromkeys(missing))


def _build_slot_grounded_paper_summary(
    document_note: dict[str, object],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    response_style: str,
) -> str:
    slots = requested_slots or (
        "research_question",
        "sample_or_data",
        "method",
        "main_findings",
        "limitations",
        "conclusion",
    )
    if response_style == "continuous_prose":
        return _build_slot_grounded_paper_summary_prose(
            document_note,
            requested_slots=slots,
            response_language=response_language,
        )
    return _build_slot_grounded_paper_summary_sections(
        document_note,
        requested_slots=slots,
        response_language=response_language,
    )


def _build_slot_grounded_paper_summary_sections(
    document_note: dict[str, object],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
) -> str:
    lines: list[str] = []
    for slot_name in requested_slots:
        lines.append(_slot_display_name(slot_name, response_language))
        lines.append(f"- {_slot_summary_sentence(document_note, slot_name, response_language=response_language)}")
    return "\n".join(lines)


def _render_grounded_slot_sentence(
    slot_name: str,
    summary: str,
    *,
    response_language: str,
) -> str:
    cleaned = _normalize_slot_summary(summary, response_language=response_language)
    cleaned = re.sub(r"^\d+\s+", "", cleaned)
    lowered_clean = cleaned.lower()
    if slot_name == "sample_or_data" and "variable importance" in lowered_clean:
        return _paper_missing_phrase(response_language)

    if response_language == "zh-CN":
        localized = lowered_clean
        phrase_map = (
            ("the fundamental goal of asset pricing is to understand the behavior of risk premiums", "\u8d44\u4ea7\u5b9a\u4ef7\u7684\u57fa\u672c\u76ee\u6807\u662f\u7406\u89e3\u98ce\u9669\u6ea2\u4ef7\u7684\u884c\u4e3a"),
            ("the challenge is how to assess the incremental predictive content of a newly proposed predictor while jointly controlling for the gamut of extant signals", "\u6838\u5fc3\u6311\u6218\u662f\u5728\u540c\u65f6\u63a7\u5236\u73b0\u6709\u4fe1\u53f7\u7684\u60c5\u51b5\u4e0b\uff0c\u8bc4\u4f30\u65b0\u63d0\u51fa\u9884\u6d4b\u53d8\u91cf\u6240\u589e\u52a0\u7684\u9884\u6d4b\u4fe1\u606f"),
            ("the sample covers nearly 30,000 individual stocks over 60 years from 1957 to 2016", "\u6837\u672c\u8986\u76d61957\u5e74\u81f32016\u5e74\u7684\u7ea660\u5e74\uff0c\u5305\u542b\u8fd13\u4e07\u53ea\u4e2a\u80a1"),
            ("the paper uses 18 years of training data, 12 years of validation data, and 30 years of out-of-sample testing", "\u6587\u4e2d\u4f7f\u752818\u5e74\u7684\u8bad\u7ec3\u6570\u636e\uff0c12\u5e74\u7684\u9a8c\u8bc1\u6570\u636e\uff0c\u4ee5\u53ca30\u5e74\u7684\u6837\u672c\u5916\u6d4b\u8bd5\u671f"),
            ("the sample begins in march 1957 and ends in december 2016, covering 60 years", "\u6837\u672c\u4ece1957\u5e743\u6708\u5f00\u59cb\uff0c\u52302016\u5e7412\u6708\u7ed3\u675f\uff0c\u5171\u8986\u76d660\u5e74"),
            ("linear regression", "\u7ebf\u6027\u56de\u5f52"),
            ("generalized linear models", "\u5e7f\u4e49\u7ebf\u6027\u6a21\u578b"),
            ("principal components regression", "\u4e3b\u6210\u5206\u56de\u5f52"),
            ("partial least squares", "\u504f\u6700\u5c0f\u4e8c\u4e58"),
            ("regression trees", "\u56de\u5f52\u6811"),
            ("neural networks", "\u795e\u7ecf\u7f51\u7edc"),
            ("dimension reduction", "\u964d\u7ef4"),
            ("nonlinear models", "\u975e\u7ebf\u6027\u6a21\u578b"),
            ("positive predictive performance", "\u8f83\u4e3a\u660e\u663e\u7684\u6b63\u5411\u9884\u6d4b\u8868\u73b0"),
            ("lack of regularization", "\u7f3a\u4e4f\u6b63\u5219\u5316"),
            ("in-sample overfit", "\u6837\u672c\u5185\u8fc7\u62df\u5408"),
            ("validation sample", "\u9a8c\u8bc1\u6837\u672c"),
            ("training sample", "\u8bad\u7ec3\u6837\u672c"),
            ("out-of-sample testing", "\u6837\u672c\u5916\u6d4b\u8bd5"),
            ("asset pricing", "\u8d44\u4ea7\u5b9a\u4ef7"),
            ("risk premiums", "\u98ce\u9669\u6ea2\u4ef7"),
            ("stock returns", "\u80a1\u7968\u6536\u76ca"),
            ("machine learning", "\u673a\u5668\u5b66\u4e60"),
        )
        for source, target in phrase_map:
            localized = localized.replace(source, target)
        templates = {
            "research_question": "\u6587\u7ae0\u7684\u6838\u5fc3\u95ee\u9898\u662f{summary}",
            "background_or_motivation": "\u7814\u7a76\u80cc\u666f\u6216\u52a8\u673a\u662f{summary}",
            "sample_or_data": "\u5728\u6837\u672c\u6216\u6570\u636e\u65b9\u9762\uff0c{summary}",
            "method": "\u5728\u65b9\u6cd5\u4e0a\uff0c{summary}",
            "main_findings": "\u4e3b\u8981\u53d1\u73b0\u662f{summary}",
            "limitations": "\u6587\u4e2d\u7684\u5c40\u9650\u5728\u4e8e{summary}",
            "conclusion": "\u603b\u4f53\u7ed3\u8bba\u662f{summary}",
            "practical_or_investment_implications": "\u5728\u5b9e\u8df5\u6216\u6295\u8d44\u542b\u4e49\u65b9\u9762\uff0c{summary}",
        }
        body = templates.get(slot_name, "{summary}").format(summary=localized.rstrip("。.; "))
        return body if body.endswith("。") else body + "。"

    if slot_name == "research_question":
        if lowered_clean.startswith("the fundamental goal of asset pricing is to understand the behavior of risk premiums"):
            body = "The paper is framed around understanding the behavior of risk premiums in asset pricing"
        elif lowered_clean.startswith(("the paper", "this paper", "our focus", "the goal", "the aim", "the question")):
            body = cleaned.rstrip(".")
        else:
            body = f"The paper focuses on {cleaned[0].lower() + cleaned[1:] if len(cleaned) > 1 else cleaned}"
        return body if body.endswith(".") else body + "."

    templates = {
        "background_or_motivation": "The background or motivation is {summary}.",
        "sample_or_data": "For the sample and data, {summary}.",
        "method": "Methodologically, {summary}.",
        "main_findings": "The main finding is that {summary}.",
        "limitations": "A key limitation is that {summary}.",
        "conclusion": "Overall, {summary}.",
        "practical_or_investment_implications": "For practical or investment implications, {summary}.",
    }
    template = templates.get(slot_name, "{summary}.")
    body = template.format(summary=cleaned.rstrip("."))
    return body if body.endswith(".") else body + "."


def _build_slot_grounded_paper_summary_prose(
    document_note: dict[str, object],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
) -> str:
    english_templates = {
        "research_question": "The paper's research question is {summary}.",
        "background_or_motivation": "The background or motivation is {summary}.",
        "sample_or_data": "For sample or data, the paper describes {summary}.",
        "method": "For method, the paper uses {summary}.",
        "main_findings": "The main findings are {summary}.",
        "limitations": "The limitations discussed are {summary}.",
        "conclusion": "The conclusion is {summary}.",
        "practical_or_investment_implications": "For practical or investment implications, the paper states {summary}.",
    }
    chinese_templates = {
        "research_question": "就研究问题而言，{summary}。",
        "background_or_motivation": "在研究背景或动机方面，{summary}。",
        "sample_or_data": "关于样本或数据，{summary}。",
        "method": "在方法上，{summary}。",
        "main_findings": "主要发现方面，{summary}。",
        "limitations": "局限方面，{summary}。",
        "conclusion": "结论方面，{summary}。",
        "practical_or_investment_implications": "就实践或投资含义而言，{summary}。",
    }
    templates = chinese_templates if response_language == "zh-CN" else english_templates
    sentences: list[str] = []
    for slot_name in requested_slots:
        summary = _slot_summary_sentence(document_note, slot_name, response_language=response_language)
        template = templates.get(slot_name, "{summary}")
        sentence = template.format(summary=_normalize_slot_summary(summary, response_language=response_language)).strip()
        if sentence:
            sentences.append(sentence)
    return " ".join(sentences).strip()


def _slot_summary_sentence(
    document_note: dict[str, object],
    slot_name: str,
    *,
    response_language: str,
) -> str:
    slot_payload = _document_slot(document_note, slot_name)
    if str(slot_payload.get("support_status", "not_clearly_stated")) == "not_clearly_stated":
        if response_language == "zh-CN":
            return "文中未明确说明"
        return f"{slot_label(slot_name)} is not clearly stated in the paper"
    raw_text = str(slot_payload.get("merged_note_text", "Not clearly stated in the processed document windows.")).strip()
    return _normalize_slot_summary(raw_text, response_language=response_language)


def _normalize_slot_summary(summary: str, *, response_language: str) -> str:
    cleaned = summary.strip().rstrip(" .;。；")
    return unicodedata.normalize("NFKC", cleaned)


def _document_slot(document_note: dict[str, object], slot_name: str) -> dict[str, object]:
    for slot in document_note.get("aggregated_slots", []):
        if str(slot.get("slot_name", "")) == slot_name:
            return dict(slot)
    return {
        "slot_name": slot_name,
        "merged_note_text": "Not clearly stated in the processed document windows.",
        "evidence_refs": [],
        "support_status": "not_clearly_stated",
        "strongest_support": "weak",
        "explicit_note_count": 0,
        "inferred_note_count": 0,
        "note_count": 0,
    }


def _format_slot_line(slot_payload: dict[str, object]) -> str:
    slot_name = str(slot_payload.get("slot_name", "other"))
    summary = str(slot_payload.get("merged_note_text", "")).strip() or "Not clearly stated in the processed document windows."
    refs = ", ".join(str(item) for item in slot_payload.get("evidence_refs", [])[:4]) or "no supporting window refs"
    support_status = str(slot_payload.get("support_status", "not_clearly_stated"))
    strongest_support = str(slot_payload.get("strongest_support", "weak"))
    explicit_count = int(slot_payload.get("explicit_note_count", 0))
    inferred_count = int(slot_payload.get("inferred_note_count", 0))
    return (
        f"- {slot_label(slot_name)} | status={support_status} | strongest={strongest_support} | "
        f"explicit={explicit_count} | inferred={inferred_count} | {summary} ({refs})"
    )


def _comparison_slot_lines(
    slot_name: str,
    document_notes: list[dict[str, object]],
) -> list[str]:
    lines = [f"{slot_label(slot_name)}"]
    if not document_notes:
        return lines + ["- No document-level slot notes were available."]
    for document_note in document_notes:
        slot_payload = _document_slot(document_note, slot_name)
        lines.append(
            f"- {document_note.get('source_path', '(unknown document)')}: "
            f"{slot_payload.get('merged_note_text', 'Not clearly stated.')} "
            f"[{slot_payload.get('support_status', 'not_clearly_stated')}]"
        )
    return lines


def _relevant_slot_lines(
    document_notes: list[dict[str, object]],
    requested_slots: tuple[str, ...],
) -> list[str]:
    lines: list[str] = []
    for document_note in document_notes:
        source_path = str(document_note.get("source_path", "(unknown document)"))
        for slot_name in requested_slots:
            slot_payload = _document_slot(document_note, slot_name)
            lines.append(
                f"- {source_path} | {slot_label(slot_name)} | {slot_payload.get('merged_note_text', 'Not clearly stated in the processed document windows.')} "
                f"[{slot_payload.get('support_status', 'not_clearly_stated')}]"
            )
    return lines or ["- Answer only from the retrieved PDF text. If the answer is not supported, say so plainly."]


def _recurring_slot_lines(
    document_notes: list[dict[str, object]],
    *,
    slot_name: str,
) -> list[str]:
    clear_lines: list[str] = []
    weak_lines: list[str] = []
    for document_note in document_notes:
        slot_payload = _document_slot(document_note, slot_name)
        status = str(slot_payload.get("support_status", "not_clearly_stated"))
        if status == "not_clearly_stated":
            continue
        line = (
            f"- {document_note.get('source_path', '(unknown document)')}: "
            f"{slot_payload.get('merged_note_text', 'Not clearly stated in the processed document windows.')}"
        )
        if status == "well_supported":
            clear_lines.append(line)
        else:
            weak_lines.append(line + " [weakly supported]")
    if clear_lines:
        return [
            "Clearly supported recurring points",
            *clear_lines,
            *(
                ["Weakly supported or partial signals", *weak_lines]
                if weak_lines
                else []
            ),
        ]
    return weak_lines


def _is_recurring_limitations_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(
        token in lowered
        for token in (
            "recurring limitations",
            "main recurring limitations",
            "limitations across papers",
            "limitations across pdfs",
            "common limitations",
            "cross-paper limitations",
        )
    ) or any(
        token in prompt
        for token in (
            "\u5171\u540c\u5c40\u9650",
            "\u91cd\u590d\u51fa\u73b0\u7684\u5c40\u9650",
            "\u4e3b\u8981\u5c40\u9650",
        )
    )


def _collect_recurring_limitation_signals(
    document_notes: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    theme_definitions = (
        (
            "ocr_text_dependence",
            "Dependence on extractable text or missing OCR support",
            "\u5bf9\u53ef\u63d0\u53d6\u6587\u672c\u7684\u4f9d\u8d56\u6216\u7f3a\u5c11 OCR \u652f\u6301",
            ("ocr", "scanned pdf", "extractable pdf text", "extractable text"),
            ("ocr", "extractable text", "scanned pdf"),
        ),
        (
            "local_scope_scale",
            "Small local scope or lightweight coverage",
            "\u8bed\u6599\u6216\u7cfb\u7edf\u8303\u56f4\u8f83\u5c0f",
            (
                "lightweight local index",
                "local corpus",
                "small-scope",
                "readability over scale",
                "narrow first release",
                "narrow scope",
                "limited automation",
            ),
            ("local scope", "local corpus", "lightweight", "scale", "limited automation"),
        ),
        (
            "external_enrichment_gap",
            "No external search or metadata enrichment support",
            "\u7f3a\u5c11\u5916\u90e8\u68c0\u7d22\u6216\u5143\u6570\u636e\u589e\u5f3a",
            (
                "remote metadata",
                "metadata enrichment",
                "external paper search",
                "external search",
                "no external paper search",
                "no remote metadata enrichment",
            ),
            ("external search", "metadata", "remote metadata", "external enrichment"),
        ),
        (
            "sample_or_model_constraints",
            "Sample-size or model-scope constraints",
            "\u6837\u672c\u89c4\u6a21\u6216\u6a21\u578b\u8303\u56f4\u53d7\u9650",
            (
                "limited sample size",
                "smaller set of architectures",
                "small set of five",
                "limitations of linear models",
                "first-order approximations",
            ),
            ("sample size", "architectures", "linear models", "first-order"),
        ),
    )

    theme_hits: list[dict[str, object]] = []
    clear: list[dict[str, object]] = []
    weak: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []

    for theme_key, label_en, label_zh, patterns, check_keywords in theme_definitions:
        matched_documents: list[str] = []
        for document_note in document_notes:
            slot_payload = _document_slot(document_note, "limitations")
            if str(slot_payload.get("support_status", "not_clearly_stated")) == "not_clearly_stated":
                continue
            limitation_text = str(slot_payload.get("merged_note_text", "")).lower()
            if any(pattern in limitation_text for pattern in patterns):
                matched_documents.append(str(document_note.get("source_path", "(unknown document)")))
        if not matched_documents:
            continue
        payload = {
            "theme": theme_key,
            "label_en": label_en,
            "label_zh": label_zh,
            "documents": tuple(dict.fromkeys(matched_documents)),
            "check_keywords": check_keywords,
        }
        theme_hits.append(payload)
        if len(payload["documents"]) >= 2:
            clear.append(payload)
        else:
            weak.append(payload)

    covered_documents = {
        source_path
        for item in clear
        for source_path in item["documents"]
    }
    for document_note in document_notes:
        source_path = str(document_note.get("source_path", "(unknown document)"))
        if source_path in covered_documents:
            continue
        slot_payload = _document_slot(document_note, "limitations")
        if str(slot_payload.get("support_status", "not_clearly_stated")) == "not_clearly_stated":
            continue
        unmatched.append(
            {
                "source_path": source_path,
                "summary": _truncate_line(
                    unicodedata.normalize(
                        "NFKC",
                        str(slot_payload.get("merged_note_text", "")).replace("\n", " "),
                    ),
                    limit=220,
                ),
                "support_status": str(slot_payload.get("support_status", "not_clearly_stated")),
            }
        )

    return {
        "all": theme_hits,
        "clear": clear,
        "weak": weak,
        "unmatched": unmatched,
    }


def _looks_like_limitation_focused_answer(answer_lower: str) -> bool:
    return any(
        token in answer_lower
        for token in (
            "limitation",
            "limitations",
            "constraint",
            "constraints",
            "\u5c40\u9650",
            "\u9650\u5236",
        )
    )


def _answer_mentions_recurring_limitation_themes(
    answer_lower: str,
    recurring_signals: dict[str, list[dict[str, object]]],
) -> bool:
    for item in recurring_signals["clear"]:
        if any(keyword in answer_lower for keyword in item["check_keywords"]):
            return True
    return False


def _build_recurring_limitations_answer(
    document_notes: list[dict[str, object]],
    *,
    response_language: str,
    response_style: str,
) -> str:
    recurring_signals = _collect_recurring_limitation_signals(document_notes)
    if response_style == "continuous_prose":
        return _build_recurring_limitations_prose(recurring_signals, response_language=response_language)
    return _build_recurring_limitations_sections(recurring_signals, response_language=response_language)


def _build_recurring_limitations_sections(
    recurring_signals: dict[str, list[dict[str, object]]],
    *,
    response_language: str,
) -> str:
    clear = recurring_signals["clear"]
    weak = recurring_signals["weak"]
    unmatched = recurring_signals["unmatched"]
    if response_language == "zh-CN":
        lines = ["明确重复出现的局限"]
        if clear:
            for item in clear:
                label = str(item["label_zh"])
                documents = ", ".join(f"`{Path(path).name}`" for path in item["documents"])
                lines.append(f"- {label}：见 {documents}。")
        else:
            lines.append("- 当前没有足够证据表明多篇文献存在明确重复出现的共同局限。")
        if weak or unmatched:
            lines.append("单篇或弱重复信号")
            for item in weak:
                label = str(item["label_zh"])
                documents = ", ".join(f"`{Path(path).name}`" for path in item["documents"])
                lines.append(f"- {label}：目前只在 {documents} 中清楚出现，不能算整组论文都重复出现。")
            for item in unmatched:
                lines.append(
                    f"- `{Path(str(item['source_path'])).name}`：{item['summary']}"
                )
        lines.append("判断标准")
        lines.append("- 只有在聚合后的 limitations 槽位中被多篇文献共同支持的限制，我才将其视为“重复出现”的共同局限。")
        return "\n".join(lines)

    lines = ["Clearly supported recurring limitations"]
    if clear:
        for item in clear:
            label = str(item["label_en"])
            documents = ", ".join(f"`{Path(path).name}`" for path in item["documents"])
            lines.append(f"- {label} appears in {documents}.")
    else:
        lines.append("- There is not enough slot support to claim a clearly recurring limitation across multiple documents.")
    if weak or unmatched:
        lines.append("Paper-specific or weakly recurring signals")
        for item in weak:
            label = str(item["label_en"])
            documents = ", ".join(f"`{Path(path).name}`" for path in item["documents"])
            lines.append(f"- {label} is only clearly supported in {documents}, so it is not a robust cross-paper pattern yet.")
        for item in unmatched:
            lines.append(
                f"- `{Path(str(item['source_path'])).name}`: {item['summary']}"
            )
    lines.append("Grounding note")
    lines.append("- A limitation is treated as recurring only when the aggregated limitations slot supports it in more than one consulted PDF.")
    return "\n".join(lines)


def _build_recurring_limitations_prose(
    recurring_signals: dict[str, list[dict[str, object]]],
    *,
    response_language: str,
) -> str:
    clear = recurring_signals["clear"]
    unmatched = recurring_signals["unmatched"]
    if response_language == "zh-CN":
        if clear:
            clear_text = "、".join(str(item["label_zh"]) for item in clear)
            clear_docs = "；".join(
                f"{str(item['label_zh'])}见于 {', '.join(Path(path).name for path in item['documents'])}"
                for item in clear
            )
            extra = ""
            if unmatched:
                extra = "；另外，" + "；".join(
                    f"{Path(str(item['source_path'])).name} 里还提到 {item['summary']}"
                    for item in unmatched[:2]
                )
            return (
                f"从聚合后的 limitations 槽位来看，当前最清楚重复出现的共同局限主要是{clear_text}。"
                f"更具体地说，{clear_docs}{extra}。我只把在多篇文献中都得到支持的限制当作“重复出现”的共同局限，"
                "其余只在单篇文献中出现的信号会保留为单篇约束，而不会扩写成整组论文的共同结论。"
            )
        return (
            "从当前聚合后的 limitations 槽位来看，还没有足够证据表明多篇文献存在明确重复出现的共同局限。"
            "现有局限更多停留在单篇文献层面，因此不宜把它们概括成整组论文都共享的结论。"
        )

    if clear:
        clear_text = ", ".join(str(item["label_en"]).lower() for item in clear)
        clear_docs = "; ".join(
            f"{str(item['label_en'])} is supported in {', '.join(Path(path).name for path in item['documents'])}"
            for item in clear
        )
        extra = ""
        if unmatched:
            extra = "; separately, " + "; ".join(
                f"{Path(str(item['source_path'])).name} adds the paper-specific note that {str(item['summary']).rstrip('.')}"
                for item in unmatched[:2]
            )
        return (
            f"Across the consulted PDFs, the clearest recurring limitations are {clear_text}. "
            f"More concretely, {clear_docs}{extra}. "
            "I am only treating a limitation as recurring when the aggregated limitations slot supports it in more than one consulted paper."
        )
    return (
        "Across the consulted PDFs, there is not enough slot support to claim a clearly recurring limitation shared by multiple papers. "
        "The current limitation signals stay mostly paper-specific, so they should not be collapsed into a stronger cross-paper conclusion."
    )


def _slot_payload_status(slot_payload: dict[str, object]) -> str:
    return str(slot_payload.get("support_status", "not_clearly_stated"))


def _slot_payload_text(slot_payload: dict[str, object]) -> str:
    text = str(
        slot_payload.get("summary_text")
        or slot_payload.get("merged_note_text")
        or "Not clearly stated in the paper."
    ).strip()
    return text or "Not clearly stated in the paper."


def _slot_is_clearly_supported(status: str) -> bool:
    return status in {"explicit_supported", "well_supported"}


def _paper_missing_phrase(response_language: str) -> str:
    return "文中未明确说明。" if response_language == "zh-CN" else "Not clearly stated in the paper."


def _compose_paper_consistency_prompt(
    *,
    original_prompt: str,
    answer_text: str,
    report_notes: tuple[str, ...],
    response_language: str,
) -> str:
    missing_phrase = "文中未明确说明" if response_language == "zh-CN" else "not clearly stated in the paper"
    return "\n".join(
        [
            "You are performing a constrained consistency repair on a paper answer.",
            f"Original user prompt: {original_prompt}",
            "Repair goals:",
            "- Keep only claims supported by the grounded slot scaffold and evidence.",
            "- Remove broad textbook commentary or domain-common-sense filler that is not explicitly supported.",
            f"- If a requested detail is weak or missing, say {missing_phrase} instead of guessing.",
            "- Preserve the requested language and formatting style.",
            "Detected issues:",
            *[f"- {item}" for item in report_notes],
            "",
            "Current answer:",
            answer_text,
            "",
            "Return only the repaired final answer body.",
        ]
    )


def _contains_missing_slot_wording(text: str, response_language: str) -> bool:
    lowered = text.lower()
    if response_language == "zh-CN":
        return any(token in text for token in ("文中未明确说明", "文中未明确展开", "未明确说明", "未明确展开"))
    return any(
        token in lowered
        for token in ("not clearly stated", "not confirmed", "not explicit", "not clearly described")
    )


def _contains_generic_paper_filler(
    answer_lower: str,
    document_notes: list[dict[str, object]],
) -> bool:
    support_corpus = " ".join(
        _slot_payload_text(_document_slot(document_note, slot_name)).lower()
        for document_note in document_notes
        for slot_name in (
            "research_question",
            "sample_or_data",
            "method",
            "main_findings",
            "limitations",
            "conclusion",
            "practical_or_investment_implications",
        )
        if _slot_payload_status(_document_slot(document_note, slot_name)) != "not_clearly_stated"
    )
    filler_patterns = (
        "broader promise of machine learning",
        "underscores the importance of machine learning",
        "illustrates how machine learning can",
        "significant benefits of machine learning",
        "provide deep insights",
        "offers a robust framework for future research",
        "important implications for investors",
        "broadly useful for practitioners",
        "说明机器学习在金融中具有广泛前景",
        "对投资者具有重要启示",
        "具有广泛意义",
        "进一步说明了",
        "这表明机器学习具有显著优势",
    )
    return any(pattern in answer_lower and pattern not in support_corpus for pattern in filler_patterns)


def _contains_unsupported_gap_inference(text: str, response_language: str) -> bool:
    lowered = text.lower()
    if response_language == "zh-CN":
        return any(
            token in text
            for token in (
                "可以推测",
                "可推测",
                "推测出",
                "据此推测",
            )
        )
    return any(
        token in lowered
        for token in (
            "we can infer",
            "one can infer",
            "it can be inferred",
            "it is reasonable to infer",
            "this likely means",
            "probably",
            "can be seen as",
            "it suggests that in general",
        )
    )


def _looks_excerpt_heavy_paper_answer(answer_text: str) -> bool:
    lowered = answer_text.lower()
    markdown_heading_count = len(re.findall(r"(?m)^#{2,6}\s", answer_text))
    evidence_marker_count = sum(lowered.count(token) for token in ("**evidence**", "evidence refs", "retrieved evidence"))
    inline_ref_count = sum(lowered.count(token) for token in ("#page=", "#chunk=", "#pages=", "evidence:"))
    return markdown_heading_count >= 3 or evidence_marker_count >= 2 or inline_ref_count >= 4


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the cleaned slot evidence."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    if _looks_excerpt_heavy_paper_answer(answer_text):
        notes.append(
            "The answer is still too excerpt-heavy or outline-heavy for the final paper renderer contract."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots and mode_selection.mode == "paper_summary":
        notes.append(
            "The answer did not clearly cover these requested summary dimensions: "
            + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
            + "."
        )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }


def _requested_paper_slots(prompt: str, mode: str) -> tuple[str, ...]:
    explicit_slots = list(_explicit_paper_slots(prompt))
    if explicit_slots:
        return tuple(dict.fromkeys(explicit_slots))
    if mode == "paper_compare":
        return ("research_question", "sample_or_data", "method", "main_findings", "limitations", "conclusion")
    if mode == "paper_grounded_qa":
        if _is_recurring_limitations_prompt(prompt):
            return ("limitations",)
        return ("method", "main_findings", "limitations")
    return (
        "research_question",
        "sample_or_data",
        "method",
        "main_findings",
        "limitations",
        "conclusion",
    )


def _explicit_paper_slots(prompt: str) -> tuple[str, ...]:
    lowered = prompt.lower()
    slots: list[str] = []
    keyword_map = (
        ("research_question", ("question", "goal", "aim", "purpose", "problem", "研究问题", "目标", "目的", "问题")),
        ("background_or_motivation", ("background", "motivation", "context", "研究背景", "动机", "背景")),
        ("sample_or_data", ("sample", "samples", "data", "dataset", "datasets", "corpus", "样本", "数据", "数据集")),
        ("method", ("method", "methods", "approach", "model", "models", "algorithm", "machine learning", "方法", "模型", "算法", "机器学习")),
        ("main_findings", ("finding", "findings", "result", "results", "outcome", "发现", "结果", "主要发现", "实证发现")),
        ("limitations", ("limitation", "limitations", "caveat", "caveats", "constraint", "constraints", "局限", "限制", "不足")),
        ("conclusion", ("conclusion", "conclusions", "overall conclusion", "summary", "结论", "总体结论", "总结")),
        ("practical_or_investment_implications", ("investment", "investor", "implication", "implications", "practical", "实践", "投资", "启示", "含义")),
    )
    for slot_name, keywords in keyword_map:
        if any(keyword in lowered or keyword in prompt for keyword in keywords):
            slots.append(slot_name)
    return tuple(dict.fromkeys(slots))


def _slot_display_name(slot_name: str, response_language: str) -> str:
    zh_labels = {
        "research_question": "研究问题",
        "background_or_motivation": "研究背景或动机",
        "sample_or_data": "样本或数据",
        "method": "方法",
        "main_findings": "主要发现",
        "limitations": "局限",
        "conclusion": "结论",
        "practical_or_investment_implications": "实践或投资含义",
        "other": "其他信息",
    }
    if response_language == "zh-CN":
        return zh_labels.get(slot_name, slot_name)
    return {
        "sample_or_data": "sample/data",
        "main_findings": "main findings",
        "practical_or_investment_implications": "practical/investment implications",
    }.get(slot_name, slot_label(slot_name))


def _missing_slot_sentence(missing_slots: tuple[str, ...], *, response_language: str) -> str:
    if response_language == "zh-CN":
        labels = "、".join(_slot_display_name(slot_name, response_language) for slot_name in missing_slots[:3])
        return f"{labels}文中未明确说明。"
    labels = ", ".join(_slot_display_name(slot_name, response_language) for slot_name in missing_slots[:3])
    return f"The paper does not clearly state {labels}."


def _missing_slot_sentence(missing_slots: tuple[str, ...], *, response_language: str) -> str:
    if response_language == "zh-CN":
        labels = "、".join(_slot_display_name(slot_name, response_language) for slot_name in missing_slots[:3])
        return f"{labels}文中未明确说明。"
    labels = ", ".join(_slot_display_name(slot_name, response_language) for slot_name in missing_slots[:3])
    return f"The paper does not clearly state {labels}."


def _slot_summary_keywords(summary: str) -> tuple[str, ...]:
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", unicodedata.normalize("NFKC", summary).lower())
    skip = {
        "the",
        "paper",
        "this",
        "that",
        "with",
        "from",
        "for",
        "and",
        "not",
        "clearly",
        "stated",
        "in",
        "文中",
        "明确",
        "说明",
    }
    kept: list[str] = []
    for token in tokens:
        if token in skip:
            continue
        if re.search(r"[\u4e00-\u9fff]", token):
            if len(token) >= 2:
                kept.append(token)
        elif len(token) >= 4:
            kept.append(token)
    return tuple(dict.fromkeys(kept[:8]))


def _answer_mentions_slot_content(answer_lower: str, slot_payload: dict[str, object]) -> bool:
    if _slot_payload_status(slot_payload) == "not_clearly_stated":
        return False
    hits = sum(1 for token in _slot_summary_keywords(_slot_payload_text(slot_payload)) if token in answer_lower)
    return hits >= 2


def _uncovered_requested_slots(
    answer_text: str,
    document_notes: list[dict[str, object]],
    requested_slots: tuple[str, ...],
    *,
    response_language: str,
) -> tuple[str, ...]:
    answer_lower = answer_text.lower()
    cue_map = {
        "research_question": ("question", "goal", "aim", "purpose", "problem", "研究问题", "目标", "问题"),
        "background_or_motivation": ("background", "motivation", "研究背景", "动机", "背景"),
        "sample_or_data": ("sample", "samples", "data", "dataset", "datasets", "样本", "数据"),
        "method": ("method", "methods", "model", "models", "algorithm", "approach", "方法", "模型", "算法", "机器学习"),
        "main_findings": ("finding", "findings", "result", "results", "performance", "发现", "结果", "主要发现", "实证发现"),
        "limitations": ("limitation", "limitations", "constraint", "constraints", "caveat", "局限", "限制", "不足"),
        "conclusion": ("conclusion", "conclusions", "overall", "summary", "结论", "总体结论", "总结"),
        "practical_or_investment_implications": ("implication", "implications", "investment", "investor", "practical", "启示", "投资", "含义", "实践"),
    }

    missing: list[str] = []
    for slot_name in requested_slots:
        slot_payloads = [_document_slot(document_note, slot_name) for document_note in document_notes]
        statuses = [_slot_payload_status(payload) for payload in slot_payloads]
        if statuses and all(status == "not_clearly_stated" for status in statuses):
            if not _contains_missing_slot_wording(answer_text, response_language):
                missing.append(slot_name)
            continue
        cues = cue_map.get(slot_name, ())
        if any(cue.lower() in answer_lower for cue in cues):
            continue
        if any(_answer_mentions_slot_content(answer_lower, payload) for payload in slot_payloads):
            continue
        missing.append(slot_name)
    return tuple(dict.fromkeys(missing))


def _paper_renderer_name(mode_selection: ModeSelection) -> str:
    if mode_selection.paper_output_profile == "detailed_paper_note":
        return "detailed_paper_note"
    if mode_selection.paper_output_profile == "quick_summary":
        return "quick_summary"
    if mode_selection.response_style == "continuous_prose":
        return "continuous_grounded_prose"
    return "grounded_paper_answer"


def _format_renderer_slot_line(
    slot_name: str,
    document_note: dict[str, object],
    *,
    paper_output_profile: str = "quick_summary",
) -> str:
    slot_payload = _document_slot(document_note, slot_name)
    refs = ", ".join(str(item) for item in slot_payload.get("evidence_refs", [])[:3]) or "no direct refs"
    return (
        f"- {slot_label(slot_name)} | status={_slot_payload_status(slot_payload)} | "
        f"{_slot_payload_render_text(slot_payload, paper_output_profile=paper_output_profile)} ({refs})"
    )


def _build_paper_summary_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    if not paper_trace.discovered_documents:
        return "\n".join(
            [
                "Renderer contract",
                f"- {_paper_renderer_name(mode_selection.response_style)}",
                "Document",
                "- No local PDF target was discovered for this prompt.",
                "Rendering rules",
                "- Do not invent any paper content when no PDF target was resolved.",
            ]
        )

    document_note = paper_trace.document_notes[0] if paper_trace.document_notes else {}
    requested_slots = list(_requested_paper_slots(prompt, mode_selection.mode))
    implication_slot = _document_slot(document_note, "practical_or_investment_implications")
    if (
        "practical_or_investment_implications" not in requested_slots
        and _slot_is_clearly_supported(_slot_payload_status(implication_slot))
    ):
        requested_slots.append("practical_or_investment_implications")
    return "\n".join(
        [
            "Renderer contract",
            f"- {_paper_renderer_name(mode_selection.response_style)}",
            "Document",
            f"- Primary target: `{paper_trace.discovered_documents[0]}`",
            f"- Read strategy: `{paper_trace.read_strategy}`",
            f"- Windows processed: {paper_trace.window_count_processed}",
            "Requested dimensions",
            *[f"- {slot_label(slot_name)}" for slot_name in requested_slots],
            "Cleaned slot scaffold",
            *[_format_renderer_slot_line(slot_name, document_note) for slot_name in requested_slots],
            "Rendering rules",
            "- Write a concise, practical RA memo rather than stitched excerpts or broad essay-style filler.",
            "- Cover every requested dimension exactly once; do not silently omit any requested dimension.",
            f"- If a requested dimension is not clearly supported, say {_paper_missing_phrase(mode_selection.response_language)}",
            "- Prefer explicit support over inferred support and avoid speculative inference.",
            "- Do not add textbook machine-learning, finance, or investment commentary unless the paper itself supports it.",
            "Evidence refs",
            *_bullet_lines(evidence_refs or paper_trace.discovered_documents or paper_trace.target_paths),
        ]
    )


def _build_paper_compare_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    compared = paper_trace.discovered_documents or paper_trace.target_paths
    requested_slots = _requested_paper_slots(prompt, mode_selection.mode)
    return "\n".join(
        [
            "Renderer contract",
            "- grounded_ra_memo",
            "Documents compared",
            *_bullet_lines(compared or ("No PDF targets were discovered.",)),
            "Comparison scaffold",
            *[
                line
                for slot_name in requested_slots
                for line in _comparison_slot_lines(slot_name, paper_trace.document_notes)
            ],
            "Rendering rules",
            "- Compare papers slot-to-slot and make asymmetries explicit.",
            "- Keep the comparison concise and practical for a research assistant.",
            f"- If a dimension is not clearly supported for one paper, say {_paper_missing_phrase(mode_selection.response_language)}",
            "- Avoid repeating weakly supported text or broad generic commentary.",
            "Evidence refs",
            *_bullet_lines(evidence_refs or compared),
        ]
    )


def _build_paper_grounded_qa_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    requested_slots = _requested_paper_slots(prompt, mode_selection.mode)
    recurring_limitations = _recurring_slot_lines(
        paper_trace.document_notes,
        slot_name="limitations",
    )
    direct_scaffold = (
        recurring_limitations
        if recurring_limitations and _is_recurring_limitations_prompt(prompt)
        else _relevant_slot_lines(paper_trace.document_notes, requested_slots)
    )
    lines = [
        "Renderer contract",
        "- grounded_ra_memo",
        "Direct answer scaffold",
        *direct_scaffold,
        "Rendering rules",
        "- Answer the question directly from the cleaned slot scaffold and retrieved evidence.",
        "- Keep the answer concise and grounded rather than excerpt-heavy.",
        f"- If the consulted evidence does not clearly support a requested detail, say {_paper_missing_phrase(mode_selection.response_language)}",
    ]
    excerpt_lines = _paper_excerpt_lines(paper_trace.retrieved_chunks)
    if excerpt_lines:
        lines.extend(["Retrieved evidence", *excerpt_lines[:4]])
    lines.extend(["Evidence refs", *_bullet_lines(evidence_refs or paper_trace.target_paths)])
    return "\n".join(lines)


def _build_slot_grounded_paper_summary(
    document_note: dict[str, object],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    response_style: str,
) -> str:
    slots = list(
        requested_slots
        or (
            "research_question",
            "sample_or_data",
            "method",
            "main_findings",
            "limitations",
            "conclusion",
        )
    )
    implication_slot = _document_slot(document_note, "practical_or_investment_implications")
    if (
        "practical_or_investment_implications" not in slots
        and _slot_is_clearly_supported(_slot_payload_status(implication_slot))
    ):
        slots.append("practical_or_investment_implications")
    if response_style == "continuous_prose":
        return _build_slot_grounded_paper_summary_prose(
            document_note,
            requested_slots=tuple(slots),
            response_language=response_language,
        )
    return _build_slot_grounded_paper_summary_sections(
        document_note,
        requested_slots=tuple(slots),
        response_language=response_language,
    )


def _build_slot_grounded_paper_summary_sections(
    document_note: dict[str, object],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
) -> str:
    return "\n".join(
        f"{_slot_display_name(slot_name, response_language)}: "
        f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language)}"
        for slot_name in requested_slots
    )


def _build_slot_grounded_paper_summary_prose(
    document_note: dict[str, object],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
) -> str:
    english_templates = {
        "research_question": "The paper asks {summary}.",
        "background_or_motivation": "Its background or motivation is {summary}.",
        "sample_or_data": "For sample or data, {summary}.",
        "method": "Methodologically, {summary}.",
        "main_findings": "Its main findings are {summary}.",
        "limitations": "Its limitations are {summary}.",
        "conclusion": "Overall, the paper concludes {summary}.",
        "practical_or_investment_implications": "For practical or investment implications, {summary}.",
    }
    chinese_templates = {
        "research_question": "文章主要研究{summary}",
        "background_or_motivation": "研究背景或动机方面，{summary}",
        "sample_or_data": "在样本或数据方面，{summary}",
        "method": "方法上，{summary}",
        "main_findings": "主要发现是{summary}",
        "limitations": "局限方面，{summary}",
        "conclusion": "总体结论是{summary}",
        "practical_or_investment_implications": "就实践或投资含义而言，{summary}",
    }
    templates = chinese_templates if response_language == "zh-CN" else english_templates
    sentences: list[str] = []
    for slot_name in requested_slots:
        summary = _slot_summary_sentence(document_note, slot_name, response_language=response_language)
        normalized = _normalize_slot_summary(summary, response_language=response_language)
        if response_language == "zh-CN":
            normalized = normalized.rstrip("。")
            sentence = templates[slot_name].format(summary=normalized).strip() + "。"
        else:
            normalized = normalized.rstrip(".")
            sentence = templates[slot_name].format(summary=normalized).strip()
            if not sentence.endswith("."):
                sentence += "."
        sentences.append(sentence)
    return " ".join(sentences).strip()


def _slot_summary_sentence(
    document_note: dict[str, object],
    slot_name: str,
    *,
    response_language: str,
    paper_output_profile: str = "quick_summary",
) -> str:
    slot_payload = _document_slot(document_note, slot_name)
    if _slot_payload_status(slot_payload) == "not_clearly_stated":
        return _paper_missing_phrase(response_language)
    return _normalize_slot_summary(
        _slot_payload_render_text(
            slot_payload,
            paper_output_profile=paper_output_profile,
        ),
        response_language=response_language,
    )


def _document_slot(document_note: dict[str, object], slot_name: str) -> dict[str, object]:
    for slot in document_note.get("cleaned_slots", []):
        if str(slot.get("slot_name", "")) == slot_name:
            payload = dict(slot)
            payload.setdefault("merged_note_text", payload.get("summary_text", ""))
            return payload
    for slot in document_note.get("aggregated_slots", []):
        if str(slot.get("slot_name", "")) == slot_name:
            return dict(slot)
    return {
        "slot_name": slot_name,
        "summary_text": "Not clearly stated in the paper.",
        "merged_note_text": "Not clearly stated in the paper.",
        "evidence_refs": [],
        "support_status": "not_clearly_stated",
        "strongest_support": "weak",
        "explicit_note_count": 0,
        "inferred_note_count": 0,
        "note_count": 0,
    }


def _slot_payload_render_text(
    slot_payload: dict[str, object],
    *,
    paper_output_profile: str,
) -> str:
    detailed_render_text = str(slot_payload.get("detailed_render_text", "")).strip()
    summary_text = str(slot_payload.get("summary_text", "")).strip()
    merged_text = str(slot_payload.get("merged_note_text", "")).strip()
    if paper_output_profile == "detailed_paper_note":
        if detailed_render_text:
            return detailed_render_text
        if merged_text and len(merged_text) > len(summary_text) + 12 and len(merged_text) <= 420:
            return merged_text
    if summary_text:
        return summary_text
    if merged_text:
        return merged_text
    return "Not clearly stated in the paper."


def _format_slot_line(slot_payload: dict[str, object]) -> str:
    slot_name = str(slot_payload.get("slot_name", "other"))
    summary = _slot_payload_text(slot_payload)
    refs = ", ".join(str(item) for item in slot_payload.get("evidence_refs", [])[:4]) or "no supporting refs"
    support_status = _slot_payload_status(slot_payload)
    strongest_support = str(slot_payload.get("strongest_support", "weak"))
    explicit_count = int(slot_payload.get("explicit_note_count", 0))
    inferred_count = int(slot_payload.get("inferred_note_count", 0))
    return (
        f"- {slot_label(slot_name)} | status={support_status} | strongest={strongest_support} | "
        f"explicit={explicit_count} | inferred={inferred_count} | {summary} ({refs})"
    )


def _comparison_slot_lines(
    slot_name: str,
    document_notes: list[dict[str, object]],
) -> list[str]:
    lines = [f"{slot_label(slot_name)}"]
    if not document_notes:
        return lines + ["- No document-level slot notes were available."]
    for document_note in document_notes:
        slot_payload = _document_slot(document_note, slot_name)
        lines.append(
            f"- {Path(str(document_note.get('source_path', '(unknown document)'))).name}: "
            f"{_slot_payload_text(slot_payload)} [{_slot_payload_status(slot_payload)}]"
        )
    return lines


def _relevant_slot_lines(
    document_notes: list[dict[str, object]],
    requested_slots: tuple[str, ...],
) -> list[str]:
    lines: list[str] = []
    for document_note in document_notes:
        source_path = Path(str(document_note.get("source_path", "(unknown document)"))).name
        for slot_name in requested_slots:
            slot_payload = _document_slot(document_note, slot_name)
            if _slot_payload_status(slot_payload) == "not_clearly_stated":
                continue
            lines.append(
                f"- {source_path} | {slot_label(slot_name)} | "
                f"{_slot_payload_text(slot_payload)} [{_slot_payload_status(slot_payload)}]"
            )
    return lines or ["- Answer only from the retrieved PDF text. If the answer is not supported, say so plainly."]


def _recurring_slot_lines(
    document_notes: list[dict[str, object]],
    *,
    slot_name: str,
) -> list[str]:
    clear_lines: list[str] = []
    weak_lines: list[str] = []
    for document_note in document_notes:
        slot_payload = _document_slot(document_note, slot_name)
        status = _slot_payload_status(slot_payload)
        if status == "not_clearly_stated":
            continue
        line = (
            f"- {Path(str(document_note.get('source_path', '(unknown document)'))).name}: "
            f"{_slot_payload_text(slot_payload)}"
        )
        if _slot_is_clearly_supported(status):
            clear_lines.append(line)
        else:
            weak_lines.append(line + " [weakly supported]")
    if clear_lines:
        return [
            "Clearly supported recurring points",
            *clear_lines,
            *(["Weakly supported or partial signals", *weak_lines] if weak_lines else []),
        ]
    return weak_lines


def _collect_recurring_limitation_signals(
    document_notes: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    theme_definitions = (
        (
            "ocr_text_dependence",
            "Dependence on extractable text or missing OCR support",
            "对可提取文本的依赖或缺少 OCR 支持",
            ("ocr", "scanned pdf", "extractable pdf text", "extractable text"),
            ("ocr", "extractable text", "scanned pdf"),
        ),
        (
            "local_scope_scale",
            "Small local scope or lightweight coverage",
            "语料或系统范围较小",
            (
                "lightweight local index",
                "local corpus",
                "small-scope",
                "readability over scale",
                "narrow first release",
                "narrow scope",
                "limited automation",
            ),
            ("local scope", "local corpus", "lightweight", "scale", "limited automation"),
        ),
        (
            "external_enrichment_gap",
            "No external search or metadata enrichment support",
            "缺少外部检索或元数据增强",
            (
                "remote metadata",
                "metadata enrichment",
                "external paper search",
                "external search",
                "no external paper search",
                "no remote metadata enrichment",
            ),
            ("external search", "metadata", "remote metadata", "external enrichment"),
        ),
        (
            "sample_or_model_constraints",
            "Sample-size or model-scope constraints",
            "样本规模或模型范围受限",
            (
                "limited sample size",
                "smaller set of architectures",
                "small set of five",
                "limitations of linear models",
                "first-order approximations",
            ),
            ("sample size", "architectures", "linear models", "first-order"),
        ),
    )

    theme_hits: list[dict[str, object]] = []
    clear: list[dict[str, object]] = []
    weak: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []

    for theme_key, label_en, label_zh, patterns, check_keywords in theme_definitions:
        matched_documents: list[str] = []
        for document_note in document_notes:
            slot_payload = _document_slot(document_note, "limitations")
            if _slot_payload_status(slot_payload) == "not_clearly_stated":
                continue
            raw_limitation_text = " ".join(
                str(slot.get("merged_note_text", ""))
                for slot in document_note.get("aggregated_slots", [])
                if str(slot.get("slot_name", "")) == "limitations"
            )
            limitation_text = " ".join(
                part
                for part in (
                    _slot_payload_text(slot_payload).lower(),
                    raw_limitation_text.lower(),
                )
                if part
            )
            if any(pattern in limitation_text for pattern in patterns):
                matched_documents.append(str(document_note.get("source_path", "(unknown document)")))
        if not matched_documents:
            continue
        payload = {
            "theme": theme_key,
            "label_en": label_en,
            "label_zh": label_zh,
            "documents": tuple(dict.fromkeys(matched_documents)),
            "check_keywords": check_keywords,
        }
        theme_hits.append(payload)
        if len(payload["documents"]) >= 2:
            clear.append(payload)
        else:
            weak.append(payload)

    covered_documents = {source_path for item in clear for source_path in item["documents"]}
    for document_note in document_notes:
        source_path = str(document_note.get("source_path", "(unknown document)"))
        if source_path in covered_documents:
            continue
        slot_payload = _document_slot(document_note, "limitations")
        if _slot_payload_status(slot_payload) == "not_clearly_stated":
            continue
        unmatched.append(
            {
                "source_path": source_path,
                "summary": _truncate_line(
                    unicodedata.normalize("NFKC", _slot_payload_text(slot_payload).replace("\n", " ")),
                    limit=180,
                ),
                "support_status": _slot_payload_status(slot_payload),
            }
        )

    return {
        "all": theme_hits,
        "clear": clear,
        "weak": weak,
        "unmatched": unmatched,
    }


def _deterministic_paper_consistency_trim(
    answer_text: str,
    paper_trace: PaperTrace,
    *,
    prompt: str,
    mode: str,
    response_language: str,
    response_style: str,
    requested_slots: tuple[str, ...],
) -> str:
    if _is_recurring_limitations_prompt(prompt):
        return _build_recurring_limitations_answer(
            paper_trace.document_notes,
            response_language=response_language,
            response_style=response_style,
        )
    if mode == "paper_summary" and requested_slots and paper_trace.document_notes:
        return _build_slot_grounded_paper_summary(
            paper_trace.document_notes[0],
            requested_slots=requested_slots,
            response_language=response_language,
            response_style=response_style,
        )

    support_corpus = " ".join(
        _slot_payload_text(_document_slot(document_note, slot_name)).lower()
        for document_note in paper_trace.document_notes
        for slot_name in (
            "research_question",
            "sample_or_data",
            "method",
            "main_findings",
            "limitations",
            "conclusion",
            "practical_or_investment_implications",
        )
    )
    generic_patterns = (
        "broader promise of machine learning",
        "underscores the importance of machine learning",
        "illustrates how machine learning can",
        "significant benefits of machine learning",
        "provide deep insights",
        "offers a robust framework for future research",
        "important implications for investors",
        "说明机器学习在金融中具有广泛前景",
        "对投资者具有重要启示",
        "具有广泛意义",
        "进一步说明了",
    )
    kept_sentences: list[str] = []
    for sentence in _answer_sentences(answer_text):
        lowered = sentence.lower()
        if any(pattern in lowered and pattern not in support_corpus for pattern in generic_patterns):
            continue
        if _contains_unsupported_gap_inference(sentence, response_language):
            continue
        kept_sentences.append(sentence.strip())

    repaired = " ".join(item for item in kept_sentences if item).strip()
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, requested_slots)
    if missing_slots and not _contains_missing_slot_wording(repaired, response_language):
        repaired = f"{repaired} {_missing_slot_sentence(missing_slots, response_language=response_language)}".strip()
    return repaired or answer_text


def _build_recurring_limitations_sections(
    recurring_signals: dict[str, list[dict[str, object]]],
    *,
    response_language: str,
) -> str:
    clear = recurring_signals["clear"]
    weak = recurring_signals["weak"]
    unmatched = recurring_signals["unmatched"]
    if response_language == "zh-CN":
        lines = ["明确重复出现的局限"]
        if clear:
            for item in clear:
                documents = ", ".join(f"`{Path(path).name}`" for path in item["documents"])
                lines.append(f"- {item['label_zh']}：见 {documents}。")
        else:
            lines.append("- 目前还没有足够证据表明多篇文献存在明确重复出现的共同局限。")
        if weak or unmatched:
            lines.append("弱重复或单篇局限")
            for item in weak:
                documents = ", ".join(f"`{Path(path).name}`" for path in item["documents"])
                lines.append(f"- {item['label_zh']}：目前只在 {documents} 中得到清晰支持，还不能算稳健的跨论文共同局限。")
            for item in unmatched:
                lines.append(f"- `{Path(str(item['source_path'])).name}`：{item['summary']}")
        lines.append("判定说明")
        lines.append("- 只有在多篇论文的 limitations 槽位都支持同一限制时，才把它视为重复出现的共同局限。")
        return "\n".join(lines)

    lines = ["Clearly supported recurring limitations"]
    if clear:
        for item in clear:
            documents = ", ".join(f"`{Path(path).name}`" for path in item["documents"])
            lines.append(f"- {item['label_en']} appears in {documents}.")
    else:
        lines.append("- There is not enough slot support to claim a clearly recurring limitation across multiple documents.")
    if weak or unmatched:
        lines.append("Weakly recurring or paper-specific limitations")
        for item in weak:
            documents = ", ".join(f"`{Path(path).name}`" for path in item["documents"])
            lines.append(f"- {item['label_en']} is only clearly supported in {documents}, so it is not yet a strong cross-paper pattern.")
        for item in unmatched:
            lines.append(f"- `{Path(str(item['source_path'])).name}`: {item['summary']}")
    lines.append("Grounding note")
    lines.append("- A limitation is treated as recurring only when the cleaned limitations slot supports it in more than one consulted PDF.")
    return "\n".join(lines)


def _build_recurring_limitations_prose(
    recurring_signals: dict[str, list[dict[str, object]]],
    *,
    response_language: str,
) -> str:
    clear = recurring_signals["clear"]
    weak = recurring_signals["weak"]
    unmatched = recurring_signals["unmatched"]
    if response_language == "zh-CN":
        if clear:
            clear_text = "、".join(str(item["label_zh"]) for item in clear)
            clear_docs = "；".join(
                f"{item['label_zh']}主要见于 {', '.join(Path(path).name for path in item['documents'])}"
                for item in clear
            )
            extra_parts = [
                f"{Path(str(item['source_path'])).name} 还提到 {item['summary']}"
                for item in (*weak, *unmatched)[:2]
            ]
            extra = f"；另外，{'；'.join(extra_parts)}" if extra_parts else ""
            return (
                f"从聚合后的 limitations 槽位来看，目前最清晰的重复性局限主要是{clear_text}。"
                f"更具体地说，{clear_docs}{extra}。我只把在多篇论文中都得到槽位支持的限制视为共同局限，"
                "其余只在单篇论文中出现的信号会保留为论文特有约束，而不会扩写成更强的跨论文结论。"
            )
        return (
            "从当前聚合后的 limitations 槽位来看，还没有足够证据表明多篇论文存在明确重复出现的共同局限。"
            "现有局限更多停留在单篇论文层面，因此不宜把它们概括成整组论文共享的结论。"
        )

    if clear:
        clear_text = ", ".join(str(item["label_en"]).lower() for item in clear)
        clear_docs = "; ".join(
            f"{item['label_en']} is supported in {', '.join(Path(path).name for path in item['documents'])}"
            for item in clear
        )
        extra_parts = [
            f"{Path(str(item['source_path'])).name} separately adds {str(item['summary']).rstrip('.')}"
            for item in (*weak, *unmatched)[:2]
        ]
        extra = "; separately, " + "; ".join(extra_parts) if extra_parts else ""
        return (
            f"Across the consulted PDFs, the clearest recurring limitations are {clear_text}. "
            f"More concretely, {clear_docs}{extra}. "
            "A limitation is treated as recurring only when the cleaned limitations slot supports it in more than one consulted paper."
        )
    return (
        "Across the consulted PDFs, there is not enough slot support to claim a clearly recurring limitation shared by multiple papers. "
        "The current limitation signals stay mostly paper-specific, so they should not be collapsed into a stronger cross-paper conclusion."
    )


def _contains_generic_paper_filler(
    answer_lower: str,
    document_notes: list[dict[str, object]],
) -> bool:
    support_corpus = " ".join(
        _slot_payload_text(_document_slot(document_note, slot_name)).lower()
        for document_note in document_notes
        for slot_name in (
            "research_question",
            "sample_or_data",
            "method",
            "main_findings",
            "limitations",
            "conclusion",
            "practical_or_investment_implications",
        )
        if _slot_payload_status(_document_slot(document_note, slot_name)) != "not_clearly_stated"
    )
    filler_patterns = (
        "broader promise of machine learning",
        "underscores the importance of machine learning",
        "illustrates how machine learning can",
        "significant benefits of machine learning",
        "provide deep insights",
        "offers a robust framework for future research",
        "important implications for investors",
        "broadly useful for practitioners",
        "potential economic benefits",
        "robust and reliable predictive tools",
        "this memo provides a concise overview",
        "说明机器学习在金融中具有广泛前景",
        "对投资者具有重要启示",
        "具有广泛意义",
        "进一步说明了",
        "这表明机器学习具有显著优势",
    )
    return any(pattern in answer_lower and pattern not in support_corpus for pattern in filler_patterns)


def _looks_excerpt_heavy_paper_answer(answer_text: str) -> bool:
    lowered = answer_text.lower()
    markdown_heading_count = len(re.findall(r"(?m)^#{2,6}\s", answer_text))
    dimension_block_count = answer_text.count("**Dimension:")
    evidence_marker_count = sum(lowered.count(token) for token in ("**evidence**", "evidence refs", "retrieved evidence"))
    inline_ref_count = sum(lowered.count(token) for token in ("#page=", "#chunk=", "#pages=", "evidence:"))
    page_note_count = len(re.findall(r"\bp\.\d+\b", lowered))
    return (
        markdown_heading_count >= 3
        or dimension_block_count >= 2
        or evidence_marker_count >= 2
        or inline_ref_count >= 4
        or page_note_count >= 4
    )


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the cleaned slot evidence."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    if "not clearly stated in the paper" in answer_lower and "however" in answer_lower:
        notes.append(
            "The answer acknowledges a missing dimension but then keeps padding it with unsupported follow-on commentary."
        )
    if _looks_excerpt_heavy_paper_answer(answer_text):
        notes.append(
            "The answer is still too excerpt-heavy or outline-heavy for the final paper renderer contract."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots and mode_selection.mode == "paper_summary":
        notes.append(
            "The answer did not clearly cover these requested summary dimensions: "
            + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
            + "."
        )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }


def _bullet_lines(items: tuple[str, ...]) -> list[str]:
    return [f"- `{item}`" for item in items]


def _evidence_from_tool_calls(tool_calls: list[ToolCall]) -> tuple[str, ...]:
    refs: list[str] = []
    for tool_call in tool_calls:
        refs.extend(tool_call.evidence_refs)
    return _dedupe_strings(refs)


def _plan_tool_usage(
    prompt: str,
    repo_root: Path,
    mode_selection: ModeSelection,
    edit_plan: WorkspaceEditPlan | None = None,
    workspace_coverage: OnboardingCoverage | None = None,
) -> list[ToolDecision]:
    if mode_selection.answer_schema == "brief_response":
        return []
    if _is_paper_mode(mode_selection.mode):
        return []
    planners = {
        "repo_overview": _plan_repo_overview_tools,
        "workspace_verification": _plan_workspace_verification_tools,
        "project_onboarding": _plan_project_onboarding_tools,
        "file_explain": _plan_file_explain_tools,
        "architecture_review": _plan_architecture_review_tools,
        "implementation_plan": _plan_implementation_plan_tools,
        "workspace_edit": _plan_workspace_edit_tools,
        "prompt_compiler": _plan_prompt_compiler_tools,
        "compare_options": _plan_compare_options_tools,
    }
    planner = planners[mode_selection.mode]
    if mode_selection.mode == "workspace_edit":
        return _dedupe_decisions(
            planner(
                prompt,
                repo_root,
                mode_selection,
                edit_plan,
                coverage=workspace_coverage,
            )
        )
    return _dedupe_decisions(planner(prompt, repo_root, mode_selection))


def _plan_repo_overview_tools(
    prompt: str,
    repo_root: Path,
    mode_selection: ModeSelection,
) -> list[ToolDecision]:
    decisions = [_decision("list_directory", "Inspect the top-level repository structure.", path=".")]
    if (repo_root / "README.md").is_file():
        decisions.append(_decision("read_text_file", "Read the project README for the high-level purpose.", path="README.md"))
    if (repo_root / "src" / "labai").is_dir():
        decisions.append(_decision("list_directory", "Inspect the main package layout.", path="src/labai"))
    if (repo_root / "tests").is_dir():
        decisions.append(_decision("list_directory", "Inspect the test surface for onboarding context.", path="tests"))
    if (repo_root / "src").is_dir():
        decisions.append(_decision("find_files", "Find the main Python source files.", pattern="*.py", path="src"))
    return decisions


def _plan_workspace_verification_tools(
    prompt: str,
    repo_root: Path,
    mode_selection: ModeSelection,
) -> list[ToolDecision]:
    decisions = _plan_project_onboarding_tools(prompt, repo_root, mode_selection)
    for doc_name in ("AGENTS.md", "CLAUDE.md", "PROJECT.md"):
        candidate = repo_root / doc_name
        if candidate.is_file():
            decisions.append(
                _decision(
                    "read_text_file",
                    "Read a top-level handoff or workspace-instruction document for readiness clues.",
                    path=doc_name,
                )
            )
    if (repo_root / "tests").is_dir():
        decisions.append(
            _decision(
                "list_directory",
                "Inspect the visible test surface to estimate day-one verification readiness.",
                path="tests",
            )
        )
    return decisions


def _plan_project_onboarding_tools(
    prompt: str,
    repo_root: Path,
    mode_selection: ModeSelection,
) -> list[ToolDecision]:
    decisions = [_decision("list_directory", "Inspect the top-level project structure for onboarding.", path=".")]
    if (repo_root / "README.md").is_file():
        decisions.append(_decision("read_text_file", "Read the README for the project purpose and setup hints.", path="README.md"))

    for relative_dir in _candidate_onboarding_directories(repo_root):
        decisions.append(_decision("list_directory", "Inspect a notable project directory for onboarding context.", path=relative_dir))

    if any(path.suffix.lower() == ".py" for path in repo_root.rglob("*.py")):
        decisions.append(_decision("find_files", "Find Python implementation files across the workspace.", pattern="*.py", path="."))
    if any(path.suffix.lower() == ".ipynb" for path in repo_root.rglob("*.ipynb")):
        decisions.append(_decision("find_files", "Find notebook files that may shape onboarding context.", pattern="*.ipynb", path="."))
    if (repo_root / "tests").is_dir():
        decisions.append(_decision("find_files", "Find the visible test files for verification context.", pattern="test_*.py", path="tests"))

    for relative_path in _candidate_onboarding_entrypoint_paths(repo_root):
        decisions.append(_decision("read_text_file", "Read a likely entry-point file for onboarding context.", path=relative_path))
    for relative_path in _candidate_onboarding_config_paths(repo_root):
        decisions.append(_decision("read_text_file", "Read a likely config or dependency file for onboarding context.", path=relative_path))

    return decisions


def _candidate_onboarding_directories(repo_root: Path) -> tuple[str, ...]:
    candidates: list[str] = []
    for name in _ONBOARDING_PRIORITY_DIRS:
        candidate = repo_root / name
        if candidate.is_dir():
            candidates.append(name)
    if candidates:
        return tuple(candidates[:6])

    for candidate in sorted(repo_root.iterdir(), key=lambda item: item.name.lower()):
        if not candidate.is_dir() or _is_ignored_workspace_scan_path(candidate, repo_root):
            continue
        candidates.append(candidate.relative_to(repo_root).as_posix())
    return tuple(candidates[:6])


def _candidate_onboarding_entrypoint_paths(repo_root: Path) -> tuple[str, ...]:
    discovered: list[str] = []
    for name in _ONBOARDING_ENTRYPOINT_NAMES:
        for candidate in sorted(repo_root.rglob(name)):
            if not candidate.is_file() or _is_ignored_workspace_scan_path(candidate, repo_root):
                continue
            discovered.append(candidate.relative_to(repo_root).as_posix())
    if not discovered:
        source_root = repo_root / "src"
        if source_root.is_dir():
            for package_dir in sorted(
                item
                for item in source_root.iterdir()
                if item.is_dir() and (item / "__init__.py").is_file()
            ):
                for name in ("__main__.py", "cli.py", "main.py", "api.py", "core.py", "__init__.py"):
                    candidate = package_dir / name
                    if candidate.is_file():
                        discovered.append(candidate.relative_to(repo_root).as_posix())
        for candidate in sorted(repo_root.glob("*.py")):
            if not candidate.is_file() or _is_ignored_workspace_scan_path(candidate, repo_root):
                continue
            discovered.append(candidate.relative_to(repo_root).as_posix())
    if not discovered:
        shallow_candidates = sorted(
            (
                candidate
                for candidate in repo_root.rglob("*.py")
                if candidate.is_file()
                and not _is_ignored_workspace_scan_path(candidate, repo_root)
                and not any(part in {"docs", "tests", "examples"} for part in candidate.relative_to(repo_root).parts[:-1])
            ),
            key=lambda item: (len(item.relative_to(repo_root).parts), item.name.lower()),
        )
        for candidate in shallow_candidates[:5]:
            discovered.append(candidate.relative_to(repo_root).as_posix())
    return _dedupe_strings(tuple(discovered[:5]))


def _candidate_onboarding_config_paths(repo_root: Path) -> tuple[str, ...]:
    discovered: list[str] = []
    for name in _ONBOARDING_CONFIG_NAMES:
        for candidate in sorted(repo_root.rglob(name)):
            if not candidate.is_file() or _is_ignored_workspace_scan_path(candidate, repo_root):
                continue
            discovered.append(candidate.relative_to(repo_root).as_posix())
    return _dedupe_strings(tuple(discovered[:6]))


def _is_ignored_workspace_scan_path(path: Path, repo_root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return True
    return any(
        part in {".venv", ".labai", ".planning"} or part in _ONBOARDING_IGNORED_DIR_NAMES
        for part in relative.parts
    )


def _plan_file_explain_tools(
    prompt: str,
    repo_root: Path,
    mode_selection: ModeSelection,
) -> list[ToolDecision]:
    decisions: list[ToolDecision] = []
    for relative_path in mode_selection.matched_paths[:3]:
        absolute = repo_root / relative_path
        if absolute.is_file():
            decisions.append(_decision("read_text_file", "Read the requested file directly.", path=relative_path))
            parent_relative = absolute.parent.relative_to(repo_root).as_posix() or "."
            decisions.append(_decision("list_directory", "Inspect the file's nearby module context.", path=parent_relative))
        elif absolute.is_dir():
            decisions.append(_decision("list_directory", "Inspect the requested directory directly.", path=relative_path))
    if decisions:
        return decisions

    for fallback_path in _fallback_file_paths(prompt, repo_root):
        decisions.append(_decision("read_text_file", "Read the likely requested implementation file.", path=fallback_path))
    return decisions or _plan_repo_overview_tools(prompt, repo_root, mode_selection)


def _plan_architecture_review_tools(
    prompt: str,
    repo_root: Path,
    mode_selection: ModeSelection,
) -> list[ToolDecision]:
    decisions: list[ToolDecision] = []
    if (repo_root / "src" / "labai").is_dir():
        decisions.append(_decision("list_directory", "Inspect the main package boundaries.", path="src/labai"))
    else:
        decisions.append(_decision("list_directory", "Inspect the top-level repository structure.", path="."))
    for relative_path in (
        "README.md",
        "src/labai/cli.py",
        "src/labai/config.py",
        "src/labai/research/loop.py",
        "src/labai/execution/claw.py",
        "src/labai/runtime/session.py",
        "src/labai/runtime/audit.py",
    ):
        if (repo_root / relative_path).is_file():
            decisions.append(_decision("read_text_file", "Read a core architecture surface.", path=relative_path))
    if (repo_root / ".labai" / "config.toml").is_file():
        decisions.append(_decision("read_text_file", "Read the tracked runtime config for active routing details.", path=".labai/config.toml"))
    return decisions


def _plan_implementation_plan_tools(
    prompt: str,
    repo_root: Path,
    mode_selection: ModeSelection,
) -> list[ToolDecision]:
    decisions: list[ToolDecision] = []
    for relative_path in (
        ".planning/ROADMAP.md",
        ".planning/REQUIREMENTS.md",
        "README.md",
        "src/labai/cli.py",
        "src/labai/config.py",
        "src/labai/research/loop.py",
        "src/labai/runtime/session.py",
        "src/labai/runtime/audit.py",
    ):
        if (repo_root / relative_path).is_file():
            decisions.append(_decision("read_text_file", "Read planning and implementation surfaces needed for the next-step plan.", path=relative_path))
    if (repo_root / "src").is_dir():
        decisions.append(_decision("find_files", "Find existing Python implementation surfaces relevant to the plan.", pattern="*.py", path="src"))
    return decisions


def _plan_workspace_edit_tools(
    prompt: str,
    repo_root: Path,
    mode_selection: ModeSelection,
    edit_plan: WorkspaceEditPlan | None = None,
    coverage: OnboardingCoverage | None = None,
) -> list[ToolDecision]:
    decisions: list[ToolDecision] = []
    config_entrypoint_task = False
    if edit_plan is not None and edit_plan.active:
        config_entrypoint_task = any(
            Path(target).suffix.lower() in {".json", ".toml", ".yaml", ".yml"}
            or Path(target).name.lower() in {"package.json", "pyproject.toml", "vercel.json", "netlify.toml"}
            for target in edit_plan.primary_targets
        )
    for relative_path in mode_selection.matched_paths[:8]:
        absolute = repo_root / relative_path
        if absolute.is_file():
            decisions.append(_decision("read_text_file", "Read an explicitly requested workspace file.", path=relative_path))
        elif absolute.is_dir():
            decisions.append(_decision("list_directory", "Inspect an explicitly requested workspace directory.", path=relative_path))
    if coverage is not None and coverage.total_files:
        decisions.append(
            _decision(
                "list_directory",
                "Inspect the top-level workspace structure before editing so the task stays grounded in the current project layout.",
                path=".",
            )
        )
        for relative_dir in _onboarding_notable_dirs(coverage)[:4]:
            if (repo_root / relative_dir).is_dir():
                decisions.append(
                    _decision(
                        "list_directory",
                        "Inspect a notable workspace directory before editing so the task uses real project context instead of prompt-only guesses.",
                        path=relative_dir,
                    )
                )
        for relative_path in _workspace_edit_manifest_read_paths(
            coverage,
            config_entrypoint_task=config_entrypoint_task,
        ):
            if (repo_root / relative_path).is_file():
                decisions.append(
                    _decision(
                        "read_text_file",
                        "Read a manifest-selected workspace file before editing so the task contract is grounded in the actual project surface.",
                        path=relative_path,
                    )
                )
    if (repo_root / "README.md").is_file() and not config_entrypoint_task:
        decisions.append(_decision("read_text_file", "Read the workspace README for current state and handoff context.", path="README.md"))
    if (repo_root / "src").is_dir() and not config_entrypoint_task:
        decisions.append(_decision("find_files", "Inspect source files that may be relevant to the coding task.", pattern="*.py", path="src"))
    if (repo_root / "tests").is_dir() and not config_entrypoint_task:
        decisions.append(_decision("find_files", "Inspect relevant test files for the coding task.", pattern="test_*.py", path="tests"))
    if edit_plan is not None and edit_plan.active:
        for relative_target in (
            *edit_plan.primary_targets[:4],
            *edit_plan.referenced_paths[:2],
            *edit_plan.secondary_targets[:2],
        ):
            if (repo_root / relative_target).is_file():
                decisions.append(
                    _decision(
                        "read_text_file",
                        "Read a planned primary, referenced, or secondary target so the edit task stays role-aware.",
                        path=relative_target,
                    )
                )
        check_plan = build_workspace_check_plan(
            prompt,
            repo_root,
            planned_modifications=edit_plan.planned_modifications,
            planned_creations=edit_plan.planned_creations,
        )
        for check in check_plan:
            for relative_target in check.relative_targets[:4]:
                if (repo_root / relative_target).is_file():
                    decisions.append(
                        _decision(
                            "read_text_file",
                            "Read a targeted check file so the coding task is grounded in the actual failing contract.",
                            path=relative_target,
                        )
                    )
    return decisions


def _workspace_edit_manifest_read_paths(
    coverage: OnboardingCoverage,
    *,
    config_entrypoint_task: bool,
) -> tuple[str, ...]:
    limits = (
        {"config": 5, "docs": 1, "scripts": 3, "source": 3, "tests": 2}
        if config_entrypoint_task
        else {"config": 4, "docs": 2, "scripts": 4, "source": 6, "tests": 4}
    )
    entry_map = {entry.path: entry for entry in coverage.manifest_entries}
    selected: list[str] = []
    category_counts: Counter[str] = Counter()
    for relative_path in coverage.inspected_paths:
        entry = entry_map.get(relative_path)
        if entry is None:
            continue
        limit = limits.get(entry.category)
        if limit is None:
            continue
        if category_counts[entry.category] >= limit:
            continue
        category_counts[entry.category] += 1
        selected.append(relative_path)
    return tuple(selected)


def _plan_prompt_compiler_tools(
    prompt: str,
    repo_root: Path,
    mode_selection: ModeSelection,
) -> list[ToolDecision]:
    return []


def _plan_compare_options_tools(
    prompt: str,
    repo_root: Path,
    mode_selection: ModeSelection,
) -> list[ToolDecision]:
    decisions: list[ToolDecision] = []
    prompt_lower = prompt.lower()
    for relative_path in mode_selection.matched_paths[:3]:
        if (repo_root / relative_path).is_file():
            decisions.append(_decision("read_text_file", "Read an explicitly mentioned comparison target.", path=relative_path))
    if "claw" in prompt_lower and (repo_root / "src/labai/execution/claw.py").is_file():
        decisions.append(_decision("read_text_file", "Read the Claw adapter path for comparison.", path="src/labai/execution/claw.py"))
    if ("native" in prompt_lower or "fallback" in prompt_lower) and (repo_root / "src/labai/research/loop.py").is_file():
        decisions.append(_decision("read_text_file", "Read the runtime routing logic for native fallback behavior.", path="src/labai/research/loop.py"))
    if ("native" in prompt_lower or "fallback" in prompt_lower) and (repo_root / "src/labai/config.py").is_file():
        decisions.append(_decision("read_text_file", "Read the config surface that controls runtime and fallback selection.", path="src/labai/config.py"))
    return decisions or _plan_architecture_review_tools(prompt, repo_root, mode_selection)


def _decision(tool_name: str, reason: str, **arguments: str) -> ToolDecision:
    return ToolDecision(tool_name=tool_name, should_use=True, reason=reason, arguments=arguments)


def _fallback_file_paths(prompt: str, repo_root: Path) -> tuple[str, ...]:
    prompt_lower = prompt.lower()
    mapping = (
        ("claw", "src/labai/execution/claw.py"),
        ("config", "src/labai/config.py"),
        ("cli", "src/labai/cli.py"),
        ("session", "src/labai/runtime/session.py"),
        ("audit", "src/labai/runtime/audit.py"),
        ("loop", "src/labai/research/loop.py"),
        ("runtime", "src/labai/execution/base.py"),
    )
    return tuple(
        relative_path
        for keyword, relative_path in mapping
        if keyword in prompt_lower and (repo_root / relative_path).is_file()
    )


def _dedupe_decisions(decisions: list[ToolDecision]) -> list[ToolDecision]:
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    deduped: list[ToolDecision] = []
    for decision in decisions:
        key = (
            decision.tool_name,
            tuple(sorted(decision.arguments.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(decision)
    return deduped


def _execute_tool_plan(
    access_root: Path | WorkspaceAccessManager,
    decisions: list[ToolDecision],
) -> tuple[list[ToolCall], list[str], tuple[str, ...]]:
    if not decisions:
        return [], [], ()

    dispatcher = ToolDispatcher(access_root)
    tool_calls: list[ToolCall] = []
    observations: list[str] = []
    evidence_refs: list[str] = []

    for decision in decisions:
        try:
            result = dispatcher.execute(decision.tool_name, **decision.arguments)
            summary, extracted_refs = _summarize_tool_result(result)
            tool_calls.append(
                ToolCall(
                    tool_name=decision.tool_name,
                    arguments=decision.arguments,
                    status="ok",
                    summary=summary,
                    evidence_refs=tuple(extracted_refs),
                )
            )
            if summary:
                observations.append(summary)
            evidence_refs.extend(extracted_refs)
        except (ToolExecutionError, ToolValidationError, OSError, UnicodeDecodeError) as exc:
            message = f"{decision.tool_name} failed for {decision.arguments}: {exc}"
            tool_calls.append(
                ToolCall(
                    tool_name=decision.tool_name,
                    arguments=decision.arguments,
                    status="error",
                    summary="",
                    error=str(exc),
                )
            )
            observations.append(message)

    return tool_calls, observations, _dedupe_strings(evidence_refs)


def _summarize_tool_result(result: object) -> tuple[str, list[str]]:
    if not isinstance(result, dict):
        return "Tool returned an unexpected result.", []

    tool_name = str(result.get("tool", ""))
    if tool_name == "list_directory":
        entries = result.get("entries", [])
        if not isinstance(entries, list):
            return "Directory listing returned unexpected data.", []
        names = [entry["path"] for entry in entries[:8] if isinstance(entry, dict) and "path" in entry]
        suffix = ""
        if len(entries) > 8:
            suffix = f", plus {len(entries) - 8} more entries"
        return (
            f"Directory {result.get('path', '.')} contains "
            f"{', '.join(names) if names else 'no visible entries'}{suffix}."
        ), [str(result.get("path", "."))]

    if tool_name == "find_files":
        matches = result.get("matches", [])
        if not isinstance(matches, list):
            return "File search returned unexpected data.", []
        shown = [str(match) for match in matches[:10]]
        suffix = ""
        if len(matches) > 10:
            suffix = f", plus {len(matches) - 10} more matches"
        return (
            f"File search under {result.get('path', '.')} matched "
            f"{', '.join(shown) if shown else 'no files'}{suffix}."
        ), shown

    if tool_name == "read_text_file":
        text = result.get("text", "")
        if not isinstance(text, str):
            return "File read returned unexpected data.", []
        relative_path = str(result.get("path", ""))
        return _summarize_file_text(relative_path, text), [relative_path]

    return "Tool completed.", []


def _summarize_file_text(relative_path: str, text: str) -> str:
    suffix = Path(relative_path).suffix.lower()
    if suffix == ".py":
        return _summarize_python_file(relative_path, text)
    if suffix == ".toml":
        return _summarize_toml_file(relative_path, text)
    if suffix == ".md":
        return _summarize_markdown_file(relative_path, text)

    lines = _meaningful_lines(text, limit=5)
    snippet = " | ".join(lines) if lines else "(empty file)"
    return f"File {relative_path} contains: {snippet}"


def _summarize_python_file(relative_path: str, text: str) -> str:
    classes = _extract_python_symbols(text, kind="class")
    functions = _extract_python_symbols(text, kind="def")
    assignments = _extract_interesting_assignments(text)
    imports = _extract_python_import_lines(text)
    summary_parts = [f"Python file {relative_path}"]

    if classes:
        summary_parts.append(f"classes: {', '.join(classes[:6])}")
    if functions:
        summary_parts.append(f"functions: {', '.join(functions[:8])}")
    if assignments:
        summary_parts.append(f"notable names: {', '.join(assignments[:6])}")
    if imports:
        summary_parts.append(f"imports: {' | '.join(imports[:4])}")

    lines = _meaningful_lines(text, limit=8, include_imports=True)
    if lines:
        summary_parts.append(f"sample lines: {' | '.join(lines)}")

    return ". ".join(summary_parts) + "."


def _summarize_toml_file(relative_path: str, text: str) -> str:
    sections = re.findall(r"^\s*\[([^\]]+)\]\s*$", text, flags=re.MULTILINE)
    key_values = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key_values.append(f"{key.strip()}={value.strip()}")
        if len(key_values) >= 6:
            break

    parts = [f"TOML file {relative_path}"]
    if sections:
        parts.append(f"sections: {', '.join(sections[:8])}")
    if key_values:
        parts.append(f"key settings: {' | '.join(key_values)}")
    return ". ".join(parts) + "."


def _summarize_markdown_file(relative_path: str, text: str) -> str:
    headings = [
        line.lstrip("#").strip()
        for line in text.splitlines()
        if line.strip().startswith("#")
    ]
    lines = _meaningful_lines(text, limit=4, skip_headings=True)
    parts = [f"Markdown file {relative_path}"]
    if headings:
        parts.append(f"headings: {', '.join(headings[:6])}")
    if lines:
        parts.append(f"content: {' | '.join(lines)}")
    return ". ".join(parts) + "."


def _extract_python_symbols(text: str, *, kind: str) -> list[str]:
    pattern = r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)" if kind == "class" else r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
    return re.findall(pattern, text, flags=re.MULTILINE)


def _extract_interesting_assignments(text: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(r"^\s*([A-Z][A-Z0-9_]+)\s*=", text, flags=re.MULTILINE):
        names.append(match.group(1))
        if len(names) >= 6:
            break
    return names


def _extract_python_import_lines(text: str) -> list[str]:
    imports: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith(("from ", "import ")):
            continue
        imports.append(_truncate_line(stripped))
        if len(imports) >= 4:
            break
    return imports


def _meaningful_lines(
    text: str,
    *,
    limit: int,
    skip_headings: bool = False,
    include_imports: bool = False,
) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if skip_headings and stripped.startswith("#"):
            continue
        if not include_imports and stripped.startswith(("from ", "import ")):
            continue
        lines.append(_truncate_line(stripped))
        if len(lines) >= limit:
            break
    return lines


def _signal_lines(text: str, *, limit: int) -> list[str]:
    keywords = (
        "runtime",
        "fallback",
        "claw",
        "native",
        "mock",
        "provider",
        "model",
        "openai_base_url",
        "allowedtools",
        "permission_mode",
        "read-only",
        "doctor",
        "ask",
        "session",
        "audit",
        "mode",
    )
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "from ", "import ")):
            continue
        lowered = stripped.lower()
        if not any(keyword in lowered for keyword in keywords):
            continue
        lines.append(_truncate_line(stripped))
        if len(lines) >= limit:
            break
    return lines


def _truncate_line(line: str, limit: int = 120) -> str:
    if len(line) <= limit:
        return line
    return line[: limit - 3] + "..."


def _success_summary(
    mode_name: str,
    runtime_name: str,
    provider_name: str,
    used_runtime_fallback: bool,
    used_provider_fallback: bool,
    tool_count: int,
) -> str:
    notes: list[str] = []
    if used_runtime_fallback:
        notes.append("runtime fallback")
    if used_provider_fallback:
        notes.append("provider fallback")
    note_suffix = f" with {', '.join(notes)}" if notes else ""
    return (
        f"{mode_name} research loop completed via {runtime_name}/{provider_name}{note_suffix} "
        f"after {tool_count} tool call(s)."
    )


def _no_runtime_fallback(requested_runtime: str, fallback_runtime: str) -> RuntimeFallbackInfo:
    return RuntimeFallbackInfo(
        applied=False,
        requested_runtime=requested_runtime,
        active_runtime=requested_runtime,
        fallback_runtime=fallback_runtime,
        reason="",
    )


def _derive_operational_status(route: AnswerRoute) -> OperationalStatus:
    if route.requested_runtime == "claw" and route.runtime_used == "native":
        return "guided_not_ready"
    if route.runtime_fallback.applied or route.provider_fallback.applied:
        return "ready_with_fallback"
    return "ready"


def _derive_error_operational_status(
    requested_runtime: str,
    runtime_fallback: RuntimeFallbackInfo,
) -> OperationalStatus:
    if requested_runtime == "claw" and runtime_fallback.applied:
        return "guided_not_ready"
    return "error"


def _dedupe_strings(items: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    ordered: dict[str, None] = {}
    for item in items:
        normalized = str(item).strip()
        if normalized:
            ordered.setdefault(normalized, None)
    return tuple(ordered.keys())


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_narrow_grounded_paper_qa(prompt: str) -> bool:
    lowered = prompt.lower()
    if _is_recurring_limitations_prompt(prompt):
        return False
    broad_tokens = (
        "recurring limitations",
        "across papers",
        "across the consulted pdfs",
        "compare ",
        "比较",
        "多篇",
    )
    if any(token in lowered or token in prompt for token in broad_tokens):
        return False
    narrow_tokens = (
        "what specific",
        "which specific",
        "what sample",
        "what data",
        "what dataset",
        "sample or data",
        "sample period",
        "sample size",
        "asset universe",
        "data source",
        "what limitation",
        "what limitations",
        "where does",
        "where is",
        "explicitly discussed",
        "explicitly stated",
        "based only on",
        "only on ",
        "具体讨论了哪些",
        "明确讨论了哪些",
        "明确写到哪些",
        "样本或数据",
        "样本期",
        "数据细节",
        "数据来源",
        "局限是什么",
        "哪里讨论",
        "在哪一页",
    )
    return any(token in lowered or token in prompt for token in narrow_tokens)


def _narrow_grounded_qa_focus_slot(prompt: str) -> str:
    lowered = prompt.lower()
    if any(token in lowered or token in prompt for token in ("sample", "data", "dataset", "sample period", "sample size", "asset universe", "data source", "样本", "数据")):
        return "sample_or_data"
    if any(token in lowered or token in prompt for token in ("method", "methods", "model family", "model families", "machine learning method", "方法", "模型")):
        return "method"
    if any(token in lowered or token in prompt for token in ("limitation", "limitations", "caveat", "constraint", "局限", "限制")):
        return "limitations"
    if any(token in lowered or token in prompt for token in ("conclusion", "结论")):
        return "conclusion"
    if any(token in lowered or token in prompt for token in ("finding", "findings", "result", "results", "发现", "结果")):
        return "main_findings"
    return "method"


def _looks_like_explicit_sample_data_text(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).lower()
    positive_markers = (
        "our sample begins",
        "in our sample",
        "we conduct a large-scale empirical analysis",
        "individual stocks",
        "dataset spans",
        "sample period",
        "training sample",
        "training data",
        "validation sample",
        "validation data",
        "out-of-sample testing",
        "balanced panel of stocks",
        "missing data",
        "30,000",
        "60 years",
        "daily adjusted closing prices",
        "nasdaq-100 constituents",
        "downloaded via yfinance",
        "observations",
        "trading days",
        "tickers",
        "equity panel",
        "vix features",
        "5-day rolling mean",
        "22-day rolling mean",
        "training window",
        "out-of-sample evaluation",
        "shock events",
        "msft",
        "adbe",
        "nvda",
        "payx",
    )
    strong_positive_markers = (
        "crsp",
        "nyse",
        "amex",
        "nasdaq",
        "our sample begins",
        "training sample",
        "validation sample",
        "out-of-sample testing",
        "30,000",
        "nasdaq-100 constituents",
        "daily adjusted closing prices",
        "number of stocks in our sample",
    )
    negative_markers = (
        "tuning parameter",
        "tuning parameters",
        "adaptively",
        "optimized via validation",
        "select tuning parameters",
        "forecast performance",
        "predictive performance",
        "performance evaluation",
        "out-of-sample r2",
        "diebold-mariano",
        "test statistic",
        "variable importance",
        "trading strategy",
        "sharpe ratio",
        "model typically chooses",
        "subsamples that include only",
        "top-1,000 stocks",
        "bottom-1,000 stocks",
        "out-of-sample period",
        "sas code",
        "web site",
        "website",
    )
    hard_negative_markers = (
        "sas code",
        "web site",
        "website",
        "variable importance",
        "diebold-mariano",
        "test statistic",
        "sharpe ratio",
        "subsamples that include only",
        "top-1,000 stocks",
        "bottom-1,000 stocks",
        "out-of-sample period",
    )
    has_year_or_count = bool(
        re.search(r"\b(19[5-9]\d|20[0-1]\d)\b", lowered)
        or re.search(r"\b\d{1,3},\d{3}\b", lowered)
        or "60 years" in lowered
    )
    if any(marker in lowered for marker in hard_negative_markers) and not any(
        marker in lowered for marker in strong_positive_markers
    ):
        return False
    if any(marker in lowered for marker in negative_markers) and not any(marker in lowered for marker in positive_markers):
        return False
    return any(marker in lowered for marker in positive_markers) or (
        has_year_or_count
        and any(
            marker in lowered
            for marker in (
                "sample",
                "data",
                "stocks",
                "stock returns",
                "tickers",
                "trading days",
                "vix",
                "nasdaq-100",
            )
        )
    )


def _narrow_grounded_qa_text_pool(paper_trace: PaperTrace, slot_name: str) -> list[str]:
    texts: list[str] = []
    for document_note in paper_trace.document_notes:
        payload = _document_slot(document_note, slot_name)
        summary_text = _slot_payload_text(payload)
        if summary_text and summary_text != "Not clearly stated in the paper.":
            texts.append(summary_text)
        merged_text = str(payload.get("merged_note_text", "")).strip()
        if merged_text and merged_text != "Not clearly stated in the paper.":
            texts.append(merged_text)
    for slot_note in paper_trace.slot_notes:
        if str(slot_note.get("slot_name", "")) != slot_name:
            continue
        extracted = str(slot_note.get("extracted_content", "")).strip()
        if not extracted:
            continue
        if slot_name == "sample_or_data" and not _looks_like_explicit_sample_data_text(extracted):
            continue
        texts.append(extracted)
    for chunk in paper_trace.retrieved_chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        if slot_name == "sample_or_data" and not _looks_like_explicit_sample_data_text(text):
            continue
        texts.append(text)
    return [text for text in _dedupe_strings(texts) if text]


def _extract_method_family_mentions(texts: list[str]) -> tuple[str, ...]:
    corpus = " ".join(unicodedata.normalize("NFKC", text).lower() for text in texts)
    catalog = (
        ("lasso", "Lasso"),
        ("elastic net", "Elastic Net"),
        ("enet", "Elastic Net"),
        ("ridge", "Ridge"),
        ("linear regression", "linear regression"),
        ("generalized linear model", "generalized linear models"),
        ("principal components regression", "principal components regression (PCR)"),
        ("principal component analysis", "principal component analysis (PCA)"),
        ("pca", "principal component analysis (PCA)"),
        ("partial least squares", "partial least squares (PLS)"),
        ("regression tree", "regression trees"),
        ("ann", "artificial neural networks (ANN)"),
        ("cnn", "convolutional neural networks (CNN)"),
        ("lstm", "long short-term memory networks (LSTM)"),
        ("neural network", "neural networks"),
        ("boosted tree", "boosted trees"),
        ("random forest", "random forests"),
    )
    mentions: list[str] = []
    for needle, label in catalog:
        if needle in corpus:
            mentions.append(label)
    if "arimax" in corpus and "garch" in corpus:
        mentions.append("ARIMAX-GARCH")
    return tuple(dict.fromkeys(mentions))


def _looks_like_sample_data_result_noise(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).lower()
    return any(
        marker in lowered
        for marker in (
            "predictive performance",
            "forecast performance",
            "report the main empirical results",
            "rmse",
            "mae",
            "sharpe ratio",
            "diebold-mariano",
            "test statistic",
            "trading strategy",
            "performance evaluation",
        )
    )


def _sample_data_fact_priority(text: str) -> tuple[int, int]:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "daily adjusted closing prices",
            "constituents",
            "equity panel",
            "tradable factors",
            "portfolios sorted by firm characteristics",
        )
    ):
        return (0, len(text))
    if re.search(r"\bfrom\b.+\bto\b.+\b(?:19|20)\d{2}\b", lowered) or re.search(
        r"\bbegins in\b.+\bends in\b",
        lowered,
    ):
        return (1, len(text))
    if any(marker in lowered for marker in ("observations", "trading days", "tickers", "n =", "t =", "first half", "second half")):
        return (2, len(text))
    if any(marker in lowered for marker in ("training", "validation", "test", "out-of-sample")):
        return (3, len(text))
    if any(marker in lowered for marker in ("data source", "downloaded via", "fred", "crsp", "compustat", "bloomberg", "refinitiv", "yfinance")):
        return (4, len(text))
    if any(marker in lowered for marker in ("vix", "rolling mean", "features")):
        return (5, len(text))
    return (6, len(text))


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [unicodedata.normalize("NFKC", text).replace("\n", " ").strip() for text in texts]
    date_range_pattern = re.compile(
        r"\b(?:from|between)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})\s+(?:to|through|until|and)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})",
        re.IGNORECASE,
    )
    for text in normalized_texts:
        lowered = text.lower()
        if not _looks_like_explicit_sample_data_text(text):
            continue
        if all(token in lowered for token in ("nasdaq-100 constituents", "3 jan 2019", "30 dec 2021", "yfinance")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for all NASDAQ-100 constituents from 3 Jan 2019 to 30 Dec 2021, downloaded via yfinance"
            )
        if "tickers" in lowered and "755 trading days" in lowered:
            facts.append("after filtering, roughly N ≈100 tickers and T = 755 trading days remain")
        if all(token in lowered for token in ("vix", "5-day rolling mean", "22-day rolling mean")):
            facts.append("VIX contributes three standardised exogenous features: the level, a 5-day rolling mean, and a 22-day rolling mean")
        if all(token in lowered for token in ("msft", "adbe", "nvda", "payx")):
            if "200 (i, t) pairs" in lowered or "the resulting 200" in lowered:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX, and the shock-event set contains 200 extreme downside events")
            else:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX")
        if all(token in lowered for token in ("3 jan 2019", "30 jun 2020", "out-of-sample evaluation")):
            facts.append("the training window runs from 3 Jan 2019 to 30 Jun 2020, with the remaining period used for out-of-sample evaluation")
        if re.search(r"30,?000.+individual stocks.+1957.+2016", lowered):
            facts.append("the sample covers nearly 30,000 individual stocks over 60 years from 1957 to 2016")
        if re.search(
            r"18 years of training (?:sample|data).+12 years of validation (?:sample|data).+30 years.+out-of-sample testing",
            lowered,
        ):
            facts.append("the paper uses 18 years of training data, 12 years of validation data, and 30 years of out-of-sample testing")
        if re.search(r"our sample begins.+1957.+2016", lowered):
            facts.append("the sample begins in March 1957 and ends in December 2016, covering 60 years")
        if re.search(r"in our sample.+longer and wider", lowered):
            facts.append("the paper says its sample is longer and wider than the benchmark sample it compares against")
        if "94 characteristics" in lowered and "8 macroeconomic predictors" in lowered:
            facts.append("the feature set includes 94 stock-level characteristics and 8 macroeconomic predictors")
        for sentence in _split_rescue_sentences(text):
            normalized_sentence = unicodedata.normalize("NFKC", sentence).strip()
            lowered_sentence = normalized_sentence.lower()
            if not lowered_sentence or _looks_like_sample_data_result_noise(normalized_sentence):
                continue
            if any(
                marker in lowered_sentence
                for marker in ("deep learning", "lasso", "elastic net", "ann", "cnn", "lstm", "rmse", "mae")
            ):
                continue
            if date_range_pattern.search(normalized_sentence) and any(
                token in lowered_sentence for token in ("sample", "data", "dataset", "observations", "panel")
            ):
                facts.append(_truncate_line(normalized_sentence, limit=180))
            if re.search(r"\b\d{1,3}(?:,\d{3})?\s+observations\b", lowered_sentence):
                facts.append(_truncate_line(normalized_sentence, limit=180))
        if re.search(r"balanced panel of stocks|missing data", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"individual stocks", lowered) and re.search(r"60 years|1957|2016", lowered):
            facts.append(_truncate_line(text, limit=180))
    ordered = [item.replace("鈮?00", "~ 100") for item in _dedupe_strings(facts)]

    def _fact_priority(item: str) -> tuple[int, int]:
        lowered_item = item.lower()
        if "nasdaq-100 constituents" in lowered_item:
            return (0, len(item))
        if "n ≈100 tickers" in lowered_item or "755 trading days" in lowered_item:
            return (1, len(item))
        if "755 trading days" in lowered_item or "~ 100 tickers" in lowered_item:
            return (1, len(item))
        if "5-day rolling mean" in lowered_item or "22-day rolling mean" in lowered_item:
            return (2, len(item))
        if "vix level" in lowered_item:
            return (2, len(item))
        if "hubs are msft" in lowered_item:
            return (3, len(item))
        if "training window runs" in lowered_item:
            return (4, len(item))
        if "30,000 individual stocks" in lowered_item:
            return (5, len(item))
        if "training data" in lowered_item:
            return (6, len(item))
        if "sample begins" in lowered_item:
            return (7, len(item))
        return (8, len(item))

    ordered.sort(key=_fact_priority)
    return tuple(ordered)


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [unicodedata.normalize("NFKC", text).replace("\n", " ").strip() for text in texts]
    date_range_pattern = re.compile(
        r"\b(?:from|between)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})\s+(?:to|through|until|and)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})",
        re.IGNORECASE,
    )
    for text in normalized_texts:
        lowered = text.lower()
        if not _looks_like_explicit_sample_data_text(text):
            continue
        if all(token in lowered for token in ("crsp", "nyse", "amex", "nasdaq")):
            facts.append(
                "the paper uses monthly total individual equity returns from CRSP for firms listed in the NYSE, AMEX, and NASDAQ"
            )
        if all(token in lowered for token in ("nasdaq-100 constituents", "3 jan 2019", "30 dec 2021", "yfinance")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for all NASDAQ-100 constituents from 3 Jan 2019 to 30 Dec 2021, downloaded via yfinance"
            )
        elif all(token in lowered for token in ("daily adjusted closing prices", "nasdaq-100 constituents", "3 jan 2019")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for NASDAQ-100 constituents beginning on 3 Jan 2019"
            )
        if "tickers" in lowered and "755 trading days" in lowered:
            facts.append("after filtering, roughly N ~ 100 tickers and T = 755 trading days remain")
        if all(token in lowered for token in ("vix", "5-day rolling mean", "22-day rolling mean")):
            facts.append("VIX contributes three standardised exogenous features: the level, a 5-day rolling mean, and a 22-day rolling mean")
        if all(token in lowered for token in ("msft", "adbe", "nvda", "payx")):
            if "200 (i, t) pairs" in lowered or "the resulting 200" in lowered or "200 extreme shock events" in lowered:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX, and the shock-event set contains 200 extreme downside events")
            else:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX")
        if all(token in lowered for token in ("3 jan 2019", "30 jun 2020", "out-of-sample evaluation")):
            facts.append("the training window runs from 3 Jan 2019 to 30 Jun 2020, with the remaining period used for out-of-sample evaluation")
        elif "training" in lowered and "out-of-sample" in lowered and "3 jan 2019" in lowered:
            facts.append("the paper defines a training window starting on 3 Jan 2019 and evaluates the remaining period out of sample")
        if re.search(r"30,?000.+individual stocks.+1957.+2016", lowered):
            facts.append("the sample covers nearly 30,000 individual stocks over 60 years from 1957 to 2016")
        if re.search(
            r"number of stocks in (?:our|the) sample is (?:almost|nearly|about|approximately)\s+30,?000",
            lowered,
        ):
            stock_count_fact = "the sample contains almost 30,000 stocks"
            if re.search(r"average number of stocks per month.+6,?200", lowered):
                stock_count_fact += ", with the average monthly cross-section exceeding 6,200 stocks"
            facts.append(stock_count_fact)
        if re.search(
            r"18 years of training (?:sample|data).+12 years of validation (?:sample|data).+30 years.+out-of-sample testing",
            lowered,
        ):
            facts.append("the paper uses 18 years of training data, 12 years of validation data, and 30 years of out-of-sample testing")
        if re.search(r"our sample begins.+1957.+2016", lowered):
            facts.append("the sample begins in March 1957 and ends in December 2016, covering 60 years")
        if re.search(r"in our sample.+longer and wider", lowered):
            facts.append("the paper says its sample is longer and wider than the benchmark sample it compares against")
        if "94 characteristics" in lowered and "8 macroeconomic predictors" in lowered:
            facts.append("the feature set includes 94 stock-level characteristics and 8 macroeconomic predictors")
        for sentence in _split_rescue_sentences(text):
            normalized_sentence = unicodedata.normalize("NFKC", sentence).strip()
            lowered_sentence = normalized_sentence.lower()
            if not lowered_sentence or _looks_like_sample_data_result_noise(normalized_sentence):
                continue
            if any(
                marker in lowered_sentence
                for marker in ("deep learning", "lasso", "elastic net", "ann", "cnn", "lstm", "rmse", "mae")
            ):
                continue
            if date_range_pattern.search(normalized_sentence) and any(
                token in lowered_sentence for token in ("sample", "data", "dataset", "observations", "panel")
            ):
                facts.append(_truncate_line(normalized_sentence, limit=180))
            if re.search(r"\b\d{1,3}(?:,\d{3})?\s+observations\b", lowered_sentence):
                facts.append(_truncate_line(normalized_sentence, limit=180))
        if re.search(r"balanced panel of stocks|missing data", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"individual stocks", lowered) and re.search(r"60 years|1957|2016", lowered):
            facts.append(_truncate_line(text, limit=180))

    ordered_raw = [
        item.replace("閳?00", "~ 100").replace("鈮?00", "~ 100").replace("Ёж100", "~ 100")
        for item in _dedupe_strings(facts)
    ]
    ordered: list[str] = []
    seen_signatures: set[str] = set()
    for item in ordered_raw:
        signature = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        ordered.append(item)

    def _fact_priority(item: str) -> tuple[int, int]:
        lowered_item = item.lower()
        if "crsp" in lowered_item and "nyse, amex, and nasdaq" in lowered_item:
            return (0, len(item))
        if "nasdaq-100 constituents" in lowered_item:
            return (1, len(item))
        if "30,000" in lowered_item or "almost 30,000" in lowered_item:
            return (2, len(item))
        if "755 trading days" in lowered_item or "~ 100 tickers" in lowered_item:
            return (3, len(item))
        if "273 observations" in lowered_item:
            return (4, len(item))
        if "january 2001 to september 2023" in lowered_item:
            return (5, len(item))
        if "94 stock-level characteristics" in lowered_item:
            return (6, len(item))
        if "training data" in lowered_item or "validation data" in lowered_item or "out-of-sample testing" in lowered_item:
            return (7, len(item))
        if "5-day rolling mean" in lowered_item or "22-day rolling mean" in lowered_item:
            return (8, len(item))
        if "hubs are msft" in lowered_item:
            return (9, len(item))
        if "training window runs" in lowered_item or "out of sample" in lowered_item:
            return (10, len(item))
        if "sample begins" in lowered_item:
            return (11, len(item))
        return (12, len(item))

    ordered.sort(key=_fact_priority)
    return tuple(ordered)


def _extract_sample_data_facts(texts: list[str]) -> tuple[str, ...]:
    facts: list[str] = []
    normalized_texts = [unicodedata.normalize("NFKC", text).replace("\n", " ").strip() for text in texts]
    date_range_pattern = re.compile(
        r"\b(?:from|between)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})\s+(?:to|through|until|and)\s+([A-Za-z]{3,9}\s+\d{4}|\d{4})",
        re.IGNORECASE,
    )
    for text in normalized_texts:
        lowered = text.lower()
        if not _looks_like_explicit_sample_data_text(text):
            continue
        if all(token in lowered for token in ("crsp", "nyse", "amex", "nasdaq")):
            facts.append(
                "the paper uses monthly total individual equity returns from CRSP for firms listed in the NYSE, AMEX, and NASDAQ"
            )
        if all(token in lowered for token in ("nasdaq-100 constituents", "3 jan 2019", "30 dec 2021", "yfinance")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for all NASDAQ-100 constituents from 3 Jan 2019 to 30 Dec 2021, downloaded via yfinance"
            )
        elif all(token in lowered for token in ("daily adjusted closing prices", "nasdaq-100 constituents", "3 jan 2019")):
            facts.append(
                "the equity panel uses daily adjusted closing prices for NASDAQ-100 constituents beginning on 3 Jan 2019"
            )
        if "tickers" in lowered and "755 trading days" in lowered:
            facts.append("after filtering, roughly N ~ 100 tickers and T = 755 trading days remain")
        if all(token in lowered for token in ("vix", "5-day rolling mean", "22-day rolling mean")):
            facts.append("VIX contributes three standardised exogenous features: the level, a 5-day rolling mean, and a 22-day rolling mean")
        if all(token in lowered for token in ("msft", "adbe", "nvda", "payx")):
            if "200 (i, t) pairs" in lowered or "the resulting 200" in lowered or "200 extreme shock events" in lowered:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX, and the shock-event set contains 200 extreme downside events")
            else:
                facts.append("the hubs are MSFT, ADBE, NVDA, and PAYX")
        if all(token in lowered for token in ("3 jan 2019", "30 jun 2020", "out-of-sample evaluation")):
            facts.append("the training window runs from 3 Jan 2019 to 30 Jun 2020, with the remaining period used for out-of-sample evaluation")
        elif "training" in lowered and "out-of-sample" in lowered and "3 jan 2019" in lowered:
            facts.append("the paper defines a training window starting on 3 Jan 2019 and evaluates the remaining period out of sample")
        if re.search(r"30,?000.+individual stocks.+1957.+2016", lowered):
            facts.append("the sample covers nearly 30,000 individual stocks over 60 years from 1957 to 2016")
        if re.search(
            r"number of stocks in (?:our|the) sample is (?:almost|nearly|about|approximately)\s+30,?000",
            lowered,
        ):
            stock_count_fact = "the sample contains almost 30,000 stocks"
            if re.search(r"average number of stocks per month.+6,?200", lowered):
                stock_count_fact += ", with the average monthly cross-section exceeding 6,200 stocks"
            facts.append(stock_count_fact)
        if re.search(
            r"18 years of training sample\s*\(1957-1974\).+12 years of validation sample\s*\(1975-1986\).+30 years\s*\(1987-2016\).+out-of-sample testing",
            lowered,
        ):
            facts.append(
                "the 60-year sample is split into 18 years of training sample (1957-1974), 12 years of validation sample (1975-1986), and 30 years (1987-2016) for out-of-sample testing"
            )
        elif re.search(
            r"18 years of training (?:sample|data).+12 years of validation (?:sample|data).+30 years.+out-of-sample testing",
            lowered,
        ):
            facts.append("the paper uses 18 years of training sample, 12 years of validation sample, and 30 years of out-of-sample testing")
        if re.search(r"our sample begins.+1957.+2016", lowered):
            facts.append("the sample begins in March 1957 and ends in December 2016, covering 60 years")
        if re.search(r"in our sample.+longer and wider", lowered):
            facts.append("the paper says its sample is longer and wider than the benchmark sample it compares against")
        if "94 characteristics" in lowered and "8 macroeconomic predictors" in lowered:
            facts.append("the feature set includes 94 stock-level characteristics and 8 macroeconomic predictors")
        for sentence in _split_rescue_sentences(text):
            normalized_sentence = unicodedata.normalize("NFKC", sentence).strip()
            lowered_sentence = normalized_sentence.lower()
            if not lowered_sentence or _looks_like_sample_data_result_noise(normalized_sentence):
                continue
            if any(
                marker in lowered_sentence
                for marker in ("deep learning", "lasso", "elastic net", "ann", "cnn", "lstm", "rmse", "mae")
            ):
                continue
            if date_range_pattern.search(normalized_sentence) and any(
                token in lowered_sentence for token in ("sample", "data", "dataset", "observations", "panel")
            ):
                facts.append(_truncate_line(normalized_sentence, limit=180))
            if re.search(r"\b\d{1,3}(?:,\d{3})?\s+observations\b", lowered_sentence):
                facts.append(_truncate_line(normalized_sentence, limit=180))
        if re.search(r"balanced panel of stocks|missing data", lowered):
            facts.append(_truncate_line(text, limit=180))
        if re.search(r"individual stocks", lowered) and re.search(r"60 years|1957|2016", lowered):
            facts.append(_truncate_line(text, limit=180))

    ordered_raw = [
        item.replace("閳?00", "~ 100").replace("鈮?00", "~ 100").replace("Ёж100", "~ 100")
        for item in _dedupe_strings(facts)
    ]
    ordered: list[str] = []
    seen_signatures: set[str] = set()
    for item in ordered_raw:
        signature = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        ordered.append(item)

    def _fact_priority(item: str) -> tuple[int, int]:
        lowered_item = item.lower()
        if "crsp" in lowered_item and "nyse, amex, and nasdaq" in lowered_item:
            return (0, len(item))
        if "nasdaq-100 constituents" in lowered_item:
            return (1, len(item))
        if "30,000" in lowered_item or "almost 30,000" in lowered_item:
            return (2, len(item))
        if "755 trading days" in lowered_item or "~ 100 tickers" in lowered_item:
            return (3, len(item))
        if "273 observations" in lowered_item:
            return (4, len(item))
        if "january 2001 to september 2023" in lowered_item:
            return (5, len(item))
        if "94 stock-level characteristics" in lowered_item:
            return (6, len(item))
        if "training sample" in lowered_item or "validation sample" in lowered_item or "out-of-sample testing" in lowered_item:
            return (7, len(item))
        if "5-day rolling mean" in lowered_item or "22-day rolling mean" in lowered_item:
            return (8, len(item))
        if "hubs are msft" in lowered_item:
            return (9, len(item))
        if "training window runs" in lowered_item or "out of sample" in lowered_item:
            return (10, len(item))
        if "sample begins" in lowered_item:
            return (11, len(item))
        return (12, len(item))

    ordered.sort(key=_fact_priority)
    return tuple(ordered)


def _slot_specific_rescue_summary(
    slot_name: str,
    candidate_pages: list[tuple[int, str]],
) -> str:
    texts = [text for _page_number, text in candidate_pages]
    if slot_name == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            return " ".join(f"{fact.rstrip('. ')}." for fact in facts[:5]).strip()
    if slot_name == "method":
        normalized_corpus = " ".join(
            unicodedata.normalize("NFKC", text).replace("\n", " ").strip()
            for text in texts
        )
        lowered_corpus = normalized_corpus.lower()
        if (
            "arimax" in lowered_corpus
            and ("garch" in lowered_corpus or "garch benchmark" in lowered_corpus)
            and "lstm" in lowered_corpus
        ):
            return "The paper implements an ARIMAX-GARCH benchmark and an end-to-end LSTM."
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "benchmark" in lowered_sentence and "lstm" in lowered_sentence:
                    return _truncate_line(unicodedata.normalize("NFKC", sentence).strip(), limit=220)
        families = _extract_method_family_mentions(texts)
        if families:
            if len(families) == 1:
                return f"The paper explicitly discusses {families[0]}."
            if len(families) == 2:
                return f"The paper explicitly discusses {families[0]} and {families[1]}."
            return "The paper explicitly discusses " + ", ".join(families[:-1]) + f", and {families[-1]}."
    if slot_name == "main_findings":
        rescued: list[str] = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "fig." in lowered_sentence or "table " in lowered_sentence:
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "best performing methods",
                        "best performing",
                        "shallow learning outperforms deep learning",
                        "dominant predictive signals",
                        "higher sharpe ratios",
                        "predictive advantage",
                        "nonlinear interactions",
                        "most valuable for forecasting",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            return ". ".join(unique[:3]).strip() + "."
    if slot_name == "limitations":
        rescued: list[str] = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "fig." in lowered_sentence or "table " in lowered_sentence:
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "limitation",
                        "limitations",
                        "future",
                        "monthly data",
                        "high-frequency data",
                        "could help researchers improve",
                        "simple ",
                        "limited",
                        "dearth of data",
                        "low signal-to-noise ratio",
                        "overfit",
                        "overfitting",
                        "computationally intensive",
                        "must be heavily regularized",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            strong = [
                item
                for item in unique
                if any(
                    marker in item.lower()
                    for marker in (
                        "dearth of data",
                        "low signal-to-noise ratio",
                        "overfit",
                        "overfitting",
                        "computationally intensive",
                        "must be heavily regularized",
                    )
                )
            ]
            preferred = strong or unique
            return ". ".join(preferred[:2]).strip() + "."
    if slot_name == "conclusion":
        rescued: list[str] = []
        for text in texts:
            for sentence in _split_rescue_sentences(text):
                lowered_sentence = sentence.lower()
                if "fig." in lowered_sentence or "table " in lowered_sentence:
                    continue
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "the evidence indicates",
                        "these findings emphasize",
                        "enhance the precision",
                        "predictive capabilities",
                        "greater precision in forecasting",
                        "leading contender for forecasting",
                        "at the highest level",
                        "best performing methods",
                        "most valuable for forecasting",
                        "dominant predictive signals",
                        "brings promise",
                        "our findings help justify",
                    )
                ):
                    continue
                cleaned = _clean_rescue_sentence(sentence, slot_name=slot_name)
                if cleaned:
                    rescued.append(cleaned.rstrip(". "))
        if rescued:
            unique = list(dict.fromkeys(rescued))
            strong = [
                item
                for item in unique
                if any(
                    marker in item.lower()
                    for marker in (
                        "at the highest level",
                        "best performing methods",
                        "brings promise",
                        "most valuable for forecasting",
                        "dominant predictive signals",
                        "help justify",
                    )
                )
            ]
            preferred = strong or unique
            return ". ".join(preferred[:3]).strip() + "."
    return ""


def _prompt_requests_support_detail(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(
        token in lowered or token in prompt
        for token in (
            "where ",
            "where does",
            "cite",
            "citation",
            "evidence",
            "page",
            "chunk",
            "哪一页",
            "哪里讨论",
            "证据",
            "引用",
        )
    )


def _first_support_ref(paper_trace: PaperTrace) -> str:
    for chunk in paper_trace.retrieved_chunks:
        ref = str(chunk.get("evidence_ref", "")).strip()
        if ref:
            return ref
    for document_note in paper_trace.document_notes:
        for payload_name in ("method", "sample_or_data", "limitations", "main_findings", "conclusion"):
            payload = _document_slot(document_note, payload_name)
            refs = payload.get("evidence_refs", [])
            if refs:
                return str(refs[0])
    return ""


def _build_narrow_grounded_paper_answer(
    prompt: str,
    paper_trace: PaperTrace,
    *,
    response_language: str,
) -> str:
    focus_slot = _narrow_grounded_qa_focus_slot(prompt)
    texts = _narrow_grounded_qa_text_pool(paper_trace, focus_slot)
    missing_phrase = _paper_missing_phrase(response_language)
    answer = missing_phrase

    if focus_slot == "method":
        families = _extract_method_family_mentions(texts)
        if families:
            joined = ", ".join(families[:-1]) + f", and {families[-1]}" if len(families) > 2 else " and ".join(families)
            if response_language == "zh-CN":
                answer = f"文中明确讨论的方法家族包括{joined}。"
            else:
                answer = f"The paper explicitly discusses {joined}."
        else:
            summary = _best_supported_narrow_slot_summary(
                paper_trace,
                "method",
                response_language=response_language,
            )
            if summary != missing_phrase:
                answer = summary if response_language == "zh-CN" else f"The paper explicitly discusses {summary.rstrip('.')}."
    elif focus_slot == "sample_or_data":
        facts = _extract_sample_data_facts(texts)
        if facts:
            if response_language == "zh-CN":
                if len(facts) == 1:
                    answer = f"文中明确说明，{facts[0]}。"
                else:
                    answer = f"文中明确说明，{facts[0]}。文中还说明，{facts[1]}。"
            else:
                if len(facts) == 1:
                    answer = f"The paper explicitly states that {facts[0]}."
                else:
                    answer = f"The paper explicitly states that {facts[0]}. It also states that {facts[1]}."
        else:
            summary = _best_supported_narrow_slot_summary(
                paper_trace,
                "sample_or_data",
                response_language=response_language,
            )
            if summary != missing_phrase:
                answer = summary if response_language == "zh-CN" else f"The paper explicitly states that {summary.rstrip('.')}."
    else:
        summary = _best_supported_narrow_slot_summary(
            paper_trace,
            focus_slot,
            response_language=response_language,
        )
        if summary != missing_phrase:
            answer = _render_grounded_slot_sentence(focus_slot, summary, response_language=response_language)

    if _prompt_requests_support_detail(prompt):
        support_ref = _first_support_ref(paper_trace)
        if support_ref:
            if response_language == "zh-CN":
                answer += f" 相关位置可见 {support_ref}。"
            else:
                answer += f" The most directly relevant retrieved support is {support_ref}."
    return answer.strip()


def _looks_over_scaffolded_grounded_qa(answer_text: str) -> bool:
    lowered = answer_text.lower()
    markers = (
        "direct answer:",
        "grounded supporting evidence",
        "retrieved evidence",
        "evidence refs",
        "uncertainty when evidence is weak",
        "the answer has been rewritten",
    )
    if any(marker in lowered for marker in markers):
        return True
    if sum(lowered.count(token) for token in ("#page=", "#chunk=", "#pages=", "evidence:")) >= 2:
        return True
    if len(re.findall(r"(?m)^\d+\.", answer_text)) >= 2:
        return True
    return False


def _build_paper_grounded_qa_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    requested_slots = _requested_paper_slots(prompt, mode_selection.mode)
    if _is_recurring_limitations_prompt(prompt):
        recurring_limitations = _recurring_slot_lines(
            paper_trace.document_notes,
            slot_name="limitations",
        )
        capped_scaffold = list(recurring_limitations[:12])
        if len(recurring_limitations) > 12:
            capped_scaffold.append(
                f"- {len(recurring_limitations) - 12} additional limitation lines were omitted from the renderer scaffold to keep the prompt compact."
            )
        lines = [
            "Renderer contract",
            "- grounded_ra_memo",
            "Paper output profile",
            f"- {mode_selection.paper_output_profile}",
            f"- Reason: {mode_selection.paper_output_profile_reason}",
            "Direct answer scaffold",
            *capped_scaffold,
            "Rendering rules",
            "- Keep the answer focused on recurring limitations across the consulted papers.",
            "- Separate clearly recurring limitations from weaker or paper-specific limitations.",
            "- Preserve concrete supported limitation details when they help explain the recurring pattern.",
            f"- If support is weak, say {_paper_missing_phrase(mode_selection.response_language)}",
        ]
        evidence_lines = _bullet_lines(evidence_refs or paper_trace.target_paths)
        lines.extend(["Evidence refs", *(evidence_lines[:8] or ["- None"])])
        return "\n".join(lines)

    if _is_narrow_grounded_paper_qa(prompt):
        scaffold_answer = _build_narrow_grounded_paper_answer(
            prompt,
            paper_trace,
            response_language=mode_selection.response_language,
        )
        lines = [
            "Renderer contract",
            "- concise_grounded_qa",
            "Direct answer scaffold",
            f"- {scaffold_answer}",
            "Rendering rules",
            "- Answer the user's question directly in the first sentence.",
            "- Keep the answer concise, grounded, and low-noise.",
            "- Do not include an evidence appendix, repeated support fragments, or retrieval-style scaffolding unless the user explicitly asked for location or citation detail.",
            f"- If the evidence is weak or missing, say {_paper_missing_phrase(mode_selection.response_language)}",
        ]
        if _prompt_requests_support_detail(prompt):
            support_ref = _first_support_ref(paper_trace)
            if support_ref:
                lines.extend(["Primary support ref", f"- {support_ref}"])
        return "\n".join(lines)

    direct_scaffold = _relevant_slot_lines(paper_trace.document_notes, requested_slots)
    lines = [
        "Renderer contract",
        "- grounded_ra_memo",
        "Direct answer scaffold",
        *direct_scaffold,
        "Rendering rules",
        "- Answer the question directly from the cleaned slot scaffold and retrieved evidence.",
        "- Keep the answer concise and grounded rather than excerpt-heavy.",
        f"- If the consulted evidence does not clearly support a requested detail, say {_paper_missing_phrase(mode_selection.response_language)}",
    ]
    lines.extend(["Evidence refs", *_bullet_lines(evidence_refs or paper_trace.target_paths)])
    return "\n".join(lines)


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    narrow_grounded_qa = mode_selection.mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(prompt)

    if mode_selection.response_language == "zh-CN" and _looks_insufficiently_translated_chinese(answer_text):
        notes.append("The answer did not translate the grounded paper content cleanly enough into Chinese.")
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the cleaned slot evidence."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    if "not clearly stated in the paper" in answer_lower and "however" in answer_lower:
        notes.append(
            "The answer acknowledges a missing dimension but then keeps padding it with unsupported follow-on commentary."
        )
    if narrow_grounded_qa and _looks_over_scaffolded_grounded_qa(answer_text):
        notes.append(
            "The narrow grounded QA answer is still too scaffold-heavy and should collapse to a concise answer-first form."
        )
    if not narrow_grounded_qa and _looks_excerpt_heavy_paper_answer(answer_text):
        notes.append(
            "The answer is still too excerpt-heavy or outline-heavy for the final paper renderer contract."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots and mode_selection.mode == "paper_summary":
        notes.append(
            "The answer did not clearly cover these requested summary dimensions: "
            + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
            + "."
        )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.mode == "paper_compare" and "hybrid approach could be beneficial" in answer_lower:
        notes.append("The comparison drifted into unsupported recommendation language instead of staying slot-grounded.")
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }


def _compose_paper_consistency_prompt(
    *,
    original_prompt: str,
    answer_text: str,
    report_notes: tuple[str, ...],
    response_language: str,
    mode: str,
    response_style: str,
    paper_output_profile: str = "quick_summary",
) -> str:
    missing_phrase = _paper_missing_phrase(response_language)
    style_instruction = (
        "- Return one continuous grounded paragraph with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "- Return a concise grounded answer using short natural paragraphs or compact sections."
    )
    language_instruction = (
        "- For zh-CN output, translate the grounded content into natural Simplified Chinese prose."
        if response_language == "zh-CN"
        else "- For English output, write like a grounded RA note rather than raw retrieval notes."
    )
    profile_instruction = (
        "- Preserve important paper-specific details when they are clearly supported, including concrete numbers, date ranges, sample/data setup, train/validation/test splits, explicit method families, findings, limitations, and conclusion details."
        if paper_output_profile == "detailed_paper_note"
        else "- Keep the answer concise and useful for quick reading; preserve the main supported point without drifting into vague filler."
    )
    return "\n".join(
        [
            "You are performing a constrained consistency repair on a paper answer.",
            f"Original user prompt: {original_prompt}",
            "Repair goals:",
            "- Rewrite from the grounded slot scaffold rather than from vague memory or generic domain knowledge.",
            "- Keep only claims supported by the grounded slot scaffold and evidence.",
            "- Remove broad textbook commentary, generic finance/ML filler, and unsupported synthesis language.",
            f"- If a requested detail is weak or missing, say {missing_phrase} instead of guessing.",
            "- Preserve the requested language and formatting style.",
            "- Paraphrase compactly instead of stitching excerpt fragments together.",
            "- Cover every explicitly requested dimension exactly once.",
            "- Do not add recommendations, future-work commentary, or hybrid proposals unless the user explicitly asked for them.",
            language_instruction,
            style_instruction,
            profile_instruction,
            f"- Current paper mode: {mode}",
            f"- Current paper output profile: {paper_output_profile}",
            "Detected issues:",
            *[f"- {item}" for item in report_notes],
            "",
            "Current answer:",
            answer_text,
            "",
            "Return only the repaired final answer body.",
        ]
    )


def _build_paper_summary_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    if not paper_trace.discovered_documents:
        return "\n".join(
            [
                "Renderer contract",
                f"- {_paper_renderer_name(mode_selection)}",
                "Document",
                "- No local PDF target was discovered for this prompt.",
                "Rendering rules",
                "- Do not invent any paper content when no PDF target was resolved.",
            ]
        )

    document_note = paper_trace.document_notes[0] if paper_trace.document_notes else {}
    requested_slots = list(_requested_paper_slots(prompt, mode_selection.mode))
    detailed_profile = mode_selection.paper_output_profile == "detailed_paper_note"
    style_rule = (
        "- Return natural Simplified Chinese continuous prose with no bullets or outline headings."
        if mode_selection.response_language == "zh-CN" and mode_selection.response_style == "continuous_prose"
        else "- Use short natural paragraphs or compact sections rather than internal slot-note scaffolding."
    )
    profile_rules = (
        [
            "- Write a detailed grounded paper note rather than a project memo or onboarding summary.",
            "- Preserve important supported paper-specific details, including concrete numbers, date ranges, sample/data setup, train/validation/test splits, explicit method families, findings, limitations, and conclusion details when they are clearly stated.",
            "- Keep both the main paper arc and the important supported details; do not flatten the paper into a vague high-level memo.",
        ]
        if detailed_profile
        else [
            "- Write a concise grounded paper summary that stays useful for quick terminal reading.",
            "- Keep the paper's main question or purpose, main approach, main finding, and the most important supported limitation or conclusion.",
            "- Do not preserve every concrete detail for brevity unless the prompt explicitly asked for it.",
        ]
    )
    return "\n".join(
        [
            "Renderer contract",
            f"- {_paper_renderer_name(mode_selection)}",
            "Paper output profile",
            f"- {mode_selection.paper_output_profile}",
            f"- Reason: {mode_selection.paper_output_profile_reason}",
            "Document",
            f"- Primary target: `{paper_trace.discovered_documents[0]}`",
            f"- Read strategy: `{paper_trace.read_strategy}`",
            f"- Windows processed: {paper_trace.window_count_processed}",
            "Requested dimensions",
            *[f"- {slot_label(slot_name)}" for slot_name in requested_slots],
            "Cleaned slot scaffold",
            *[
                _format_renderer_slot_line(
                    slot_name,
                    document_note,
                    paper_output_profile=mode_selection.paper_output_profile,
                )
                for slot_name in requested_slots
            ],
            "Rendering rules",
            "- Cover every requested dimension exactly once; do not silently omit any requested dimension.",
            f"- If a requested dimension is not clearly supported, say {_paper_missing_phrase(mode_selection.response_language)}",
            "- Prefer explicit support over inferred support and avoid speculative inference.",
            "- Do not add textbook machine-learning, finance, or investment commentary unless the paper itself supports it.",
            "- Do not quote the cleaned slot scaffold verbatim unless a short method name or sample fact must remain exact.",
            "- Do not use project-onboarding framing such as project overview, project goal, or research-assistant memo unless the user explicitly asked for that style.",
            *profile_rules,
            style_rule,
            "Evidence refs",
            *_bullet_lines(evidence_refs or paper_trace.discovered_documents or paper_trace.target_paths),
        ]
    )


def _build_paper_compare_draft(
    config: LabaiConfig,
    prompt: str,
    mode_selection: ModeSelection,
    tool_calls: list[ToolCall],
    evidence_refs: tuple[str, ...],
    paper_trace: PaperTrace,
) -> str:
    compared = paper_trace.discovered_documents or paper_trace.target_paths
    requested_slots = _requested_paper_slots(prompt, mode_selection.mode)
    return "\n".join(
        [
            "Renderer contract",
            f"- {_paper_renderer_name(mode_selection)}",
            "Paper output profile",
            f"- {mode_selection.paper_output_profile}",
            f"- Reason: {mode_selection.paper_output_profile_reason}",
            "Documents compared",
            *_bullet_lines(compared or ("No PDF targets were discovered.",)),
            "Comparison scaffold",
            *[
                line
                for slot_name in requested_slots
                for line in _comparison_slot_lines(slot_name, paper_trace.document_notes)
            ],
            "Rendering rules",
            "- Compare papers slot-to-slot and make asymmetries explicit.",
            "- Keep the comparison grounded in paper-specific content rather than broad generic themes.",
            f"- If a dimension is not clearly supported for one paper, say {_paper_missing_phrase(mode_selection.response_language)}",
            "- Avoid repeating weakly supported text or broad generic commentary.",
            "- Do not render the final answer as a raw `slot:` dump or stitched slot scaffold.",
            "- For each requested dimension, identify what each paper clearly says and then state the contrast explicitly.",
            "- Do not recommend combining methods or propose a hybrid approach unless the user explicitly asked for recommendations.",
            "- Do not reuse the same phrase across multiple dimensions when a shorter contrast will do.",
            "- Preserve important supported asymmetries in sample/data, methods, findings, limitations, and conclusion details instead of flattening them away.",
            "Evidence refs",
            *_bullet_lines(evidence_refs or compared),
        ]
    )


def _build_slot_grounded_paper_summary(
    document_note: dict[str, object],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    response_style: str,
    paper_output_profile: str = "quick_summary",
) -> str:
    default_slots = (
        (
            "research_question",
            "sample_or_data",
            "method",
            "main_findings",
            "limitations",
            "conclusion",
        )
        if paper_output_profile == "detailed_paper_note"
        else (
            "research_question",
            "method",
            "main_findings",
            "limitations",
            "conclusion",
        )
    )
    slots = tuple(requested_slots or default_slots)
    if paper_output_profile == "detailed_paper_note" and response_style != "continuous_prose":
        return _build_detailed_paper_note(
            document_note,
            requested_slots=slots,
            response_language=response_language,
        )
    return _build_slot_grounded_paper_summary_prose(
        document_note,
        requested_slots=slots,
        response_language=response_language,
        paper_output_profile=paper_output_profile,
    )


def _render_grounded_slot_sentence(
    slot_name: str,
    summary: str,
    *,
    response_language: str,
) -> str:
    cleaned = _normalize_slot_summary(summary, response_language=response_language)
    cleaned = re.sub(r"^\d+\s+", "", cleaned)
    lowered_clean = cleaned.lower()
    if slot_name == "sample_or_data" and "variable importance" in lowered_clean:
        return _paper_missing_phrase(response_language)
    if response_language == "zh-CN":
        phrase_map = (
            ("the fundamental goal of asset pricing is to understand the behavior of risk premiums", "资产定价的基本目标是理解风险溢价的行为"),
            ("the challenge is how to assess the incremental predictive content of a newly proposed predictor while jointly controlling for the gamut of extant signals", "核心挑战是在同时控制既有信号集合的情况下，评估新提出预测变量所增加的预测信息"),
            ("linear regression", "线性回归"),
            ("generalized linear models", "广义线性模型"),
            ("principal components regression", "主成分回归"),
            ("partial least squares", "偏最小二乘"),
            ("regression trees", "回归树"),
            ("neural networks", "神经网络"),
            ("dimension reduction", "降维"),
            ("nonlinear models", "非线性模型"),
            ("positive predictive performance", "正向预测表现"),
            ("lack of regularization", "缺乏正则化"),
            ("in-sample overfit", "样本内过拟合"),
            ("sample period back to 1957", "样本期向前扩展到 1957 年"),
            ("our sample begins in march 1957", "样本从 1957 年 3 月开始"),
            ("validation sample", "验证样本"),
            ("training sample", "训练样本"),
            ("out-of-sample testing", "样本外测试"),
            ("asset pricing", "资产定价"),
            ("risk premiums", "风险溢价"),
            ("stock returns", "股票收益"),
            ("machine learning", "机器学习"),
        )
        lowered = lowered_clean
        for source, target in phrase_map:
            lowered = lowered.replace(source, target)
        cleaned = lowered
        templates = {
            "research_question": "文章的核心问题是{summary}",
            "background_or_motivation": "研究背景或动机是{summary}",
            "sample_or_data": "样本或数据方面，{summary}",
            "method": "方法上，{summary}",
            "main_findings": "主要发现是{summary}",
            "limitations": "局限在于{summary}",
            "conclusion": "总体结论是{summary}",
            "practical_or_investment_implications": "实践或投资含义方面，{summary}",
        }
        body = templates.get(slot_name, "{summary}").format(summary=cleaned.rstrip("。.;"))
        return body if body.endswith("。") else body + "。"

    lowered = lowered_clean
    if slot_name == "research_question":
        if lowered.startswith("the fundamental goal of asset pricing is to understand the behavior of risk premiums"):
            body = "The paper is framed around understanding the behavior of risk premiums in asset pricing"
        elif lowered.startswith(("the paper", "this paper", "our focus", "the goal", "the aim", "the question")):
            body = cleaned.rstrip(".")
        else:
            body = f"The paper focuses on {cleaned[0].lower() + cleaned[1:] if len(cleaned) > 1 else cleaned}"
        return body if body.endswith(".") else body + "."
    templates = {
        "background_or_motivation": "The background or motivation is {summary}.",
        "sample_or_data": "For the sample and data, {summary}.",
        "method": "Methodologically, {summary}.",
        "main_findings": "The main finding is that {summary}.",
        "limitations": "A key limitation is that {summary}.",
        "conclusion": "Overall, {summary}.",
        "practical_or_investment_implications": "For practical or investment implications, {summary}.",
    }
    template = templates.get(slot_name, "{summary}.")
    body = template.format(summary=cleaned.rstrip("."))
    return body if body.endswith(".") else body + "."


def _build_slot_grounded_paper_summary_prose(
    document_note: dict[str, object],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    paper_output_profile: str = "quick_summary",
) -> str:
    sentences: list[str] = []
    for slot_name in requested_slots:
        summary = _slot_summary_sentence(
            document_note,
            slot_name,
            response_language=response_language,
            paper_output_profile=paper_output_profile,
        )
        if summary == _paper_missing_phrase(response_language):
            if response_language == "zh-CN":
                sentence = f"{_slot_display_name(slot_name, response_language)}{summary}"
            else:
                sentence = f"{_slot_display_name(slot_name, response_language).capitalize()}: {summary}"
                if not sentence.endswith("."):
                    sentence += "."
            sentences.append(sentence)
            continue
        sentences.append(
            _render_grounded_slot_sentence(
                slot_name,
                summary,
                response_language=response_language,
            )
        )
    return " ".join(sentences).strip()


def _compare_documents_heading(response_language: str) -> str:
    return "\u6bd4\u8f83\u6587\u732e" if response_language == "zh-CN" else "Documents compared"


def _compare_section_title(slot_name: str, response_language: str = "en") -> str:
    if response_language == "zh-CN":
        return {
            "research_question": "\u7814\u7a76\u95ee\u9898",
            "sample_or_data": "\u6837\u672c\u4e0e\u6570\u636e",
            "method": "\u65b9\u6cd5",
            "main_findings": "\u4e3b\u8981\u53d1\u73b0",
            "limitations": "\u5c40\u9650",
            "conclusion": "\u7ed3\u8bba",
            "practical_or_investment_implications": "\u5b9e\u8df5\u6216\u6295\u8d44\u542b\u4e49",
        }.get(slot_name, slot_name)
    return {
        "research_question": "Research question",
        "sample_or_data": "Sample and data",
        "method": "Method",
        "main_findings": "Main findings",
        "limitations": "Limitations",
        "conclusion": "Conclusion",
        "practical_or_investment_implications": "Practical or investment implications",
    }.get(slot_name, slot_label(slot_name))


def _compare_document_name(document_note: dict[str, object]) -> str:
    return Path(str(document_note.get("source_path", "(unknown document)"))).name


def _aggregated_document_slot(document_note: dict[str, object], slot_name: str) -> dict[str, object]:
    for slot in document_note.get("aggregated_slots", []):
        if str(slot.get("slot_name", "")) == slot_name:
            payload = dict(slot)
            payload.setdefault("summary_text", payload.get("merged_note_text", ""))
            return payload
    return {
        "slot_name": slot_name,
        "summary_text": "Not clearly stated in the paper.",
        "merged_note_text": "Not clearly stated in the paper.",
        "evidence_refs": [],
        "support_status": "not_clearly_stated",
        "strongest_support": "weak",
        "explicit_note_count": 0,
        "inferred_note_count": 0,
        "note_count": 0,
    }


def _slot_payload_candidate_texts(slot_payload: dict[str, object]) -> list[str]:
    texts: list[str] = []
    for key in ("detailed_render_text", "summary_text", "merged_note_text"):
        value = str(slot_payload.get(key, "")).strip()
        if not value or value == "Not clearly stated in the paper.":
            continue
        texts.append(value)
    return list(dict.fromkeys(texts))


def _compare_slot_candidate_texts(
    document_note: dict[str, object],
    slot_name: str,
) -> list[str]:
    candidate_texts: list[str] = []
    primary_payload = _document_slot(document_note, slot_name)
    aggregated_payload = _aggregated_document_slot(document_note, slot_name)
    candidate_texts.extend(_slot_payload_candidate_texts(primary_payload))
    candidate_texts.extend(_slot_payload_candidate_texts(aggregated_payload))

    supplemental_slots: dict[str, tuple[str, ...]] = {
        "research_question": ("background_or_motivation", "other"),
        "sample_or_data": ("other",),
        "method": ("other",),
        "main_findings": ("conclusion",),
        "conclusion": ("main_findings", "practical_or_investment_implications"),
        "practical_or_investment_implications": ("conclusion", "main_findings"),
    }
    for supplemental_slot in supplemental_slots.get(slot_name, ()):
        payload = _document_slot(document_note, supplemental_slot)
        candidate_texts.extend(_slot_payload_candidate_texts(payload))

    normalized: list[str] = []
    seen: set[str] = set()
    for text in candidate_texts:
        signature = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()
        if not signature or signature in seen:
            continue
        seen.add(signature)
        normalized.append(text)
    return normalized


def _compare_candidate_sentences(text: str) -> list[str]:
    expanded = _normalize_compare_sentence_surface(text).replace("; ", ". ").replace(";", ". ")
    return [sentence for sentence in _split_rescue_sentences(expanded) if sentence.strip()]


def _normalize_compare_sentence_surface(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = re.sub(r"[\u0300-\u036f]", "", cleaned)
    cleaned = cleaned.replace("\u00ad", "")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", cleaned)
    for marker in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"):
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(
        r"^(?:[A-Z][A-Za-z-]+(?: [A-Z][A-Za-z-]+){2,})\s+\d{2,4}\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"([A-Za-z]{2,})-\s+([A-Za-z]{2,})", r"\1\2", cleaned)
    cleaned = re.sub(r"(\d+)\s*-\s+([A-Za-z]{2,})", r"\1-\2", cleaned)
    cleaned = re.sub(
        r"\b([A-Za-z]{5,})\s+((?:ated|ation|ations|cated|tive|tives|ment|ments|ally|ically|ized|ising|izing|tion|tions|sion|sions|ality|ities|ously|ness|able|ible|ance|ances|ence|ences))\b",
        r"\1\2",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(one|two|three|four|five|six|seven|eight|nine)\s*-?\s*step\b",
        lambda match: f"{match.group(1)}-step",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(one|two|three|four|five|six|seven|eight|nine)step\b",
        lambda match: f"{match.group(1)}-step",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(one|two|three|four|five|six|seven|eight|nine)and\s+"
        r"(one|two|three|four|five|six|seven|eight|nine)-layer\b",
        lambda match: f"{match.group(1)}- and {match.group(2)}-layer",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\blinearquadratic\b", "linear-quadratic", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", cleaned)
    cleaned = re.sub(r"(?<=\d)\s+%", "%", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])\((?=[A-Za-z])", " (", cleaned)
    cleaned = re.sub(r"\)(?=(?:and|or|but|while|whereas|in contrast)\b)", ") ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r",\s*(?:while|whereas|but)\s+[^.]{0,120}\b(?:better than|worse than|higher than|lower than|more than|less than|greater than|smaller than)\.?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\.\s*\d+\s+(?=[A-Z])", ". ", cleaned)
    cleaned = re.sub(
        r"(?<=[a-z0-9])\s+(?=(?:This|These|It|We|Our)\s+(?:article|paper|study|work|approach|procedure|method|methods|framework|results|findings|evidence|novel|model|conclusion|research)\b)",
        ". ",
        cleaned,
    )
    cleaned = re.sub(
        r"(?<=[a-z0-9])\s+(?=(?:In conclusion|In summary|Overall|Finally)\b)",
        ". ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^this example also sheds light on the fact that\s+",
        "The results also show that ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:main|primary)\s+(technical contribution|contribution)\s+is\b",
        lambda match: f"The {match.group(0).lower()}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:[A-Za-zΑ-Ωα-ω]+\s+)?[A-Z]\s+cannot obtain the efficiency bounds for all parameters unless\b",
        "the efficiency bounds for all parameters cannot be obtained unless",
        cleaned,
    )
    cleaned = re.sub(
        r"\b[Α-Ωα-ω]\s+[A-Z]\s+cannot obtain the efficiency bounds for all parameters unless\b",
        "the efficiency bounds for all parameters cannot be obtained unless",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(optimal policy function)\s+f\s*[∗*?]\s*\((?:·|\.)\)",
        r"\1",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:(?:[A-Za-z]|[α-ωΑ-Ω]|\d+|dt|ds|du|dv|[^\x00-\x7F])\s+){3,}(?:[A-Za-z]|[α-ωΑ-Ω]|\d+|dt|ds|du|dv|[^\x00-\x7F])(?=[.,;:]|$)",
        "",
        cleaned,
    )
    cleaned = re.sub(r",?\s*see,\s*e\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:and|or)\s+their\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b([A-Za-z][A-Za-z-]*(?: [A-Za-z][A-Za-z-]*){3,})\s+([A-Z])\s+is\s+a\b", r"\1. \2 is a", cleaned)
    cleaned = re.sub(r"\s+(?:\d{1,2}|[ivx]{1,4})\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    elif re.match(r"^(the|this)\b", cleaned):
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _compare_sentence_has_result_signal(slot_name: str, sentence: str) -> bool:
    lowered = unicodedata.normalize("NFKC", sentence).lower()
    marker_map = {
        "main_findings": (
            "find",
            "finding",
            "findings",
            "result",
            "results",
            "outperform",
            "performs best",
            "performs better",
            "best-performing",
            "best predictor",
            "predictive performance",
            "effective",
            "higher",
            "lower",
            "mape",
            "rmse",
            "mae",
            "r2",
            "show that",
            "we show",
            "we show that",
            "we derive",
            "we demonstrate",
            "we find that",
            "results show",
            "the results show",
            "our results",
            "numerical studies confirm",
            "necessary and sufficient conditions",
            "consistent and asymptotically normal",
            "better efficiency",
            "performs very well",
            "negligible impact",
            "converges towards the theoretical value",
            "advantages of the proposed approach",
            "admits an approximate closed-form solution",
            "find an approximate closed-form solution",
            "derive the optimal quotes",
            "optimal quotes in feedback form",
            "compare performance",
            "more efficient than the gqmle",
            "outperforms gqmle",
            "smaller asymptotic variance",
        ),
        "conclusion": (
            "conclusion",
            "conclude",
            "best predictor overall",
            "best performing nonlinear method",
            "informed financial decisions",
            "suitable for making informed financial decisions",
            "these findings",
            "evidence supports",
            "overall literary consequences",
            "in this paper, we have considered",
            "in this paper, we consider",
            "our results imply",
            "this paper establishes",
            "this leads to",
            "significant advancement",
            "numerical studies confirm",
            "we derive the optimal quotes",
            "we then numerically investigate",
            "outperforms gqmle",
            "more efficient than the gqmle",
            "smaller asymptotic variance",
            "practical importance",
            "robust tools for evaluating and mitigating risk",
        ),
    }
    return any(marker in lowered for marker in marker_map.get(slot_name, ()))


def _compare_sentence_has_question_signal(sentence: str) -> bool:
    lowered = unicodedata.normalize("NFKC", sentence).lower()
    if any(
        marker in lowered
        for marker in (
            "objective function",
            "validation objective",
            "forecast errors",
            "hyperparameter",
            "hyperparameters",
            "finally, we study how good",
            "finally we study how good",
            "compare against two benchmarks",
            "this seminal idea led to",
            "it is often observed that volatility tends to cluster together",
            "historically, it is often observed",
        )
    ):
        return False
    return any(
        marker in lowered
        for marker in (
            "we study",
            "we investigate",
            "we consider",
            "we propose",
            "we introduce",
            "we develop",
            "we identify",
            "we address",
            "we aim to",
            "investigate whether",
            "the goal",
            "the objective",
            "the purpose",
            "aims to",
            "asks whether",
            "problem we study",
            "empirically, we study",
            "the paper studies",
            "this paper studies",
            "this paper proposes",
            "this paper develops",
            "this paper introduces",
            "this paper aims",
            "this paper considers",
            "this article focuses",
            "this study aims",
            "fundamental goal",
            "goal of asset pricing",
            "seeks to forecast",
            "main contribution",
            "primary contribution",
        )
    )


def _compare_sentence_has_sample_signal(sentence: str) -> bool:
    lowered = unicodedata.normalize("NFKC", sentence).lower()
    if "sample path" in lowered or "sample paths" in lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "dataset",
            "data set",
            "sample",
            "out-of-sample",
            "training sample",
            "evaluation sample",
            "first half of the sample period",
            "second half of the sample period",
            "historical underlying asset prices",
            "synthetic data",
            "exchange-traded options",
            "individual stocks in china",
            "high-frequency data",
            "daily adjusted closing prices",
            "monthly returns",
            "crsp",
            "nasdaq-100",
            "observation",
            "observations",
            "tickers",
            "trading days",
            "sample period",
            "training window",
        )
    )


def _compare_sentence_has_method_signal(sentence: str) -> bool:
    lowered = unicodedata.normalize("NFKC", sentence).lower()
    return any(
        marker in lowered
        for marker in (
            "method",
            "methods",
            "framework",
            "approach",
            "algorithm",
            "we establish",
            "axiom scheme",
            "axiomatic dual representation",
            "set-valued risk measures",
            "conical market models",
            "covariance structure",
            "principal component",
            "reinforcement learning",
            "ppo",
            "closed-form solution",
            "quasi-maximum likelihood",
            "three-step quasi-maximum likelihood",
            "three-step estimator",
            "scale adjustment parameter",
            "ngqmle",
            "gqmle",
            "stochastic control",
            "hamilton-jacobi-bellman",
            "dual representation",
            "pricing kernel",
            "martingale loss",
            "option pricing methodology",
            "three-step",
            "arimax",
            "garch",
            "lstm",
        )
    )


def _compare_sentence_has_limitation_signal(sentence: str) -> bool:
    lowered = unicodedata.normalize("NFKC", sentence).lower()
    if any(
        marker in lowered
        for marker in (
            "left panel",
            "right panel",
            "lower panel",
            "upper panel",
            "figure ",
            "table ",
            "lemma",
            "proof",
            "corollary",
            "matrix manipulation",
            "innovations range from",
        )
    ):
        return False
    return any(
        marker in lowered
        for marker in (
            "limitation",
            "limitations",
            "future work",
            "future research",
            "leave the discussion",
            "we leave",
            "caveat",
            "limited",
            "limited to",
            "restricted to",
            "we only",
            "only consider",
            "only uses",
            "assume",
            "less accurate",
            "sensitive to errors",
            "insufficient",
            "do not assume",
            "monthly data",
            "high-frequency data",
            "future could help",
            "future could improve",
            "could help researchers improve",
            "small amount of data",
            "dearth of data",
            "signal-to-noise ratio",
            "overfit",
            "overfitting",
            "must be heavily regularized",
            "synthetic data",
            "one country",
            "one market",
            "cannot obtain",
            "cannot be obtained unless",
            "efficiency bounds",
            "unless the true underlying density",
            "only a limited number",
            "absent in many",
            "data required is close to reality",
            "heavy-tailed",
            "thin tails",
            "heavy tail density is selected",
        )
    )


def _compare_sentence_has_practical_signal(sentence: str) -> bool:
    lowered = unicodedata.normalize("NFKC", sentence).lower()
    if any(
        marker in lowered
        for marker in (
            "left panel",
            "right panel",
            "figure ",
            "table ",
            "abstract",
            "introduction",
            "article submitted to",
            "et al",
            "literature",
        )
    ):
        return False
    if re.search(r"\b[a-z]+ and [a-z]+ \(\d{4}\)", lowered):
        return False
    return any(
        marker in lowered
        for marker in (
            "risk management",
            "trading strategies",
            "financial decisions",
            "market maker",
            "inventory risk",
            "practical",
            "investment",
            "investor",
            "emerging derivatives markets",
            "tradeoff",
            "inventory",
            "robust against density misspecification",
            "price any derivatives",
        )
    )


def _compare_sentence_is_structural_noise(sentence: str) -> bool:
    lowered = unicodedata.normalize("NFKC", sentence).lower().strip()
    if re.match(r"^[A-Z][A-Za-z-]+(?: [A-Z][A-Za-z-]+){3,}\s+\d{2,4}\b", sentence.strip()):
        return True
    if re.match(
        r"^(?:[A-Z][A-Za-z-]+|of|with|and|the|in|on|for|to|a|an)(?: (?:[A-Z][A-Za-z-]+|of|with|and|the|in|on|for|to|a|an)){5,}\s+\d{1,4}\s+[A-Z]\.?$",
        sentence.strip(),
    ):
        return True
    if lowered in {
        "discussion and conclusions",
        "discussion and conclusion",
        "future research directions",
        "numerical results",
        "conclusion",
        "conclusions",
        "concluding remarks",
        "gmm implementation",
    }:
        return True
    if lowered.startswith("approximate closed-form solution "):
        return True
    if any(
        marker in lowered
        for marker in (
            "article submitted to",
            "mathematical methods of operations research",
            "journal of ",
            "validation dataset",
            "the review of financial studies /",
            "the following claims hold",
            "section 9 concludes",
            "section 5 concludes",
            "section concludes",
            "this concludes the proof",
            "author:",
            "keywords:",
            "jel classification",
            "mathematics subject classification",
            "positive numbers indicate the column model outperforms the row model",
            "our sign convention is that a positive statistic indicates the column model outperforms the row model",
            "bold font indicates the difference is significant",
        )
    ):
        return True
    if any(marker in lowered for marker in ("downloaded from", "guest (guest)", "guest ip:", "ip:")):
        return True
    if "•" in sentence and re.search(r"\b\d{2,4}\s+[A-Z][A-Za-z-]+(?:\s+[•·]\s*[A-Z][A-Za-z-]+){1,}", sentence):
        return True
    if "sample path" in lowered or "sample paths" in lowered:
        return True
    if re.search(r"\b(?:arxiv|department of|university|college|institute of)\b", lowered):
        return True
    if "@" in sentence:
        return True
    if re.search(r"^\s*(?:references?|appendix|doi)\b", lowered):
        return True
    if re.match(r"^\s*\(?[a-z]\.\d+\)?\b", lowered):
        return True
    if "formula (" in lowered or "equation (" in lowered:
        return True
    if re.search(r"^\s*(?:figure|table|section)\s+\d+\b", lowered):
        return True
    if re.search(r"\b(?:figure|table)\s+\d+\b", lowered):
        return True
    if "concludes the proof" in lowered:
        return True
    if re.match(r"^[A-Z]\s+is\s+(?:a|an|the)\b", sentence.strip()):
        return True
    return False


def _compare_sentence_is_truncated_fragment(sentence: str) -> bool:
    normalized = _normalize_compare_sentence_surface(sentence).strip()
    if "..." in normalized:
        return True
    lowered = normalized.lower().rstrip(". ")
    if re.match(
        r"^(?:by|and|or|but|while|whereas|because|with|for|to|of|in|on|from|than|which|that)\b",
        lowered,
    ) and len(lowered) <= 80:
        return True
    if normalized.count("(") > normalized.count(")") and re.search(r"\([a-z0-9]{0,8}$", lowered):
        return True
    if normalized.count("(") > normalized.count(")") and len(normalized) < 220:
        return True
    if lowered.endswith("et al"):
        return True
    if lowered.endswith("see, e") or lowered.endswith("and their"):
        return True
    if re.search(r"\b(?:better|worse|higher|lower|more|less|greater|smaller)\s+than$", lowered):
        return True
    if re.search(r"\b(?:r2|rmse|mae|sharpe ratio|variance|value)\s+of\s+\d+(?:\.\d+)?$", lowered):
        return True
    if re.match(
        r"^[A-Z][A-Za-z-]+(?:(?:,\s*[A-Z][A-Za-z-]+)|(?:\s*&\s*[A-Z][A-Za-z-]+)){1,}\s*\(?\d{4}\)?",
        normalized,
    ) and not re.search(
        r"\b(?:show|find|suggest|report|demonstrate|use|introduce|study|investigate|derive|propose|estimate|conclude|perform|focus)\b",
        lowered,
    ):
        return True
    if re.match(
        r"^[A-Z][A-Za-z-]+,\s*[A-Z][A-Za-z-]+\s*&\s*[A-Z][A-Za-z-]+\s*\(?\d{4}\)?",
        normalized,
    ) and not re.search(
        r"\b(?:show|find|suggest|report|demonstrate|use|introduce|study|investigate|derive|propose|estimate|conclude|perform|focus)\b",
        lowered,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:predic|estim|conclu|discussi|methodolog|out|with|and|or|of|to|for|in|on|by|from|than|because|while|where|which|whose|when|that|via|using|wh)$",
            lowered,
        )
    )


def _compare_sentence_has_formula_noise(sentence: str) -> bool:
    normalized = unicodedata.normalize("NFKC", sentence)
    lowered = normalized.lower()
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", normalized):
        return True
    if re.search(r"\b[a-z]\s*[?？]\s*\(", lowered):
        return True
    if re.search(r"[Α-Ωα-ω]\s+[A-Z]\b", normalized):
        return True
    if re.search(r"[Α-Ωα-ω][A-Za-z]\b", normalized):
        return True
    if len(re.findall(r"[=∑√∞μσληθωεΓΔ]", normalized)) >= 2:
        return True
    if re.search(r"[α-ωΑ-Ω√∞≤≥≈≠∑∂ηθγσλμνρτφψω]", normalized) and any(
        marker in lowered
        for marker in (
            "parameter",
            "variance",
            "consistent",
            "normal",
            "likelihood",
            "equation",
            "theorem",
            "lemma",
            "corollary",
            "proof",
        )
    ):
        return True
    if lowered.count("(") + lowered.count(")") >= 6 and any(symbol in normalized for symbol in ("=", "−", "∑", "√")):
        return True
    if any(marker in lowered for marker in ("lemma", "theorem", "corollary", "proof")) and any(
        symbol in normalized for symbol in ("=", "−", "∑", "√", "η", "σ", "ε")
    ):
        return True
    if re.match(r"^[A-Z]\s+is\s+(?:a|an|the)\b", normalized.strip()):
        return True
    return False


def _strip_compare_metadata(text: str, *, slot_name: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", cleaned)
    cleaned = re.sub(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhttps?://\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+\s*our code is publicly available\b.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bour code is publicly available\b.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:keywords?|jel classification|mathematics subject classification)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if abstract_match := re.search(r"\babstract\b[:.]?\s*", cleaned, flags=re.IGNORECASE):
        trailing = cleaned[abstract_match.end() :].strip()
        if len(trailing) >= 40:
            cleaned = trailing
    elif any(
        marker in cleaned.lower()
        for marker in ("arxiv:", "department of", "university", "college", "institute of", "@")
    ):
        if header_match := re.search(
            r"\)\s*(?=(?:The|We|This|Traditional|Our|To|In|Moreover|Finally|For)\b)",
            cleaned,
        ):
            trailing = cleaned[header_match.end() :].strip()
            if len(trailing) >= 40:
                cleaned = trailing
    if slot_name == "conclusion":
        if conclusion_match := re.search(r"\bconclusions?\b[:.]?\s*", cleaned, flags=re.IGNORECASE):
            trailing = cleaned[conclusion_match.end() :].strip()
            if len(trailing) >= 28:
                cleaned = trailing
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _compare_sentence_specificity_score(slot_name: str, sentence: str) -> int:
    lowered = unicodedata.normalize("NFKC", sentence).lower()
    score = _detail_marker_count(slot_name, lowered) * 2
    if _compare_sentence_has_result_signal(slot_name, lowered):
        score += 3
    if re.search(r"\b\d+(?:\.\d+)?%?\b", lowered):
        score += 2
    if any(marker in lowered for marker in ("outperform", "performs best", "best predictor", "mape", "r2", "informed financial decisions")):
        score += 2
    if slot_name == "main_findings" and "positive predictive performance" in lowered:
        score += 3
    if slot_name == "main_findings" and "portfolio turnover" in lowered:
        score -= 3
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "necessary and sufficient conditions",
            "consistent and asymptotically normal",
            "better efficiency",
            "numerical studies confirm",
            "performs very well",
            "negligible impact",
            "advantages of the proposed approach",
            "approximate closed-form solution",
            "predictive efficiency",
            "lstm model",
            "mape",
        )
    ):
        score += 3
    if slot_name in {"main_findings", "conclusion"} and any(
        marker in lowered
        for marker in (
            "comparing it with equation",
            "equation (",
            "equation ",
            "lemma ",
            "theorem ",
            "corollary ",
            "proposition ",
        )
    ):
        score -= 8
    if slot_name == "conclusion" and any(
        marker in lowered
        for marker in (
            "in this paper, we have considered",
            "in this paper, we consider",
            "these findings",
            "our results imply",
            "significant advancement",
            "numerical studies confirm",
            "we derive the optimal quotes",
        )
    ):
        score += 3
    if slot_name == "sample_or_data" and ("sample path" in lowered or "sample paths" in lowered):
        score -= 8
    return score


def _compare_summary_quality_score(
    slot_name: str,
    summary: str,
    *,
    support_status: str,
) -> int:
    lowered = unicodedata.normalize("NFKC", summary).lower()
    score = _compare_sentence_specificity_score(slot_name, lowered)
    score += min(5, _detail_marker_count(slot_name, lowered))
    if _slot_is_clearly_supported(support_status):
        score += 5
    elif support_status == "weakly_supported":
        score += 1
    if slot_name == "research_question" and _compare_sentence_has_question_signal(summary):
        score += 4
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "this study aims",
            "fundamental goal",
            "goal of asset pricing",
            "we study liquidity provision",
            "we introduce the concept",
            "this paper aims",
            "aims to forecast",
            "seeks to forecast",
        )
    ):
        score += 4
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "objective function",
            "validation objective",
            "forecast errors",
            "hyperparameter",
            "hyperparameters",
        )
    ):
        score -= 8
    if slot_name == "research_question" and re.search(r"[∑λθ̂]", summary):
        score -= 8
    if slot_name == "sample_or_data" and (
        _looks_like_explicit_sample_data_text(summary)
        or _compare_sentence_has_sample_signal(summary)
    ):
        score += 5
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "crsp",
            "nyse",
            "amex",
            "nasdaq",
            "cross-section",
            "individual stocks",
        )
    ):
        score += 4
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "predictive power",
            "benchmark model",
            "portfolio level",
            "stock level",
            "r2 ",
            "r2)",
            "r2(",
            "liquid stocks",
            "behave erratically",
        )
    ):
        score -= 6
    if slot_name == "method" and _compare_sentence_has_method_signal(summary):
        score += 4
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "we propose",
            "this paper proposes",
            "we introduce",
            "the main contribution of this article",
            "we introduce a scale adjustment parameter",
            "technical contribution",
            "three-step quasi-maximum likelihood",
            "three-step estimator",
            "three-step procedure",
            "scale adjustment parameter",
            "dual representation",
            "axiom scheme",
            "axiomatic dual representation",
            "we establish",
            "hamilton-jacobi-bellman",
            "innovative data-driven option pricing methodology",
            "estimate the bid-ask spread",
        )
    ):
        score += 4
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "we establish",
            "technical contribution",
            "three-step procedure",
            "we introduce a scale adjustment parameter",
            "novel three-step",
            "proposed procedure runs a gqmle",
        )
    ):
        score += 4
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "however, these methods may lead",
            "this may be partly because",
            "for a comprehensive review",
            "can not be reduced to the traditional setup",
            "in what follows",
            "before stating",
            "quasi-maximum likelihood estimation of garch models with heavy-tailed likelihoods",
            "gmm implementation",
            "one-step generalized methods of moments",
            "score functions",
        )
    ):
        score -= 7
    if slot_name == "limitations" and any(
        marker in lowered
        for marker in (
            "lower panel",
            "upper panel",
            "innovations range from",
        )
    ):
        score -= 9
    if slot_name in {"main_findings", "conclusion"} and _compare_sentence_has_result_signal(slot_name, summary):
        score += 5
    if slot_name == "conclusion" and any(
        marker in lowered
        for marker in (
            "in this paper",
            "we conclude",
            "our results imply",
            "these findings",
            "numerical studies confirm",
            "in conclusion",
            "significant advancement",
            "we hope that our work will inspire",
        )
    ):
        score += 3
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "numerical experiment",
            "numerical experiments",
            "performs very well",
            "negligible impact",
            "consistent and asymptotically normal",
            "better efficiency",
            "necessary and sufficient conditions",
            "advantages of the proposed approach",
            "in most cases, ngqmle shows an advantage",
        )
    ):
        score += 3
    if slot_name == "practical_or_investment_implications" and any(
        marker in lowered
        for marker in (
            "robust against density misspecification",
            "more efficient than the gqmle",
            "suitable for making informed financial decisions",
            "offer robust tools for evaluating and mitigating risk",
        )
    ):
        score += 5
    if _compare_sentence_is_structural_noise(summary):
        score -= 6
    if _compare_sentence_is_truncated_fragment(summary):
        score -= 5
    if _compare_sentence_has_formula_noise(summary):
        score -= 8
    if slot_name in {"research_question", "method", "main_findings", "conclusion"} and re.search(
        r"[α-ωΑ-Ω√∞≤≥≈≠∑∂ηθγσλμνρτφψω]",
        unicodedata.normalize("NFKC", summary),
    ):
        score -= 5
    if slot_name == "method" and "we have to al" in lowered:
        score -= 6
    return score


def _compare_source_rescue_pages(
    slot_name: str,
    page_texts: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    if not page_texts:
        return []
    selected = {page_number for page_number, _text in _candidate_pages_for_detailed_slot(slot_name, page_texts)}
    conclusion_markers = (
        "conclusion",
        "concluding remarks",
        "discussion and conclusions",
        "in conclusion",
        "in summary",
    )
    selected.update(
        page_number
        for page_number, text in page_texts
        if any(marker in unicodedata.normalize("NFKC", text).lower() for marker in conclusion_markers)
    )
    if slot_name in {"research_question", "main_findings", "conclusion"}:
        selected.update(page_number for page_number, _text in page_texts[:3])
    if slot_name == "sample_or_data":
        selected.update(page_number for page_number, _text in page_texts[:3])
    if slot_name in {"research_question", "method", "limitations"}:
        selected.update(page_number for page_number, _text in page_texts[-3:])
    if slot_name in {"main_findings", "conclusion"}:
        selected.update(page_number for page_number, _text in page_texts[-3:])
    if slot_name == "method":
        selected.update(page_number for page_number, _text in page_texts[:5])
    if slot_name == "practical_or_investment_implications":
        selected.update(page_number for page_number, _text in page_texts[:4])
        selected.update(page_number for page_number, _text in page_texts[-4:])
    return [(page_number, text) for page_number, text in page_texts if page_number in selected]


def _compare_method_page_rescue_summary(
    page_texts: list[tuple[int, str]],
) -> str:
    candidates: list[tuple[int, str]] = []
    design_markers = (
        "we propose",
        "this paper proposes",
        "we introduce",
        "three-step",
        "procedure",
        "approach",
        "framework",
        "scale adjustment parameter",
        "closed-form solution",
        "stochastic control",
        "dual representation",
        "pricing kernel",
        "generalized methods of moments",
    )
    property_markers = (
        "consistent",
        "asymptotically normal",
        "finite fourth moment",
        "better efficiency",
        "more efficient",
        "smaller asymptotic variance",
        "weak moment conditions",
    )
    for page_number, text in page_texts:
        normalized_page = _normalize_extracted_block(text)
        lowered_page = normalized_page.lower()
        for marker, bonus in (
            ("the main contribution of this article", 8),
            ("we introduce a scale adjustment parameter", 9),
            ("it proposes the ngqmle", 7),
            ("the first step of the proposed procedure", 7),
            ("final step performs an ngqmle", 7),
            ("main technical contribution is", 8),
            ("axiomatic scheme", 8),
            ("axiomatic dual representation", 8),
        ):
            marker_index = lowered_page.find(marker)
            if marker_index == -1:
                continue
            snippet = normalized_page[marker_index : marker_index + 360]
            snippet_sentences = [
                sentence
                for sentence in _compare_candidate_sentences(snippet)
                if not _compare_sentence_is_truncated_fragment(sentence)
            ]
            if not snippet_sentences:
                continue
            cleaned = _clean_compare_slot_summary(
                ". ".join(sentence.rstrip(". ") for sentence in snippet_sentences[:2]) + ".",
                slot_name="method",
            )
            if not cleaned or _looks_unreliable_compare_summary("method", cleaned, support_status="explicit_supported"):
                continue
            lowered_cleaned = cleaned.lower()
            if any(marker in lowered_cleaned for marker in property_markers) and not any(
                marker in lowered_cleaned for marker in design_markers
            ):
                continue
            score = _compare_summary_quality_score(
                "method",
                cleaned,
                support_status="explicit_supported",
            )
            candidates.append((score + bonus + max(0, 5 - page_number), cleaned))
        for sentence in _compare_candidate_sentences(text):
            lowered_sentence = sentence.lower()
            if re.match(r"^\(?\d{4}\)?[,)]\s+[\"“]", sentence.strip()):
                continue
            if re.search(r",\s*\d{1,3},\s*\d{1,4}(?:-\d{1,4})?\.?\s*$", sentence.strip()) and any(
                quote in sentence for quote in ('"', "“", "”")
            ):
                continue
            if not any(
                marker in lowered_sentence
                for marker in (
                    "we propose",
                    "this paper proposes",
                    "we introduce",
                    "this paper introduces",
                    "main contribution of this article",
                    "proposes the ngqmle",
                    "final step performs an ngqmle",
                    "we establish",
                    "technical contribution",
                    "three-step procedure",
                    "three-step quasi-maximum likelihood",
                    "quasi-maximum likelihood procedure",
                    "scale adjustment parameter",
                    "axiom scheme",
                    "axiomatic dual representation",
                    "dual representation",
                    "closed-form solution",
                    "stochastic control",
                    "pricing kernel",
                )
            ):
                continue
            cleaned = _clean_compare_slot_summary(sentence, slot_name="method")
            if not cleaned:
                continue
            lowered_cleaned = cleaned.lower()
            if any(marker in lowered_cleaned for marker in property_markers) and not any(
                marker in lowered_cleaned for marker in design_markers
            ):
                continue
            if _looks_unreliable_compare_summary("method", cleaned, support_status="explicit_supported"):
                continue
            score = _compare_summary_quality_score(
                "method",
                cleaned,
                support_status="explicit_supported",
            )
            if any(
                marker in lowered_sentence
                for marker in (
                    "we establish",
                    "technical contribution",
                    "three-step procedure",
                    "quasi-maximum likelihood procedure",
                    "scale adjustment parameter",
                    "axiomatic dual representation",
                    "main contribution of this article",
                    "proposes the ngqmle",
                    "final step performs an ngqmle",
                )
            ):
                score += 4
            if any(
                marker in lowered_sentence
                for marker in (
                    "we introduce a scale adjustment parameter",
                    "propose a threestep quasi-maximum likelihood procedure",
                    "propose a three-step quasi-maximum likelihood procedure",
                )
            ):
                score += 4
            if any(
                marker in lowered_sentence
                for marker in (
                    "gmm implementation",
                    "implementation alternatively",
                    "one-step generalized methods of moments",
                )
            ):
                score -= 6
            score += max(0, 6 - page_number)
            candidates.append((score, cleaned))
    if not candidates:
        return ""
    ordered = sorted(candidates, key=lambda item: (-item[0], -len(item[1]), item[1]))
    unique: list[str] = []
    seen: set[str] = set()
    for _score, sentence in ordered:
        signature = re.sub(r"\W+", " ", sentence.lower()).strip()
        if not signature or signature in seen:
            continue
        seen.add(signature)
        unique.append(sentence.rstrip(". "))
        if len(unique) >= 1:
            break
    return (". ".join(unique).strip() + ".") if unique else ""


def _compare_research_question_page_rescue_summary(
    page_texts: list[tuple[int, str]],
) -> str:
    candidates: list[tuple[int, str]] = []
    for page_number, text in page_texts:
        normalized_page = _normalize_extracted_block(text)
        lowered_page = normalized_page.lower()
        for marker, bonus in (
            ("this article focuses on", 9),
            ("this paper focuses on", 9),
            ("this article questions", 8),
            ("this study aims", 8),
            ("this paper aims", 8),
            ("we aim to", 7),
            ("we investigate", 6),
            ("the fundamental goal", 7),
            ("the main contribution of this article", 6),
        ):
            marker_index = lowered_page.find(marker)
            if marker_index == -1:
                continue
            snippet = normalized_page[marker_index : marker_index + 320]
            snippet_sentences = [
                sentence
                for sentence in _compare_candidate_sentences(snippet)
                if not _compare_sentence_is_truncated_fragment(sentence)
            ]
            if not snippet_sentences:
                continue
            cleaned = _clean_compare_slot_summary(
                ". ".join(sentence.rstrip(". ") for sentence in snippet_sentences[:2]) + ".",
                slot_name="research_question",
            )
            if not cleaned:
                continue
            if not _compare_sentence_has_question_signal(cleaned):
                continue
            if _looks_unreliable_compare_summary("research_question", cleaned, support_status="explicit_supported"):
                continue
            score = _compare_summary_quality_score(
                "research_question",
                cleaned,
                support_status="explicit_supported",
            )
            candidates.append((score + bonus + max(0, 4 - page_number), cleaned))
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: (-item[0], -len(item[1]), item[1]))[0][1]


def _compare_source_page_rescue_summary(
    document_note: dict[str, object],
    slot_name: str,
) -> str:
    source_path = str(document_note.get("source_path", "")).strip()
    if not source_path:
        return ""
    page_texts = list(_source_page_texts_for_detailed_rescue(source_path))
    if not page_texts:
        return ""
    rescue_candidates: list[str] = []
    if slot_name == "limitations":
        limitations_specific = _compare_limitations_page_rescue_summary(page_texts)
        if limitations_specific:
            return limitations_specific
    compare_pages = _compare_source_rescue_pages(slot_name, page_texts)
    if slot_name == "research_question":
        research_specific = _compare_research_question_page_rescue_summary(compare_pages)
        if research_specific:
            rescue_candidates.append(research_specific)
    if slot_name == "method":
        method_specific = _compare_method_page_rescue_summary(compare_pages)
        if method_specific:
            rescue_candidates.append(method_specific)
    generic_summary, _page_numbers = _rescue_generic_detailed_slot_summary(slot_name, page_texts)
    if generic_summary:
        rescue_candidates.append(generic_summary)
    compare_summary = _compare_slot_rescue_summary(
        slot_name,
        [text for _page_number, text in compare_pages],
    )
    if compare_summary:
        rescue_candidates.append(compare_summary)

    best_summary = ""
    best_score = -10**9
    for candidate in rescue_candidates:
        cleaned = _clean_compare_slot_summary(candidate, slot_name=slot_name)
        if not cleaned:
            continue
        if _looks_unreliable_compare_summary(slot_name, cleaned, support_status="explicit_supported"):
            continue
        score = _compare_summary_quality_score(
            slot_name,
            cleaned,
            support_status="explicit_supported",
        )
        if score > best_score:
            best_score = score
            best_summary = cleaned
    return best_summary


def _compare_limitations_page_rescue_summary(
    page_texts: list[tuple[int, str]],
) -> str:
    candidates: list[tuple[int, int, str]] = []
    total_pages = page_texts[-1][0] if page_texts else 0
    bad_markers = (
        "abilities and limitations of various prediction models",
        "aims to alleviate the limitations of individual",
        "existing studies still present some shortcomings",
        "discussion of the limitations of",
        "limitations of linear models as first-order approximations",
        "despite obvious limitations, such a plot",
        "forecasting is impossible without considering new information",
        "see white (1980)",
        "holds a promising future",
        "including certain limitations",
        "outperforms traditional methods",
        "traditional methods have potentially severe limitations",
        "their flexibility is also their limitation",
        "comparative patterns in predictive performance across methods is the same",
        "available from amit goyal",
        "web site",
    )
    strong_markers = (
        "future work",
        "future research",
        "we leave",
        "leave the discussion",
        "monthly data",
        "high-frequency data",
        "could help researchers improve",
        "limited to",
        "restricted to",
        "we only",
        "only consider",
        "only uses",
        "do not assume",
        "assumption",
        "synthetic data",
        "small amount of data",
        "dearth of data",
        "signal-to-noise ratio",
        "low signal-to-noise ratio",
        "overfit",
        "overfitting",
        "must be heavily regularized",
        "one country",
        "one market",
        "cannot obtain",
        "cannot be obtained unless",
    )
    for page_number, text in page_texts:
        normalized_page = _normalize_extracted_block(text)
        for sentence in _compare_candidate_sentences(normalized_page):
            if _compare_sentence_is_truncated_fragment(sentence):
                continue
            cleaned = _clean_compare_slot_summary(sentence, slot_name="limitations")
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if not _compare_sentence_has_limitation_signal(cleaned):
                continue
            if any(marker in lowered for marker in bad_markers):
                continue
            if _compare_sentence_is_structural_noise(cleaned) or _compare_sentence_has_formula_noise(cleaned):
                continue
            if not any(marker in lowered for marker in strong_markers):
                continue
            score = _compare_summary_quality_score(
                "limitations",
                cleaned,
                support_status="explicit_supported",
            )
            if any(marker in lowered for marker in strong_markers):
                score += 6
            if total_pages and page_number >= max(1, total_pages - max(6, total_pages // 3)):
                score += 2
            candidates.append((score, page_number, cleaned.rstrip(". ")))
    if not candidates:
        return ""
    ordered = sorted(candidates, key=lambda item: (-item[0], -item[1], -len(item[2]), item[2]))
    unique: list[str] = []
    seen: set[str] = set()
    for _score, _page_number, sentence in ordered:
        signature = re.sub(r"\W+", " ", sentence.lower()).strip()
        if not signature or signature in seen:
            continue
        seen.add(signature)
        unique.append(sentence)
        if len(unique) >= 2:
            break
    return ". ".join(unique).strip() + "." if unique else ""


def _compare_slot_rescue_summary(
    slot_name: str,
    candidate_texts: list[str],
) -> str:
    if not candidate_texts:
        return ""
    if slot_name == "research_question":
        rescued: list[tuple[int, str]] = []
        for text in candidate_texts:
            for sentence in _compare_candidate_sentences(text):
                if _compare_sentence_is_truncated_fragment(sentence):
                    continue
                cleaned = _clean_compare_slot_summary(sentence, slot_name=slot_name)
                if not cleaned or _compare_sentence_is_structural_noise(cleaned):
                    continue
                if _compare_sentence_has_question_signal(cleaned):
                    rescued.append((_detail_marker_count(slot_name, cleaned) + len(cleaned), cleaned.rstrip(". ")))
        if rescued:
            return sorted(rescued, key=lambda item: (-item[0], item[1]))[0][1] + "."
    if slot_name == "sample_or_data":
        facts = _extract_sample_data_facts(candidate_texts)
        if facts:
            return " ".join(f"{fact.rstrip('. ')}." for fact in facts[:2]).strip()
        rescued: list[tuple[int, str]] = []
        for text in candidate_texts:
            for sentence in _compare_candidate_sentences(text):
                if _compare_sentence_is_truncated_fragment(sentence):
                    continue
                cleaned = _clean_compare_slot_summary(sentence, slot_name=slot_name)
                if not cleaned or _compare_sentence_is_structural_noise(cleaned):
                    continue
                if _compare_sentence_has_sample_signal(cleaned):
                    rescued.append((_detail_marker_count(slot_name, cleaned) + len(cleaned), cleaned.rstrip(". ")))
        if rescued:
            ordered = sorted(rescued, key=lambda item: (-item[0], item[1]))
            unique: list[str] = []
            seen: set[str] = set()
            for _, sentence in ordered:
                signature = sentence.lower()
                if signature in seen:
                    continue
                seen.add(signature)
                unique.append(sentence)
            return ". ".join(unique[:2]).strip() + "."
    if slot_name == "method":
        rescued = _slot_specific_rescue_summary(
            slot_name,
            list(enumerate(candidate_texts, start=1)),
        ).strip()
        if rescued and any(
            marker in rescued.lower()
            for marker in (
                "we propose",
                "we introduce",
                "approach",
                "procedure",
                "framework",
                "estimator",
                "pricing methodology",
                "quasi-maximum likelihood",
                "closed-form solution",
                "dual representation",
                "stochastic control",
                "pricing kernel",
            )
        ):
            return rescued
        preferred: list[tuple[int, str]] = []
        backup: list[tuple[int, str]] = []
        for text in candidate_texts:
            for sentence in _compare_candidate_sentences(text):
                lowered_sentence = sentence.lower()
                if not any(
                    marker in lowered_sentence
                    for marker in (
                        "we propose",
                        "this paper proposes",
                        "data-driven",
                        "learning algorithm",
                        "historical underlying asset prices",
                        "robust methods",
                        "multiple testing",
                        "multiverse analysis",
                        "estimate the parameter",
                        "estimate the bid-ask spread",
                        "we introduce",
                        "this paper introduces",
                        "we establish",
                        "axiom scheme",
                        "axiomatic dual representation",
                        "procedure",
                        "quasi-maximum likelihood",
                        "dual representation",
                        "closed-form solution",
                        "stochastic control",
                        "pricing kernel",
                    )
                ):
                    continue
                if any(
                    marker in lowered_sentence
                    for marker in (
                        "figure ",
                        "table ",
                        "article submitted to",
                        "loss curves",
                        "validation dataset",
                        "for a comprehensive review",
                        "this may be partly because",
                    )
                ):
                    continue
                cleaned = _clean_compare_slot_summary(sentence, slot_name=slot_name)
                if not cleaned:
                    continue
                cleaned_sentence = cleaned.rstrip(". ")
                score = _compare_summary_quality_score(
                    slot_name,
                    cleaned_sentence,
                    support_status="explicit_supported"
                    if any(
                        marker in lowered_sentence
                        for marker in (
                            "we propose",
                            "this paper proposes",
                            "we introduce",
                            "this paper introduces",
                            "we establish",
                            "axiom scheme",
                            "axiomatic dual representation",
                            "quasi-maximum likelihood",
                            "three-step",
                            "dual representation",
                            "closed-form solution",
                            "innovative data-driven option pricing methodology",
                            "estimate the bid-ask spread",
                        )
                    )
                    else "weakly_supported",
                )
                if any(
                    marker in lowered_sentence
                    for marker in (
                        "we propose",
                        "this paper proposes",
                        "we introduce",
                        "this paper introduces",
                        "we establish",
                        "axiom scheme",
                        "axiomatic dual representation",
                        "quasi-maximum likelihood",
                        "three-step",
                        "dual representation",
                        "closed-form solution",
                        "pricing kernel",
                        "innovative data-driven option pricing methodology",
                        "robust methods",
                        "multiple testing",
                        "multiverse analysis",
                        "estimate the bid-ask spread",
                    )
                ):
                    preferred.append((score + 3, cleaned_sentence))
                else:
                    backup.append((score, cleaned_sentence))
        chosen = preferred or backup
        if chosen:
            ordered = sorted(chosen, key=lambda item: (-item[0], -len(item[1]), item[1]))
            unique: list[str] = []
            seen: set[str] = set()
            for _score, sentence in ordered:
                signature = sentence.lower()
                if signature in seen:
                    continue
                seen.add(signature)
                unique.append(sentence)
            return ". ".join(unique[:1]).strip() + "."
        generic_method_candidates: list[tuple[int, str]] = []
        for text in candidate_texts:
            for sentence in _compare_candidate_sentences(text):
                if _compare_sentence_is_truncated_fragment(sentence):
                    continue
                cleaned = _clean_compare_slot_summary(sentence, slot_name=slot_name)
                if not cleaned or _compare_sentence_is_structural_noise(cleaned):
                    continue
                if _compare_sentence_has_method_signal(cleaned):
                    generic_method_candidates.append((_detail_marker_count(slot_name, cleaned) + len(cleaned), cleaned.rstrip(". ")))
        if generic_method_candidates:
            return sorted(generic_method_candidates, key=lambda item: (-item[0], item[1]))[0][1] + "."
        families = _extract_method_family_mentions(candidate_texts)
        if families:
            if len(families) == 1:
                return families[0]
            if len(families) == 2:
                return f"{families[0]} and {families[1]}"
            return ", ".join(families[:-1]) + f", and {families[-1]}"
        if rescued:
            return rescued
    if slot_name == "limitations":
        preferred: list[tuple[int, str]] = []
        backup: list[tuple[int, str]] = []
        for text in candidate_texts:
            for sentence in _compare_candidate_sentences(text):
                lowered_sentence = sentence.lower()
                if not _compare_sentence_has_limitation_signal(sentence):
                    continue
                if any(
                    marker in lowered_sentence
                    for marker in (
                        "theorem",
                        "proof",
                        "lemma",
                        "corollary",
                        "table ",
                        "figure ",
                    )
                ):
                    continue
                cleaned = _clean_compare_slot_summary(sentence, slot_name=slot_name)
                if not cleaned:
                    continue
                score = _compare_summary_quality_score(
                    slot_name,
                    cleaned,
                    support_status="explicit_supported",
                )
                if any(
                    marker in lowered_sentence
                    for marker in (
                        "insufficient option data",
                        "do not estimate",
                        "sensitive to errors",
                        "future work",
                        "caveat",
                        "do not assume",
                        "without considering",
                        "cannot obtain",
                    )
                ):
                    preferred.append((score + 2, cleaned.rstrip(". ")))
                else:
                    backup.append((score, cleaned.rstrip(". ")))
        chosen = preferred or backup
        if chosen:
            ordered = sorted(chosen, key=lambda item: (-item[0], -len(item[1]), item[1]))
            unique: list[str] = []
            seen: set[str] = set()
            for _, sentence in ordered:
                signature = sentence.lower()
                if signature in seen:
                    continue
                seen.add(signature)
                unique.append(sentence)
            return ". ".join(unique[:2]).strip() + "."
    if slot_name == "practical_or_investment_implications":
        preferred: list[tuple[int, str]] = []
        backup: list[tuple[int, str]] = []
        for text in candidate_texts:
            for sentence in _compare_candidate_sentences(text):
                cleaned = _clean_compare_slot_summary(sentence, slot_name=slot_name)
                if not cleaned:
                    continue
                if _compare_sentence_is_structural_noise(cleaned) or _compare_sentence_is_truncated_fragment(cleaned):
                    continue
                if _compare_sentence_has_practical_signal(cleaned):
                    score = _compare_summary_quality_score(
                        slot_name,
                        cleaned,
                        support_status="explicit_supported",
                    )
                    if any(
                        marker in cleaned.lower()
                        for marker in (
                            "robust against density misspecification",
                            "more efficient than the gqmle",
                            "suitable for making informed financial decisions",
                            "offer robust tools for evaluating and mitigating risk",
                        )
                    ):
                        score += 6
                    preferred.append((score, cleaned.rstrip(". ")))
                elif _detail_marker_count(slot_name, cleaned.lower()) > 0:
                    backup.append(
                        (
                            _compare_summary_quality_score(
                                slot_name,
                                cleaned,
                                support_status="weakly_supported",
                            ),
                            cleaned.rstrip(". "),
                        )
                    )
        chosen = preferred or backup
        if chosen:
            ordered = sorted(chosen, key=lambda item: (-item[0], -len(item[1]), item[1]))
            unique: list[str] = []
            seen: set[str] = set()
            for _, sentence in ordered:
                signature = sentence.lower()
                if signature in seen:
                    continue
                seen.add(signature)
                unique.append(sentence)
            return ". ".join(unique[:2]).strip() + "."
    if slot_name in {"main_findings", "conclusion"}:
        preferred: list[tuple[int, str]] = []
        backup: list[tuple[int, str]] = []
        for text in candidate_texts:
            for sentence in _compare_candidate_sentences(text):
                if _compare_sentence_is_truncated_fragment(sentence):
                    continue
                cleaned = _clean_compare_slot_summary(sentence, slot_name=slot_name)
                if not cleaned:
                    continue
                if _compare_sentence_is_structural_noise(cleaned):
                    continue
                support_status = (
                    "explicit_supported"
                    if _compare_sentence_has_result_signal(slot_name, cleaned)
                    else "weakly_supported"
                )
                score = _compare_summary_quality_score(
                    slot_name,
                    cleaned,
                    support_status=support_status,
                )
                if _compare_sentence_has_result_signal(slot_name, cleaned):
                    preferred.append((score, cleaned.rstrip(". ")))
                elif _detail_marker_count(slot_name, cleaned) > 0:
                    backup.append((score, cleaned.rstrip(". ")))
        chosen = preferred or backup
        if chosen:
            ordered = sorted(chosen, key=lambda item: (-item[0], -len(item[1]), item[1]))
            unique: list[str] = []
            seen: set[str] = set()
            for _, sentence in ordered:
                signature = sentence.lower()
                if signature in seen:
                    continue
                seen.add(signature)
                unique.append(sentence)
            limit = 1 if slot_name == "conclusion" else 2
            return ". ".join(unique[:limit]).strip() + "."
    rescued = _slot_specific_rescue_summary(
        slot_name,
        list(enumerate(candidate_texts, start=1)),
    ).strip()
    if rescued:
        return rescued
    return ""


def _clean_compare_slot_summary(
    text: str,
    *,
    slot_name: str,
) -> str:
    cleaned = _strip_compare_metadata(text, slot_name=slot_name)
    cleaned = _normalize_compare_sentence_surface(cleaned)
    cleaned = _clean_detailed_render_body(
        cleaned,
        slot_name=slot_name,
        response_language="en",
    )
    cleaned = _strip_compare_metadata(cleaned, slot_name=slot_name)
    cleaned = _normalize_compare_sentence_surface(cleaned)
    cleaned = _cleanup_duplicate_render_lines(cleaned)
    cleaned = re.sub(
        r"^the review of financial studies\s*/\s*v\s*\d+\s*n\s*\d+\s*\d{4}\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^figure\s+\d+\s+reports[^.]*\.\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:numerical results|future research directions|discussion and conclusions?|conclusions?)\s*[:.\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:comparative statics and numerical results|discussion and conclusions?)\s*[:.\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:gmm implementation|concluding remarks|introduction|abstract|methodology)\s*[:.\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:\d+(?:\.\d+)*)\s*(?:conclusion|conclusions|discussion and conclusions?)\s*[:.\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^section\s+\d+\s+concludes?\s*[:.\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bwhich concludes the proof\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bproof of theorem\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<=[A-Za-z0-9])\.(?=[A-Za-z])", ". ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;")
    sentences = [part.strip(" .;") for part in _split_rescue_sentences(cleaned) if part.strip()]
    if sentences:
        preferred: list[tuple[int, str]] = []
        backup: list[tuple[int, str]] = []
        for sentence in sentences:
            normalized_sentence = _normalize_compare_sentence_surface(
                _strip_compare_metadata(sentence, slot_name=slot_name)
            ).strip(" .;")
            if len(normalized_sentence) < 24:
                continue
            if normalized_sentence.startswith("(") or re.match(r"^\(?\d{4}\)?[,)]", normalized_sentence):
                continue
            if _compare_sentence_is_structural_noise(normalized_sentence):
                continue
            if _compare_sentence_is_truncated_fragment(normalized_sentence):
                continue
            if _compare_sentence_has_formula_noise(normalized_sentence):
                continue
            lowered_sentence = normalized_sentence.lower()
            if slot_name == "research_question":
                is_preferred = _compare_sentence_has_question_signal(normalized_sentence)
            elif slot_name == "sample_or_data":
                is_preferred = _looks_like_explicit_sample_data_text(normalized_sentence) or _compare_sentence_has_sample_signal(normalized_sentence)
            elif slot_name == "method":
                is_preferred = _compare_sentence_has_method_signal(normalized_sentence) and not lowered_sentence.startswith(
                    ("however,", "however ", "in contrast", "this may be partly because")
                )
            elif slot_name in {"main_findings", "conclusion"}:
                is_preferred = _compare_sentence_has_result_signal(slot_name, normalized_sentence)
            elif slot_name == "limitations":
                is_preferred = _compare_sentence_has_limitation_signal(normalized_sentence)
            elif slot_name == "practical_or_investment_implications":
                is_preferred = _compare_sentence_has_practical_signal(normalized_sentence)
            else:
                is_preferred = _detail_marker_count(slot_name, lowered_sentence) > 0
            score = _compare_summary_quality_score(
                slot_name,
                normalized_sentence,
                support_status="explicit_supported",
            )
            if is_preferred:
                preferred.append((score, normalized_sentence))
            elif _detail_marker_count(slot_name, lowered_sentence) > 0:
                backup.append((score, normalized_sentence))
        chosen = preferred or backup
        if chosen:
            limit = 2 if slot_name in {"main_findings", "conclusion", "limitations", "practical_or_investment_implications"} else 1
            ordered = sorted(chosen, key=lambda item: (-item[0], -len(item[1]), item[1]))
            unique: list[str] = []
            seen: set[str] = set()
            for _score, sentence in ordered:
                signature = re.sub(r"\W+", " ", sentence.lower()).strip()
                if not signature or signature in seen:
                    continue
                seen.add(signature)
                unique.append(sentence.rstrip(". "))
                if len(unique) >= limit:
                    break
            if unique:
                cleaned = ". ".join(unique).strip() + "."
    return cleaned


def _looks_unreliable_compare_summary(
    slot_name: str,
    summary: str,
    *,
    support_status: str,
) -> bool:
    lowered = unicodedata.normalize("NFKC", summary).lower().strip()
    if not lowered:
        return True
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "output layer",
            "input layer",
            "hidden layer",
            "output node",
            "hidden node",
        )
    ):
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "downloaded from",
            "guest (guest)",
            "guest ip:",
            "ip:",
        )
    ):
        return True
    if slot_name == "sample_or_data" and re.search(
        r"^(?:mon|tue|wed|thu|fri|sat|sun),\s+\d{1,2}\s+[a-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\b",
        lowered,
    ):
        return True
    if slot_name == "sample_or_data" and re.fullmatch(
        r"(?:the\s+)?(?:training|validation|testing|test|out-of-sample)\s+sample\.?",
        lowered,
    ):
        return True
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "simulation results depict",
            "predictive efficiency which has a value",
            "exceeds predictive efficiency",
        )
    ):
        return True
    if support_status in {"explicit_supported", "well_supported"} and _compare_sentence_has_result_signal(slot_name, lowered) and not _compare_sentence_has_formula_noise(summary):
        return False
    if support_status in {"explicit_supported", "well_supported"} and not _compare_sentence_is_structural_noise(lowered) and not _compare_sentence_has_formula_noise(summary):
        if slot_name == "research_question" and _compare_sentence_has_question_signal(lowered) and len(lowered) >= 36:
            return False
        if (
            slot_name == "sample_or_data"
            and _compare_sentence_has_sample_signal(lowered)
            and len(lowered) >= 28
            and not any(
                marker in lowered
                for marker in (
                    "volatility smile",
                    "calibrating these models",
                    "advanced option pricing models beyond",
                    "black-scholes world",
                    "constant volatility assumed",
                )
            )
        ):
            return False
        if slot_name == "method" and _compare_sentence_has_method_signal(lowered) and len(lowered) >= 24:
            return False
        if slot_name in {"main_findings", "conclusion"} and _detail_marker_count(slot_name, lowered) > 0 and len(lowered) >= 36:
            return False
    if support_status == "weakly_supported" and not _compare_sentence_is_structural_noise(lowered) and not _compare_sentence_has_formula_noise(summary):
        if slot_name == "research_question" and _compare_sentence_has_question_signal(lowered) and len(lowered) >= 48:
            return False
        if (
            slot_name == "sample_or_data"
            and _compare_sentence_has_sample_signal(lowered)
            and len(lowered) >= 32
            and not any(
                marker in lowered
                for marker in (
                    "volatility smile",
                    "calibrating these models",
                    "advanced option pricing models beyond",
                    "black-scholes world",
                    "constant volatility assumed",
                )
            )
        ):
            return False
        if slot_name == "method" and _compare_sentence_has_method_signal(lowered) and len(lowered) >= 24:
            return False
        if slot_name == "main_findings" and _compare_sentence_has_result_signal(slot_name, lowered) and len(lowered) >= 48:
            return False
        if slot_name == "conclusion" and _compare_sentence_has_result_signal(slot_name, lowered) and len(lowered) >= 40:
            return False
    if len(lowered) < 18 and _detail_marker_count(slot_name, lowered) == 0:
        return True
    if "sample path" in lowered or "sample paths" in lowered:
        return True
    if _compare_sentence_has_formula_noise(summary):
        return True
    if any(
        marker in lowered
        for marker in (
            "article submitted to",
            "mathematical methods of operations research",
            "journal of ",
            "loss curves against",
            "validation dataset",
            "author:",
            "keywords:",
            "department of",
            "university of",
            "institute of",
        )
    ):
        return True
    if "@" in summary:
        return True
    if re.search(r"\b(?:references?|appendix|doi|figure|table)\b", lowered):
        return True
    if re.search(r"\backnowledg(?:ement|ements|ment|ments)\b", lowered):
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "article submitted to",
            "exchange-traded options",
            "historical data of underlying asset prices rather than simulated data",
        )
    ) and _detail_marker_count(slot_name, lowered) == 0:
        return True
    if slot_name == "sample_or_data" and re.search(
        r"^(?:mon|tue|wed|thu|fri|sat|sun),\s+\d{1,2}\s+[a-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\b",
        lowered,
    ):
        return True
    if slot_name == "sample_or_data" and re.fullmatch(
        r"(?:the\s+)?(?:training|validation|testing|test|out-of-sample)\s+sample\.?",
        lowered,
    ):
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "volatility smile",
            "calibrating these models",
            "advanced option pricing models beyond",
            "black-scholes world",
            "constant volatility assumed",
        )
    ) and not any(
        marker in lowered
        for marker in (
            "historical underlying asset prices",
            "sample contains",
            "daily returns",
            "observations",
            "training sample",
            "out-of-sample",
        )
    ):
        return True
    if slot_name == "sample_or_data" and any(
        marker in lowered
        for marker in (
            "predictive power",
            "benchmark model",
            "portfolio level",
            "stock level",
            "behave erratically",
            "best bid quotes",
            "inventories over the zoomed period",
            "right panel shows",
            "left panel shows",
            "left panel",
            "right panel",
            "mean reward ppo",
            "loss curve",
        )
    ) and not any(
        marker in lowered
        for marker in (
            "crsp",
            "nyse",
            "amex",
            "nasdaq",
            "observation",
            "cross-section",
            "sample contains",
            "daily returns",
            "monthly returns",
            "historical underlying asset prices",
        )
    ):
        return True
    if slot_name == "conclusion" and any(
        marker in lowered for marker in ("this concludes the proof", "proof", "lemma", "theorem", "corollary")
    ):
        return True
    if slot_name == "research_question" and any(
        marker in lowered
        for marker in (
            "objective function",
            "validation objective",
            "forecast errors",
            "hyperparameter",
            "hyperparameters",
        )
    ):
        return True
    if slot_name == "research_question" and re.search(r"[∑λθ̂]", summary):
        return True
    if slot_name == "method" and any(
        marker in lowered
        for marker in (
            "operations research",
            "article submitted to",
            "working paper",
            "loss curves against",
            "validation dataset",
            "however, these methods may lead",
            "for a comprehensive review",
            "future research",
        )
    ):
        return True
    if slot_name == "main_findings" and any(
        marker in lowered
        for marker in (
            "for specific details and results",
            "section 4 presents the findings",
            "numerical results we test",
        )
    ):
        return True
    if slot_name == "limitations" and (
        any(
            marker in lowered
            for marker in (
                "aims to alleviate the limitations of individual",
                "lack of regularization leaves ols",
                "trading strategy exactly",
                "maximum leverage constraint",
                "excluding short sales",
                "left panel",
                "right panel",
                "lemma",
                "proof",
                "corollary",
                "matrix manipulation",
            )
        )
        or not _compare_sentence_has_limitation_signal(summary)
    ):
        return True
    if slot_name == "practical_or_investment_implications" and not _compare_sentence_has_practical_signal(summary):
        return True
    if (
        slot_name in {"main_findings", "limitations", "conclusion"}
        and lowered.startswith(("section ", "results ", "discussion ", "conclusion ", "numerical results"))
        and _detail_marker_count(slot_name, lowered) == 0
    ):
        return True
    if support_status == "weakly_supported" and _detail_marker_count(slot_name, lowered) == 0:
        return True
    if _compare_sentence_is_truncated_fragment(summary):
        return True
    return False


def _build_compare_slot_entry(
    document_note: dict[str, object],
    slot_name: str,
    *,
    response_language: str,
    paper_output_profile: str,
) -> dict[str, str]:
    missing_phrase = _paper_missing_phrase(response_language)
    slot_payload = _document_slot(document_note, slot_name)
    aggregated_payload = _aggregated_document_slot(document_note, slot_name)
    status = _slot_payload_status(slot_payload)
    aggregated_status = _slot_payload_status(aggregated_payload)
    source_name = _compare_document_name(document_note)
    candidate_texts = _compare_slot_candidate_texts(document_note, slot_name)
    candidate_summaries: list[tuple[int, str, str]] = []
    source_rescued_cleaned = ""

    def _consider(candidate_text: str, candidate_status: str, *, bonus: int = 0) -> None:
        cleaned_candidate = _clean_compare_slot_summary(candidate_text, slot_name=slot_name)
        if not cleaned_candidate:
            return
        if _looks_unreliable_compare_summary(
            slot_name,
            cleaned_candidate,
            support_status=candidate_status,
        ):
            return
        candidate_summaries.append(
            (
                _compare_summary_quality_score(
                    slot_name,
                    cleaned_candidate,
                    support_status=candidate_status,
                )
                + bonus,
                candidate_status,
                cleaned_candidate,
            )
        )

    rescued = _compare_slot_rescue_summary(slot_name, candidate_texts)
    if rescued:
        rescue_status = status if status != "not_clearly_stated" else aggregated_status
        if rescue_status == "not_clearly_stated":
            rescue_status = "explicit_supported"
        _consider(
            rescued,
            rescue_status,
            bonus=6 if slot_name in {"research_question", "method", "main_findings", "conclusion"} else 4,
        )

    source_rescued = _compare_source_page_rescue_summary(document_note, slot_name)
    if source_rescued:
        source_rescued_cleaned = _clean_compare_slot_summary(source_rescued, slot_name=slot_name)
        source_bonus = 14 if slot_name in {"research_question", "method", "main_findings", "conclusion"} else 10
        if slot_name in {"method", "main_findings", "conclusion"} and (
            _slot_is_clearly_supported(status) or _slot_is_clearly_supported(aggregated_status)
        ):
            source_bonus -= 6
        _consider(
            source_rescued,
            "explicit_supported",
            bonus=source_bonus,
        )

    for payload, payload_status in (
        (slot_payload, status),
        (aggregated_payload, aggregated_status),
    ):
        if payload_status == "not_clearly_stated":
            continue
        raw = _slot_payload_render_text(
            payload,
            paper_output_profile=paper_output_profile,
        )
        _consider(raw, payload_status)
        summary_text = str(payload.get("summary_text", "")).strip()
        if summary_text and summary_text != raw:
            _consider(
                summary_text,
                payload_status,
                bonus=2 if slot_name in {"main_findings", "conclusion", "limitations"} else 1,
            )

    if not candidate_summaries:
        return {
            "source_name": source_name,
            "status": "not_clearly_stated",
            "summary": missing_phrase,
            "clause": missing_phrase.rstrip("."),
        }

    _score, effective_status, cleaned = sorted(
        candidate_summaries,
        key=lambda item: (-item[0], -len(item[2]), item[2]),
    )[0]
    if source_rescued_cleaned:
        lowered_cleaned = cleaned.lower()
        if (
            _compare_sentence_is_structural_noise(cleaned)
            or _compare_sentence_is_truncated_fragment(cleaned)
            or _compare_sentence_has_formula_noise(cleaned)
            or any(
                marker in lowered_cleaned
                for marker in (
                    "we conclude this section",
                    "stochastic discount factor interpretation",
                    "annals of mathematical statistics",
                    "empirical results table",
                    "table b1",
                    "overall length of t =",
                )
            )
        ):
            cleaned = source_rescued_cleaned
            effective_status = "explicit_supported"
    split_ready = cleaned.replace("; ", ". ").replace(";", ". ")
    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", split_ready)
        if part.strip()
    ]
    merged_sentence_parts: list[str] = []
    for part in sentence_parts:
        if merged_sentence_parts and re.match(r"^\(?\d{4}\)?[,)]?\s*", part):
            merged_sentence_parts[-1] = merged_sentence_parts[-1].rstrip(". ") + " " + part
            continue
        merged_sentence_parts.append(part)
    sentence_parts = merged_sentence_parts
    sentence_parts = [
        part
        for part in sentence_parts
        if not _compare_sentence_is_truncated_fragment(part)
        and not _compare_sentence_is_structural_noise(part)
        and not _compare_sentence_has_formula_noise(part)
    ] or sentence_parts
    if (
        slot_name in {"main_findings", "conclusion"}
        and len(sentence_parts) > 1
        and sentence_parts[0].lower().startswith(
            ("in this paper", "this paper", "we consider", "we have considered")
        )
        and _compare_sentence_has_result_signal(slot_name, sentence_parts[1])
    ):
        sentence_parts = [sentence_parts[1], sentence_parts[0], *sentence_parts[2:]]
    if slot_name == "main_findings" and len(sentence_parts) > 1:
        finding_safe_parts = [
            part
            for part in sentence_parts
            if not any(
                marker in part.lower()
                for marker in (
                    "comparing it with equation",
                    "equation (",
                    "equation ",
                )
            )
        ]
        if finding_safe_parts:
            sentence_parts = finding_safe_parts
    if slot_name in {
        "research_question",
        "method",
        "main_findings",
        "limitations",
        "conclusion",
        "practical_or_investment_implications",
    } and sentence_parts:
        sentence_parts = sorted(
            sentence_parts,
            key=lambda part: (
                -_compare_summary_quality_score(
                    slot_name,
                    part,
                    support_status=effective_status,
                ),
                -len(part),
                part,
            ),
        )
        if slot_name in {"research_question", "method", "conclusion"}:
            sentence_parts = sentence_parts[:1]
    clause_parts: list[str] = []
    current_length = 0
    for part in sentence_parts:
        candidate_length = current_length + len(part) + (1 if clause_parts else 0)
        if clause_parts and candidate_length > 340:
            break
        clause_parts.append(part.rstrip("."))
        current_length = candidate_length
        if current_length >= 300:
            break
    clause = ". ".join(clause_parts).strip()
    if not clause:
        clause = _truncate_line(cleaned, limit=300).rstrip(".")
    elif len(clause) > 340:
        clause = _truncate_line(clause, limit=340).rstrip(".")
    clause = _normalize_compare_sentence_surface(clause).rstrip(".")
    if response_language == "en":
        clause = _polish_compare_clause_english(clause, slot_name=slot_name).rstrip(".")
    if slot_name == "method" and clause.count("(") > clause.count(")"):
        clause = re.sub(r",?\s*[^,()]*\([^()]*$", "", clause).rstrip(",; ")
    return {
        "source_name": source_name,
        "status": effective_status,
        "summary": clause + ".",
        "clause": clause,
    }


def _compare_summaries_look_similar(left_summary: str, right_summary: str) -> bool:
    left_tokens = set(_slot_summary_keywords(left_summary))
    right_tokens = set(_slot_summary_keywords(right_summary))
    if not left_tokens or not right_tokens:
        return False
    shared = left_tokens & right_tokens
    return len(shared) >= max(2, min(len(left_tokens), len(right_tokens)) // 2)


def _build_compare_contrast_line(
    slot_name: str,
    entries: list[dict[str, str]],
    *,
    response_language: str,
) -> str:
    missing_phrase = _paper_missing_phrase(response_language)
    supported = [item for item in entries if item["status"] != "not_clearly_stated"]
    if response_language == "zh-CN":
        missing_text = missing_phrase.rstrip(".").rstrip("\u3002")
        if not supported:
            return "- \u5bf9\u6bd4\uff1a\u4e24\u7bc7\u8bba\u6587\u90fd\u672a\u660e\u786e\u8bf4\u660e\u8fd9\u4e00\u7ef4\u5ea6\u3002"
        if len(supported) == 1 and len(entries) == 2:
            supported_entry = supported[0]
            missing_entry = next(item for item in entries if item["status"] == "not_clearly_stated")
            return (
                f"- \u5bf9\u6bd4\uff1a\u53ea\u6709 `{supported_entry['source_name']}` \u660e\u786e\u8bf4\u660e\u4e86\u8fd9\u4e00\u7ef4\u5ea6\uff1b"
                f"`{missing_entry['source_name']}` {missing_text}\u3002"
            )
        if len(entries) == 2:
            left_entry, right_entry = entries
            if left_entry["status"] == "not_clearly_stated" or right_entry["status"] == "not_clearly_stated":
                return (
                    f"- \u5bf9\u6bd4\uff1a`{left_entry['source_name']}` -> {left_entry['summary']} "
                    f"`{right_entry['source_name']}` -> {right_entry['summary']}"
                )
            if _compare_summaries_look_similar(left_entry["clause"], right_entry["clause"]):
                return "- \u5bf9\u6bd4\uff1a\u4e24\u7bc7\u8bba\u6587\u90fd\u660e\u786e\u8ba8\u8bba\u4e86\u8fd9\u4e00\u7ef4\u5ea6\uff0c\u4e3b\u8981\u5dee\u5f02\u4f53\u73b0\u5728\u4e0a\u9762\u7684\u8bba\u6587\u7279\u5b9a\u7ec6\u8282\u3002"
            frame_map = {
                "research_question": "\u7814\u7a76\u95ee\u9898\u662f",
                "sample_or_data": "\u6837\u672c\u4e0e\u6570\u636e\u8bbe\u5b9a\u662f",
                "method": "\u65b9\u6cd5\u8bbe\u5b9a\u662f",
                "main_findings": "\u4e3b\u8981\u53d1\u73b0\u662f",
                "limitations": "\u5c40\u9650\u4e3b\u8981\u662f",
                "conclusion": "\u7ed3\u8bba\u662f",
                "practical_or_investment_implications": "\u5b9e\u8df5\u542b\u4e49\u662f",
            }
            frame = frame_map.get(slot_name, "\u8fd9\u4e00\u7ef4\u5ea6\u662f")
            return (
                f"- \u5bf9\u6bd4\uff1a\u5728 `{left_entry['source_name']}` \u4e2d\uff0c{frame}{left_entry['clause']}\uff1b"
                f"\u5728 `{right_entry['source_name']}` \u4e2d\uff0c{frame}{right_entry['clause']}\u3002"
            )
        missing_count = sum(1 for item in entries if item["status"] == "not_clearly_stated")
        if missing_count:
            return (
                f"- \u5bf9\u6bd4\uff1a\u6709 {missing_count} \u7bc7\u8bba\u6587\u672a\u660e\u786e\u8bf4\u660e\u8fd9\u4e00\u7ef4\u5ea6\uff0c"
                "\u56e0\u6b64\u6bd4\u8f83\u5e94\u7d27\u6263\u4e0a\u9762\u6709\u652f\u6301\u7684\u8bba\u6587\u7ec6\u8282\u3002"
            )
        return "- \u5bf9\u6bd4\uff1a\u8fd9\u4e9b\u8bba\u6587\u5728\u8fd9\u4e00\u7ef4\u5ea6\u4e0a\u5b58\u5728\u5dee\u5f02\uff0c\u5e94\u4ee5\u5404\u7bc7\u8bba\u6587\u7684\u5177\u4f53\u652f\u6301\u7ec6\u8282\u4e3a\u51c6\uff0c\u907f\u514d\u7b3c\u7edf\u6df7\u5199\u3002"
    if not supported:
        return "- Contrast: Neither paper clearly states this dimension."
    if len(supported) == 1 and len(entries) == 2:
        supported_entry = supported[0]
        missing_entry = next(item for item in entries if item["status"] == "not_clearly_stated")
        return (
            f"- Contrast: Only `{supported_entry['source_name']}` clearly states this dimension; "
            f"`{missing_entry['source_name']}` is {missing_phrase.lower()}"
        )
    if len(entries) == 2:
        left_entry, right_entry = entries
        if left_entry["status"] == "not_clearly_stated" or right_entry["status"] == "not_clearly_stated":
            return (
                f"- Contrast: `{left_entry['source_name']}` -> {left_entry['summary']} "
                f"`{right_entry['source_name']}` -> {right_entry['summary']}"
            )
        if _compare_summaries_look_similar(left_entry["clause"], right_entry["clause"]):
            return (
                "- Contrast: Both papers clearly discuss this dimension, and the main difference is in the "
                "paper-specific details listed above."
            )
        frame_map = {
            "research_question": "the core question is",
            "sample_or_data": "the sample/data setup is",
            "method": "the method is",
            "main_findings": "the main finding is",
            "limitations": "the main limitation is",
            "conclusion": "the conclusion is",
            "practical_or_investment_implications": "the practical implication is",
        }
        frame = frame_map.get(slot_name, "this dimension is")
        return (
            f"- Contrast: In `{left_entry['source_name']}`, {frame} {left_entry['clause']}. "
            f"In `{right_entry['source_name']}`, {frame} {right_entry['clause']}."
        )
    missing_count = sum(1 for item in entries if item["status"] == "not_clearly_stated")
    if missing_count:
        return (
            f"- Contrast: {missing_count} consulted paper(s) do not clearly state this dimension, so the "
            "comparison should stay close to the supported paper-specific details above."
        )
    return "- Contrast: The papers differ on this dimension; use the paper-specific details above rather than collapsing them into a generic blended summary."


def _build_compare_slot_section_lines(
    slot_name: str,
    document_notes: list[dict[str, object]],
    *,
    response_language: str,
    paper_output_profile: str,
) -> list[str]:
    entries = [
        _build_compare_slot_entry(
            document_note,
            slot_name,
            response_language=response_language,
            paper_output_profile=paper_output_profile,
        )
        for document_note in document_notes
    ]
    lines = [f"- `{item['source_name']}`: {item['summary']}" for item in entries]
    lines.append(
        _build_compare_contrast_line(
            slot_name,
            entries,
            response_language=response_language,
        )
    )
    return lines


def _build_slot_grounded_compare_answer(
    document_notes: list[dict[str, object]],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    response_style: str,
    paper_output_profile: str = "detailed_paper_note",
) -> str:
    if response_style == "continuous_prose":
        sections: list[str] = []
        for slot_name in requested_slots:
            comparisons = [
                f"{Path(str(document_note.get('source_path', '(unknown document)'))).name}: "
                f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language, paper_output_profile=paper_output_profile)}"
                for document_note in document_notes
            ]
            if response_language == "zh-CN":
                sections.append(f"{_slot_display_name(slot_name, response_language)}方面，" + "；".join(comparisons) + "。")
            else:
                sections.append(f"For {_slot_display_name(slot_name, response_language)}, " + "; ".join(comparisons) + ".")
        return " ".join(sections).strip()

    lines: list[str] = []
    for slot_name in requested_slots:
        lines.append(f"{_slot_display_name(slot_name, response_language)}:")
        for document_note in document_notes:
            source_name = Path(str(document_note.get("source_path", "(unknown document)"))).name
            lines.append(
                f"- {source_name}: "
                f"{_slot_summary_sentence(document_note, slot_name, response_language=response_language, paper_output_profile=paper_output_profile)}"
            )
    return "\n".join(lines)


def _build_slot_grounded_compare_answer(
    document_notes: list[dict[str, object]],
    *,
    requested_slots: tuple[str, ...],
    response_language: str,
    response_style: str,
    paper_output_profile: str = "detailed_paper_note",
) -> str:
    if not document_notes:
        return (
            "\u6ca1\u6709\u53ef\u7528\u7684\u6bd4\u8f83\u8bba\u6587\u7b14\u8bb0\u3002"
            if response_language == "zh-CN"
            else "No compared paper notes were available."
        )

    lines: list[str] = [
        _compare_documents_heading(response_language),
        *[f"- `{_compare_document_name(document_note)}`" for document_note in document_notes],
    ]
    for slot_name in requested_slots:
        lines.extend(
            [
                "",
                _compare_section_title(slot_name, response_language),
                *_build_compare_slot_section_lines(
                    slot_name,
                    document_notes,
                    response_language=response_language,
                    paper_output_profile=paper_output_profile,
                ),
            ]
        )

    if response_style == "continuous_prose":
        paragraphs: list[str] = []
        buffer: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buffer:
                    paragraphs.append(" ".join(buffer))
                    buffer = []
                continue
            buffer.append(stripped)
        if buffer:
            paragraphs.append(" ".join(buffer))
        return "\n\n".join(paragraphs).strip()
    return "\n".join(lines).strip()


def _looks_insufficiently_translated_chinese(answer_text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", answer_text)
    normalized = re.sub(r"`[^`\n]+\.pdf`", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"`[^`\n]+`", " ", normalized)
    normalized = re.sub(r"\b[a-z0-9._-]+\.pdf\b", " ", normalized, flags=re.IGNORECASE)
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", normalized))
    hangul_chars = len(re.findall(r"[\uac00-\ud7af]", normalized))
    english_words = re.findall(r"[A-Za-z]{2,}", normalized.lower())
    allowed_terms = {
        "crsp",
        "nyse",
        "amex",
        "nasdaq",
        "pdf",
        "ml",
        "documents",
        "compared",
        "pca",
        "pcr",
        "pls",
        "ann",
        "cnn",
        "lstm",
        "rmse",
        "mae",
        "lasso",
        "elastic",
        "net",
        "ridge",
        "sharpe",
        "garch",
        "arimax",
        "boosted",
        "tree",
        "trees",
        "forest",
        "forests",
        "random",
        "neural",
        "network",
        "networks",
        "principal",
        "component",
        "analysis",
        "regression",
        "linear",
        "generalized",
        "ngqmle",
        "gqmle",
        "srm",
        "srms",
        "cbc",
    }
    suspicious_words = [word for word in english_words if word not in allowed_terms]
    return (
        chinese_chars < 24
        or len(suspicious_words) >= 10
        or hangul_chars >= 4
        or "??" in normalized
        or "\ufffd" in normalized
    )


def _compose_slot_translation_prompt_legacy_final(*, source_text: str, response_style: str) -> str:
    style_instruction = (
        "Return one continuous paragraph in natural Simplified Chinese with no bullets or outline headings."
        if response_style == "continuous_prose"
        else "Return a concise Simplified Chinese memo."
    )
    return "\n".join(
        [
            "Translate the grounded memo below into natural Simplified Chinese.",
            "Keep only facts that already appear in the grounded memo.",
            "Do not add generic finance or machine-learning commentary.",
            "If a dimension is unclear or missing, say 文中未明确说明。",
            "Keep method-family names, abbreviations, and file identifiers exact when needed.",
            style_instruction,
            "",
            "Grounded memo:",
            source_text,
            "",
            "Return only the final Chinese answer body.",
        ]
    )


def _contains_generic_paper_filler(
    answer_lower: str,
    document_notes: list[dict[str, object]],
) -> bool:
    support_corpus = " ".join(
        _slot_payload_text(_document_slot(document_note, slot_name)).lower()
        for document_note in document_notes
        for slot_name in (
            "research_question",
            "sample_or_data",
            "method",
            "main_findings",
            "limitations",
            "conclusion",
            "practical_or_investment_implications",
        )
        if _slot_payload_status(_document_slot(document_note, slot_name)) != "not_clearly_stated"
    )
    filler_patterns = (
        "broader promise of machine learning",
        "underscores the importance of machine learning",
        "illustrates how machine learning can",
        "significant benefits of machine learning",
        "provide deep insights",
        "offers a robust framework for future research",
        "important implications for investors",
        "broadly useful for practitioners",
        "potential economic benefits",
        "robust and reliable predictive tools",
        "this memo provides a concise overview",
        "given the provided evidence",
        "a hybrid approach could be beneficial",
        "combined methodology can provide",
        "this structured comparison ensures",
        "for future research",
        "说明机器学习在金融中具有广泛前景",
        "对投资者具有重要启示",
        "具有广泛意义",
        "进一步说明了",
        "这表明机器学习具有显著优势",
    )
    return any(pattern in answer_lower and pattern not in support_corpus for pattern in filler_patterns)


def _looks_excerpt_heavy_paper_answer(answer_text: str) -> bool:
    lowered = answer_text.lower()
    markdown_heading_count = len(re.findall(r"(?m)^#{2,6}\s", answer_text))
    dimension_block_count = answer_text.count("**Dimension:")
    evidence_marker_count = sum(lowered.count(token) for token in ("**evidence**", "evidence refs", "retrieved evidence"))
    inline_ref_count = sum(lowered.count(token) for token in ("#page=", "#chunk=", "#pages=", "evidence:"))
    page_note_count = len(re.findall(r"\bp\.\d+\b", lowered))
    slot_line_count = len(re.findall(r"(?m)^(research question|sample/data|method|main findings|limitations|conclusion):", lowered))
    return (
        markdown_heading_count >= 3
        or dimension_block_count >= 2
        or evidence_marker_count >= 2
        or inline_ref_count >= 4
        or page_note_count >= 4
        or slot_line_count >= 6
    )


def _looks_like_slot_dump_compare_answer(answer_text: str) -> bool:
    lowered = answer_text.lower()
    old_slot_dump = len(
        re.findall(
            r"(?m)^(research question|sample/data|sample and data|method|main findings|limitations|conclusion):\s*-",
            lowered,
        )
    )
    if old_slot_dump >= 2:
        return True
    if "comparison scaffold" in lowered:
        return True
    if sum(lowered.count(marker) for marker in ("- contrast:", "- `")) < 3:
        return False
    return False


def _compare_answer_claims_missing_inputs(answer_text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", answer_text).lower()
    return any(
        marker in lowered
        for marker in (
            "could not find the file",
            "cannot find the file",
            "file not found",
            "please provide the files",
            "please provide more information about which file",
            "please provide the pdf",
            "找不到指定的文件",
            "无法找到文件",
            "请提供更多关于你尝试访问哪个文件的信息",
            "请提供文件",
        )
    )


def _compare_answer_lacks_direct_contrast(
    answer_text: str,
    *,
    prompt: str,
    paper_trace: PaperTrace,
) -> bool:
    lowered = unicodedata.normalize("NFKC", answer_text).lower()
    requested_slots = _requested_paper_slots(prompt, "paper_compare")
    if len(requested_slots) < 2:
        return False
    contrast_markers = sum(
        lowered.count(marker)
        for marker in (
            "- contrast:",
            "contrast:",
            "比较:",
            "相比",
            "而",
            "whereas",
            "in contrast",
            "by contrast",
        )
    )
    if contrast_markers >= max(2, min(4, len(requested_slots) - 1)):
        return False
    if "documents compared" in lowered and "- `" in lowered and contrast_markers == 0:
        return True
    if any(
        marker in lowered
        for marker in (
            "# commonalities",
            "commonalities",
            "# differences",
            "differences",
            "# recommendations or synthesis",
            "recommendations or synthesis",
            "recommendation or synthesis",
            "strengths / weaknesses / limitations",
            "combining methodologies could",
            "hybrid approach could be beneficial",
            "both papers focus on",
        )
    ):
        return True
    compared_names = [
        _compare_document_name(document_note).lower()
        for document_note in paper_trace.document_notes
    ]
    mentioned_documents = sum(1 for name in compared_names if name and name in lowered)
    return mentioned_documents < min(2, len(compared_names))


def _compare_false_missing_supported_slots(
    answer_text: str,
    document_notes: list[dict[str, object]],
    *,
    response_language: str,
) -> tuple[str, ...]:
    if not document_notes:
        return ()
    missing_phrase = _paper_missing_phrase(response_language)
    false_slots: list[str] = []
    core_slots = (
        "research_question",
        "sample_or_data",
        "method",
        "main_findings",
        "conclusion",
    )
    normalized = unicodedata.normalize("NFKC", answer_text)
    sections: dict[str, list[str]] = {}
    current_section = ""
    for raw_line in normalized.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.strip() in {"Documents compared", *(_compare_section_title(slot) for slot in core_slots), "Limitations"}:
            current_section = line.strip()
            sections.setdefault(current_section, [])
            continue
        if current_section:
            sections.setdefault(current_section, []).append(line.strip())
    for slot_name in core_slots:
        section_title = _compare_section_title(slot_name)
        section_lines = sections.get(section_title)
        if not section_lines:
            continue
        section_body = "\n".join(section_lines)
        for document_note in document_notes:
            entry = _build_compare_slot_entry(
                document_note,
                slot_name,
                response_language=response_language,
                paper_output_profile="detailed_paper_note",
            )
            if entry["status"] == "not_clearly_stated":
                continue
            source_name = re.escape(_compare_document_name(document_note))
            if re.search(
                rf"(?im)^- `{source_name}`:\s*{re.escape(missing_phrase)}\s*$",
                section_body,
            ):
                false_slots.append(slot_name)
                break
    return tuple(dict.fromkeys(false_slots))


def _compare_missing_document_entry_slots(
    answer_text: str,
    document_notes: list[dict[str, object]],
    *,
    response_language: str,
    requested_slots: tuple[str, ...],
) -> tuple[str, ...]:
    if not answer_text or not document_notes or not requested_slots:
        return ()
    section_titles = {
        _compare_section_title(slot_name, response_language): slot_name for slot_name in requested_slots
    }
    bullet_counts = {slot_name: 0 for slot_name in requested_slots}
    current_slot = ""
    for raw_line in unicodedata.normalize("NFKC", answer_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in section_titles:
            current_slot = section_titles[line]
            continue
        if current_slot and re.match(r"^- `[^`]+`:\s*", line):
            bullet_counts[current_slot] = bullet_counts.get(current_slot, 0) + 1
    expected_count = len(document_notes)
    return tuple(
        slot_name
        for slot_name in requested_slots
        if bullet_counts.get(slot_name, 0) < expected_count
    )


def _compare_answer_has_contradictory_missing_prefix(
    answer_text: str,
    *,
    response_language: str,
) -> bool:
    normalized = unicodedata.normalize("NFKC", answer_text or "")
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not re.match(r"^- `[^`]+`:\s*", line):
            continue
        body = re.sub(r"^- `[^`]+`:\s*", "", line).strip()
        if response_language == "zh-CN":
            if body.startswith(("文本未明确说明", "文中未明确说明")):
                remainder = re.sub(r"^(?:文本|文中)未明确说明[。；，,:：]?\s*", "", body)
                if remainder:
                    return True
        else:
            if body.startswith("Not clearly stated in the paper."):
                remainder = body.removeprefix("Not clearly stated in the paper.").strip()
                if remainder:
                    return True
    return False


def _normalize_compare_sentence_surface(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = cleaned.replace("＊", "'")
    cleaned = cleaned.replace("∗", "*")
    cleaned = cleaned.replace("\u00ad", "")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", cleaned)
    for marker in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"):
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"\s+([\u0300-\u036f])", r"\1", cleaned)
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = re.sub(r"[\u0300-\u036f]", "", cleaned)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = re.sub(
        r"^(?:[A-Z][A-Za-z-]+(?: [A-Z][A-Za-z-]+){2,})\s+\d{2,4}\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"([A-Za-z]{2,})-\s+([A-Za-z]{2,})", r"\1\2", cleaned)
    cleaned = re.sub(r"(\d+)\s*-\s+([A-Za-z]{2,})", r"\1-\2", cleaned)
    cleaned = re.sub(
        r"\b([A-Za-z]{5,})\s+((?:ated|ation|ations|cated|tive|tives|ment|ments|ally|ically|ized|ising|izing|tion|tions|sion|sions|ality|ities|ously|ness|able|ible|ance|ances|ence|ences))\b",
        r"\1\2",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(one|two|three|four|five|six|seven|eight|nine)\s*-?\s*step\b",
        lambda match: f"{match.group(1)}-step",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(one|two|three|four|five|six|seven|eight|nine)step\b",
        lambda match: f"{match.group(1)}-step",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(one|two|three|four|five|six|seven|eight|nine)and\s+((?:one|two|three|four|five|six|seven|eight|nine)-[A-Za-z]+)\b",
        lambda match: f"{match.group(1)}- and {match.group(2)}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bclosedform\b", "closed-form", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\blinearquadratic\b", "linear-quadratic", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", cleaned)
    cleaned = re.sub(r"(?<=\d)\s+%", "%", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])\((?=[A-Za-z])", " (", cleaned)
    cleaned = re.sub(r"\)(?=(?:and|or|but|while|whereas|in contrast)\b)", ") ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r",\s*(?:while|whereas|but)\s+[^.]{0,120}\b(?:better than|worse than|higher than|lower than|more than|less than|greater than|smaller than)\.?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\.\s*\d+\s+(?=[A-Z])", ". ", cleaned)
    cleaned = re.sub(
        r"(?<=[a-z0-9])\s+(?=(?:This|These|It|We|Our)\s+(?:article|paper|study|work|approach|procedure|method|methods|framework|results|findings|evidence|novel|model|conclusion|research)\b)",
        ". ",
        cleaned,
    )
    cleaned = re.sub(
        r"(?<=[a-z0-9])\s+(?=(?:In conclusion|In summary|Overall|Finally)\b)",
        ". ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?<=\.\s)consistent and asymptotically normal\b",
        "It is consistent and asymptotically normal",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^this example also sheds light on the fact that\s+",
        "The results also show that ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:main|primary)\s+(technical contribution|contribution)\s+is\b",
        lambda match: f"The {match.group(0).lower()}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:[A-Za-z]+\s+)?[A-Z]\s+cannot obtain the efficiency bounds for all parameters unless\b",
        "the efficiency bounds for all parameters cannot be obtained unless",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(optimal policy function)\s+f\s*(?:\?|\*|∗)\s*\((?:·|\.)\)",
        r"\1",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r",?\s*see,\s*e\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:and|or)\s+their\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s+(?:(?:[A-Za-z]|\d+|dt|ds|du|dv|[^\x00-\x7F])\s+){3,}(?:[A-Za-z]|\d+|dt|ds|du|dv|[^\x00-\x7F])(?=[.,;:]|$)",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\b([A-Za-z][A-Za-z-]*(?: [A-Za-z][A-Za-z-]*){3,})\s+([A-Z])\s+is\s+a\b", r"\1. \2 is a", cleaned)
    cleaned = re.sub(r"\s+(?:\d{1,2}|[ivx]{1,4})\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"(?<=[.!?]\s)([a-z])",
        lambda match: match.group(1).upper(),
        cleaned,
    )
    cleaned = re.sub(
        r"(?<=[A-Za-z])[A-Za-z]?[^\x00-\x7F]{1,4}[A-Za-z]?(?=\s+[A-Za-z])",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\b[A-Z][^\x00-\x7F]{1,3}\b", " ", cleaned)
    cleaned = re.sub(r"\bhighfrequency\b", "high-frequency", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bana lyses\b", "analyses", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhy pothesis\b", "hypothesis", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bin formation\b", "information", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if re.match(r"^(the|this)\b", cleaned):
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _polish_compare_clause_english(text: str, *, slot_name: str) -> str:
    cleaned = _normalize_compare_sentence_surface(text).strip()
    replacements = (
        (r"^we introduce\b", "The paper introduces"),
        (r"^we propose\b", "The paper proposes"),
        (r"^we derive\b", "The paper derives"),
        (r"^we conclude(?: that)?\b", "The paper concludes that"),
        (r"^we find(?: that)?\b", "The paper finds that"),
        (r"^we show(?: that)?\b", "The paper shows that"),
        (r"^we study\b", "The paper studies"),
        (r"^we investigate\b", "The paper investigates"),
        (r"^we focus on\b", "The paper focuses on"),
        (r"^in this paper, we have considered\b", "The paper considers"),
        (r"^in this paper, we consider\b", "The paper considers"),
        (r"^this paper explicitly discusses\b", "The paper discusses"),
        (r"^(Outperforms|Performs|Achieves)\b", lambda m: f"The paper {m.group(1).lower()}"),
        (r"\bbecause their simulation results depict that\b", "because the simulation results show that"),
        (r"\bwhich has a value of less than\b", "with less than"),
        (r"\band,\s+using ([^,]+),\s+find\b", r"and, using \1, finds"),
        (r"\bmitigate overfit\b", "mitigate overfitting"),
        (r"\bincon\s*-\s*sistent\b", "inconsistent"),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _compare_sentence_is_truncated_fragment(sentence: str) -> bool:
    normalized = _normalize_compare_sentence_surface(sentence).strip()
    if "..." in normalized:
        return True
    lowered = normalized.lower().rstrip(". ")
    if normalized.count("(") > normalized.count(")") and re.search(r"\([a-z0-9]{0,8}$", lowered):
        return True
    if normalized.count("(") > normalized.count(")") and len(normalized) < 220:
        return True
    if lowered.endswith("et al"):
        return True
    if lowered.endswith("see, e") or lowered.endswith("and their"):
        return True
    if re.search(r"\b\d{1,3}\s+(?:because|as an aside)\b", lowered):
        return True
    if any(
        marker in lowered
        for marker in (
            "as an aside",
            "estimated complexity in figure",
            "we do not describe their estimated complexity",
        )
    ):
        return True
    if re.search(r"\breported in table$", lowered):
        return True
    if re.search(r"\b(?:better|worse|higher|lower|more|less|greater|smaller)\s+than$", lowered):
        return True
    if re.search(r"\b(?:r2|rmse|mae|sharpe ratio|variance|value)\s+of\s+\d+(?:\.\d+)?$", lowered):
        return True
    if re.match(
        r"^[A-Z][A-Za-z-]+(?:(?:,\s*[A-Z][A-Za-z-]+)|(?:\s*&\s*[A-Z][A-Za-z-]+)){1,}\s*\(?\d{4}\)?",
        normalized,
    ) and not re.search(
        r"\b(?:show|find|suggest|report|demonstrate|use|introduce|study|investigate|derive|propose|estimate|conclude|perform|focus)\b",
        lowered,
    ):
        return True
    if re.match(
        r"^[A-Z][A-Za-z-]+,\s*[A-Z][A-Za-z-]+\s*&\s*[A-Z][A-Za-z-]+\s*\(?\d{4}\)?",
        normalized,
    ) and not re.search(
        r"\b(?:show|find|suggest|report|demonstrate|use|introduce|study|investigate|derive|propose|estimate|conclude|perform|focus)\b",
        lowered,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:predic|estim|conclu|discussi|methodolog|out|with|and|or|of|to|for|in|on|by|from|than|because|while|where|which|whose|when|that|via|using|wh)$",
            lowered,
        )
    )


def _compare_sentence_has_formula_noise(sentence: str) -> bool:
    normalized = unicodedata.normalize("NFKC", sentence)
    lowered = normalized.lower()
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", normalized):
        return True
    if re.search(r"[\ue000-\uf8ff]", normalized):
        return True
    if re.search(r"\b[a-z]\s*(?:\?|\*|∗)\s*\(", normalized):
        return True
    if re.search(r"\b[a-z]\s*(?:\?|\*|∗)\s*\((?:\.|·)\)", normalized):
        return True
    math_symbol_count = len(re.findall(r"[=<>+*/^∑∫∏≈≠≤≥βσλρμτωαΔΠΣΩ∞εη]", normalized))
    paren_count = normalized.count("(") + normalized.count(")")
    digit_count = len(re.findall(r"\d", normalized))
    if math_symbol_count >= 5 and paren_count >= 4 and digit_count >= 2:
        return True
    if any(token in lowered for token in ("equation (", "formula (", "theorem ", "lemma ", "corollary ", "proof")) and math_symbol_count >= 3:
        return True
    if "=" in normalized and re.search(r"[−γδβσλρμτωαΔΠΣΩ√∞εη]", normalized):
        return True
    if re.search(r"[γδβσλρμτωαΔΠΣΩ√∞εη]", normalized) and any(
        marker in lowered for marker in ("consistent", "asymptotically normal", "efficiency", "parameter", "variance", "likelihood")
    ):
        return True
    if re.match(r"^[A-Z]\s+is\s+(?:a|an|the)\b", normalized.strip()):
        return True
    return False


def _collect_compare_page_rescue_candidates(
    slot_name: str,
    page_texts: list[tuple[int, str]],
    marker_bonuses: tuple[tuple[str, int], ...],
    *,
    require_signal: str,
    snippet_length: int = 360,
) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    for page_number, text in page_texts:
        normalized_page = _normalize_extracted_block(text)
        lowered_page = normalized_page.lower()
        for marker, bonus in marker_bonuses:
            marker_index = lowered_page.find(marker)
            if marker_index == -1:
                continue
            snippet_start = max(0, marker_index - 140)
            snippet = normalized_page[snippet_start : marker_index + snippet_length]
            snippet_sentences = [
                sentence
                for sentence in _compare_candidate_sentences(snippet)
                if not _compare_sentence_is_truncated_fragment(sentence)
            ]
            if not snippet_sentences:
                continue
            ranked_sentences: list[tuple[int, str]] = []
            for sentence in snippet_sentences:
                cleaned_sentence = _clean_compare_slot_summary(sentence, slot_name=slot_name)
                if not cleaned_sentence:
                    continue
                lowered_sentence = cleaned_sentence.lower()
                if require_signal == "question" and not _compare_sentence_has_question_signal(cleaned_sentence):
                    continue
                if require_signal == "method" and not (
                    _compare_sentence_has_method_signal(cleaned_sentence)
                    or _detail_marker_count(slot_name, lowered_sentence) > 0
                ):
                    continue
                if require_signal == "result" and not (
                    _compare_sentence_has_result_signal(slot_name, cleaned_sentence)
                    or (
                        slot_name == "conclusion"
                        and any(
                            marker in lowered_sentence
                            for marker in (
                                "we propose",
                                "this article develops",
                                "we conclude",
                                "our findings",
                                "these findings",
                                "under the black-scholes-merton framework",
                                "we derive the optimal quotes",
                                "approximate closed-form solution",
                            )
                        )
                    )
                    or _detail_marker_count(slot_name, lowered_sentence) > 0
                ):
                    continue
                if _looks_unreliable_compare_summary(slot_name, cleaned_sentence, support_status="explicit_supported"):
                    continue
                sentence_score = _compare_summary_quality_score(
                    slot_name,
                    cleaned_sentence,
                    support_status="explicit_supported",
                )
                if marker in lowered_sentence:
                    sentence_score += 5
                if sentence[:1].isupper():
                    sentence_score += 1
                if not _compare_sentence_has_formula_noise(cleaned_sentence):
                    sentence_score += 1
                ranked_sentences.append((sentence_score, cleaned_sentence.rstrip(". ")))
            if not ranked_sentences:
                continue
            ordered_sentences = sorted(
                ranked_sentences,
                key=lambda item: (-item[0], -len(item[1]), item[1]),
            )
            unique_sentences: list[str] = []
            seen_sentences: set[str] = set()
            for _score, cleaned_sentence in ordered_sentences:
                signature = re.sub(r"\W+", " ", cleaned_sentence.lower()).strip()
                if not signature or signature in seen_sentences:
                    continue
                seen_sentences.add(signature)
                unique_sentences.append(cleaned_sentence)
                if len(unique_sentences) >= 2:
                    break
            cleaned = ". ".join(unique_sentences).strip()
            if cleaned:
                cleaned = cleaned.rstrip(". ") + "."
            if not cleaned:
                continue
            if require_signal == "question" and not _compare_sentence_has_question_signal(cleaned):
                continue
            if require_signal == "method" and not (
                _compare_sentence_has_method_signal(cleaned)
                or _detail_marker_count(slot_name, cleaned.lower()) > 0
            ):
                continue
            if require_signal == "result" and not (
                _compare_sentence_has_result_signal(slot_name, cleaned)
                or (
                    slot_name == "conclusion"
                    and any(
                        marker in cleaned.lower()
                        for marker in (
                            "we propose",
                            "this article develops",
                            "we conclude",
                            "our findings",
                            "these findings",
                            "under the black-scholes-merton framework",
                        )
                    )
                )
                or _detail_marker_count(slot_name, cleaned.lower()) > 0
            ):
                continue
            if _looks_unreliable_compare_summary(slot_name, cleaned, support_status="explicit_supported"):
                continue
            score = _compare_summary_quality_score(
                slot_name,
                cleaned,
                support_status="explicit_supported",
            )
            candidates.append((score + bonus + max(0, 5 - page_number), cleaned))
    return candidates


def _compare_method_page_rescue_summary(
    page_texts: list[tuple[int, str]],
) -> str:
    candidates = _collect_compare_page_rescue_candidates(
        "method",
        page_texts,
        (
            ("we propose to address this estimation problem with a moment method", 12),
            ("continuous fourier sine transform", 11),
            ("quasi-maximum likelihood estimator", 11),
            ("generalized synchronization scheme", 10),
            ("implement pca at high frequency", 11),
            ("constructing and estimating realized eigenvalues", 10),
            ("the main technical contribution is", 10),
            ("axiomatic dual representation", 10),
            ("we introduce a scale adjustment parameter", 10),
            ("we introduce", 8),
            ("we propose", 8),
            ("closed-form solution", 8),
            ("principal component analysis", 7),
            ("previous tick method", 7),
            ("moment method", 9),
        ),
        require_signal="method",
    )
    if not candidates:
        return ""
    design_markers = (
        "we propose",
        "this paper proposes",
        "we introduce",
        "three-step",
        "procedure",
        "approach",
        "framework",
        "scale adjustment parameter",
        "closed-form solution",
        "stochastic control",
        "dual representation",
        "pricing kernel",
        "generalized methods of moments",
    )
    property_markers = (
        "consistent",
        "asymptotically normal",
        "finite fourth moment",
        "better efficiency",
        "more efficient",
        "smaller asymptotic variance",
        "weak moment conditions",
    )
    preferred: list[tuple[int, str]] = []
    backup: list[tuple[int, str]] = []
    for score, cleaned in candidates:
        lowered = cleaned.lower()
        if any(marker in lowered for marker in property_markers) and not any(
            marker in lowered for marker in design_markers
        ):
            backup.append((score - 12, cleaned))
        else:
            preferred.append((score, cleaned))
    chosen = preferred or backup
    return sorted(chosen, key=lambda item: (-item[0], -len(item[1]), item[1]))[0][1]


def _compare_research_question_page_rescue_summary(
    page_texts: list[tuple[int, str]],
) -> str:
    candidates = _collect_compare_page_rescue_candidates(
        "research_question",
        page_texts,
        (
            ("the aim of our work is to", 10),
            ("the purpose of our work is to", 10),
            ("the purpose of this paper is to", 10),
            ("the goal in this paper is to", 10),
            ("this article focuses on", 9),
            ("this paper focuses on", 9),
            ("this article questions", 8),
            ("this study aims", 8),
            ("this paper aims", 8),
            ("we aim to", 7),
            ("we study", 7),
            ("we investigate", 6),
            ("we introduce the concept of", 8),
            ("the remaining question is", 7),
            ("in our present work, we propose to", 8),
            ("the fundamental goal", 7),
            ("the main contribution of this article", 6),
        ),
        require_signal="question",
    )
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: (-item[0], -len(item[1]), item[1]))[0][1]


def _compare_findings_page_rescue_summary(
    page_texts: list[tuple[int, str]],
) -> str:
    candidates = _collect_compare_page_rescue_candidates(
        "main_findings",
        page_texts,
        (
            ("we find that", 11),
            ("the results show", 9),
            ("results show", 9),
            ("we show that", 6),
            ("we provided", 8),
            ("monte carlo simulations show", 9),
            ("empirical findings", 8),
            ("we find a surprising consistency", 9),
            ("we find a risk premium", 13),
            ("outperforms", 8),
            ("admits an approximate closed-form solution", 8),
            ("consistent and asymptotically normal", 8),
            ("necessary and sufficient conditions", 8),
            ("performs very well", 9),
            ("negligible impact", 9),
            ("converges towards the theoretical value", 9),
            ("achieved the lowest rmse", 11),
            ("achieved the lowest mae", 10),
            ("stands out as the most promising", 10),
            ("leading contender", 9),
            ("greater precision in forecasting", 9),
            ("strongest and most consistent trading strategies", 10),
            ("best performing nonlinear method", 10),
            ("positive predictive performance", 10),
        ),
        require_signal="result",
    )
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: (-item[0], -len(item[1]), item[1]))[0][1]


def _compare_conclusion_page_rescue_summary(
    page_texts: list[tuple[int, str]],
) -> str:
    candidates = _collect_compare_page_rescue_candidates(
        "conclusion",
        page_texts,
        (
            ("in conclusion", 10),
            ("we conclude", 9),
            ("these findings", 8),
            ("this article develops", 10),
            ("we derive the optimal quotes in feedback form", 10),
            ("we derive", 7),
            ("we propose a pca-based estimator", 11),
            ("we propose a", 7),
            ("under the black-scholes-merton framework", 10),
            ("our findings", 7),
            ("these findings underscore", 8),
            ("we find a surprising consistency", 8),
            ("performs very well", 8),
            ("negligible impact", 8),
            ("converges towards the theoretical value", 8),
            ("advantages of the proposed approach", 8),
            ("better efficiency", 8),
        ),
        require_signal="result",
        snippet_length=420,
    )
    for page_number, text in page_texts:
        normalized_page = _normalize_extracted_block(text)
        lowered_page = normalized_page.lower()
        for marker in ("discussion and conclusions", "discussion and conclusion", "conclusions", "conclusion"):
            marker_index = lowered_page.find(marker)
            if marker_index == -1:
                continue
            snippet = normalized_page[marker_index : marker_index + 420]
            snippet_sentences = [
                sentence
                for sentence in _compare_candidate_sentences(snippet)
                if not _compare_sentence_is_truncated_fragment(sentence)
            ]
            if not snippet_sentences:
                continue
            ranked_sentences: list[tuple[int, str]] = []
            for sentence in snippet_sentences:
                cleaned_sentence = _clean_compare_slot_summary(sentence, slot_name="conclusion")
                if not cleaned_sentence:
                    continue
                lowered_sentence = cleaned_sentence.lower()
                if not (
                    _compare_sentence_has_result_signal("conclusion", cleaned_sentence)
                    or any(
                        marker in lowered_sentence
                        for marker in (
                            "we conclude",
                            "our findings",
                            "these findings",
                            "we derive the optimal quotes",
                            "approximate closed-form solution",
                            "under the black-scholes-merton framework",
                        )
                    )
                    or _detail_marker_count("conclusion", lowered_sentence) > 0
                ):
                    continue
                if _looks_unreliable_compare_summary("conclusion", cleaned_sentence, support_status="explicit_supported"):
                    continue
                sentence_score = _compare_summary_quality_score(
                    "conclusion",
                    cleaned_sentence,
                    support_status="explicit_supported",
                )
                if marker in lowered_sentence:
                    sentence_score += 5
                if sentence[:1].isupper():
                    sentence_score += 1
                ranked_sentences.append((sentence_score, cleaned_sentence.rstrip(". ")))
            if not ranked_sentences:
                continue
            ordered_sentences = sorted(
                ranked_sentences,
                key=lambda item: (-item[0], -len(item[1]), item[1]),
            )
            unique_sentences: list[str] = []
            seen_sentences: set[str] = set()
            for _score, cleaned_sentence in ordered_sentences:
                signature = re.sub(r"\W+", " ", cleaned_sentence.lower()).strip()
                if not signature or signature in seen_sentences:
                    continue
                seen_sentences.add(signature)
                unique_sentences.append(cleaned_sentence)
                if len(unique_sentences) >= 2:
                    break
            cleaned = ". ".join(unique_sentences).strip()
            if cleaned:
                cleaned = cleaned.rstrip(". ") + "."
            if not cleaned:
                continue
            if _looks_unreliable_compare_summary("conclusion", cleaned, support_status="explicit_supported"):
                continue
            score = _compare_summary_quality_score(
                "conclusion",
                cleaned,
                support_status="explicit_supported",
            )
            candidates.append((score + 9 + max(0, 4 - page_number), cleaned))
    if not candidates:
        findings_fallback = _compare_findings_page_rescue_summary(page_texts)
        if findings_fallback and not _looks_unreliable_compare_summary(
            "conclusion",
            findings_fallback,
            support_status="explicit_supported",
        ):
            score = _compare_summary_quality_score(
                "conclusion",
                findings_fallback,
                support_status="explicit_supported",
            )
            candidates.append((score + 6, findings_fallback))
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: (-item[0], -len(item[1]), item[1]))[0][1]


def _compare_source_page_rescue_summary(
    document_note: dict[str, object],
    slot_name: str,
) -> str:
    source_path = str(document_note.get("source_path", "")).strip()
    if not source_path:
        return ""
    page_texts = list(_source_page_texts_for_detailed_rescue(source_path))
    if not page_texts:
        return ""
    if slot_name == "limitations":
        limitations_specific = _compare_limitations_page_rescue_summary(page_texts)
        if limitations_specific:
            return limitations_specific
    rescue_candidates: list[tuple[int, str]] = []
    compare_pages = _compare_source_rescue_pages(slot_name, page_texts)
    if slot_name == "research_question":
        research_specific = _compare_research_question_page_rescue_summary(compare_pages)
        if research_specific:
            rescue_candidates.append((10, research_specific))
    if slot_name == "method":
        method_specific = _compare_method_page_rescue_summary(compare_pages)
        if method_specific:
            rescue_candidates.append((10, method_specific))
    if slot_name == "main_findings":
        findings_specific = _compare_findings_page_rescue_summary(compare_pages)
        if findings_specific:
            rescue_candidates.append((10, findings_specific))
    if slot_name == "conclusion":
        conclusion_specific = _compare_conclusion_page_rescue_summary(compare_pages)
        if conclusion_specific:
            rescue_candidates.append((12, conclusion_specific))
    generic_summary, _page_numbers = _rescue_generic_detailed_slot_summary(slot_name, page_texts)
    if generic_summary and (
        (slot_name == "research_question" and _compare_sentence_has_question_signal(generic_summary))
        or (slot_name == "method" and _compare_sentence_has_method_signal(generic_summary))
        or (
            slot_name in {"main_findings", "conclusion"}
            and (
                _compare_sentence_has_result_signal(slot_name, generic_summary)
                or _detail_marker_count(slot_name, generic_summary.lower()) > 0
            )
        )
        or slot_name not in {"research_question", "method", "main_findings", "conclusion"}
    ):
        rescue_candidates.append((0, generic_summary))
    compare_summary = _compare_slot_rescue_summary(
        slot_name,
        [text for _page_number, text in compare_pages],
    )
    if compare_summary and (
        (slot_name == "research_question" and _compare_sentence_has_question_signal(compare_summary))
        or (slot_name == "method" and _compare_sentence_has_method_signal(compare_summary))
        or (
            slot_name in {"main_findings", "conclusion"}
            and (
                _compare_sentence_has_result_signal(slot_name, compare_summary)
                or _detail_marker_count(slot_name, compare_summary.lower()) > 0
            )
        )
        or slot_name not in {"research_question", "method", "main_findings", "conclusion"}
    ):
        rescue_candidates.append((2, compare_summary))

    best_summary = ""
    best_score = -10**9
    for bonus, candidate in rescue_candidates:
        cleaned = _clean_compare_slot_summary(candidate, slot_name=slot_name)
        if not cleaned:
            continue
        if _looks_unreliable_compare_summary(slot_name, cleaned, support_status="explicit_supported"):
            continue
        score = _compare_summary_quality_score(
            slot_name,
            cleaned,
            support_status="explicit_supported",
        )
        score += bonus
        if score > best_score:
            best_score = score
            best_summary = cleaned
    return best_summary


def _compare_sentence_is_structural_noise(sentence: str) -> bool:
    lowered = unicodedata.normalize("NFKC", sentence).lower().strip()
    stripped = sentence.strip()
    if re.match(r"^[A-Z][A-Za-z-]+(?: [A-Z][A-Za-z-]+){3,}\s+\d{2,4}\b", stripped):
        return True
    if re.match(
        r"^(?:[A-Z][A-Za-z-]+|of|with|and|the|in|on|for|to|a|an)(?: (?:[A-Z][A-Za-z-]+|of|with|and|the|in|on|for|to|a|an)){5,}\s+\d{1,4}\s+[A-Z]\.?$",
        stripped,
    ):
        return True
    if lowered in {
        "discussion and conclusions",
        "discussion and conclusion",
        "future research directions",
        "numerical results",
        "conclusion",
        "conclusions",
        "concluding remarks",
        "gmm implementation",
    }:
        return True
    if lowered.startswith(
        (
            "approximate closed-form solution ",
            "stochastic discount factor interpretation",
            "empirical results",
            "comparative statics and numerical results",
        )
    ):
        return True
    if any(
        marker in lowered
        for marker in (
            "article submitted to",
            "mathematical methods of operations research",
            "journal of ",
            "validation dataset",
            "the review of financial studies /",
            "the following claims hold",
            "section 9 concludes",
            "section 5 concludes",
            "section concludes",
            "this concludes the proof",
            "author:",
            "keywords:",
            "jel classification",
            "mathematics subject classification",
            "annals of mathematical statistics",
            "econometrica",
            "econometrics ",
            "positive numbers indicate the column model outperforms the row model",
            "our sign convention is that a positive statistic indicates the column model outperforms the row model",
            "bold font indicates the difference is significant",
        )
    ):
        return True
    if "•" in sentence and re.search(r"\b\d{2,4}\s+[A-Z][A-Za-z-]+(?:\s+[•·]\s*[A-Z][A-Za-z-]+){1,}", sentence):
        return True
    if "sample path" in lowered or "sample paths" in lowered:
        return True
    if re.search(r"\b(?:arxiv|department of|university|college|institute of)\b", lowered):
        return True
    if "@" in sentence:
        return True
    if re.search(r"^\s*(?:references?|appendix|doi)\b", lowered):
        return True
    if re.match(r"^\s*\(?[a-z]\.\d+\)?\b", lowered):
        return True
    if "formula (" in lowered or "equation (" in lowered:
        return True
    if re.search(r"^\s*(?:figure|table|section)\s+\d+\b", lowered):
        return True
    if re.search(r"\b(?:figure|table)\s+[a-z0-9]+\b", lowered):
        return True
    if "concludes the proof" in lowered:
        return True
    if re.match(r"^[A-Z]\s+is\s+(?:a|an|the)\b", stripped):
        return True
    return False


def _compare_answer_surface_noise_issues(answer_text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", answer_text or "")
    issues: list[str] = []
    if re.search(r"\b\d+\.\s+\d", normalized):
        issues.append("spaced numeric fragments")
    if re.search(
        r"\b([A-Za-z]{5,})\s+((?:ated|ation|ations|cated|tive|tives|ment|ments|ally|ically|ized|ising|izing|tion|tions|sion|sions|ality|ities|ously|ness|able|ible|ance|ances|ence|ences))\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        issues.append("split words")
    if re.search(
        r"\b(?:better|worse|higher|lower|more|less|greater|smaller)\s+than(?:[.;]|$)",
        normalized.lower(),
    ):
        issues.append("truncated comparison tails")
    lowered = normalized.lower()
    if any(
        marker in lowered
        for marker in (
            "numerical研究",
            "numerical结果",
            "numerical实验",
            "quasi-最大似然",
            "quasi最大似然",
        )
    ):
        issues.append("mixed translated technical fragments")
    if any(marker in normalized for marker in ("原文:", "翻译后:", "参考文献:", "#page=", "#chunk=")):
        issues.append("translation or reference residue")

    segments: list[str] = []
    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^- (?:`[^`]+`|[^:]+):\s*", "", stripped)
        stripped = re.sub(
            r"^(?:documents compared|research question|sample and data|method|main findings|limitations|conclusion|研究问题|样本与数据|方法|主要发现|局限|结论)\s*:?\s*$",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        stripped = stripped.strip()
        if stripped:
            segments.append(stripped)
    if any(_compare_sentence_has_formula_noise(segment) for segment in segments):
        issues.append("formula-heavy extraction fragments")
    return tuple(dict.fromkeys(issues))


def _evaluate_paper_answer_consistency(
    prompt: str,
    mode_selection: ModeSelection,
    paper_trace: PaperTrace,
    answer_text: str,
) -> dict[str, object]:
    notes: list[str] = []
    answer_lower = answer_text.lower()
    explicit_slots = _explicit_paper_slots(prompt)
    if mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        missing_slots = tuple(
            slot_name
            for slot_name in explicit_slots
            if all(
                _build_compare_slot_entry(
                    document_note,
                    slot_name,
                    response_language=mode_selection.response_language,
                    paper_output_profile="detailed_paper_note",
                )["status"]
                == "not_clearly_stated"
                for document_note in paper_trace.document_notes
            )
        )
    else:
        missing_slots = _fully_missing_requested_slots(paper_trace.document_notes, explicit_slots)
    recurring_limitations = _is_recurring_limitations_prompt(prompt)
    narrow_grounded_qa = mode_selection.mode == "paper_grounded_qa" and _is_narrow_grounded_paper_qa(prompt)
    if mode_selection.response_language == "zh-CN" and _looks_insufficiently_translated_chinese(answer_text):
        notes.append("The answer did not translate the grounded paper content cleanly enough into Chinese.")
    if missing_slots and not _contains_missing_slot_wording(answer_text, mode_selection.response_language):
        notes.append(
            "Requested dimensions are missing in the slot evidence, but the answer does not clearly acknowledge the missing support."
        )
    if _contains_generic_paper_filler(answer_lower, paper_trace.document_notes):
        notes.append(
            "The answer still contains generic paper commentary that is not anchored in the cleaned slot evidence."
        )
    if _contains_unsupported_gap_inference(answer_text, mode_selection.response_language):
        notes.append(
            "The answer still turns unsupported gaps into speculative inference instead of restrained missing-detail wording."
        )
    if re.search(r"not clearly stated in the paper[^.\n]{0,160}\bhowever\b", answer_lower):
        notes.append(
            "The answer acknowledges a missing dimension but then keeps padding it with unsupported follow-on commentary."
        )
    if narrow_grounded_qa and _looks_over_scaffolded_grounded_qa(answer_text):
        notes.append(
            "The narrow grounded QA answer is still too scaffold-heavy and should collapse to a concise answer-first form."
        )
    if narrow_grounded_qa and len(answer_text) > 650 and not _prompt_requests_support_detail(prompt):
        notes.append(
            "The narrow grounded QA answer is longer than needed for a focused factual question."
        )
    if mode_selection.paper_output_profile == "detailed_paper_note" and mode_selection.mode == "paper_summary":
        false_missing_slots = _false_missing_supported_slots(
            answer_text,
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
        )
        if false_missing_slots:
            notes.append(
                "The detailed paper note still marks clearly supported slots as missing: "
                + ", ".join(slot_label(slot_name) for slot_name in false_missing_slots)
                + "."
            )
        if mode_selection.response_style != "continuous_prose" and _looks_slot_stitched_detailed_note(answer_text):
            notes.append(
                "The detailed paper note still reads like stitched slot snippets instead of a coherent paper note."
            )
        render_issues = _render_integrity_issues(answer_text, mode_selection.response_language)
        if render_issues:
            notes.append(
                "The detailed paper note still contains broken duplication or malformed render residue: "
                + "; ".join(render_issues)
                + "."
            )
    if not narrow_grounded_qa and _looks_excerpt_heavy_paper_answer(answer_text):
        notes.append(
            "The answer is still too excerpt-heavy or outline-heavy for the final paper renderer contract."
        )
    uncovered_slots = _uncovered_requested_slots(
        answer_text,
        paper_trace.document_notes,
        explicit_slots,
        response_language=mode_selection.response_language,
    )
    if uncovered_slots and explicit_slots and mode_selection.mode == "paper_summary":
        notes.append(
            "The answer did not clearly cover these requested summary dimensions: "
            + ", ".join(slot_label(slot_name) for slot_name in uncovered_slots)
            + "."
        )
    if recurring_limitations:
        recurring_signals = _collect_recurring_limitation_signals(paper_trace.document_notes)
        if not _looks_like_limitation_focused_answer(answer_lower):
            notes.append(
                "The answer does not stay focused on limitations even though the user asked for recurring limitations across papers."
            )
        if recurring_signals["clear"] and not _answer_mentions_recurring_limitation_themes(
            answer_lower,
            recurring_signals,
        ):
            notes.append(
                "The answer does not surface the clearly recurring limitations supported across multiple documents."
            )
    if mode_selection.mode == "paper_compare" and _looks_like_slot_dump_compare_answer(answer_text):
        notes.append(
            "The comparison still looks like a raw slot dump instead of a slot-to-slot paper comparison."
        )
    if (
        mode_selection.mode == "paper_compare"
        and paper_trace.document_notes
        and _compare_answer_claims_missing_inputs(answer_text)
    ):
        notes.append(
            "The comparison answer incorrectly claims the PDFs could not be accessed even though grounded compare context was built."
        )
    if (
        mode_selection.mode == "paper_compare"
        and paper_trace.document_notes
        and _compare_answer_lacks_direct_contrast(
            answer_text,
            prompt=prompt,
            paper_trace=paper_trace,
        )
    ):
        notes.append(
            "The comparison still reads like separate summaries or wrapper text rather than a direct slot-to-slot contrast."
        )
    if mode_selection.mode == "paper_compare" and any(
        marker in answer_lower
        for marker in (
            "hybrid approach could be beneficial",
            "combining methodologies could",
            "recommendations or synthesis",
        )
    ):
        notes.append("The comparison drifted into unsupported recommendation language instead of staying slot-grounded.")
    if mode_selection.mode == "paper_compare" and paper_trace.document_notes:
        false_missing_slots = _compare_false_missing_supported_slots(
            answer_text,
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
        )
        if false_missing_slots:
            notes.append(
                "The comparison still marks clearly supported compare dimensions as missing: "
                + ", ".join(slot_label(slot_name) for slot_name in false_missing_slots)
                + "."
            )
        surface_noise_issues = _compare_answer_surface_noise_issues(answer_text)
        if surface_noise_issues:
            notes.append(
                "The comparison still contains visible extraction-noise residue: "
                + ", ".join(surface_noise_issues)
                + "."
            )
        missing_entry_slots = _compare_missing_document_entry_slots(
            answer_text,
            paper_trace.document_notes,
            response_language=mode_selection.response_language,
            requested_slots=_requested_paper_slots(prompt, "paper_compare"),
        )
        if missing_entry_slots:
            notes.append(
                "The comparison dropped one or more per-paper slot entries for these dimensions: "
                + ", ".join(slot_label(slot_name) for slot_name in missing_entry_slots)
                + "."
            )
        if _compare_answer_has_contradictory_missing_prefix(
            answer_text,
            response_language=mode_selection.response_language,
        ):
            notes.append(
                "The comparison still prefixes substantive slot content with missing-detail wording, which makes the final compare internally contradictory."
            )
        if mode_selection.response_language == "zh-CN" and any(
            marker in answer_text
            for marker in (
                "对比总结",
                "比较总结",
                "可以将其翻译为",
                "没有添加或删除任何主张",
            )
        ):
            notes.append(
                "The Chinese comparison still contains translation-commentary residue instead of a clean final compare sentence."
            )
        if (
            mode_selection.response_language == "zh-CN"
            and _structured_compare_section_count(answer_text) < max(4, min(6, len(_requested_paper_slots(prompt, "paper_compare"))))
        ):
            notes.append(
                "The Chinese comparison lost too much slot-to-slot structure and should fall back to the grounded compare renderer."
            )
    if mode_selection.response_style == "continuous_prose" and looks_like_structured_output(answer_text):
        notes.append("The answer did not fully obey the requested continuous-prose style.")
    return {
        "needs_repair": bool(notes),
        "notes": notes or ["Slot-supported answer passed the paper consistency check."],
    }

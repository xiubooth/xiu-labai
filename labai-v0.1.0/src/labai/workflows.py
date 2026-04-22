from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Sequence

from labai.config import LabaiConfig
from labai.editing import (
    WorkspaceEditPlan,
    build_workspace_check_plan,
    build_workspace_edit_plan,
    classify_output_intent,
    inspect_git_workspace,
)
from labai.execution import build_local_runtime_report
from labai.papers import build_paper_library_report
from labai.research import evaluate_research_readiness
from labai.research.modes import ModeSelection, select_mode
from labai.workspace import WorkspaceAccessError, WorkspaceAccessManager

WorkflowKind = Literal["answer_only", "edit_capable"]
WorkflowOutputType = Literal[
    "paper_summary",
    "paper_compare",
    "compiled_prompt",
    "project_overview",
    "repro_summary",
    "workspace_edit",
    "workspace_verification",
]

WORKFLOW_EXECUTION_FLOW = "resolve -> preview -> execute -> summarize -> artifact"


class WorkflowCommandError(RuntimeError):
    """Raised when a workflow command cannot resolve inputs or execute safely."""


@dataclass(frozen=True)
class WorkflowCommandSpec:
    name: str
    purpose: str
    input_schema: str
    default_mode: str
    workflow_kind: WorkflowKind
    may_create_deliverables: bool
    output_type: WorkflowOutputType
    output_contract: str
    resolver: Callable[[LabaiConfig, tuple[str, ...]], "WorkflowResolution"]
    supports_preview: bool = True
    absorbs_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeprecatedWorkflowCommand:
    name: str
    guidance: str


@dataclass(frozen=True)
class WorkflowResolution:
    spec: WorkflowCommandSpec
    prompt: str
    command_label: str
    command_path: str
    target_workspace_root: str
    resolved_inputs: tuple[str, ...]
    accepted_paths: tuple[str, ...]
    selected_mode: str
    selected_model: str
    read_strategy: str
    response_style: str
    response_language: str
    paper_output_profile: str
    paper_output_profile_reason: str
    output_intent: str
    output_intent_reason: str
    expected_output_type: WorkflowOutputType
    config_for_execution: LabaiConfig
    workflow_notes: tuple[str, ...] = ()
    planned_reads: tuple[str, ...] = ()
    planned_modifications: tuple[str, ...] = ()
    planned_creations: tuple[str, ...] = ()
    expected_deliverables: tuple[str, ...] = ()
    expected_checks: tuple[str, ...] = ()
    git_repo_detected: bool = False
    git_repo_root: str = ""
    preview_document_count: int = 0


def list_workflow_specs() -> tuple[WorkflowCommandSpec, ...]:
    return _WORKFLOW_SPECS


def list_deprecated_workflow_commands() -> tuple[DeprecatedWorkflowCommand, ...]:
    return _DEPRECATED_WORKFLOW_COMMANDS


def get_deprecated_workflow_command(name: str) -> DeprecatedWorkflowCommand | None:
    normalized = name.strip().lower()
    for command in _DEPRECATED_WORKFLOW_COMMANDS:
        if command.name == normalized:
            return command
    return None


def get_workflow_spec(name: str) -> WorkflowCommandSpec:
    normalized = name.strip().lower()
    for spec in _WORKFLOW_SPECS:
        if spec.name == normalized:
            return spec
    available = ", ".join(spec.name for spec in _WORKFLOW_SPECS)
    raise WorkflowCommandError(
        f"Unknown workflow command '{name}'. Available commands: {available}"
    )


def resolve_workflow_command(
    config: LabaiConfig,
    command_name: str,
    arguments: Sequence[str],
) -> WorkflowResolution:
    spec = get_workflow_spec(command_name)
    return spec.resolver(config, tuple(arguments))


def build_workflow_trace(resolution: WorkflowResolution, *, preview: bool) -> dict[str, object]:
    return {
        "active": True,
        "command": resolution.spec.name,
        "command_label": resolution.command_label,
        "command_path": resolution.command_path,
        "purpose": resolution.spec.purpose,
        "input_schema": resolution.spec.input_schema,
        "preview": preview,
        "workflow_kind": resolution.spec.workflow_kind,
        "may_create_deliverables": resolution.spec.may_create_deliverables,
        "absorbs_capabilities": list(resolution.spec.absorbs_capabilities),
        "workflow_path": WORKFLOW_EXECUTION_FLOW,
        "resolved_inputs": list(resolution.resolved_inputs),
        "accepted_paths": list(resolution.accepted_paths),
        "target_workspace_root": resolution.target_workspace_root,
        "default_internal_mode": resolution.spec.default_mode,
        "selected_mode": resolution.selected_mode,
        "selected_model": resolution.selected_model,
        "read_strategy": resolution.read_strategy,
        "response_style": resolution.response_style,
        "response_language": resolution.response_language,
        "paper_output_profile": resolution.paper_output_profile,
        "paper_output_profile_reason": resolution.paper_output_profile_reason,
        "output_intent": resolution.output_intent,
        "output_intent_reason": resolution.output_intent_reason,
        "expected_output_type": resolution.expected_output_type,
        "planned_reads": list(resolution.planned_reads),
        "planned_modifications": list(resolution.planned_modifications),
        "planned_creations": list(resolution.planned_creations),
        "expected_deliverables": list(resolution.expected_deliverables),
        "expected_checks": list(resolution.expected_checks),
        "preview_document_count": resolution.preview_document_count,
        "git_repo_detected": resolution.git_repo_detected,
        "git_repo_root": resolution.git_repo_root,
        "notes": list(resolution.workflow_notes),
    }


def render_workflow_preview(
    resolution: WorkflowResolution,
    *,
    session_path: Path,
    audit_path: Path,
) -> list[str]:
    lines = [
        f"labai workflow {resolution.spec.name} --preview",
        f"purpose: {resolution.spec.purpose}",
        f"workflow_path: {WORKFLOW_EXECUTION_FLOW}",
        f"target_workspace_root: {resolution.target_workspace_root}",
        f"selected_mode: {resolution.selected_mode}",
        f"selected_model: {resolution.selected_model}",
        f"read_strategy: {resolution.read_strategy}",
        f"paper_output_profile: {resolution.paper_output_profile}",
        f"paper_output_profile_reason: {resolution.paper_output_profile_reason}",
        f"output_intent: {resolution.output_intent}",
        f"expected_output_type: {resolution.expected_output_type}",
        f"may_create_deliverables: {str(resolution.spec.may_create_deliverables).lower()}",
        "writes: no | preview mode only",
    ]
    if resolution.resolved_inputs:
        lines.append("resolved_inputs:")
        lines.extend(f"- {item}" for item in resolution.resolved_inputs)
    if resolution.accepted_paths:
        lines.append("accepted_paths:")
        lines.extend(f"- {item}" for item in resolution.accepted_paths)
    if resolution.preview_document_count:
        lines.append(f"preview_document_count: {resolution.preview_document_count}")
    if resolution.planned_reads:
        lines.append("planned_reads:")
        lines.extend(f"- {item}" for item in resolution.planned_reads)
    if resolution.planned_modifications:
        lines.append("planned_modifications:")
        lines.extend(f"- {item}" for item in resolution.planned_modifications)
    if resolution.planned_creations:
        lines.append("planned_creations:")
        lines.extend(f"- {item}" for item in resolution.planned_creations)
    if resolution.expected_deliverables:
        lines.append("expected_deliverables:")
        lines.extend(f"- {item}" for item in resolution.expected_deliverables)
    if resolution.expected_checks:
        lines.append("expected_checks:")
        lines.extend(f"- {item}" for item in resolution.expected_checks)
    if resolution.workflow_notes:
        lines.append("notes:")
        lines.extend(f"- {item}" for item in resolution.workflow_notes)
    lines.append(f"session_file: {session_path}")
    lines.append(f"audit_log: {audit_path}")
    return lines


def render_workflow_result(
    resolution: WorkflowResolution,
    result,
    *,
    session_path: Path,
    audit_path: Path,
    artifact,
) -> list[str]:
    lines = [f"labai workflow {resolution.spec.name}"]
    if result.workspace_trace.edit_intent:
        lines.extend(
            [
                "plan:",
                f"- summary={result.workspace_trace.edit_plan_summary or resolution.spec.output_contract}",
            ]
        )
        if result.workspace_trace.planned_reads:
            lines.append(f"- reads={', '.join(result.workspace_trace.planned_reads)}")
        if result.workspace_trace.planned_modifications:
            lines.append(
                f"- modifies={', '.join(result.workspace_trace.planned_modifications)}"
            )
        if result.workspace_trace.planned_creations:
            lines.append(
                f"- creates={', '.join(result.workspace_trace.planned_creations)}"
            )
        if result.workspace_trace.intended_changes:
            lines.append("intended_changes:")
            lines.extend(f"- {item}" for item in result.workspace_trace.intended_changes)
        lines.append("")

    lines.extend(
        [
            "result:",
            result.final_answer or "(no answer)",
            "",
        ]
    )
    if result.workspace_trace.edit_intent:
        lines.append("post_change_summary:")
        if result.workspace_trace.modified_files:
            lines.append(f"- changed_files={', '.join(result.workspace_trace.modified_files)}")
        if result.workspace_trace.created_files:
            lines.append(f"- created_files={', '.join(result.workspace_trace.created_files)}")
        if result.workspace_trace.skipped_files:
            lines.append(f"- skipped_files={', '.join(result.workspace_trace.skipped_files)}")
        if result.workspace_trace.planned_checks:
            lines.append(f"- planned_checks={', '.join(result.workspace_trace.planned_checks)}")
        if result.workspace_trace.checks_run:
            lines.append(f"- checks_run={', '.join(result.workspace_trace.checks_run)}")
        if result.workspace_trace.check_status != "not_run":
            lines.append(f"- check_status={result.workspace_trace.check_status}")
        if result.workspace_trace.check_failures:
            lines.append("check_failures:")
            lines.extend(f"- {item}" for item in result.workspace_trace.check_failures)
        if result.workspace_trace.repair_rounds:
            lines.append(f"- repair_rounds={result.workspace_trace.repair_rounds}")
        if result.workspace_trace.file_change_summaries:
            lines.append("file_summaries:")
            lines.extend(f"- {item}" for item in result.workspace_trace.file_change_summaries)
        if result.workspace_trace.rollback_notes:
            lines.append("rollback_notes:")
            lines.extend(f"- {item}" for item in result.workspace_trace.rollback_notes)
        if result.workspace_trace.git_repo_detected:
            lines.append(f"- git_repo_root={result.workspace_trace.git_repo_root}")
            if result.workspace_trace.git_changed_files:
                lines.append(
                    f"- git_changed_files={', '.join(result.workspace_trace.git_changed_files)}"
                )
            if result.workspace_trace.git_untracked_files:
                lines.append(
                    f"- git_untracked_files={', '.join(result.workspace_trace.git_untracked_files)}"
                )
            if result.workspace_trace.git_commit_message_draft:
                lines.append(
                    f"- git_commit_message_draft={result.workspace_trace.git_commit_message_draft}"
                )
        lines.append("")

    lines.extend(
        [
            "what_happened:",
            f"- status={result.status}",
            f"- mode={result.selected_mode}",
            f"- model={result.provider_model or '(unknown)'}",
            f"- read_strategy={result.read_strategy}",
            f"- paper_output_profile={resolution.paper_output_profile}",
            f"- runtime={result.runtime_used}",
            f"- runtime_fallback={_format_runtime_fallback(result)}",
            f"- output_intent={result.output_intent}",
            f"- workspace_root={result.workspace_trace.active_workspace_root}",
        ]
    )
    if resolution.resolved_inputs:
        lines.append(f"- resolved_inputs={', '.join(resolution.resolved_inputs)}")
    if result.paper_trace.active:
        lines.append(f"- paper_documents={len(result.paper_trace.discovered_documents)}")
        lines.append(f"- paper_targets={len(result.paper_trace.target_paths)}")
    if result.workspace_trace.edit_intent:
        lines.append(f"- edit_intent={result.workspace_trace.edit_intent}")
        if result.workspace_trace.edit_plan_summary:
            lines.append(f"- plan_summary={result.workspace_trace.edit_plan_summary}")
        if result.workspace_trace.check_status != "not_run":
            lines.append(f"- check_status={result.workspace_trace.check_status}")
        if result.workspace_trace.repair_rounds:
            lines.append(f"- repair_rounds={result.workspace_trace.repair_rounds}")
    if result.workspace_trace.modified_files:
        lines.append(f"- changed_files={', '.join(result.workspace_trace.modified_files)}")
    if result.workspace_trace.created_files:
        lines.append(f"- created_files={', '.join(result.workspace_trace.created_files)}")
    if result.workspace_trace.skipped_files:
        lines.append(f"- skipped_files={', '.join(result.workspace_trace.skipped_files)}")
    if result.workspace_trace.user_deliverable_file:
        lines.append(f"- deliverable_file={result.workspace_trace.user_deliverable_file}")
    if artifact.path:
        lines.append(f"- artifact_file={artifact.path}")
    lines.append(f"- artifact_status={artifact.status}")
    if result.outcome_summary:
        lines.append(f"- outcome_summary={result.outcome_summary}")
    lines.append(f"- session_file={session_path}")
    lines.append(f"- audit_log={audit_path}")
    return lines


def build_preview_metadata(config: LabaiConfig, resolution: WorkflowResolution) -> dict[str, object]:
    readiness = evaluate_research_readiness(resolution.config_for_execution)
    runtime_report = build_local_runtime_report(resolution.config_for_execution)
    paper_report = build_paper_library_report(resolution.config_for_execution)
    return {
        "timestamp": _utc_timestamp(),
        "readiness_status": readiness.status,
        "doctor_status": runtime_report.doctor_status,
        "paper_status": paper_report.status,
        "runtime_summary": runtime_report.summary,
        "paper_summary": paper_report.summary,
    }


def _resolve_read_paper(
    config: LabaiConfig,
    arguments: tuple[str, ...],
) -> WorkflowResolution:
    if len(arguments) != 1:
        raise WorkflowCommandError("read-paper expects exactly one PDF path.")
    access_manager = WorkspaceAccessManager(config)
    pdf_path = _resolve_pdf_path(access_manager, arguments[0], label="read-paper")
    display_path = _display_for_prompt(access_manager, pdf_path)
    prompt = (
        f"Treat {display_path} as a paper that needs a complete read of the whole paper. "
        "Write a detailed grounded paper note in English for an RA. Preserve concrete supported details, including sample/data setup, "
        "date ranges, train/validation/test splits when stated, method families, main findings, limitations, and conclusion details. "
        "Do not omit important supported details for brevity. Use only what the paper clearly supports; if a detail is not explicit in "
        "the paper, say Not clearly stated in the paper. Do not add generic machine-learning or finance commentary."
    )
    return _build_resolution(
        config,
        spec=get_workflow_spec("read-paper"),
        prompt=prompt,
        resolved_inputs=(display_path,),
        accepted_paths=(display_path,),
        preview_document_count=1,
    )


def _resolve_compare_papers(
    config: LabaiConfig,
    arguments: tuple[str, ...],
) -> WorkflowResolution:
    if len(arguments) < 2:
        raise WorkflowCommandError("compare-papers expects at least two PDF paths.")
    access_manager = WorkspaceAccessManager(config)
    resolved = tuple(
        _resolve_pdf_path(access_manager, item, label="compare-papers") for item in arguments
    )
    display_paths = tuple(_display_for_prompt(access_manager, item) for item in resolved)
    prompt = (
        f"Compare {_join_for_prompt(display_paths)} on goal, sample/data, method, findings, limitations, and conclusion. "
        "Write a detailed grounded paper note in English for an RA. Compare papers slot-to-slot rather than blending them into broad themes. "
        "Preserve concrete supported asymmetries and important paper-specific details instead of flattening them away. "
        "If a comparison point is not explicit in the consulted PDFs, say Not clearly stated in the paper rather than inferring."
    )
    return _build_resolution(
        config,
        spec=get_workflow_spec("compare-papers"),
        prompt=prompt,
        resolved_inputs=display_paths,
        accepted_paths=display_paths,
        preview_document_count=len(display_paths),
    )


def _resolve_onboard_project(
    config: LabaiConfig,
    arguments: tuple[str, ...],
) -> WorkflowResolution:
    workspace_root = _resolve_workspace_root_argument(config, arguments, label="onboard-project")
    prompt = (
        "Onboard a new RA to this project. Keep it practical and grounded in the consulted workspace evidence only. "
        "Base the summary on broad workspace coverage rather than a few sampled files, and make it clear when the repo contains repeated work areas or many scripts. "
        "Cover the project purpose, main directories or modules, likely entry points, visible config or environment assumptions, "
        "risks or missing pieces, what to read first, and the most useful next steps. Respond in English. "
        "If something is not visible in the files, say not confirmed."
    )
    preview_notes = _workspace_preview_notes(config, workspace_root)
    return _build_resolution(
        config,
        spec=get_workflow_spec("onboard-project"),
        prompt=prompt,
        resolved_inputs=(str(workspace_root),),
        accepted_paths=(str(workspace_root),),
        workspace_root=workspace_root,
        workflow_notes=preview_notes,
    )


def _resolve_repro_check(
    config: LabaiConfig,
    arguments: tuple[str, ...],
) -> WorkflowResolution:
    workspace_root = _resolve_workspace_root_argument(config, arguments, label="repro-check")
    prompt = (
        "Assess how reproducible this project looks for a new RA. Explain what appears ready, "
        "what environment assumptions are visible, what may be missing, and the most practical next steps. Respond in English. "
        "Base the assessment only on visible workspace evidence, and say not confirmed when the files do not show something."
    )
    preview_notes = _workspace_preview_notes(config, workspace_root)
    return _build_resolution(
        config,
        spec=get_workflow_spec("repro-check"),
        prompt=prompt,
        resolved_inputs=(str(workspace_root),),
        accepted_paths=(str(workspace_root),),
        workspace_root=workspace_root,
        workflow_notes=preview_notes,
    )


def _resolve_edit_task(
    config: LabaiConfig,
    arguments: tuple[str, ...],
) -> WorkflowResolution:
    if len(arguments) != 1:
        raise WorkflowCommandError("edit-task expects a single quoted instruction string.")
    instruction = arguments[0].strip()
    if not instruction:
        raise WorkflowCommandError("edit-task instruction must not be empty.")
    prompt = (
        "In this active workspace, carry out the following coding-oriented edit task with a short plan first, "
        f"then apply the changes: {instruction}"
    )
    preview_notes = (
        "This workflow reuses the Phase 9 plan-first, grouped-apply edit loop.",
        "Phase 13 extends it with structured file output, targeted checks, and a bounded repair loop.",
    )
    resolution = _build_resolution(
        config,
        spec=get_workflow_spec("edit-task"),
        prompt=prompt,
        resolved_inputs=(instruction,),
        accepted_paths=(),
        workflow_notes=preview_notes,
    )
    if not resolution.planned_modifications and not resolution.planned_creations:
        raise WorkflowCommandError(
            "edit-task could not resolve any planned file modifications or creations from the instruction."
        )
    return replace(
        resolution,
        prompt=_build_edit_task_prompt(instruction, resolution),
    )


def _resolve_compile_prompt(
    config: LabaiConfig,
    arguments: tuple[str, ...],
) -> WorkflowResolution:
    if len(arguments) != 1:
        raise WorkflowCommandError("compile-prompt expects a single quoted need or rough request.")
    rough_need = arguments[0].strip()
    if not rough_need:
        raise WorkflowCommandError("compile-prompt input must not be empty.")
    prompt = (
        "Turn the following rough need into a stronger executable prompt for another coding or research agent. "
        "Respond entirely in English and do not solve the task itself. "
        "Return these sections exactly: Strong prompt, Constraints, Acceptance criteria, Missing assumptions or open questions, "
        "Recommendation, Compact variant, Strict executable variant. "
        "Preserve the user's scope, keep it practical for a real RA handoff, and avoid prompt-engineering theory. "
        f"User need: {rough_need}"
    )
    return _build_resolution(
        config,
        spec=get_workflow_spec("compile-prompt"),
        prompt=prompt,
        resolved_inputs=(rough_need,),
        accepted_paths=(),
    )


def _resolve_verify_workspace(
    config: LabaiConfig,
    arguments: tuple[str, ...],
) -> WorkflowResolution:
    workspace_root = _resolve_workspace_root_argument(config, arguments, label="verify-workspace")
    prompt = (
        "Treat this as a workspace a new RA may need to start using today. "
        "Classify the readiness state as ready, ready_with_gaps, partially_ready, blocked, or uncertain when the evidence supports it. "
        "Cover why that status was chosen, likely entry points or run surfaces, visible config or dependency assumptions, what is missing, what looks risky, what to read first, and the first three practical next steps. Respond in English. "
        "Use only the consulted workspace evidence, and say not confirmed when a detail is not visible in the files."
    )
    preview_notes = _workspace_preview_notes(config, workspace_root)
    return _build_resolution(
        config,
        spec=get_workflow_spec("verify-workspace"),
        prompt=prompt,
        resolved_inputs=(str(workspace_root),),
        accepted_paths=(str(workspace_root),),
        workspace_root=workspace_root,
        workflow_notes=preview_notes,
    )


def _build_resolution(
    config: LabaiConfig,
    *,
    spec: WorkflowCommandSpec,
    prompt: str,
    resolved_inputs: tuple[str, ...],
    accepted_paths: tuple[str, ...],
    workspace_root: Path | None = None,
    workflow_notes: tuple[str, ...] = (),
    preview_document_count: int = 0,
) -> WorkflowResolution:
    config_for_execution = _override_workflow_config(
        config,
        workspace_root=workspace_root,
        mode_override=spec.default_mode,
    )
    access_manager = WorkspaceAccessManager(config_for_execution)
    mode_selection = select_mode(config_for_execution, prompt)
    edit_plan = build_workspace_edit_plan(prompt, mode_selection, access_manager)
    check_plan = build_workspace_check_plan(
        prompt,
        access_manager.active_workspace_root,
        planned_modifications=edit_plan.planned_modifications,
        planned_creations=edit_plan.planned_creations,
    )
    output_intent, output_intent_reason = classify_output_intent(edit_plan)
    expected_deliverables = edit_plan.planned_creations
    return WorkflowResolution(
        spec=spec,
        prompt=prompt,
        command_label=f"labai workflow {spec.name}",
        command_path=WORKFLOW_EXECUTION_FLOW,
        target_workspace_root=str(access_manager.active_workspace_root),
        resolved_inputs=resolved_inputs,
        accepted_paths=accepted_paths,
        selected_mode=mode_selection.mode,
        selected_model=mode_selection.selected_model,
        read_strategy=mode_selection.read_strategy,
        response_style=mode_selection.response_style,
        response_language=mode_selection.response_language,
        paper_output_profile=mode_selection.paper_output_profile,
        paper_output_profile_reason=mode_selection.paper_output_profile_reason,
        output_intent=output_intent,
        output_intent_reason=output_intent_reason,
        expected_output_type=spec.output_type,
        config_for_execution=config_for_execution,
        workflow_notes=workflow_notes,
        planned_reads=edit_plan.planned_reads,
        planned_modifications=edit_plan.planned_modifications,
        planned_creations=edit_plan.planned_creations,
        expected_deliverables=expected_deliverables,
        expected_checks=tuple(check.summary for check in check_plan),
        preview_document_count=preview_document_count,
    )


def _build_edit_task_prompt(instruction: str, resolution: WorkflowResolution) -> str:
    file_targets = (*resolution.planned_modifications, *resolution.planned_creations)
    file_lines = "\n".join(f"- {item}" for item in file_targets)
    check_lines = "\n".join(f"- {item}" for item in _render_edit_task_prompt_checks(resolution.expected_checks))
    if not check_lines:
        check_lines = "- No automatic checks were planned for this task."
    return (
        "You are carrying out a focused coding task inside the active workspace.\n"
        "Start by reasoning about the task briefly, then produce structured file updates that can be applied directly.\n"
        "Use this exact format for every file you want to change or create:\n"
        "=== FILE: relative/path ===\n"
        "```language\n"
        "<full file content>\n"
        "```\n"
        "After the file blocks, include a short section titled SUMMARY with concise notes about what changed.\n"
        "Only include files that should actually change. Preserve unrelated content and keep changes scoped to the request.\n"
        "Task:\n"
        f"{instruction}\n\n"
        "Planned file targets:\n"
        f"{file_lines}\n\n"
        "Expected automatic checks after apply:\n"
        f"{check_lines}\n"
    )


def _render_edit_task_prompt_checks(checks: tuple[str, ...]) -> tuple[str, ...]:
    rendered: list[str] = []
    for item in checks:
        lowered = item.lower()
        if "pytest" in lowered:
            rendered.append("Run the planned targeted pytest checks.")
            continue
        if "py_compile" in lowered or "python syntax" in lowered:
            rendered.append("Run the planned Python syntax checks.")
            continue
        rendered.append("Run the planned automatic checks.")
    return tuple(dict.fromkeys(rendered))


def _override_workflow_config(
    config: LabaiConfig,
    *,
    workspace_root: Path | None,
    mode_override: str,
) -> LabaiConfig:
    resolved_workspace_root = (
        workspace_root.resolve()
        if workspace_root is not None
        else config.workspace.active_workspace_root.resolve()
    )
    workspace_settings = replace(
        config.workspace,
        active_workspace_root=resolved_workspace_root,
    )
    research_settings = replace(
        config.research,
        mode_override=mode_override,
    )
    return replace(
        config,
        workspace=workspace_settings,
        research=research_settings,
    )


def _resolve_pdf_path(
    access_manager: WorkspaceAccessManager,
    raw_path: str,
    *,
    label: str,
) -> Path:
    resolved = _resolve_existing_path(access_manager, raw_path, label=label)
    if not resolved.is_file() or resolved.suffix.lower() != ".pdf":
        raise WorkflowCommandError(f"{label} expects a PDF file path: {raw_path}")
    return resolved


def _resolve_existing_path(
    access_manager: WorkspaceAccessManager,
    raw_path: str,
    *,
    label: str,
) -> Path:
    candidate = access_manager.resolve_prompt_path(raw_path, must_exist=True)
    if candidate is None:
        raise WorkflowCommandError(
            f"{label} could not resolve an allowed existing path from: {raw_path}"
        )
    return candidate.resolve()


def _resolve_workspace_root_argument(
    config: LabaiConfig,
    arguments: tuple[str, ...],
    *,
    label: str,
) -> Path:
    access_manager = WorkspaceAccessManager(config)
    if not arguments:
        return access_manager.active_workspace_root
    if len(arguments) != 1:
        raise WorkflowCommandError(f"{label} accepts at most one optional path argument.")
    try:
        resolved = access_manager.resolve_user_path(arguments[0], for_write=False, must_exist=True)
    except WorkspaceAccessError as exc:
        explicit_root = access_manager.resolve_explicit_read_root(arguments[0])
        if explicit_root is None:
            raise WorkflowCommandError(str(exc)) from exc
        resolved = explicit_root
    if resolved.is_file():
        return resolved.parent.resolve()
    return resolved.resolve()


def _display_for_prompt(access_manager: WorkspaceAccessManager, path: Path) -> str:
    return access_manager.display_path(path)


def _expand_pdf_preview_targets(targets: tuple[Path, ...]) -> tuple[Path, ...]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        resolved = target.resolve()
        if resolved.is_file() and resolved.suffix.lower() == ".pdf":
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(resolved)
            continue
        if not resolved.is_dir():
            continue
        for pdf_path in sorted(resolved.rglob("*.pdf")):
            if pdf_path not in seen:
                seen.add(pdf_path)
                discovered.append(pdf_path)
    return tuple(discovered)


def _document_preview_notes(paths: tuple[Path, ...]) -> tuple[str, ...]:
    if not paths:
        return ("No previewable PDF documents were found under the accepted targets.",)
    shown = [str(path) for path in paths[:4]]
    notes = [f"Preview found {len(paths)} PDF document(s)."]
    notes.extend(f"Document: {item}" for item in shown)
    if len(paths) > 4:
        notes.append(f"Plus {len(paths) - 4} more document(s).")
    return tuple(notes)


def _workspace_preview_notes(config: LabaiConfig, workspace_root: Path) -> tuple[str, ...]:
    config_for_workspace = _override_workflow_config(
        config,
        workspace_root=workspace_root,
        mode_override=config.research.mode_override or "repo_overview",
    )
    access_manager = WorkspaceAccessManager(config_for_workspace)
    write_decision = access_manager.describe_path(workspace_root, for_write=True)
    git_summary = inspect_git_workspace(workspace_root)
    notes = [
        f"Workspace write access is {'allowed' if write_decision.allowed else 'blocked'} for {workspace_root}.",
        f"Git repo detected: {'yes' if git_summary.detected else 'no'}.",
    ]
    if git_summary.repo_root:
        notes.append(f"Git root: {git_summary.repo_root}")
    return tuple(notes)


def _format_runtime_fallback(result) -> str:
    if not result.runtime_fallback.applied:
        return "none"
    return (
        f"requested={result.runtime_fallback.requested_runtime} | "
        f"used={result.runtime_fallback.active_runtime} | "
        f"fallback={result.runtime_fallback.fallback_runtime} | "
        f"reason={result.runtime_fallback.reason}"
    )


def _join_for_prompt(items: tuple[str, ...]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_WORKFLOW_SPECS = (
    WorkflowCommandSpec(
        name="read-paper",
        purpose="Read one paper or PDF as a complete practical RA summary.",
        input_schema="read-paper <pdf>",
        default_mode="paper_summary",
        workflow_kind="answer_only",
        may_create_deliverables=False,
        output_type="paper_summary",
        output_contract="Practical grounded paper summary with a short what-happened summary.",
        resolver=_resolve_read_paper,
    ),
    WorkflowCommandSpec(
        name="compare-papers",
        purpose="Compare two or more papers on goal, method, limitations, and findings.",
        input_schema="compare-papers <pdf_a> <pdf_b> [more ...]",
        default_mode="paper_compare",
        workflow_kind="answer_only",
        may_create_deliverables=False,
        output_type="paper_compare",
        output_contract="Grounded paper comparison with a short what-happened summary.",
        resolver=_resolve_compare_papers,
    ),
    WorkflowCommandSpec(
        name="onboard-project",
        purpose="Give a practical project overview for a new RA.",
        input_schema="onboard-project [path]",
        default_mode="project_onboarding",
        workflow_kind="answer_only",
        may_create_deliverables=False,
        output_type="project_overview",
        output_contract="Practical workspace onboarding summary with a short what-happened summary.",
        resolver=_resolve_onboard_project,
    ),
    WorkflowCommandSpec(
        name="repro-check",
        purpose="Assess how reproducible a project looks and what is likely missing.",
        input_schema="repro-check [path]",
        default_mode="implementation_plan",
        workflow_kind="answer_only",
        may_create_deliverables=False,
        output_type="repro_summary",
        output_contract="Practical reproducibility summary with likely steps and missing pieces.",
        resolver=_resolve_repro_check,
    ),
    WorkflowCommandSpec(
        name="edit-task",
        purpose="Run one focused coding or workspace edit task with plan, apply, checks, and bounded retries.",
        input_schema='edit-task "<instruction>"',
        default_mode="workspace_edit",
        workflow_kind="edit_capable",
        may_create_deliverables=True,
        output_type="workspace_edit",
        output_contract="Short plan first, then result and post-change summary.",
        resolver=_resolve_edit_task,
        absorbs_capabilities=("debug triage", "issue-to-plan", "review-diff"),
    ),
    WorkflowCommandSpec(
        name="verify-workspace",
        purpose="Summarize whether the current or specified workspace looks ready for labai usage.",
        input_schema="verify-workspace [path]",
        default_mode="workspace_verification",
        workflow_kind="answer_only",
        may_create_deliverables=False,
        output_type="workspace_verification",
        output_contract="Practical workspace readiness summary plus next-step advice.",
        resolver=_resolve_verify_workspace,
    ),
    WorkflowCommandSpec(
        name="compile-prompt",
        purpose="Turn a vague Chinese or English need into a stronger executable English prompt.",
        input_schema='compile-prompt "<rough need>"',
        default_mode="prompt_compiler",
        workflow_kind="answer_only",
        may_create_deliverables=False,
        output_type="compiled_prompt",
        output_contract="Executable prompt package with constraints, acceptance criteria, and compact/strict variants.",
        resolver=_resolve_compile_prompt,
    ),
)


_DEPRECATED_WORKFLOW_COMMANDS = (
    DeprecatedWorkflowCommand(
        name="paper-limitations",
        guidance=(
            "paper-limitations has been deprecated from the top-level workflow surface. "
            "Use `labai ask` for cross-paper recurring limitation synthesis, or use `read-paper` / `compare-papers` for focused paper work."
        ),
    ),
)

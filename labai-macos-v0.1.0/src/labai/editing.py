from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import sys
import tomllib
from typing import TYPE_CHECKING, Literal

from labai.execution import RuntimeAdapterError
from labai.runtime_exec import run_runtime_command
from labai.structured_edits import (
    apply_unified_diff_text,
    extract_structured_file_block_specs,
)
from labai.typed_validation import build_typed_validation_result
from labai.workspace import WorkspaceAccessError, WorkspaceAccessManager, _iter_prompt_candidates

if TYPE_CHECKING:
    from labai.config import LabaiConfig
    from labai.research.modes import ModeSelection


EditIntentName = Literal["", "deliverable_update", "direct_file_edit", "multi_file_edit"]
WorkspaceCheckStatus = Literal["passed", "failed", "blocked"]
WorkspaceTaskContract = dict[str, object]

_TEXT_WRITE_SUFFIXES = frozenset({".md", ".txt", ".csv", ".json", ".py", ".ipynb", ".toml", ".yaml", ".yml", ".xlsx"})
_CREATE_TOKENS = (
    "create ",
    "generate ",
    "write ",
    "save ",
    "save as ",
    "export ",
    "write this to",
    "write the result to",
    "put the result in",
    "\u4fdd\u5b58\u5230",
    "\u4fdd\u5b58\u6210",
    "\u5199\u6210",
    "\u5199\u5230",
    "\u751f\u6210",
    "\u521b\u5efa",
    "\u65b0\u5efa",
    "\u5bfc\u51fa",
)
_UPDATE_TOKENS = (
    "open ",
    "fix ",
    "modify ",
    "edit ",
    "update ",
    "change ",
    "repair ",
    "refactor ",
    "implement ",
    "synchronize ",
    "sync ",
    "refine ",
    "tidy ",
    "improve ",
    "add ",
    "\u4fee\u590d",
    "\u4fee\u6539",
    "\u7f16\u8f91",
    "\u66f4\u65b0",
    "\u91cd\u6784",
    "\u6539\u5199",
    "\u8865\u5145",
    "\u52a0\u4e0a",
)
_SAME_FOLDER_TOKENS = (
    "same folder",
    "same directory",
    "\u540c\u4e00\u6587\u4ef6\u5939",
    "\u540c\u4e00\u6587\u4ef6\u5939\u4e0b",
    "\u540c\u4e00\u76ee\u5f55",
    "\u540c\u76ee\u5f55",
)
_WORKSPACE_ROOT_TOKENS = (
    "current directory",
    "current workspace",
    "project root",
    "this project root",
    "in this workspace",
    "\u5f53\u524d\u76ee\u5f55",
    "\u5f53\u524d\u5de5\u4f5c\u533a",
    "\u5728\u8fd9\u4e2a\u9879\u76ee\u76ee\u5f55",
)
_DOCSTRING_TOKENS = (
    "module docstring",
    "docstring at the top",
    "\u6a21\u5757\u6587\u6863\u5b57\u7b26\u4e32",
)
_OPTIONAL_UPDATE_TOKENS = (
    "if needed",
    "if necessary",
    "if required",
    "when needed",
)
_EXPLICIT_TEST_EDIT_TOKENS = (
    "update test",
    "update tests",
    "edit test",
    "edit tests",
    "modify test",
    "modify tests",
    "fix test",
    "fix tests",
    "rewrite test",
    "rewrite tests",
    "add test",
    "add tests",
    "create test",
    "create tests",
)


@dataclass(frozen=True)
class WorkspaceEditOperation:
    action: Literal["create_file", "update_file"]
    target_path: str
    strategy: str
    reason: str
    destination_policy: str
    display_target: str
    target_role: str = ""
    overwrite: bool = False


@dataclass(frozen=True)
class WorkspaceCheckPlan:
    name: str
    command: tuple[str, ...]
    summary: str
    relative_targets: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    check_id: str = ""
    check_type: str = ""
    working_directory: str = "."
    relevance_reason: str = ""
    exists_preflight: bool = True
    created_in_run: bool = False
    belongs_to_current_workspace: bool = True
    validator_origin: str = "not_applicable"
    syntax_preflight_passed: bool | None = None
    syntax_preflight_error: str = ""
    validator_task_run_id: str = ""
    source_target: str = ""


@dataclass(frozen=True)
class WorkspaceCheckResult:
    name: str
    command: tuple[str, ...]
    status: WorkspaceCheckStatus
    summary: str
    output_excerpt: str = ""
    output_full: str = ""
    runtime_exec_result: dict[str, object] = field(default_factory=dict)
    typed_validation: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CriterionEvidence:
    criterion_text: str
    status: Literal["pass", "fail"]
    evidence: str = ""
    source: str = ""
    raw_line: str = ""


@dataclass(frozen=True)
class WorkspaceCheckPlanningIssue:
    check_id: str
    status: Literal["validation_plan_error", "check_scheduler_error"]
    summary: str
    relative_targets: tuple[str, ...] = ()


def preflight_workspace_check_plan(
    workspace_root: Path,
    checks: tuple[WorkspaceCheckPlan, ...],
    *,
    prompt: str,
    task_contract: WorkspaceTaskContract | None = None,
    current_run_created: tuple[str, ...] = (),
) -> tuple[tuple[WorkspaceCheckPlan, ...], tuple[WorkspaceCheckPlanningIssue, ...], tuple[dict[str, object], ...]]:
    executable_checks: list[WorkspaceCheckPlan] = []
    planning_issues: list[WorkspaceCheckPlanningIssue] = []
    records: list[dict[str, object]] = []
    contract = task_contract or {}
    explicit_files = {
        _normalize_workspace_relative_path(item)
        for item in tuple(contract.get("explicit_files", ()) or ())
        if item
    }
    likely_relevant = {
        _normalize_workspace_relative_path(item)
        for item in tuple(contract.get("likely_relevant_files", ()) or ())
        if item
    }
    current_created = {
        _normalize_workspace_relative_path(item)
        for item in current_run_created
        if item
    }

    for index, check in enumerate(checks, start=1):
        check_id = check.check_id or f"{check.name}-{index}"
        relative_targets = tuple(_dedupe(tuple(check.relative_targets)))
        normalized_targets = tuple(_normalize_workspace_relative_path(item) for item in relative_targets)
        missing_targets = tuple(
            item
            for item in normalized_targets
            if not _resolve_workspace_relative_path(workspace_root, item).exists()
        )
        belongs_to_workspace = all(
            _path_belongs_to_workspace(workspace_root, _resolve_workspace_relative_path(workspace_root, item))
            for item in normalized_targets
        )
        validator_origin = "not_applicable"
        created_in_run = any(item in current_created for item in normalized_targets)
        relevance_reason = check.relevance_reason or _default_check_relevance_reason(
            check.name,
            normalized_targets,
        )
        record = {
            "check_id": check_id,
            "check_type": check.check_type or check.name,
            "command": list(check.command),
            "working_directory": check.working_directory or ".",
            "files_or_scripts": list(normalized_targets),
            "exists": not missing_targets,
            "created_in_run": created_in_run,
            "belongs_to_current_workspace": belongs_to_workspace,
            "relevance_reason": relevance_reason,
            "validator_origin": validator_origin,
            "syntax_preflight_passed": None,
            "syntax_preflight_error": "",
            "validator_task_run_id": "",
        }

        if check.name == "python_validate":
            validator_path = normalized_targets[0] if normalized_targets else ""
            validator_task_run_id = _extract_validation_task_run_id(validator_path)
            explicitly_selected = validator_path in explicit_files or validator_path in likely_relevant
            preexisting = bool(validator_path) and _resolve_workspace_relative_path(workspace_root, validator_path).is_file()
            if created_in_run:
                validator_origin = "generated_current_run"
            elif explicitly_selected and preexisting:
                validator_origin = "project_existing"
            elif preexisting and _looks_like_generated_validation_target(validator_path):
                validator_origin = "generated_prior_run"
            elif preexisting:
                validator_origin = "project_existing"
            else:
                validator_origin = "missing"
            record["validator_origin"] = validator_origin
            record["validator_task_run_id"] = validator_task_run_id
            if missing_targets:
                if created_in_run:
                    planning_issues.append(
                        WorkspaceCheckPlanningIssue(
                            check_id=check_id,
                            status="validation_plan_error",
                            summary=(
                                f"validation-plan error: expected current-run validator `{validator_path}` "
                                "does not exist."
                            ),
                            relative_targets=relative_targets,
                        )
                    )
                else:
                    planning_issues.append(
                        WorkspaceCheckPlanningIssue(
                            check_id=check_id,
                            status="validation_plan_error",
                            summary=(
                                f"validation-plan error: planned validator `{validator_path}` does not exist in "
                                "the current workspace and was not created in this task run."
                            ),
                            relative_targets=relative_targets,
                        )
                    )
                records.append(record)
                continue
            if validator_origin == "generated_prior_run":
                planning_issues.append(
                    WorkspaceCheckPlanningIssue(
                        check_id=check_id,
                        status="check_scheduler_error",
                        summary=(
                            f"check-scheduler error: `{validator_path}` looks like a validator from a prior task "
                            "run and is not part of the current task contract."
                        ),
                        relative_targets=relative_targets,
                    )
                )
                records.append(record)
                continue
            validator_absolute = _resolve_workspace_relative_path(workspace_root, validator_path)
            syntax_preflight_passed, syntax_preflight_error = _python_file_syntax_preflight(validator_absolute)
            record["syntax_preflight_passed"] = syntax_preflight_passed
            record["syntax_preflight_error"] = syntax_preflight_error
            if not syntax_preflight_passed:
                planning_issues.append(
                    WorkspaceCheckPlanningIssue(
                        check_id=check_id,
                        status="validation_plan_error",
                        summary=(
                            f"validation-plan error: validator `{validator_path}` failed Python syntax preflight. "
                            f"{syntax_preflight_error}"
                        ),
                        relative_targets=relative_targets,
                    )
                )
                records.append(record)
                continue

        elif missing_targets:
            if check.name == "py_compile":
                retained_targets = tuple(
                    item
                    for item in normalized_targets
                    if item not in missing_targets
                )
                trimmed_missing_targets = tuple(
                    item
                    for item in missing_targets
                    if _looks_like_generated_validation_target(item)
                )
                if retained_targets and len(trimmed_missing_targets) == len(missing_targets):
                    records.append(
                        {
                            **record,
                            "trimmed_missing_targets": list(trimmed_missing_targets),
                        }
                    )
                    executable_checks.append(
                        replace(
                            check,
                            command=(sys.executable, "-m", "py_compile", *retained_targets),
                            relative_targets=retained_targets,
                            check_id=check_id,
                            check_type=check.check_type or check.name,
                            working_directory=check.working_directory or ".",
                            relevance_reason=relevance_reason,
                            exists_preflight=True,
                            created_in_run=created_in_run,
                            belongs_to_current_workspace=belongs_to_workspace,
                            validator_origin=validator_origin,
                            syntax_preflight_passed=getattr(check, "syntax_preflight_passed", None),
                            syntax_preflight_error=getattr(check, "syntax_preflight_error", ""),
                            validator_task_run_id=getattr(check, "validator_task_run_id", ""),
                        )
                    )
                    continue
            planning_issues.append(
                WorkspaceCheckPlanningIssue(
                    check_id=check_id,
                    status="check_scheduler_error",
                    summary=(
                        "check-scheduler error: planned check references missing current-task files: "
                        + ", ".join(missing_targets)
                    ),
                    relative_targets=relative_targets,
                )
            )
            records.append(record)
            continue

        if not belongs_to_workspace:
            planning_issues.append(
                WorkspaceCheckPlanningIssue(
                    check_id=check_id,
                    status="check_scheduler_error",
                    summary="check-scheduler error: planned check references paths outside the current workspace.",
                    relative_targets=relative_targets,
                )
            )
            records.append(record)
            continue

        records.append(
            {
                **record,
                "validator_origin": validator_origin,
            }
        )
        executable_checks.append(
            replace(
                check,
                check_id=check_id,
                check_type=check.check_type or check.name,
                working_directory=check.working_directory or ".",
                relevance_reason=relevance_reason,
                exists_preflight=not missing_targets,
                created_in_run=created_in_run,
                belongs_to_current_workspace=belongs_to_workspace,
                validator_origin=validator_origin,
                syntax_preflight_passed=record["syntax_preflight_passed"],
                syntax_preflight_error=record["syntax_preflight_error"],
                validator_task_run_id=record["validator_task_run_id"],
            )
        )

    return tuple(executable_checks), tuple(planning_issues), tuple(records)


@dataclass(frozen=True)
class WorkspaceEditPlan:
    active: bool = False
    edit_intent: EditIntentName = ""
    reason: str = ""
    summary: str = ""
    operations: tuple[WorkspaceEditOperation, ...] = ()
    planned_reads: tuple[str, ...] = ()
    planned_modifications: tuple[str, ...] = ()
    planned_creations: tuple[str, ...] = ()
    primary_targets: tuple[str, ...] = ()
    secondary_targets: tuple[str, ...] = ()
    referenced_paths: tuple[str, ...] = ()
    skipped_files: tuple[str, ...] = ()
    skipped_notes: tuple[str, ...] = ()
    intended_changes: tuple[str, ...] = ()
    task_contract: WorkspaceTaskContract = field(default_factory=dict)


@dataclass(frozen=True)
class GitWorkspaceSummary:
    detected: bool = False
    repo_root: str = ""
    before_status: tuple[str, ...] = ()
    after_status: tuple[str, ...] = ()
    changed_tracked_files: tuple[str, ...] = ()
    untracked_files: tuple[str, ...] = ()
    commit_message_draft: str = ""
    note: str = ""


@dataclass(frozen=True)
class WorkspaceEditApplyResult:
    status: str
    operation: str = ""
    primary_file: str = ""
    destination_policy: str = ""
    created_files: tuple[str, ...] = ()
    modified_files: tuple[str, ...] = ()
    skipped_files: tuple[str, ...] = ()
    skipped_notes: tuple[str, ...] = ()
    file_change_summaries: tuple[str, ...] = ()
    rollback_notes: tuple[str, ...] = ()
    git_summary: GitWorkspaceSummary = field(default_factory=GitWorkspaceSummary)
    display_answer: str = ""
    error: str = ""


@dataclass(frozen=True)
class _StagedTextWrite:
    final_path: Path
    display_path: str
    content: str
    created: bool
    summary: str
    destination_policy: str
    collision_suffix: int = 0
    original_content: str | None = None


def build_workspace_edit_plan(
    prompt: str,
    mode_selection: ModeSelection,
    access_manager: WorkspaceAccessManager,
) -> WorkspaceEditPlan:
    if mode_selection.mode == "prompt_compiler":
        return WorkspaceEditPlan(
            reason="prompt_compiler is an answer-only workflow and must not mutate the workspace."
        )

    task_prompt = _extract_current_task_prompt(prompt)
    prompt_lower = task_prompt.lower()
    candidate_prompt = prompt if _prompt_contains_explicit_retry_targets(prompt) else task_prompt
    task1_promoted_targets = _promote_task1_daily_crsp_targets(
        task_prompt,
        access_manager.active_workspace_root,
    )
    candidate_targets = _dedupe(
        (
            *_extract_write_candidates(candidate_prompt, access_manager=access_manager),
            *_infer_contextual_write_candidates(candidate_prompt, access_manager),
            *task1_promoted_targets,
        )
    )
    candidate_roles = _classify_workspace_target_roles(
        candidate_targets,
        candidate_prompt,
        access_manager,
    )
    candidate_roles = _refine_candidate_roles_for_workspace_context(
        candidate_targets,
        candidate_prompt,
        access_manager,
        candidate_roles,
    )
    create_requested = _contains_any(prompt_lower, prompt, _CREATE_TOKENS)
    update_requested = _contains_any(prompt_lower, prompt, _UPDATE_TOKENS)
    docstring_requested = _contains_any(prompt_lower, prompt, _DOCSTRING_TOKENS)
    workspace_requested = _contains_any(prompt_lower, prompt, _WORKSPACE_ROOT_TOKENS)
    optional_update_requested = _contains_any(prompt_lower, prompt, _OPTIONAL_UPDATE_TOKENS)
    explicit_test_edit_requested = _contains_any(prompt_lower, prompt, _EXPLICIT_TEST_EDIT_TOKENS)

    operations: list[WorkspaceEditOperation] = []
    planned_reads: list[str] = []
    intended_changes: list[str] = []
    skipped_files: list[str] = []
    skipped_notes: list[str] = []
    primary_targets: list[str] = []
    secondary_targets: list[str] = []
    referenced_paths: list[str] = []
    seen_targets: set[str] = set()

    for candidate in candidate_targets:
        target_role = candidate_roles.get(candidate, "primary_source_target")
        lowered_name = Path(candidate).name.lower()
        normalized_candidate = candidate.replace("\\", "/").lower()
        looks_like_test_target = (
            normalized_candidate.startswith("tests/")
            or "/tests/" in normalized_candidate
            or lowered_name.startswith("test_")
        )
        if looks_like_test_target and not explicit_test_edit_requested:
            planned_reads.append(candidate)
            intended_changes.append(
                f"{candidate}: treat the targeted test as read-only verification context unless the user explicitly asks to edit tests."
            )
            continue
        if target_role == "referenced_path":
            referenced_target = _resolve_workspace_reference_path(access_manager, candidate)
            display_reference = _display_target(
                access_manager,
                referenced_target,
                referenced_target.relative_to(access_manager.active_workspace_root).as_posix()
                if referenced_target is not None
                else candidate.lstrip("/\\") or candidate,
            )
            planned_reads.append(display_reference)
            referenced_paths.append(display_reference)
            intended_changes.append(
                f"{display_reference}: treat this as a referenced path inside config or route metadata; inspect it as context and only edit it if the task directly proves the entrypoint file itself is broken."
            )
            continue
        resolved_existing = access_manager.resolve_prompt_path(candidate, must_exist=True)
        resolved_target, destination_policy = _resolve_target_path(
            candidate,
            task_prompt,
            mode_selection,
            access_manager,
        )
        display_target = _display_target(access_manager, resolved_target, candidate)
        exists = resolved_existing is not None

        if target_role == "context_read_target":
            planned_reads.append(display_target)
            intended_changes.append(
                f"{display_target}: inspect this file as read-only context first; do not edit it unless later evidence proves that it owns the real requested code path."
            )
            continue

        action: Literal["create_file", "update_file"] | None = None
        strategy = ""
        reason = ""
        overwrite = False

        if lowered_name == "readme.md" and exists and optional_update_requested:
            planned_reads.append(display_target)
            intended_changes.append(
                f"{display_target}: consult the README as optional context and only rewrite it if the task still requires a doc sync after the code fix."
            )
            continue
        if lowered_name == "readme.md":
            action = "update_file" if exists or update_requested else "create_file"
            strategy = "readme_handoff_refresh"
            reason = "Refresh the workspace README with a short RA handoff snapshot."
            overwrite = action == "update_file"
        elif lowered_name == "pyproject.toml" and not _looks_like_config_entrypoint_task(prompt_lower):
            action = "update_file"
            strategy = "pyproject_comment_refresh"
            reason = "Refine top-level pyproject comments when appropriate."
            overwrite = True
        elif lowered_name in {"next_steps.md", "handoff_notes.md"}:
            action = "update_file" if exists else "create_file"
            strategy = "handoff_markdown"
            overwrite = action == "update_file"
            if exists:
                reason = f"Refresh {Path(candidate).name} with focused handoff notes."
            else:
                reason = f"Create {Path(candidate).name} with focused handoff notes."
        elif docstring_requested and lowered_name.endswith(".py") and exists:
            action = "update_file"
            strategy = "python_module_docstring"
            reason = "Add a short module docstring while preserving the rest of the file."
            overwrite = True
        elif exists and update_requested and Path(candidate).suffix.lower() in {".md", ".txt"}:
            action = "update_file"
            strategy = "managed_text_refresh"
            reason = "Refresh the requested text file with a managed handoff-oriented section."
            overwrite = True
        elif exists and update_requested and Path(candidate).suffix.lower() in {".py", ".ipynb", ".toml", ".json", ".yaml", ".yml"}:
            action = "update_file"
            strategy = "structured_text_refresh"
            reason = "Apply a structured in-place update to the requested coding or config file."
            overwrite = True
        elif create_requested:
            action = "create_file"
            suffix = Path(candidate).suffix.lower()
            if suffix in {".py", ".ipynb", ".toml", ".json", ".yaml", ".yml"}:
                strategy = "structured_text_file"
                reason = "Create the requested coding or config file from a structured file block."
            else:
                strategy = "answer_body_file"
                reason = "Create the explicitly requested deliverable file."
        elif exists and update_requested:
            action = "update_file"
            strategy = "structured_text_refresh"
            reason = "Apply a structured in-place update to the requested file."
            overwrite = True

        if action is None:
            continue

        if target_role == "primary_config_target":
            reason = "Repair the primary config or deployment file that controls the route, manifest, or entrypoint."
        elif target_role == "primary_notebook_target":
            reason = "Modify the primary notebook deliverable in place and keep any helper scripts secondary to that notebook."
        elif target_role == "primary_source_target" and _looks_like_config_entrypoint_task(prompt_lower):
            reason = "Repair the primary entrypoint or runnable source file that the config or route depends on."
        elif target_role == "secondary_docs_target" and lowered_name == "readme.md":
            reason = "Sync the visible README after the primary repair so the hosted path and handoff text stay accurate."
        elif target_role == "secondary_docs_target":
            reason = f"Sync {Path(candidate).name} after the primary repair so the handoff stays accurate."

        key = f"{action}:{display_target}"
        if key in seen_targets:
            continue
        seen_targets.add(key)
        operations.append(
            WorkspaceEditOperation(
                action=action,
                target_path=str(resolved_target) if resolved_target.is_absolute() else resolved_target.as_posix(),
                strategy=strategy,
                reason=reason,
                destination_policy=destination_policy,
                display_target=display_target,
                target_role=target_role,
                overwrite=overwrite,
            )
        )
        intended_changes.append(f"{display_target}: {reason}")
        if exists:
            planned_reads.append(display_target)
        if target_role.startswith("primary_"):
            primary_targets.append(display_target)
        elif target_role == "secondary_docs_target":
            secondary_targets.append(display_target)

    for candidate in task1_promoted_targets:
        resolved_target = access_manager.active_workspace_root / candidate
        display_target = candidate
        key = f"update_file:{display_target}"
        if key in seen_targets:
            continue
        seen_targets.add(key)
        operations.append(
            WorkspaceEditOperation(
                action="update_file",
                target_path=str(resolved_target),
                strategy="structured_text_refresh",
                reason="Update the daily rolling or prep script so the centralized Task 1 start-date rule is enforced in the real downstream code path.",
                destination_policy="workspace_relative",
                display_target=display_target,
                target_role="primary_source_target",
                overwrite=True,
            )
        )
        intended_changes.append(
            f"{display_target}: update this daily rolling or prep script so the centralized Task 1 start-date rule is enforced in the real downstream code path."
        )
        planned_reads.append(display_target)
        primary_targets.append(display_target)

    if not operations:
        if workspace_requested and create_requested and not candidate_targets:
            skipped_notes.append("Workspace deliverable was requested without a concrete file name.")
        return WorkspaceEditPlan(
            skipped_files=tuple(skipped_files),
            skipped_notes=tuple(skipped_notes),
        )

    planned_modifications = tuple(op.display_target for op in operations if op.action == "update_file")
    planned_creations = tuple(op.display_target for op in operations if op.action == "create_file")
    prompt_context_reads = _discover_prompt_context_reads(
        task_prompt,
        access_manager.active_workspace_root,
    )
    for item in prompt_context_reads:
        planned_reads.append(item)
        intended_changes.append(
            f"{item}: inspect this prompt-named source file before editing because it is part of the requested code path."
        )
    planned_reads = _dedupe((*planned_reads, *planned_modifications))
    task_contract = build_workspace_task_contract(
        task_prompt,
        planned_reads=planned_reads,
        planned_modifications=planned_modifications,
        planned_creations=planned_creations,
        referenced_paths=_dedupe(tuple(referenced_paths)),
    )

    if len(operations) >= 2:
        edit_intent: EditIntentName = "multi_file_edit"
        if _looks_like_config_entrypoint_task(prompt_lower):
            summary = "Plan the config or entrypoint repair, update the primary target first, then sync any secondary docs or handoff files."
        elif any(token in prompt_lower for token in ("refactor", "\u91cd\u6784")):
            summary = "Plan the requested refactor, then apply grouped multi-file changes."
        elif any(token in prompt_lower for token in ("fix", "repair", "bug", "\u4fee\u590d")):
            summary = "Plan the requested bug fix, then apply grouped multi-file changes."
        else:
            summary = "Plan the requested coding task, then apply grouped multi-file changes."
    elif planned_creations and not planned_modifications:
        edit_intent = "deliverable_update"
        summary = "Create the requested workspace deliverable after a short preflight."
    else:
        edit_intent = "direct_file_edit"
        summary = "Apply the requested in-place workspace edit after a short preflight."

    return WorkspaceEditPlan(
        active=True,
        edit_intent=edit_intent,
        reason="Prompt requests workspace file creation or modification inside the allowlisted roots.",
        summary=summary,
        operations=tuple(operations),
        planned_reads=planned_reads,
        planned_modifications=planned_modifications,
        planned_creations=planned_creations,
        primary_targets=_dedupe(tuple(primary_targets)),
        secondary_targets=_dedupe(tuple(secondary_targets)),
        referenced_paths=_dedupe(tuple(referenced_paths)),
        skipped_files=tuple(skipped_files),
        skipped_notes=tuple(skipped_notes),
        intended_changes=_dedupe(tuple(intended_changes)),
        task_contract=task_contract,
    )


def classify_output_intent(plan: WorkspaceEditPlan) -> tuple[str, str]:
    if plan.active:
        return "deliverable_requested", plan.reason
    return "answer_only", "Prompt is an ordinary terminal answer request."


def build_workspace_task_contract(
    prompt: str,
    *,
    planned_reads: tuple[str, ...] = (),
    planned_modifications: tuple[str, ...] = (),
    planned_creations: tuple[str, ...] = (),
    referenced_paths: tuple[str, ...] = (),
    workspace_understanding_summary: str = "",
    workspace_inspected_files: tuple[str, ...] = (),
    workspace_skipped_files: tuple[str, ...] = (),
    workspace_manifest_categories: dict[str, int] | None = None,
    likely_code_paths: tuple[str, ...] = (),
    workspace_relevant_files: tuple[str, ...] = (),
    workspace_full_relevant_coverage: bool = False,
) -> WorkspaceTaskContract:
    task_prompt = _extract_current_task_prompt(prompt)
    explicit_files = _dedupe((*planned_modifications, *planned_creations))
    likely_relevant_files = _dedupe(
        (
            *planned_reads,
            *explicit_files,
            *referenced_paths,
            *workspace_inspected_files,
            *likely_code_paths,
        )
    )
    behavior_requirements = _normalize_task_contract_clauses(
        _extract_task_contract_clauses(task_prompt, category="behavior")
    )
    acceptance_criteria = _normalize_task_contract_clauses(
        _extract_task_contract_clauses(task_prompt, category="acceptance")
    )
    forbidden_shortcuts = _extract_task_contract_clauses(task_prompt, category="forbidden")
    validation_strategy = _extract_task_contract_clauses(task_prompt, category="validation")
    behavioral_validation_required = _is_behavioral_edit_task(
        task_prompt,
        explicit_files=explicit_files,
        behavior_requirements=behavior_requirements,
        acceptance_criteria=acceptance_criteria,
    )
    if not acceptance_criteria and behavior_requirements:
        acceptance_criteria = behavior_requirements
    if behavioral_validation_required and not validation_strategy:
        validation_strategy = (
            "Prefer existing targeted tests or project checks; if they do not exist, create a focused validation harness or regression test inside the workspace.",
            "Treat syntax-only checks such as py_compile as insufficient for this behavioral task.",
        )
    numeric_python_targets = tuple(
        item
        for item in explicit_files
        if Path(item).suffix.lower() == ".py" and Path(item).name[:1].isdigit()
    )
    if numeric_python_targets:
        validation_strategy = _dedupe(
            (
                *validation_strategy,
                "One or more focused Python files start with digits; do not use normal import syntax against those filenames. Use importlib loading or extract the callable logic into a helper module with a valid Python name.",
            )
        )
    expected_checks: list[str] = []
    if behavioral_validation_required:
        expected_checks.append(
            "Run meaningful behavioral validation against the requested acceptance criteria."
        )
    if explicit_files:
        expected_checks.append(
            f"Touch the real source path(s): {', '.join(explicit_files)}."
        )
    failure_conditions = tuple(
        _dedupe(
            (
                *forbidden_shortcuts,
                "Do not declare success if any acceptance criterion lacks direct validation evidence.",
                "Do not treat a broken or syntactically invalid generated validator as a user-task blocker; repair the validator plan or validator file first.",
            )
        )
    )
    return {
        "user_goal": _summarize_task_goal(task_prompt),
        "business_requirements": behavior_requirements,
        "explicit_files": explicit_files,
        "likely_relevant_files": likely_relevant_files,
        "behavior_requirements": behavior_requirements,
        "acceptance_criteria": acceptance_criteria,
        "forbidden_shortcuts": forbidden_shortcuts,
        "validation_strategy": validation_strategy,
        "expected_changed_files": explicit_files,
        "expected_created_files": planned_creations,
        "expected_checks": tuple(expected_checks),
        "failure_conditions": failure_conditions,
        "behavioral_validation_required": behavioral_validation_required,
        "reject_syntax_only_success": behavioral_validation_required,
        "task_type": _workspace_task_type(
            planned_modifications=planned_modifications,
            planned_creations=planned_creations,
            behavioral_validation_required=behavioral_validation_required,
        ),
        "workspace_understanding_summary": workspace_understanding_summary,
        "workspace_inspected_files": workspace_inspected_files,
        "workspace_skipped_files": workspace_skipped_files,
        "workspace_relevant_files": workspace_relevant_files,
        "workspace_full_relevant_coverage": workspace_full_relevant_coverage,
        "workspace_manifest_categories": dict(workspace_manifest_categories or {}),
        "likely_code_paths": likely_code_paths,
    }


def _select_behavioral_validation_source_target(
    *,
    planned_modifications: tuple[str, ...],
    planned_creations: tuple[str, ...],
    task_contract: WorkspaceTaskContract,
) -> str:
    route2_routing = task_contract.get("route2_validator_routing")
    if isinstance(route2_routing, dict):
        for candidate in tuple(route2_routing.get("required_source_or_artifact", ()) or ()):
            if candidate and not _looks_like_generated_validation_target(candidate):
                return candidate
    route2_owner = task_contract.get("route2_owner_detection")
    if isinstance(route2_owner, dict):
        for bucket in ("primary_owner_files", "primary_artifacts"):
            for candidate in tuple(route2_owner.get(bucket, ()) or ()):
                if candidate and not _looks_like_generated_validation_target(candidate):
                    return candidate
    for candidate in planned_modifications:
        if candidate and not _looks_like_generated_validation_target(candidate):
            return candidate
    for candidate in planned_creations:
        if candidate and not _looks_like_generated_validation_target(candidate):
            return candidate
    return ""


def build_workspace_check_plan(
    prompt: str,
    workspace_root: Path,
    *,
    planned_modifications: tuple[str, ...],
    planned_creations: tuple[str, ...],
    task_contract: WorkspaceTaskContract | None = None,
) -> tuple[WorkspaceCheckPlan, ...]:
    task_prompt = _extract_current_task_prompt(prompt)
    planned_targets = _dedupe((*planned_modifications, *planned_creations))
    if not planned_targets:
        return ()

    prompt_lower = task_prompt.lower()
    python_targets = tuple(
        item
        for item in planned_targets
        if Path(item).suffix.lower() == ".py"
    )
    toml_targets = tuple(
        item
        for item in planned_targets
        if Path(item).suffix.lower() == ".toml"
    )
    json_targets = tuple(
        item
        for item in planned_targets
        if Path(item).suffix.lower() == ".json"
    )
    text_targets = tuple(
        item
        for item in planned_targets
        if Path(item).suffix.lower() in {".md", ".txt"}
    )
    config_targets = tuple(
        item
        for item in planned_targets
        if _looks_like_config_target(item)
    )
    checks: list[WorkspaceCheckPlan] = []

    if python_targets:
        checks.append(
            WorkspaceCheckPlan(
                name="py_compile",
                command=(sys.executable, "-m", "py_compile", *python_targets),
                summary=(
                    "Run Python syntax checks on "
                    + ", ".join(python_targets)
                ),
                relative_targets=python_targets,
                source_target=python_targets[0],
            )
        )

    for json_target in json_targets:
        absolute_target = (workspace_root / json_target).resolve()
        checks.append(
            WorkspaceCheckPlan(
                name="json_validate",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "import json, pathlib; "
                        f"json.loads(pathlib.Path(r'{absolute_target}').read_text(encoding='utf-8-sig'))"
                    ),
                ),
                summary=f"Validate JSON syntax for {json_target}",
                relative_targets=(json_target,),
                source_target=json_target,
            )
        )
    for toml_target in toml_targets:
        absolute_target = (workspace_root / toml_target).resolve()
        checks.append(
            WorkspaceCheckPlan(
                name="toml_validate",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, tomllib; "
                        f"tomllib.loads(pathlib.Path(r'{absolute_target}').read_text(encoding='utf-8-sig'))"
                    ),
                ),
                summary=f"Validate TOML syntax for {toml_target}",
                relative_targets=(toml_target,),
                source_target=toml_target,
            )
        )

    explicit_config_value = _extract_explicit_config_expectation(task_prompt)
    if explicit_config_value and config_targets:
        primary_config_target = config_targets[0]
        absolute_target = (workspace_root / primary_config_target).resolve()
        checks.append(
            WorkspaceCheckPlan(
                name="content_expectation",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, sys; "
                        f"text = pathlib.Path(r'{absolute_target}').read_text(encoding='utf-8-sig'); "
                        f"expected = {explicit_config_value!r}; "
                        "sys.stderr.write(f'Expected literal not found: {expected}') if expected not in text else None; "
                        "raise SystemExit(0 if expected in text else 1)"
                    ),
                ),
                summary=f"Confirm {primary_config_target} contains `{explicit_config_value}`",
                relative_targets=(primary_config_target,),
                source_target=primary_config_target,
            )
        )

    explicit_text_expectations = _extract_explicit_text_expectations(task_prompt)
    for relative_target, expected_text in explicit_text_expectations:
        if relative_target not in text_targets:
            continue
        absolute_target = (workspace_root / relative_target).resolve()
        checks.append(
            WorkspaceCheckPlan(
                name="content_expectation",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, sys; "
                        f"text = pathlib.Path(r'{absolute_target}').read_text(encoding='utf-8-sig'); "
                        f"expected = {expected_text!r}; "
                        "sys.stderr.write(f'Expected literal not found: {expected}') if expected not in text else None; "
                        "raise SystemExit(0 if expected in text else 1)"
                    ),
                ),
                summary=f"Confirm {relative_target} contains `{expected_text}`",
                relative_targets=(relative_target,),
                source_target=relative_target,
            )
        )

    handoff_targets = _discover_next_steps_validation_targets(
        task_prompt,
        planned_targets=text_targets,
    )
    for relative_target in handoff_targets:
        absolute_target = (workspace_root / relative_target).resolve()
        checks.append(
            WorkspaceCheckPlan(
                name="content_expectation",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, re, sys; "
                        f"text = pathlib.Path(r'{absolute_target}').read_text(encoding='utf-8-sig'); "
                        "matches = re.findall(r'(?m)^\\s*(?:[-*]|\\d+\\.)\\s+.+$', text); "
                        "sys.stderr.write('Expected at least three next-step list items') if len(matches) < 3 else None; "
                        "raise SystemExit(0 if len(matches) >= 3 else 1)"
                    ),
                ),
                summary=f"Confirm {relative_target} contains at least three next-step list items",
                relative_targets=(relative_target,),
                source_target=relative_target,
            )
        )

    pytest_targets = _discover_pytest_targets(
        task_prompt,
        workspace_root,
        python_targets=python_targets,
    )
    validation_script_targets = _discover_validation_script_targets(
        task_prompt,
        planned_targets=planned_targets,
        task_contract=task_contract or {},
    )
    config_only_task = bool(config_targets) and not python_targets and _looks_like_config_entrypoint_task(prompt_lower)
    explicit_test_request = any(
        token in prompt_lower
        for token in ("pytest", "tests", "test ", "failing test", "regression")
    )
    if pytest_targets and (not config_only_task or explicit_test_request):
        pytest_env = _build_pytest_env(
            workspace_root,
            python_targets=python_targets,
            pytest_targets=pytest_targets,
        )
        checks.append(
            WorkspaceCheckPlan(
                name="pytest",
                command=(sys.executable, "-m", "pytest", "-q", *pytest_targets),
                summary=(
                    "Run targeted pytest checks for "
                    + ", ".join(pytest_targets)
                ),
                relative_targets=pytest_targets,
                env=pytest_env,
                source_target=python_targets[0] if python_targets else pytest_targets[0],
            )
        )

    for validation_target in validation_script_targets:
        validation_env = _build_pytest_env(
            workspace_root,
            python_targets=python_targets,
            pytest_targets=(validation_target,),
        )
        acceptance_criteria = ()
        if task_contract and task_contract.get("behavioral_validation_required", False):
            acceptance_criteria = tuple(task_contract.get("acceptance_criteria", ()) or ())
        source_target = _select_behavioral_validation_source_target(
            planned_modifications=planned_modifications,
            planned_creations=planned_creations,
            task_contract=task_contract or {},
        )
        checks.append(
            WorkspaceCheckPlan(
                name="python_validate",
                command=(sys.executable, validation_target),
                summary=f"Run focused validation script {validation_target}",
                relative_targets=(validation_target,),
                acceptance_criteria=acceptance_criteria,
                env=validation_env,
                source_target=source_target,
            )
        )

    return tuple(checks)


def run_workspace_checks(
    workspace_root: Path,
    checks: tuple[WorkspaceCheckPlan, ...],
    *,
    timeout_seconds: int = 120,
) -> tuple[WorkspaceCheckResult, ...]:
    results: list[WorkspaceCheckResult] = []
    for check in checks:
        exec_result = run_runtime_command(
            check.command,
            cwd=workspace_root,
            timeout_seconds=timeout_seconds,
            env_overrides=check.env,
        )
        if exec_result.timeout:
            typed = build_typed_validation_result(
                validator_id=check.check_id or check.name,
                task_run_id=check.validator_task_run_id,
                check_name=check.name,
                command=check.command,
                status="blocked",
                acceptance_criteria=getattr(check, "acceptance_criteria", ()) or (),
                criterion_evidence=(),
                output_excerpt=exec_result.error or "The check timed out.",
                source_target=getattr(check, "source_target", ""),
                raw_output_path="",
            )
            results.append(
                WorkspaceCheckResult(
                    name=check.name,
                    command=check.command,
                    status="blocked",
                    summary=f"{check.name} timed out after {timeout_seconds}s.",
                    output_excerpt=exec_result.error or "The check timed out.",
                    output_full=exec_result.error or "",
                    runtime_exec_result=exec_result.to_record(),
                    typed_validation=typed.to_record(),
                )
            )
            continue
        if exec_result.error and exec_result.exit_code == -1 and not exec_result.timeout:
            typed = build_typed_validation_result(
                validator_id=check.check_id or check.name,
                task_run_id=check.validator_task_run_id,
                check_name=check.name,
                command=check.command,
                status="blocked",
                acceptance_criteria=getattr(check, "acceptance_criteria", ()) or (),
                criterion_evidence=(),
                output_excerpt=exec_result.error,
                source_target=getattr(check, "source_target", ""),
                raw_output_path="",
            )
            results.append(
                WorkspaceCheckResult(
                    name=check.name,
                    command=check.command,
                    status="blocked",
                    summary=f"{check.name} could not run: {exec_result.error}",
                    output_excerpt=exec_result.error,
                    output_full=exec_result.error,
                    runtime_exec_result=exec_result.to_record(),
                    typed_validation=typed.to_record(),
                )
            )
            continue

        full_output = "\n".join(
            part.strip()
            for part in (exec_result.stdout, exec_result.stderr)
            if part and part.strip()
        ).strip()
        excerpt = _truncate_process_output(exec_result.stdout, exec_result.stderr)
        criterion_evidence = parse_criterion_evidence_from_output(
            exec_result.stdout,
            exec_result.stderr,
        )
        failing_criteria = tuple(item for item in criterion_evidence if item.status == "fail")
        if exec_result.exit_code == 0 and not exec_result.timeout:
            typed = build_typed_validation_result(
                validator_id=check.check_id or check.name,
                task_run_id=check.validator_task_run_id,
                check_name=check.name,
                command=check.command,
                status="passed",
                acceptance_criteria=getattr(check, "acceptance_criteria", ()) or (),
                criterion_evidence=criterion_evidence,
                output_excerpt=excerpt,
                source_target=getattr(check, "source_target", ""),
                raw_output_path="",
            )
            if failing_criteria:
                results.append(
                    WorkspaceCheckResult(
                        name=check.name,
                        command=check.command,
                        status="failed",
                        summary=(
                            "python_validate failed because the validation harness reported failed current-run "
                            "criteria: "
                            + _summarize_failed_criteria(failing_criteria)
                        )
                        if check.name == "python_validate"
                        else (
                            f"{check.name} failed because the current-run output reported failed criteria: "
                            + _summarize_failed_criteria(failing_criteria)
                        ),
                        output_excerpt=excerpt,
                        output_full=full_output,
                        runtime_exec_result=exec_result.to_record(),
                        typed_validation=typed.to_record(),
                    )
                )
                continue
            if check.name == "python_validate" and _looks_like_skipped_behavioral_validation(full_output):
                results.append(
                    WorkspaceCheckResult(
                        name=check.name,
                        command=check.command,
                        status="failed",
                        summary=(
                            "python_validate failed because the validation harness skipped the required "
                            "behavioral assertions instead of proving them."
                        ),
                        output_excerpt=excerpt,
                        output_full=full_output,
                        runtime_exec_result=exec_result.to_record(),
                        typed_validation=typed.to_record(),
                    )
                )
                continue
            if check.name == "python_validate":
                missing_criteria = _missing_validator_criteria_evidence(
                    criterion_evidence,
                    getattr(check, "acceptance_criteria", ()) or (),
                )
                if missing_criteria:
                    results.append(
                        WorkspaceCheckResult(
                            name=check.name,
                            command=check.command,
                            status="failed",
                            summary=(
                                "python_validate failed because the validation harness did not emit direct "
                                "criterion-level evidence for: "
                                + ", ".join(missing_criteria)
                            ),
                            output_excerpt=excerpt,
                            output_full=full_output,
                            runtime_exec_result=exec_result.to_record(),
                            typed_validation=typed.to_record(),
                        )
                    )
                    continue
            results.append(
                WorkspaceCheckResult(
                    name=check.name,
                    command=check.command,
                    status="passed",
                    summary=f"{check.name} passed.",
                    output_excerpt=excerpt,
                    output_full=full_output,
                    runtime_exec_result=exec_result.to_record(),
                    typed_validation=typed.to_record(),
                )
            )
            continue
        typed = build_typed_validation_result(
            validator_id=check.check_id or check.name,
            task_run_id=check.validator_task_run_id,
            check_name=check.name,
            command=check.command,
            status="failed",
            acceptance_criteria=getattr(check, "acceptance_criteria", ()) or (),
            criterion_evidence=criterion_evidence,
            output_excerpt=excerpt,
            source_target=getattr(check, "source_target", ""),
            raw_output_path="",
        )
        results.append(
            WorkspaceCheckResult(
                name=check.name,
                command=check.command,
                status="failed",
                summary=(
                    f"{check.name} failed because the command timed out after {timeout_seconds}s."
                    if exec_result.timeout
                    else f"{check.name} failed with exit code {exec_result.exit_code}."
                ),
                output_excerpt=excerpt,
                output_full=full_output,
                runtime_exec_result=exec_result.to_record(),
                typed_validation=typed.to_record(),
            )
        )
    return tuple(results)


def _looks_like_skipped_behavioral_validation(output_excerpt: str) -> bool:
    lowered = output_excerpt.lower()
    return any(
        token in lowered
        for token in (
            "skipping validation",
            "skip validation",
            "sample csv",
            "sample data",
            "fixture not found",
            "not found - skipping",
            "not found – skipping",
            "not found ?c skipping",
            "no behavioral assertions ran",
        )
    )


def _missing_validator_criteria_evidence(
    output_excerpt: str | tuple[CriterionEvidence, ...],
    acceptance_criteria: tuple[str, ...],
) -> tuple[str, ...]:
    criteria = tuple(item for item in acceptance_criteria if str(item).strip())
    if not criteria:
        return ()
    markers = _extract_validator_criterion_markers(output_excerpt)
    if not markers:
        return criteria
    normalized_markers = {_normalize_validator_criterion_text(item) for item in markers if item.strip()}
    missing: list[str] = []
    for criterion in criteria:
        normalized_criterion = _normalize_validator_criterion_text(criterion)
        if not normalized_criterion:
            continue
        if any(
            normalized_criterion == marker
            or normalized_criterion in marker
            or marker in normalized_criterion
            for marker in normalized_markers
        ):
            continue
        missing.append(criterion)
    return tuple(missing)


def _extract_validator_criterion_markers(
    output_excerpt: str | tuple[CriterionEvidence, ...],
) -> tuple[str, ...]:
    if isinstance(output_excerpt, tuple):
        evidence = output_excerpt
    else:
        evidence = parse_criterion_evidence_from_output(output_excerpt)
    return tuple(
        item.criterion_text
        for item in evidence
        if item.status == "pass" and item.criterion_text.strip()
    )


def _summarize_failed_criteria(
    failing_criteria: tuple[CriterionEvidence, ...],
    *,
    limit: int = 3,
) -> str:
    rendered: list[str] = []
    for item in failing_criteria[:limit]:
        criterion = item.criterion_text.strip() or item.raw_line.strip() or "(unnamed criterion)"
        detail = item.evidence.strip()
        rendered.append(f"{criterion} :: {detail}" if detail else criterion)
    remaining = len(failing_criteria) - min(len(failing_criteria), limit)
    if remaining > 0:
        rendered.append(f"plus {remaining} more")
    return "; ".join(rendered)


def parse_criterion_evidence_from_output(
    stdout: str,
    stderr: str = "",
) -> tuple[CriterionEvidence, ...]:
    combined = "\n".join(
        part
        for part in (stdout, stderr)
        if isinstance(part, str) and part.strip()
    ).strip()
    if not combined:
        return ()
    evidence: list[CriterionEvidence] = []
    in_fenced_block = False
    in_non_current_block = False
    for raw_line in combined.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            in_fenced_block = not in_fenced_block
            continue
        if re.match(r"^(?:BEGIN|START)_(?:NON_CURRENT_EVIDENCE|HISTORICAL_LOG)\b", stripped, flags=re.IGNORECASE):
            in_non_current_block = True
            continue
        if re.match(r"^END_(?:NON_CURRENT_EVIDENCE|HISTORICAL_LOG)\b", stripped, flags=re.IGNORECASE):
            in_non_current_block = False
            continue
        if in_fenced_block or in_non_current_block:
            continue
        if re.match(
            r"^(?:NON_CURRENT_EVIDENCE|HISTORICAL_LOG|QUOTED_HISTORICAL_LOG):",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue
        evidence.extend(_parse_line_criterion_evidence(stripped))
    return tuple(evidence)


def _parse_line_criterion_evidence(line: str) -> list[CriterionEvidence]:
    evidence: list[CriterionEvidence] = []
    marker_pattern = re.compile(
        r"CRITERION(?:\s+|_)(?P<status>PASS|FAIL)\s*:\s*(?P<body>.*?)(?=(?:CRITERION(?:\s+|_)(?:PASS|FAIL)\s*:)|$)",
        flags=re.IGNORECASE,
    )
    for match in marker_pattern.finditer(line):
        body = match.group("body").strip()
        criterion_text, detail = _split_criterion_body(body)
        evidence.append(
            CriterionEvidence(
                criterion_text=criterion_text,
                status=match.group("status").lower(),
                evidence=detail,
                source="marker",
                raw_line=line,
            )
        )
    if evidence:
        return evidence

    key_value_match = re.search(r"\bcriterion_status\s*=\s*(pass|fail)\b", line, flags=re.IGNORECASE)
    if key_value_match:
        evidence.append(
            CriterionEvidence(
                criterion_text=_extract_key_value_field(
                    line,
                    ("criterion", "criterion_text", "name", "label", "text"),
                )
                or line,
                status=key_value_match.group(1).lower(),
                evidence=_extract_key_value_field(line, ("detail", "evidence", "message", "reason")),
                source="key_value",
                raw_line=line,
            )
        )
        return evidence

    if line.startswith("{") or line.startswith("["):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return evidence
        evidence.extend(_extract_json_criterion_evidence(payload, raw_line=line))
    return evidence


def _split_criterion_body(body: str) -> tuple[str, str]:
    criterion_text, _, detail = body.partition("::")
    return criterion_text.strip(), detail.strip()


def _extract_key_value_field(line: str, keys: tuple[str, ...]) -> str:
    for key in keys:
        match = re.search(
            rf"\b{re.escape(key)}\s*=\s*(?P<value>\"[^\"]*\"|'[^']*'|[^;|]+)",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        value = match.group("value").strip().strip("\"'")
        if value:
            return value
    return ""


def _extract_json_criterion_evidence(payload: object, *, raw_line: str) -> list[CriterionEvidence]:
    evidence: list[CriterionEvidence] = []
    if isinstance(payload, list):
        for item in payload:
            evidence.extend(_extract_json_criterion_evidence(item, raw_line=raw_line))
        return evidence
    if not isinstance(payload, dict):
        return evidence
    normalized_status = ""
    status_value = payload.get("status")
    passed_value = payload.get("passed")
    if isinstance(status_value, str) and status_value.lower() in {"pass", "fail"}:
        normalized_status = status_value.lower()
    elif isinstance(passed_value, bool):
        normalized_status = "pass" if passed_value else "fail"
    if normalized_status:
        criterion_text = ""
        for key in ("criterion", "criterion_text", "name", "label", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                criterion_text = value.strip()
                break
        detail = ""
        for key in ("detail", "evidence", "message", "reason"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                detail = value.strip()
                break
        evidence.append(
            CriterionEvidence(
                criterion_text=criterion_text or raw_line,
                status=normalized_status,
                evidence=detail,
                source="json",
                raw_line=raw_line,
            )
        )
    for value in payload.values():
        if isinstance(value, (dict, list)):
            evidence.extend(_extract_json_criterion_evidence(value, raw_line=raw_line))
    return evidence


def _normalize_validator_criterion_text(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value).strip().lower())
    normalized = re.sub(r"[^a-z0-9_:.\\/\- ]+", "", normalized)
    return normalized.strip(" .")


def apply_workspace_edit_plan(
    config: LabaiConfig,
    access_manager: WorkspaceAccessManager,
    prompt: str,
    plan: WorkspaceEditPlan,
    final_answer: str,
    observations: list[str],
) -> WorkspaceEditApplyResult:
    if not plan.active:
        return WorkspaceEditApplyResult(status="skipped")

    operation_name = "batch_apply" if len(plan.operations) > 1 else plan.operations[0].action

    if config.workspace.edit_mode != "auto_edit":
        return WorkspaceEditApplyResult(
            status="suggest_only",
            operation=operation_name,
            destination_policy=_primary_destination_policy(plan),
            skipped_files=tuple(op.display_target for op in plan.operations),
            skipped_notes=("Workspace edit mode is suggest-only, so no files were modified.",),
            display_answer=final_answer,
        )

    staged: list[_StagedTextWrite] = []
    skipped_files = list(plan.skipped_files)
    skipped_notes = list(plan.skipped_notes)
    file_change_summaries: list[str] = []
    structured_block_specs = extract_structured_file_block_specs(final_answer)
    structured_blocks = _extract_structured_file_blocks(final_answer)
    used_block_paths: set[str] = set()

    for operation in plan.operations:
        try:
            target_path = access_manager.resolve_user_path(
                operation.target_path,
                for_write=True,
                must_exist=operation.action == "update_file",
            )
        except WorkspaceAccessError as exc:
            skipped_files.append(operation.display_target)
            skipped_notes.append(str(exc))
            continue

        staged_write, skip_reason = _stage_operation(
            access_manager=access_manager,
            workspace_root=access_manager.active_workspace_root,
            prompt=prompt,
            operation=operation,
            target_path=target_path,
            final_answer=final_answer,
            observations=observations,
            structured_blocks=structured_blocks,
            structured_block_specs=structured_block_specs,
        )
        if staged_write is None:
            skipped_files.append(operation.display_target)
            if skip_reason:
                skipped_notes.append(skip_reason)
            continue
        staged.append(staged_write)
        file_change_summaries.append(staged_write.summary)
        used_block_paths.update(
            _structured_block_candidates(
                workspace_root=access_manager.active_workspace_root,
                display_target=operation.display_target,
                target_path=target_path,
                final_path=staged_write.final_path,
            )
        )

    extra_staged, extra_skip_notes = _stage_related_support_blocks(
        access_manager=access_manager,
        workspace_root=access_manager.active_workspace_root,
        plan=plan,
        structured_blocks=structured_blocks,
        used_block_paths=used_block_paths,
    )
    if extra_staged:
        staged.extend(extra_staged)
        file_change_summaries.extend(item.summary for item in extra_staged)
    if extra_skip_notes:
        skipped_notes.extend(extra_skip_notes)

    git_before = inspect_git_workspace(access_manager.active_workspace_root)

    if not staged and skipped_notes:
        git_after = inspect_git_workspace(access_manager.active_workspace_root)
        return WorkspaceEditApplyResult(
            status="skipped",
            operation=operation_name,
            destination_policy=_primary_destination_policy(plan),
            skipped_files=_dedupe(tuple(skipped_files)),
            skipped_notes=_dedupe(tuple(skipped_notes)),
            git_summary=_finalize_git_summary(git_before, git_after, (), ()),
            display_answer="No workspace files were changed. Review the skipped-files summary for details.",
        )

    apply_error, created_files, modified_files, rollback_notes = _apply_staged_writes(staged)
    git_after = inspect_git_workspace(access_manager.active_workspace_root)
    git_summary = _finalize_git_summary(git_before, git_after, created_files, modified_files)

    if apply_error:
        return WorkspaceEditApplyResult(
            status="error",
            operation=operation_name,
            destination_policy=_primary_destination_policy(plan),
            skipped_files=_dedupe(tuple(skipped_files)),
            skipped_notes=_dedupe(tuple(skipped_notes)),
            file_change_summaries=_dedupe(tuple(file_change_summaries)),
            rollback_notes=_dedupe(tuple(rollback_notes)),
            git_summary=git_summary,
            display_answer="The planned workspace changes were rolled back after an apply failure.",
            error=apply_error,
        )

    primary_file = created_files[0] if len(created_files) == 1 and not modified_files else ""
    return WorkspaceEditApplyResult(
        status="ok",
        operation=operation_name,
        primary_file=primary_file,
        destination_policy=_primary_destination_policy(plan),
        created_files=created_files,
        modified_files=modified_files,
        skipped_files=_dedupe(tuple(skipped_files)),
        skipped_notes=_dedupe(tuple(skipped_notes)),
        file_change_summaries=_dedupe(tuple(file_change_summaries)),
        git_summary=git_summary,
        display_answer=_build_display_answer(plan, created_files, modified_files, skipped_files),
    )


def inspect_git_workspace(workspace_root: Path) -> GitWorkspaceSummary:
    repo_root = _find_git_root(workspace_root)
    if repo_root is None:
        return GitWorkspaceSummary(note="not_a_git_repo")
    return GitWorkspaceSummary(
        detected=True,
        repo_root=str(repo_root),
        before_status=(),
        note="git_repo_detected",
    )


def _extract_write_candidates(
    prompt: str,
    access_manager: WorkspaceAccessManager | None = None,
) -> tuple[str, ...]:
    candidates: list[str] = []
    for candidate in _iter_prompt_candidates(prompt):
        normalized = _sanitize_candidate(candidate)
        suffix = Path(normalized).suffix.lower()
        if suffix not in _TEXT_WRITE_SUFFIXES:
            continue
        candidates.append(normalized)
    deduped = _dedupe(tuple(candidates))
    absolute_basenames = {
        Path(item).name.lower()
        for item in deduped
        if Path(item).is_absolute()
        and _absolute_prompt_candidate_can_shadow(item, access_manager)
    }
    filtered = tuple(
        item
        for item in deduped
        if not (
            absolute_basenames
            and not Path(item).is_absolute()
            and "/" not in item
            and "\\" not in item
            and Path(item).name.lower() in absolute_basenames
        )
    )
    if access_manager is None:
        return filtered

    disallowed_absolute_basenames = {
        Path(item).name
        for item in deduped
        if Path(item).is_absolute()
        and not _absolute_prompt_candidate_can_shadow(item, access_manager)
    }
    prompt_basenames = {
        match.group(0)
        for match in re.finditer(
            r"\b[A-Za-z0-9_.-]+\.(?:pdf|py|ipynb|md|toml|json|ya?ml|ps1|txt)\b",
            prompt,
            flags=re.IGNORECASE,
        )
    }
    restored = list(filtered)
    for basename in sorted(disallowed_absolute_basenames):
        if basename not in prompt_basenames:
            continue
        if access_manager.resolve_prompt_path(basename, must_exist=False) is None:
            continue
        if basename not in restored:
            restored.append(basename)
    return tuple(restored)


def _absolute_prompt_candidate_can_shadow(
    candidate: str,
    access_manager: WorkspaceAccessManager | None,
) -> bool:
    if access_manager is None:
        return True
    return access_manager.resolve_prompt_path(candidate, must_exist=False) is not None


def _infer_contextual_write_candidates(
    prompt: str,
    access_manager: WorkspaceAccessManager,
) -> tuple[str, ...]:
    prompt_lower = prompt.lower()
    workspace_root = access_manager.active_workspace_root
    candidates: list[str] = []

    vercel_config = workspace_root / "vercel.json"
    if vercel_config.is_file() and any(token in prompt_lower for token in ("vercel", "rewrite", "route", "entrypoint")):
        candidates.append("vercel.json")

    return _dedupe(tuple(candidates))


def _classify_workspace_target_roles(
    candidate_targets: tuple[str, ...],
    prompt: str,
    access_manager: WorkspaceAccessManager,
) -> dict[str, str]:
    prompt_lower = prompt.lower()
    config_task = _looks_like_config_entrypoint_task(prompt_lower)
    config_candidates = {
        candidate
        for candidate in candidate_targets
        if _looks_like_config_target(candidate)
    }
    roles: dict[str, str] = {}
    for candidate in candidate_targets:
        lowered_name = Path(candidate).name.lower()
        if _candidate_is_referenced_config_path(
            candidate,
            prompt_lower,
            access_manager,
            has_config_candidate=bool(config_candidates),
        ):
            roles[candidate] = "referenced_path"
        elif lowered_name == "readme.md" or Path(candidate).suffix.lower() in {".md", ".txt"}:
            roles[candidate] = "secondary_docs_target"
        elif Path(candidate).suffix.lower() == ".ipynb":
            roles[candidate] = "primary_notebook_target"
        elif config_task and _looks_like_config_target(candidate):
            roles[candidate] = "primary_config_target"
        else:
            roles[candidate] = "primary_source_target"
    return roles


def _refine_candidate_roles_for_workspace_context(
    candidate_targets: tuple[str, ...],
    prompt: str,
    access_manager: WorkspaceAccessManager,
    roles: dict[str, str],
) -> dict[str, str]:
    locked_primary_targets = set(_discover_prompt_primary_targets(prompt, access_manager))
    reference_context_targets = set(_discover_prompt_reference_targets(prompt, access_manager))
    refined = dict(roles)
    for candidate in candidate_targets:
        if candidate in reference_context_targets and candidate not in locked_primary_targets:
            refined[candidate] = "context_read_target"

    python_targets = tuple(
        candidate
        for candidate in candidate_targets
        if refined.get(candidate) == "primary_source_target"
        and candidate not in locked_primary_targets
        and Path(candidate).suffix.lower() == ".py"
    )
    if len(python_targets) < 2 or _prompt_requires_grouped_code_edits(prompt):
        return refined

    prompt_tokens = _workspace_target_signal_tokens(prompt)
    if not prompt_tokens:
        return refined

    scored_targets: list[tuple[int, str]] = []
    for candidate in python_targets:
        snippet = _read_workspace_target_snippet(access_manager, candidate).lower()
        score = 0
        for token in prompt_tokens:
            if token in snippet:
                if "_" in token or "-" in token or len(token) >= 8:
                    score += 3
                elif len(token) >= 5:
                    score += 2
                else:
                    score += 1
        if "link" in Path(candidate).stem.lower():
            score += 1
        scored_targets.append((score, candidate))

    ranked = sorted(scored_targets, key=lambda item: (-item[0], item[1].lower()))
    if len(ranked) < 2:
        return refined
    best_score, best_target = ranked[0]
    second_score = ranked[1][0]
    if best_score <= 0 or best_score < second_score + 2:
        return refined

    for _score, candidate in ranked[1:]:
        refined[candidate] = "context_read_target"
    return refined


def _prompt_requires_grouped_code_edits(prompt: str) -> bool:
    lowered = prompt.lower()
    if any(
        token in lowered
        for token in (
            "more than one file",
            "multi-file",
            "edit all",
            "all of the following files",
            "primary task files are",
            "primary task file is",
        )
    ):
        return True
    return bool(
        re.search(r"\bacross\b.+\band\b", lowered)
        or re.search(r"\bboth\b.+\bfiles?\b", lowered)
    )


def _discover_prompt_primary_targets(
    prompt: str,
    access_manager: WorkspaceAccessManager,
) -> tuple[str, ...]:
    return _discover_prompt_section_targets(
        prompt,
        access_manager,
        patterns=(
            r"(?:primary task files?)\s+(?:are|is)\s+(?P<body>[^\n]+)",
            r"existing primary deliverable file\s+(?:is|are)\s*:\s*(?P<body>(?:\n\s*-\s*.+)+)",
        ),
        existing_only=True,
    )


def _discover_prompt_reference_targets(
    prompt: str,
    access_manager: WorkspaceAccessManager,
) -> tuple[str, ...]:
    return _discover_prompt_section_targets(
        prompt,
        access_manager,
        patterns=(
            r"reference implementations?\s+to\s+match\s+(?:are|is)\s+(?P<body>[^\n]+)",
        ),
        existing_only=False,
    )


def _discover_prompt_section_targets(
    prompt: str,
    access_manager: WorkspaceAccessManager,
    *,
    patterns: tuple[str, ...],
    existing_only: bool,
) -> tuple[str, ...]:
    discovered: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE | re.MULTILINE):
            body = match.group("body")
            for wildcard in sorted(
                set(
                    re.findall(
                        r"\b[A-Za-z0-9_.-]*\*[A-Za-z0-9_.-]*\.(?:py|ipynb|toml|json|ya?ml|md|txt)\b",
                        body,
                    )
                )
            ):
                matches = [
                    path.relative_to(access_manager.active_workspace_root).as_posix()
                    for path in sorted(access_manager.active_workspace_root.glob(wildcard))
                    if path.is_file()
                ]
                if matches:
                    for item in matches:
                        if item in seen:
                            continue
                        seen.add(item)
                        discovered.append(item)
                    continue
                if existing_only:
                    continue
                if wildcard not in seen:
                    seen.add(wildcard)
                    discovered.append(wildcard)
            for candidate in _iter_prompt_candidates(body):
                normalized = _sanitize_candidate(candidate)
                if Path(normalized).suffix.lower() not in _TEXT_WRITE_SUFFIXES:
                    continue
                resolved = access_manager.resolve_prompt_path(normalized, must_exist=existing_only)
                if resolved is not None:
                    display = access_manager.display_path(resolved)
                    if display in seen:
                        continue
                    seen.add(display)
                    discovered.append(display)
                    continue
                if existing_only or _looks_like_absolute_path(normalized):
                    continue
                if normalized in seen:
                    continue
                seen.add(normalized)
                discovered.append(normalized)
    return tuple(discovered)


def _workspace_target_signal_tokens(prompt: str) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z0-9_:-]+", prompt.lower())
    stopwords = {
        "actual",
        "all",
        "any",
        "apply",
        "before",
        "but",
        "code",
        "coding",
        "complete",
        "correct",
        "declare",
        "dates",
        "edit",
        "editing",
        "existing",
        "task",
        "goal",
        "implement",
        "must",
        "other",
        "path",
        "prompt",
        "read",
        "real",
        "requested",
        "required",
        "scope",
        "short",
        "should",
        "source",
        "summary",
        "that",
        "then",
        "this",
        "table",
        "truly",
        "update",
        "use",
        "workspace",
        "workspaces",
        "file",
        "files",
        "first",
        "main",
        "minimum",
        "plan",
        "relevant",
        "grouped",
        "identify",
        "inspect",
        "keep",
        "output",
        "produces",
        "reply",
        "satisfy",
        "success",
        "validation",
        "validate",
        "behavior",
        "change",
    }
    kept: list[str] = []
    for token in tokens:
        cleaned = token.strip("`'\"")
        if len(cleaned) < 4 and cleaned not in {"lc", "lu", "pc", "cik"}:
            continue
        if cleaned in stopwords:
            continue
        if cleaned not in kept:
            kept.append(cleaned)
    return tuple(kept[:40])


def _read_workspace_target_snippet(
    access_manager: WorkspaceAccessManager,
    candidate: str,
    *,
    max_chars: int = 8000,
) -> str:
    resolved = access_manager.resolve_prompt_path(candidate, must_exist=True)
    if resolved is None or not resolved.is_file():
        return candidate
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return resolved.read_text(encoding=encoding)[:max_chars]
        except OSError:
            return candidate
        except UnicodeDecodeError:
            continue
    return candidate


def _looks_like_config_entrypoint_task(prompt_lower: str) -> bool:
    return any(
        token in prompt_lower
        for token in (
            "config key",
            "configuration key",
            "environment variable",
            "env var",
            "rewrite",
            "route",
            "entrypoint",
            "entry point",
            "console script",
            "project.scripts",
            "project.urls",
            "pyproject.toml",
            "package.json",
            "vercel.json",
            "devcontainer.json",
            "repository url",
            "homepage url",
            "documentation url",
            "script points",
            "startup",
            "launcher",
            "launch",
            "manifest",
            "deployment",
            "deploy",
            "hosted url",
            "vercel",
            "netlify",
            "package manifest",
        )
    )


def _looks_like_config_target(candidate: str) -> bool:
    normalized = candidate.replace("\\", "/")
    suffix = Path(normalized).suffix.lower()
    name = Path(normalized).name.lower()
    return name in {
        "vercel.json",
        "package.json",
        "pyproject.toml",
        "netlify.toml",
        "docker-compose.yml",
        "docker-compose.yaml",
    } or suffix in {".json", ".toml", ".yaml", ".yml"}


def _candidate_is_referenced_config_path(
    candidate: str,
    prompt_lower: str,
    access_manager: WorkspaceAccessManager,
    *,
    has_config_candidate: bool,
) -> bool:
    if not has_config_candidate:
        return False
    normalized = candidate.replace("\\", "/")
    suffix = Path(normalized).suffix.lower()
    if suffix not in {".py", ".ps1", ".txt"}:
        return False
    if _looks_like_absolute_path(candidate):
        return False
    resolved_reference = _resolve_workspace_reference_path(access_manager, candidate)
    if resolved_reference is None:
        return False
    if normalized.startswith(("/", "\\")):
        return True
    candidate_variants = _dedupe(
        (
            normalized.lower(),
            normalized.lower().lstrip("./"),
            Path(normalized).name.lower(),
        )
    )
    relation_pattern = (
        r"(?:point(?:s)?(?:\s+back)?\s+to|route(?:s)?\s+to|rewrite(?:s)?(?:\s+back)?\s+to|"
        r"source|destination|entrypoint|entry point|hosted url|url)"
    )
    return any(
        re.search(rf"{relation_pattern}[^\n]{{0,60}}{re.escape(variant)}", prompt_lower)
        for variant in candidate_variants
        if variant
    )


def _extract_explicit_config_expectation(prompt: str) -> str:
    patterns = (
        r"(?:points?\s+back\s+to|point\s+back\s+to|points?\s+to|routes?\s+to|matches?)\s+"
        r"(?P<value>`[^`]+`|\"[^\"]+\"|'[^']+'|https?://[^\s,]+|[./A-Za-z0-9_:-]+)",
        r"entry\s+is\s+(?P<value>`[^`]+`|\"[^\"]+\"|'[^']+'|https?://[^\s,]+|[./A-Za-z0-9_:-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match is None:
            continue
        value = _sanitize_candidate(match.group("value"))
        if value.lower() in {"the", "a", "an"}:
            continue
        return value
    return ""


def _extract_explicit_text_expectations(prompt: str) -> tuple[tuple[str, str], ...]:
    expectations: list[tuple[str, str]] = []
    patterns = (
        r"(?P<file>`[^`]+`|[A-Za-z0-9_./-]+\.(?:md|txt))"
        r"[^\n]{0,140}?"
        r"(?:exact\s+(?:heading|text|literal)|heading|text|literal|contains?|include(?:s)?)"
        r"[^\n]{0,40}?"
        r"(?P<value>`[^`]+`|\"[^\"]+\"|'[^']+')",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            relative_target = _sanitize_candidate(match.group("file"))
            expected_value = _sanitize_candidate(match.group("value"))
            if not relative_target or not expected_value:
                continue
            expectations.append((relative_target, expected_value))
    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in expectations:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _discover_next_steps_validation_targets(
    prompt: str,
    *,
    planned_targets: tuple[str, ...],
) -> tuple[str, ...]:
    prompt_lower = prompt.lower()
    if "next step" not in prompt_lower and "next steps" not in prompt_lower:
        return ()
    if "three" not in prompt_lower and "3" not in prompt_lower:
        return ()
    targets = []
    for item in planned_targets:
        lowered_name = Path(item).name.lower()
        if lowered_name in {"handoff_notes.md", "next_steps.md"}:
            targets.append(item)
    return tuple(_dedupe(tuple(targets)))


def infer_workspace_config_reference_targets(
    prompt: str,
    workspace_root: Path,
) -> tuple[str, ...]:
    expected = _extract_explicit_config_expectation(prompt)
    if not expected:
        return ()

    candidates: list[Path] = []
    if _looks_like_workspace_relative_reference(expected):
        normalized = expected.replace("\\", "/").lstrip("/\\")
        if normalized:
            candidates.append((workspace_root / normalized).resolve())
    else:
        module_target, _, attribute = expected.partition(":")
        if attribute and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module_target):
            module_path = Path(*module_target.split("."))
            for base in (workspace_root / "src", workspace_root):
                candidates.append((base / module_path / "__init__.py").resolve())
                candidates.append((base / Path(f"{module_path.as_posix()}.py")).resolve())

    discovered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        relative = _relative_to_workspace(workspace_root, candidate)
        if relative in seen:
            continue
        seen.add(relative)
        discovered.append(relative)
    return tuple(discovered)


def _looks_like_workspace_relative_reference(value: str) -> bool:
    normalized = value.replace("\\", "/")
    if normalized.lower().startswith(("http://", "https://")):
        return False
    return normalized.startswith(("/", "./")) or "/" in normalized


def _resolve_workspace_reference_path(
    access_manager: WorkspaceAccessManager,
    candidate: str,
) -> Path | None:
    normalized = candidate.replace("\\", "/").lstrip("/\\")
    if not normalized:
        return None
    resolved = access_manager.resolve_prompt_path(normalized, must_exist=True)
    if resolved is not None and resolved.is_file():
        return resolved
    fallback = (access_manager.active_workspace_root / normalized).resolve()
    if fallback.is_file():
        return fallback
    return None


def _extract_structured_file_blocks(answer_text: str) -> dict[str, str]:
    blocks = {
        path: _normalize_structured_block_content(block.content, block.language)
        for path, block in extract_structured_file_block_specs(answer_text).items()
    }
    for normalized_path, content in _extract_json_operation_file_blocks(answer_text).items():
        blocks.setdefault(normalized_path, content)
    return blocks


def _extract_json_operation_file_blocks(answer_text: str) -> dict[str, str]:
    if not answer_text.strip():
        return {}

    blocks: dict[str, str] = {}
    for match in re.finditer(
        r"```json\s*(?P<body>.*?)```",
        answer_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        body = match.group("body")
        path_match = re.search(r'"(?:name|path)"\s*:\s*"(?P<path>[^"\n]+)"', body)
        if not path_match:
            continue
        raw_path = _sanitize_candidate(path_match.group("path"))
        if not raw_path:
            continue

        content = ""
        backtick_match = re.search(r'"content"\s*:\s*`(?P<content>.*?)`', body, flags=re.DOTALL)
        if backtick_match:
            content = backtick_match.group("content").rstrip()
        else:
            quoted_match = re.search(
                r'"content"\s*:\s*"(?P<content>(?:\\.|[^"\\])*)"',
                body,
                flags=re.DOTALL,
            )
            if quoted_match:
                try:
                    content = json.loads(f"\"{quoted_match.group('content')}\"").rstrip()
                except json.JSONDecodeError:
                    continue
        if not content:
            continue
        blocks[_normalize_structured_block_path(raw_path)] = content
    return blocks


def _normalize_structured_block_content(content: str, language: str) -> str:
    normalized = content.rstrip()
    lowered_language = (language or "").strip().lower()
    if "diff" not in lowered_language:
        return normalized

    rebuilt: list[str] = []
    for line in normalized.splitlines():
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            continue
        if line.startswith("+"):
            rebuilt.append(line[1:])
            continue
        rebuilt.append(line)
    return "\n".join(rebuilt).rstrip()


def _match_structured_file_content(
    blocks: dict[str, str],
    *,
    workspace_root: Path,
    display_target: str,
    target_path: Path,
    final_path: Path,
) -> str:
    if not blocks:
        return ""
    candidates = (
        display_target,
        _relative_to_workspace(workspace_root, target_path),
        _relative_to_workspace(workspace_root, final_path),
        target_path.name,
        final_path.name,
        str(target_path),
        str(final_path),
    )
    for candidate in candidates:
        normalized = _normalize_structured_block_path(candidate)
        if normalized in blocks:
            return blocks[normalized]
    return ""


def _normalize_structured_block_path(path_text: str) -> str:
    return Path(_sanitize_candidate(path_text)).as_posix().lower()


def _discover_pytest_targets(
    prompt: str,
    workspace_root: Path,
    *,
    python_targets: tuple[str, ...],
) -> tuple[str, ...]:
    targets: list[str] = []
    seen: set[str] = set()

    for item in python_targets:
        target_path = _resolve_workspace_relative_path(workspace_root, item)
        name = target_path.name
        if name.startswith("test_") and name.endswith(".py"):
            relative = _relative_to_workspace(workspace_root, target_path)
            if relative not in seen:
                seen.add(relative)
                targets.append(relative)
            continue

        stem = target_path.stem
        candidate_paths = (
            workspace_root / "tests" / f"test_{stem}.py",
            workspace_root / "tests" / f"{stem}_test.py",
        )
        for candidate in candidate_paths:
            if candidate.is_file():
                relative = _relative_to_workspace(workspace_root, candidate)
                if relative not in seen:
                    seen.add(relative)
                    targets.append(relative)

    prompt_lower = prompt.lower()
    wants_test_checks = any(
        token in prompt_lower
        for token in (
            "pytest",
            "unit test",
            "tests",
            "test ",
            "failing test",
            "regression",
            "bug",
            "fix",
            "verify",
        )
    )
    tests_root = workspace_root / "tests"
    if wants_test_checks and not targets and tests_root.is_dir():
        test_files = sorted(tests_root.glob("test_*.py"))
        if 0 < len(test_files) <= 12:
            for path in test_files:
                relative = _relative_to_workspace(workspace_root, path)
                if relative not in seen:
                    seen.add(relative)
                    targets.append(relative)

    return tuple(targets)


def _discover_validation_script_targets(
    prompt: str,
    *,
    planned_targets: tuple[str, ...],
    task_contract: WorkspaceTaskContract,
) -> tuple[str, ...]:
    if not task_contract.get("behavioral_validation_required", False):
        return ()
    prompt_lower = prompt.lower()
    explicit_validation_request = any(
        token in prompt_lower
        for token in (
            "validation",
            "validate",
            "smoke",
            "assert",
            "fixture",
            "harness",
            "regression",
        )
    )
    targets: list[str] = []
    for item in planned_targets:
        normalized = item.replace("\\", "/")
        stem = Path(normalized).stem.lower()
        if Path(normalized).suffix.lower() != ".py":
            continue
        if any(
            token in stem
            for token in ("validate", "validation", "smoke", "contract", "check")
        ) or normalized.startswith("validation/") or "/validation/" in normalized or "/checks/" in normalized:
            targets.append(item)
    if not explicit_validation_request:
        return tuple(_dedupe(tuple(targets)))
    return tuple(_dedupe(tuple(targets)))


def _summarize_task_goal(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt.strip())
    if len(normalized) <= 260:
        return normalized
    return normalized[:257].rstrip() + "..."


def _discover_prompt_context_reads(prompt: str, workspace_root: Path) -> tuple[str, ...]:
    reads: list[str] = []
    seen: set[str] = set()
    patterns = set(
        re.findall(
            r"\b[A-Za-z0-9_.-]*\*[A-Za-z0-9_.-]*\.(?:py|ipynb|toml|json|ya?ml|md|txt)\b",
            prompt,
        )
    )
    for pattern in sorted(patterns):
        for path in sorted(workspace_root.glob(pattern)):
            if not path.is_file():
                continue
            relative = path.relative_to(workspace_root).as_posix()
            if relative not in seen:
                seen.add(relative)
                reads.append(relative)
    return tuple(reads)


def _promote_task1_daily_crsp_targets(prompt: str, workspace_root: Path) -> tuple[str, ...]:
    lowered = prompt.lower()
    if "15_preparedailycrsp_task*.py" not in lowered:
        return ()
    if not any(
        token in lowered
        for token in (
            "daily crsp",
            "centralized start-date rule",
            "ad-hoc truncation logic",
            "rolling scripts",
            "2010-01-01",
        )
    ):
        return ()
    promoted: list[str] = []
    for path in sorted(workspace_root.glob("15_PrepareDailyCRSP_task*.py")):
        if not path.is_file():
            continue
        try:
            contents = path.read_text(encoding="utf-8-sig").lower()
        except OSError:
            contents = ""
        if any(
            token in contents
            for token in (
                "2010-01-01",
                ">= '2010",
                '>= "2010',
                "m_dcrsp",
            )
        ):
            promoted.append(path.relative_to(workspace_root).as_posix())
    if promoted:
        return tuple(promoted)
    return tuple(
        path.relative_to(workspace_root).as_posix()
        for path in sorted(workspace_root.glob("15_PrepareDailyCRSP_task*.py"))
        if path.is_file()
    )


def _extract_current_task_prompt(prompt: str) -> str:
    marker = "Original instruction:"
    if marker in prompt:
        extracted = prompt.split(marker, 1)[1].lstrip()
        section_match = re.search(
            r"^\s*(?:Task contract:|Locked target files:|Focused files to revisit first:|Current workspace context:|Required FILE blocks? this round:|Acceptance requirements(?: the validation must prove)?:|Checks that must pass:|Failures or gaps from the previous attempt:|Repair directives:|Validation requirement for this round:)\s*$",
            extracted,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if section_match is not None:
            extracted = extracted[: section_match.start()].rstrip()
        if extracted:
            return extracted
    return prompt


def _prompt_contains_explicit_retry_targets(prompt: str) -> bool:
    return bool(
        re.search(
            r"^\s*(?:Locked target files:|Focused files to revisit first:|Required FILE blocks? this round:)\s*$",
            prompt,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )


def _normalize_task_contract_clauses(clauses: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for clause in clauses:
        expanded = _expand_multi_requirement_clause(clause)
        if expanded:
            normalized.extend(expanded)
            continue
        normalized.append(clause)
    return tuple(_dedupe(tuple(normalized)))


def _expand_multi_requirement_clause(clause: str) -> tuple[str, ...]:
    lowered = clause.lower()
    keyword_tokens = (
        "restrict to",
        "standardize",
        "preserve",
        "keep ",
        "handle ",
        "ensure ",
        "output columns",
        "datetime",
        "null ",
        "empty string",
        "validity-window",
    )
    if not any(token in lowered for token in keyword_tokens):
        return ()
    segments = re.split(r",\s+|(?:\s+and\s+)", clause)
    expanded: list[str] = []
    carry_prefix = ""
    for segment in segments:
        cleaned = segment.strip().strip(".")
        lowered_segment = cleaned.lower()
        if lowered_segment.startswith("implement the required behavior"):
            cleaned = re.sub(
                r"(?i)^implement the required behavior:\s*",
                "",
                cleaned,
            ).strip()
        if lowered_segment.startswith("the required behavior:"):
            cleaned = re.sub(r"(?i)^the required behavior:\s*", "", cleaned).strip()
        if not cleaned:
            continue
        lowered_cleaned = cleaned.lower()
        if any(token in lowered_cleaned for token in keyword_tokens):
            if "restrict to" in lowered_cleaned:
                carry_prefix = "restrict to"
            elif "standardize" in lowered_cleaned and "output columns" in lowered_cleaned:
                carry_prefix = "standardize the output columns to"
            else:
                carry_prefix = ""
            expanded.append(cleaned.rstrip(".") + ".")
            continue
        if len(cleaned) < 3:
            continue
        if carry_prefix and re.fullmatch(r"[A-Za-z0-9_./:-]+(?:\s+[A-Za-z0-9_./:-]+)*", cleaned):
            expanded.append(f"{carry_prefix} {cleaned.rstrip('.')}.")
    if len(expanded) <= 1:
        return ()
    return tuple(_dedupe(tuple(expanded)))


def _extract_task_contract_clauses(
    prompt: str,
    *,
    category: Literal["behavior", "acceptance", "forbidden", "validation"],
) -> tuple[str, ...]:
    clauses = _split_prompt_clauses(prompt)
    selected: list[str] = []
    for clause in clauses:
        lowered = clause.lower()
        if category in {"behavior", "acceptance"} and _is_operational_task_clause(lowered):
            continue
        if category == "forbidden":
            if any(
                token in lowered
                for token in (
                    "do not",
                    "don't",
                    "must not",
                    "should not",
                    "not enough",
                    "do not accept",
                    "do not satisfy",
                    "do not declare success",
                )
            ):
                selected.append(clause)
            continue
        if category == "validation":
            if _is_operational_task_clause(lowered) and not any(
                token in lowered
                for token in (
                    "assert",
                    "fixture",
                    "validation harness",
                    "existing test",
                    "existing behavioral test",
                    "py_compile",
                    "pytest",
                )
            ):
                continue
            if any(
                token in lowered
                for token in (
                    "run ",
                    "validate",
                    "validation",
                    "assert",
                    "check",
                    "pytest",
                    "fixture",
                    "smoke",
                    "py_compile",
                    "test ",
                    "tests ",
                )
            ):
                selected.append(clause)
            continue
        if category == "behavior":
            if any(
                token in lowered
                for token in (
                    "implement",
                    "fix ",
                    "repair",
                    "restrict",
                    "standardize",
                    "preserve",
                    "keep ",
                    "handle",
                    "ensure",
                    "return ",
                    "change the real code path",
                    "source edits",
                )
            ) and "run " not in lowered and "main files are" not in lowered and "treat it as" not in lowered:
                selected.append(clause)
            continue
        if category == "acceptance":
            if any(
                token in lowered
                for token in (
                    "restrict",
                    "standardize",
                    "preserve",
                    "keep ",
                    "handle",
                    "assert",
                    "acceptance criteria",
                    "must ",
                    "required",
                    "criteria",
                    "real code path",
                    "datetime",
                    "null ",
                    "empty string",
                    "output columns",
                    "validity-window",
                )
            ) and not lowered.startswith("use a short plan") and "main files are" not in lowered:
                selected.append(clause)
    return tuple(_dedupe(tuple(selected)))


def _is_operational_task_clause(lowered_clause: str) -> bool:
    return any(
        token in lowered_clause
        for token in (
            "start with a short plan",
            "use a short plan",
            "short plan first",
            "keep the changes scoped",
            "keep the scope tight",
            "read the existing code first",
            "read the code first",
            "identify the relevant files",
            "identify relevant files",
            "run relevant checks",
            "run a relevant check",
            "give a truthful final summary",
            "truthful final summary",
            "do not reply with only explanations",
            "treat it as a real ra coding task",
            "must end in actual source edits",
        )
    )


def _split_prompt_clauses(prompt: str) -> tuple[str, ...]:
    normalized = prompt.replace("\r", "\n")
    raw_parts = re.split(r"[\n;]+|(?<=[.!?])\s+(?=[A-Z0-9`])", normalized)
    clauses: list[str] = []
    for part in raw_parts:
        clause = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part.strip())
        clause = clause.strip("` ")
        if len(clause) < 12:
            continue
        if len(clause) > 280:
            clause = clause[:277].rstrip() + "..."
        clauses.append(clause)
    return tuple(_dedupe(tuple(clauses)))


def _is_behavioral_edit_task(
    prompt: str,
    *,
    explicit_files: tuple[str, ...],
    behavior_requirements: tuple[str, ...],
    acceptance_criteria: tuple[str, ...],
) -> bool:
    prompt_lower = prompt.lower()
    if acceptance_criteria or behavior_requirements:
        return True
    if any(
        token in prompt_lower
        for token in (
            "acceptance criteria",
            "behavior",
            "business rule",
            "feature",
            "bugfix",
            "bug fix",
            "dataframe",
            "table",
            "output columns",
            "datetime",
            "null ",
            "empty string",
            "validation",
            "fixture",
            "real code path",
            "must end in actual source edits",
        )
    ):
        return True
    return any(Path(item).suffix.lower() == ".py" for item in explicit_files) and any(
        token in prompt_lower
        for token in (
            "implement",
            "fix",
            "repair",
            "refactor",
            "preserve",
            "restrict",
            "standardize",
            "handle",
        )
    )


def _build_pytest_env(
    workspace_root: Path,
    *,
    python_targets: tuple[str, ...],
    pytest_targets: tuple[str, ...],
) -> dict[str, str]:
    if not pytest_targets:
        return {}

    src_root = workspace_root / "src"
    existing = os.environ.get("PYTHONPATH", "").strip()
    path_entries = [str(workspace_root.resolve())]
    if src_root.is_dir():
        path_entries.append(str(src_root.resolve()))
    if existing:
        path_entries.append(existing)
    return {"PYTHONPATH": os.pathsep.join(path_entries)}


def _resolve_workspace_relative_path(workspace_root: Path, item: str) -> Path:
    candidate = Path(item)
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


def _normalize_workspace_relative_path(item: str) -> str:
    return item.replace("\\", "/").lstrip("./")


def _path_belongs_to_workspace(workspace_root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(workspace_root.resolve())
        return True
    except ValueError:
        return False


def _looks_like_generated_validation_target(path: str) -> bool:
    normalized = _normalize_workspace_relative_path(path).lower()
    stem = Path(normalized).stem.lower()
    return (
        normalized.startswith("validation/")
        or "/validation/" in normalized
        or any(token in stem for token in ("validate", "validation", "smoke", "contract", "check"))
    )


def _extract_validation_task_run_id(path: str) -> str:
    stem = Path(_normalize_workspace_relative_path(path)).stem
    match = re.search(r"_([0-9a-f]{8,32})$", stem, flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def _python_file_syntax_preflight(path: Path) -> tuple[bool, str]:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return False, f"Could not read validator file: {exc}"
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        return False, _format_syntax_preflight_error(exc)
    return True, ""


def _format_syntax_preflight_error(exc: SyntaxError) -> str:
    line_no = exc.lineno or 0
    offset = exc.offset or 0
    detail = (exc.msg or "SyntaxError").strip()
    line = (exc.text or "").strip()
    location = f"line {line_no}"
    if offset:
        location += f", column {offset}"
    if line:
        return f"{detail} at {location}: {line}"
    return f"{detail} at {location}."


def _default_check_relevance_reason(check_name: str, relative_targets: tuple[str, ...]) -> str:
    if check_name == "python_validate":
        return "Focused behavioral validation for the current task."
    if check_name == "pytest":
        return "Existing project tests relevant to the current source changes."
    if check_name in {"json_validate", "toml_validate"}:
        return "Syntax validation for the current config target."
    if check_name == "content_expectation":
        return "Content validation for the current task deliverable or config literal."
    if check_name == "py_compile":
        return "Python syntax validation for the current source edits."
    if relative_targets:
        return f"Validation for current task targets: {', '.join(relative_targets)}."
    return "Validation for the current task."


def _workspace_task_type(
    *,
    planned_modifications: tuple[str, ...],
    planned_creations: tuple[str, ...],
    behavioral_validation_required: bool,
) -> str:
    if behavioral_validation_required:
        if planned_modifications and planned_creations:
            return "behavioral_multi_file_edit"
        if planned_modifications:
            return "behavioral_in_place_edit"
        if planned_creations:
            return "behavioral_creation"
        return "behavioral_task"
    if planned_modifications and planned_creations:
        return "mixed_workspace_edit"
    if planned_modifications:
        return "in_place_workspace_edit"
    if planned_creations:
        return "workspace_deliverable_creation"
    return "workspace_task"


def _relative_to_workspace(workspace_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _truncate_process_output(stdout: str, stderr: str, *, limit: int = 420) -> str:
    combined = "\n".join(part.strip() for part in (stdout, stderr) if part.strip()).strip()
    if not combined:
        return ""
    filtered_lines: list[str] = []
    for line in combined.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            "PytestCacheWarning" in stripped
            or "could not create cache path" in stripped
            or "pytest-cache-files-" in stripped
            or stripped.startswith("-- Docs: https://docs.pytest.org")
        ):
            continue
        filtered_lines.append(stripped)
    preferred_lines = _prioritize_process_excerpt_lines(filtered_lines)
    compact = re.sub(r"\s+", " ", "\n".join(preferred_lines) or "\n".join(filtered_lines) or combined)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _prioritize_process_excerpt_lines(lines: list[str]) -> list[str]:
    highlight_tokens = (
        "assertionerror",
        "nameerror",
        "modulenotfounderror",
        "importerror",
        "traceback",
        "failed",
        "error collecting",
        "short test summary info",
        "test_",
    )
    preferred: list[str] = []
    seen: set[str] = set()

    for line in lines:
        lowered = line.lower()
        if line.startswith("E   ") or any(token in lowered for token in highlight_tokens):
            if line not in seen:
                seen.add(line)
                preferred.append(line)

    for line in lines[:8]:
        if line not in seen:
            seen.add(line)
            preferred.append(line)

    return preferred


def _sanitize_candidate(value: str) -> str:
    return value.strip().strip("`\"'")


def _contains_any(prompt_lower: str, prompt_original: str, tokens: tuple[str, ...]) -> bool:
    return any(token.lower() in prompt_lower or token in prompt_original for token in tokens)


def _resolve_target_path(
    candidate: str,
    prompt: str,
    mode_selection: ModeSelection,
    access_manager: WorkspaceAccessManager,
) -> tuple[Path, str]:
    if _looks_like_absolute_path(candidate):
        return Path(candidate).resolve(), "explicit_path"

    source_directory = _same_folder_source_dir(prompt, mode_selection, access_manager)
    if source_directory is not None:
        return (source_directory / Path(candidate).name).resolve(), "same_folder"

    return (access_manager.active_workspace_root / candidate).resolve(), "active_workspace"


def _same_folder_source_dir(
    prompt: str,
    mode_selection: ModeSelection,
    access_manager: WorkspaceAccessManager,
) -> Path | None:
    prompt_lower = prompt.lower()
    if not _contains_any(prompt_lower, prompt, _SAME_FOLDER_TOKENS):
        return None

    candidate_files: list[Path] = []
    for item in mode_selection.matched_paths:
        resolved = access_manager.resolve_prompt_path(item, must_exist=True)
        if resolved is None or not resolved.is_file():
            continue
        candidate_files.append(resolved)

    if not candidate_files:
        return None

    parent_dirs = {path.parent.resolve() for path in candidate_files}
    if len(parent_dirs) == 1:
        return next(iter(parent_dirs))
    return None


def _display_target(
    access_manager: WorkspaceAccessManager,
    target_path: Path,
    original_candidate: str,
) -> str:
    try:
        return access_manager.display_path(target_path)
    except OSError:
        return original_candidate


def _primary_destination_policy(plan: WorkspaceEditPlan) -> str:
    if not plan.operations:
        return ""
    return plan.operations[0].destination_policy


def _stage_operation(
    *,
    access_manager: WorkspaceAccessManager,
    workspace_root: Path,
    prompt: str,
    operation: WorkspaceEditOperation,
    target_path: Path,
    final_answer: str,
    observations: list[str],
    structured_blocks: dict[str, str],
    structured_block_specs: dict[str, object],
) -> tuple[_StagedTextWrite | None, str]:
    suffix = target_path.suffix.lower()
    if suffix == ".xlsx":
        return None, f"{operation.display_target} was skipped because binary spreadsheet writes are not implemented."

    created = operation.action == "create_file"
    original_content: str | None = None
    final_path = target_path
    collision_suffix = 0
    if created and final_path.exists() and not operation.overwrite:
        final_path, collision_suffix = _resolve_collision_path(final_path)
    if not created:
        try:
            original_content = target_path.read_text(encoding="utf-8")
        except OSError as exc:
            return None, f"Could not read {operation.display_target}: {exc}"

    structured_content = _match_structured_file_content(
        structured_blocks,
        workspace_root=workspace_root,
        display_target=operation.display_target,
        target_path=target_path,
        final_path=final_path,
    )
    block_spec = _match_structured_file_block(
        structured_block_specs,
        workspace_root=workspace_root,
        display_target=operation.display_target,
        target_path=target_path,
        final_path=final_path,
    )
    if (
        block_spec is not None
        and "diff" in getattr(block_spec, "language", "")
        and any(marker in block_spec.content for marker in ("---", "+++", "@@"))
    ):
        try:
            diff_result = apply_unified_diff_text(
                diff_text=block_spec.content,
                original_content=original_content or "",
                workspace_root=workspace_root,
                expected_target=operation.display_target,
            )
            structured_content = diff_result.normalized_content
        except ValueError as exc:
            return None, (
                f"{operation.display_target} was skipped because the unified diff block was invalid: {exc}"
            )

    if operation.strategy == "python_module_docstring":
        if original_content is None:
            return None, f"{operation.display_target} is missing, so the docstring edit could not be applied."
        updated = _add_python_module_docstring(original_content, target_path)
        if updated == original_content:
            return None, f"{operation.display_target} already had a module docstring, so it was left untouched."
        summary = f"{operation.display_target}: added a short module docstring at the top."
        return _StagedTextWrite(
            final_path=final_path,
            display_path=access_manager.display_path(final_path),
            content=updated,
            created=False,
            summary=summary,
            destination_policy=operation.destination_policy,
            original_content=original_content,
        ), ""

    if operation.strategy in {"structured_text_file", "structured_text_refresh"}:
        if not structured_content:
            return None, (
                f"{operation.display_target} was skipped because the workflow answer did not include "
                "a structured file block for that path."
            )
        preflight_error = _preflight_staged_text_content(final_path, structured_content)
        if preflight_error:
            return None, (
                f"{operation.display_target} was skipped because the proposed file block failed content preflight: "
                f"{preflight_error}"
            )
        summary = _operation_summary(operation, final_path, created=(operation.action == "create_file"))
        return _StagedTextWrite(
            final_path=final_path,
            display_path=access_manager.display_path(final_path),
            content=_ensure_trailing_newline(structured_content),
            created=operation.action == "create_file",
            summary=summary,
            destination_policy=operation.destination_policy,
            collision_suffix=collision_suffix,
            original_content=original_content,
        ), ""

    if structured_content:
        preflight_error = _preflight_staged_text_content(final_path, structured_content)
        if preflight_error:
            return None, (
                f"{operation.display_target} was skipped because the proposed file block failed content preflight: "
                f"{preflight_error}"
            )
        summary = _operation_summary(operation, final_path, created=(operation.action == "create_file"))
        return _StagedTextWrite(
            final_path=final_path,
            display_path=access_manager.display_path(final_path),
            content=_ensure_trailing_newline(structured_content),
            created=operation.action == "create_file",
            summary=summary,
            destination_policy=operation.destination_policy,
            collision_suffix=collision_suffix,
            original_content=original_content,
        ), ""

    rendered = _render_operation_content(
        workspace_root=workspace_root,
        prompt=prompt,
        operation=operation,
        target_path=target_path,
        final_path=final_path,
        original_content=original_content,
        final_answer=final_answer,
        observations=observations,
    )
    if rendered is None:
        return None, f"{operation.display_target} did not need a content update."
    preflight_error = _preflight_staged_text_content(final_path, rendered)
    if preflight_error:
        return None, (
            f"{operation.display_target} was skipped because the rendered content failed preflight: "
            f"{preflight_error}"
        )

    summary = _operation_summary(operation, final_path, created=(operation.action == "create_file"))
    return _StagedTextWrite(
        final_path=final_path,
        display_path=access_manager.display_path(final_path),
        content=rendered,
        created=operation.action == "create_file",
        summary=summary,
        destination_policy=operation.destination_policy,
        collision_suffix=collision_suffix,
        original_content=original_content,
    ), ""


def _preflight_staged_text_content(target_path: Path, content: str) -> str:
    suffix = target_path.suffix.lower()
    try:
        if suffix == ".py":
            compile(content, str(target_path), "exec")
        elif suffix in {".json", ".ipynb"}:
            json.loads(content)
        elif suffix == ".toml":
            tomllib.loads(content)
    except SyntaxError as exc:
        return _format_syntax_preflight_error(exc)
    except json.JSONDecodeError as exc:
        return f"invalid JSON content: {exc.msg}"
    except tomllib.TOMLDecodeError as exc:
        return f"invalid TOML content: {exc}"
    return ""


def _structured_block_candidates(
    *,
    workspace_root: Path,
    display_target: str,
    target_path: Path,
    final_path: Path,
) -> tuple[str, ...]:
    candidates = (
        display_target,
        _relative_to_workspace(workspace_root, target_path),
        _relative_to_workspace(workspace_root, final_path),
        target_path.name,
        final_path.name,
        str(target_path),
        str(final_path),
    )
    normalized = [
        _normalize_structured_block_path(candidate)
        for candidate in candidates
        if str(candidate).strip()
    ]
    return _dedupe(tuple(normalized))


def _match_structured_file_block(
    blocks: dict[str, object],
    *,
    workspace_root: Path,
    display_target: str,
    target_path: Path,
    final_path: Path,
):
    if not blocks:
        return None
    candidates = _structured_block_candidates(
        workspace_root=workspace_root,
        display_target=display_target,
        target_path=target_path,
        final_path=final_path,
    )
    for candidate in candidates:
        if candidate in blocks:
            return blocks[candidate]
    return None


def _stage_related_support_blocks(
    *,
    access_manager: WorkspaceAccessManager,
    workspace_root: Path,
    plan: WorkspaceEditPlan,
    structured_blocks: dict[str, str],
    used_block_paths: set[str],
    max_extra_blocks: int = 2,
) -> tuple[tuple[_StagedTextWrite, ...], tuple[str, ...]]:
    if not structured_blocks or not plan.active:
        return (), ()

    staged: list[_StagedTextWrite] = []
    skipped_notes: list[str] = []

    for normalized_path, content in structured_blocks.items():
        if normalized_path in used_block_paths:
            continue
        if len(staged) >= max_extra_blocks:
            skipped_notes.append(
                "Additional structured support files were ignored after the focused support-file limit was reached."
            )
            break

        candidate = Path(normalized_path)
        if candidate.suffix.lower() not in _TEXT_WRITE_SUFFIXES:
            continue

        display_target = candidate.as_posix()
        try:
            target_path = access_manager.resolve_user_path(
                display_target,
                for_write=True,
                must_exist=False,
            )
        except WorkspaceAccessError as exc:
            skipped_notes.append(str(exc))
            continue

        relative_target = _relative_to_workspace(workspace_root, target_path)
        if not _is_related_support_path(relative_target, plan):
            skipped_notes.append(
                f"{relative_target} was not created because it falls outside the focused multi-file edit scope."
            )
            continue

        created = not target_path.exists()
        original_content: str | None = None
        if not created:
            try:
                original_content = target_path.read_text(encoding="utf-8")
            except OSError as exc:
                skipped_notes.append(f"Could not read {relative_target}: {exc}")
                continue

        summary = (
            f"{relative_target}: created a directly necessary support file for the grouped edit."
            if created
            else f"{relative_target}: updated a directly necessary support file for the grouped edit."
        )
        staged.append(
            _StagedTextWrite(
                final_path=target_path,
                display_path=access_manager.display_path(target_path),
                content=_ensure_trailing_newline(content),
                created=created,
                summary=summary,
                destination_policy="active_workspace",
                original_content=original_content,
            )
        )

    return tuple(staged), _dedupe(tuple(skipped_notes))


def _is_related_support_path(relative_target: str, plan: WorkspaceEditPlan) -> bool:
    normalized_target = relative_target.replace("\\", "/").strip("./")
    if not normalized_target:
        return False

    planned_targets = _dedupe((*plan.planned_modifications, *plan.planned_creations))
    normalized_planned = tuple(path.replace("\\", "/").strip("./") for path in planned_targets if path.strip())
    if normalized_target in normalized_planned:
        return False
    if _looks_like_validation_path(normalized_target) and any(
        _looks_like_validation_path(item) for item in normalized_planned
    ):
        return False
    target_name = Path(normalized_target).name.lower()
    if Path(normalized_target).suffix.lower() in {".md", ".txt"}:
        return False
    if "pyproject.toml" in {Path(item).name.lower() for item in normalized_planned} and target_name in {
        "setup.py",
        "setup.cfg",
    }:
        return False

    target_path = Path(normalized_target)
    target_parent = target_path.parent.as_posix()
    target_parts = target_path.parts

    planned_parents = {
        Path(item).parent.as_posix()
        for item in normalized_planned
        if Path(item).parent.as_posix() not in {"", "."}
    }
    if target_parent in planned_parents:
        return True

    planned_roots = {Path(item).parts[0] for item in normalized_planned if Path(item).parts}
    if target_parts and target_parts[0] in planned_roots:
        return True

    return False


def _looks_like_validation_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    stem = Path(normalized).stem.lower()
    return (
        normalized.startswith("validation/")
        or "/validation/" in normalized
        or any(token in stem for token in ("validate", "validation", "smoke", "contract", "check"))
    )


def _render_operation_content(
    *,
    workspace_root: Path,
    prompt: str,
    operation: WorkspaceEditOperation,
    target_path: Path,
    final_path: Path,
    original_content: str | None,
    final_answer: str,
    observations: list[str],
) -> str | None:
    if operation.strategy == "answer_body_file":
        return _ensure_trailing_newline(final_answer)
    if operation.strategy == "handoff_markdown":
        return _render_handoff_markdown(final_path, final_answer, observations)
    if operation.strategy == "readme_handoff_refresh":
        return _render_readme_refresh(workspace_root, original_content or "", final_answer, observations)
    if operation.strategy == "pyproject_comment_refresh":
        return _render_pyproject_comment_refresh(original_content or "", final_answer)
    if operation.strategy == "managed_text_refresh":
        return _render_managed_text_refresh(target_path, original_content or "", final_answer, prompt)
    return _ensure_trailing_newline(final_answer)


def _render_handoff_markdown(
    target_path: Path,
    final_answer: str,
    observations: list[str],
) -> str:
    title = target_path.stem.replace("_", " ").replace("-", " ").strip() or "Notes"
    content = [
        f"# {title}",
        "",
        _compact_answer(final_answer),
    ]
    if observations:
        content.extend(
            (
                "",
                "## Supporting context",
                *[f"- {item}" for item in observations[:4]],
            )
        )
    return _ensure_trailing_newline("\n".join(content).strip())


def _render_readme_refresh(
    workspace_root: Path,
    original_content: str,
    final_answer: str,
    observations: list[str],
) -> str:
    existing = original_content.strip()
    if not existing:
        existing = f"# {workspace_root.name}\n"
    next_steps = observations[:3]
    body_lines = [
        "## RA Handoff Snapshot",
        _compact_answer(final_answer),
    ]
    if next_steps:
        body_lines.extend(("", "### Immediate checks", *[f"- {item}" for item in next_steps]))
    block = _managed_block(
        "LABAI:README-HANDOFF",
        "\n".join(body_lines).strip(),
    )
    return _replace_or_append_managed_block(existing, "LABAI:README-HANDOFF", block)


def _render_pyproject_comment_refresh(original_content: str, final_answer: str) -> str:
    body = (
        "# Managed by labai: lightweight handoff note for this workspace.\n"
        f"# {_single_line(_compact_answer(final_answer), limit=88)}\n"
    )
    marker = "LABAI:PYPROJECT-NOTE"
    if not original_content.strip():
        return _ensure_trailing_newline(f"# {marker}\n{body}")

    lines = original_content.splitlines()
    filtered: list[str] = []
    skip = False
    for line in lines:
        if line.strip() == f"# {marker}-START":
            skip = True
            continue
        if line.strip() == f"# {marker}-END":
            skip = False
            continue
        if not skip:
            filtered.append(line)
    managed = [
        f"# {marker}-START",
        body.rstrip(),
        f"# {marker}-END",
        "",
    ]
    return _ensure_trailing_newline("\n".join((*managed, *filtered)).strip())


def _render_managed_text_refresh(
    target_path: Path,
    original_content: str,
    final_answer: str,
    prompt: str,
) -> str:
    if target_path.suffix.lower() == ".md":
        block = _managed_block(
            "LABAI:TEXT-REFRESH",
            "\n".join(
                (
                    "## Requested update",
                    _compact_answer(final_answer),
                )
            ),
        )
        return _replace_or_append_managed_block(original_content.strip(), "LABAI:TEXT-REFRESH", block)
    if original_content.strip():
        return _ensure_trailing_newline(
            f"{original_content.rstrip()}\n\n[labai update]\n{_compact_answer(final_answer)}"
        )
    return _ensure_trailing_newline(_compact_answer(final_answer))


def _apply_staged_writes(
    staged_writes: list[_StagedTextWrite],
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    applied: list[_StagedTextWrite] = []
    created_files: list[str] = []
    modified_files: list[str] = []
    rollback_notes: list[str] = []

    for item in staged_writes:
        try:
            _write_text_atomic(item.final_path, item.content)
        except OSError as exc:
            rollback_notes.append(f"Apply failed for {item.display_path}: {exc}")
            for previous in reversed(applied):
                try:
                    if previous.created:
                        if previous.final_path.exists():
                            previous.final_path.unlink()
                        rollback_notes.append(f"Rolled back created file {previous.display_path}.")
                    else:
                        _write_text_atomic(previous.final_path, previous.original_content or "")
                        rollback_notes.append(f"Restored original contents for {previous.display_path}.")
                except OSError as rollback_exc:
                    rollback_notes.append(
                        f"Rollback warning for {previous.display_path}: {rollback_exc}"
                    )
            return str(exc), (), (), _dedupe(tuple(rollback_notes))

        applied.append(item)
        if item.created:
            created_files.append(item.display_path)
        else:
            modified_files.append(item.display_path)

    return "", _dedupe(tuple(created_files)), _dedupe(tuple(modified_files)), _dedupe(tuple(rollback_notes))


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.labai-{os.getpid()}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _finalize_git_summary(
    before: GitWorkspaceSummary,
    after: GitWorkspaceSummary,
    created_files: tuple[str, ...],
    modified_files: tuple[str, ...],
) -> GitWorkspaceSummary:
    if not before.detected and not after.detected:
        return GitWorkspaceSummary(note=after.note or before.note)

    before_status = before.before_status or _synthetic_git_status_lines((), ())
    after_status_lines = after.before_status or _synthetic_git_status_lines(modified_files, created_files)
    tracked_after, untracked_after = _parse_git_status(after_status_lines)
    task_paths = _normalize_task_paths((*created_files, *modified_files))
    changed_tracked = _filter_git_paths(tracked_after, task_paths)
    untracked = _filter_git_paths(untracked_after, task_paths)
    if not changed_tracked:
        changed_tracked = tracked_after
    if not untracked:
        untracked = untracked_after

    commit_message = _draft_commit_message(created_files, modified_files, changed_tracked, untracked)
    return GitWorkspaceSummary(
        detected=True,
        repo_root=after.repo_root or before.repo_root,
        before_status=before_status,
        after_status=after_status_lines,
        changed_tracked_files=changed_tracked,
        untracked_files=untracked,
        commit_message_draft=commit_message,
        note="git_repo_detected",
    )


def _parse_git_status(status_lines: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    changed: list[str] = []
    untracked: list[str] = []
    for line in status_lines:
        if len(line) < 4:
            continue
        status = line[:2]
        path_text = line[3:].strip()
        if "->" in path_text:
            path_text = path_text.split("->", 1)[1].strip()
        normalized = path_text.replace("\\", "/")
        if status == "??":
            untracked.append(normalized)
        else:
            changed.append(normalized)
    return _dedupe(tuple(changed)), _dedupe(tuple(untracked))


def _normalize_task_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for item in paths:
        candidate = item.replace("\\", "/").strip()
        if candidate:
            normalized.append(candidate)
    return _dedupe(tuple(normalized))


def _filter_git_paths(
    git_paths: tuple[str, ...],
    task_paths: tuple[str, ...],
) -> tuple[str, ...]:
    if not task_paths:
        return ()
    matches: list[str] = []
    for git_path in git_paths:
        normalized_git = git_path.replace("\\", "/")
        if any(
            normalized_git == task_path
            or normalized_git.endswith(f"/{task_path}")
            or task_path.endswith(f"/{normalized_git}")
            for task_path in task_paths
        ):
            matches.append(normalized_git)
    return _dedupe(tuple(matches))


def _draft_commit_message(
    created_files: tuple[str, ...],
    modified_files: tuple[str, ...],
    changed_tracked: tuple[str, ...],
    untracked_files: tuple[str, ...],
) -> str:
    touched = (*modified_files, *created_files)
    lowered = " ".join(path.lower() for path in touched)
    if "readme" in lowered and ("handoff" in lowered or "next_steps" in lowered):
        return "docs: refresh handoff docs and workspace notes"
    if any(path.lower().endswith(".py") for path in touched):
        return "chore: apply workspace tidy-up updates"
    if untracked_files and not changed_tracked:
        return "docs: add requested workspace deliverables"
    if changed_tracked:
        return "chore: update requested workspace files"
    return ""


def _synthetic_git_status_lines(
    modified_files: tuple[str, ...],
    created_files: tuple[str, ...],
) -> tuple[str, ...]:
    normalized_modified = tuple(path.replace("\\", "/") for path in modified_files)
    normalized_created = tuple(path.replace("\\", "/") for path in created_files)
    lines = [f" M {path}" for path in normalized_modified]
    lines.extend(f"?? {path}" for path in normalized_created)
    return tuple(lines)


def _build_display_answer(
    plan: WorkspaceEditPlan,
    created_files: tuple[str, ...],
    modified_files: tuple[str, ...],
    skipped_files: list[str],
) -> str:
    details: list[str] = []
    if modified_files:
        details.append(f"updated {', '.join(modified_files)}")
    if created_files:
        details.append(f"created {', '.join(created_files)}")
    if skipped_files:
        details.append(f"left {', '.join(skipped_files)} untouched")
    if not details:
        return "No workspace files were changed."
    prefix = "Applied the planned workspace update" if plan.edit_intent == "multi_file_edit" else "Applied the requested workspace change"
    return f"{prefix}: {'; '.join(details)}."


def _operation_summary(
    operation: WorkspaceEditOperation,
    final_path: Path,
    *,
    created: bool,
) -> str:
    display_name = final_path.name if operation.display_target == "." else operation.display_target
    if operation.strategy == "readme_handoff_refresh":
        return f"{display_name}: refreshed the managed RA handoff snapshot."
    if operation.strategy == "pyproject_comment_refresh":
        return f"{display_name}: refreshed the lightweight handoff comment block."
    if operation.strategy == "handoff_markdown":
        return f"{display_name}: created a focused handoff note."
    if operation.strategy == "python_module_docstring":
        return f"{display_name}: added a short module docstring."
    if created:
        return f"{display_name}: created the requested deliverable file."
    return f"{display_name}: refreshed the requested file."


def _resolve_collision_path(path: Path) -> tuple[Path, int]:
    suffix = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate, suffix
        suffix += 1


def _managed_block(marker: str, body: str) -> str:
    return "\n".join(
        (
            f"<!-- {marker}-START -->",
            body.strip(),
            f"<!-- {marker}-END -->",
        )
    ).strip()


def _replace_or_append_managed_block(existing_text: str, marker: str, block: str) -> str:
    pattern = re.compile(
        rf"<!-- {re.escape(marker)}-START -->.*?<!-- {re.escape(marker)}-END -->",
        flags=re.DOTALL,
    )
    if pattern.search(existing_text):
        updated = pattern.sub(lambda _match: block, existing_text)
    else:
        spacer = "\n\n" if existing_text.strip() else ""
        updated = f"{existing_text.rstrip()}{spacer}{block}"
    return _ensure_trailing_newline(updated.strip())


def _compact_answer(answer: str) -> str:
    cleaned = _strip_workspace_edit_scaffolding(answer)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    compact = " ".join(lines)
    compact = re.sub(r"^[A-Za-z0-9_]+:\s*", "", compact)
    if len(compact) <= 600:
        return compact
    return compact[:597].rstrip() + "..."


def _strip_workspace_edit_scaffolding(answer: str) -> str:
    cleaned = re.sub(
        r"^(?:===|#{1,6})\s*FILE:.*?\n```[^\n]*\n.*?\n```",
        "",
        answer,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    cleaned = re.sub(
        r"```json\s*\{.*?\"content\"\s*:\s*`.*?`.*?```",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"```json\s*\{.*?\"content\"\s*:\s*\"(?:\\.|[^\"\\])*\".*?```",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    summary_match = re.search(
        r"\bSUMMARY\b\s*:?\s*(?P<body>.+)$",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if summary_match:
        cleaned = summary_match.group("body")
    return cleaned


def _single_line(text: str, *, limit: int) -> str:
    compact = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _ensure_trailing_newline(text: str) -> str:
    return text.rstrip() + "\n"


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
    title = stem[:1].upper() + stem[1:]
    docstring = f'"""{title} module."""\n\n'
    return "".join((*lines[:index], docstring, *lines[index:]))


def _dedupe(items: tuple[str, ...]) -> tuple[str, ...]:
    ordered: dict[str, None] = {}
    for item in items:
        normalized = item.strip()
        if normalized:
            ordered.setdefault(normalized, None)
    return tuple(ordered.keys())


def _looks_like_absolute_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or Path(value).is_absolute()


def _find_git_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        git_dir = candidate / ".git"
        if git_dir.is_dir() or git_dir.is_file():
            return candidate
    return None

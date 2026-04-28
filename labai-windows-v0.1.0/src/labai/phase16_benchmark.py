from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REQUIRED_PRIMARY_FAMILIES: tuple[str, ...] = (
    "single_file_bugfix",
    "multi_file_bugfix",
    "requirement_implementation",
    "data_processing_logic_repair",
    "config_path_env_repair",
    "entrypoint_runnable_surface_repair",
    "refactor_light",
    "docs_code_sync",
    "deliverable_creation",
    "retry_loop",
    "final_failure_reporting",
    "git_aware_summary",
)

PASS_STATUSES = {"pass"}
VALID_STATUSES = {"pass", "fail", "partial", "blocked", "skipped"}
VALID_SOURCE_TYPES = {"local_user_copy", "generated_fixture", "public_github_copy"}
VALID_PATHS = {"workflow", "ask"}
MIN_TOTAL_PASSING_TASKS = 60
MIN_PASSING_TASKS_PER_FAMILY = 5
MIN_PUBLIC_REPOS = 20
MIN_PUBLIC_EXECUTIONS = 20


@dataclass(frozen=True)
class Phase16BenchmarkSummary:
    total_passing_tasks: int
    family_counts: dict[str, int]
    public_repo_count: int
    public_execution_count: int
    workflow_count: int
    ask_count: int


def load_task_bank_jsonl(path: str | Path) -> list[dict[str, Any]]:
    bank_path = Path(path)
    tasks: list[dict[str, Any]] = []
    for line_no, line in enumerate(bank_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{bank_path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{bank_path}:{line_no}: expected JSON object")
        tasks.append(item)
    return tasks


def verify_phase16_task_bank(
    tasks: list[dict[str, Any]],
    *,
    min_total: int = MIN_TOTAL_PASSING_TASKS,
    min_per_family: int = MIN_PASSING_TASKS_PER_FAMILY,
    min_public_repos: int = MIN_PUBLIC_REPOS,
    min_public_tasks: int = MIN_PUBLIC_EXECUTIONS,
) -> tuple[Phase16BenchmarkSummary, list[str]]:
    errors: list[str] = []
    seen_task_ids: dict[str, str] = {}
    passing_tasks: list[dict[str, Any]] = []

    for index, task in enumerate(tasks, start=1):
        task_id = _string(task.get("task_id"))
        task_errors = _validate_task_record(task, index=index)
        if task_id:
            prior_family = seen_task_ids.get(task_id)
            current_family = _string(task.get("primary_family"))
            if prior_family is None:
                seen_task_ids[task_id] = current_family
            else:
                task_errors.append(
                    f"task `{task_id}` is duplicated; one task cannot count toward multiple primary families"
                )
        errors.extend(task_errors)
        if not task_errors and task.get("status") == "pass":
            passing_tasks.append(task)

    family_counts = Counter(_string(task.get("primary_family")) for task in passing_tasks)
    public_passing = [task for task in passing_tasks if task.get("source_type") == "public_github_copy"]
    public_repo_count = len({_string(task.get("repo_name")) for task in public_passing})
    workflow_count = sum(1 for task in passing_tasks if task.get("workflow_or_ask") == "workflow")
    ask_count = sum(1 for task in passing_tasks if task.get("workflow_or_ask") == "ask")

    summary = Phase16BenchmarkSummary(
        total_passing_tasks=len(passing_tasks),
        family_counts={family: family_counts.get(family, 0) for family in REQUIRED_PRIMARY_FAMILIES},
        public_repo_count=public_repo_count,
        public_execution_count=len(public_passing),
        workflow_count=workflow_count,
        ask_count=ask_count,
    )

    if summary.total_passing_tasks < min_total:
        errors.append(f"need at least {min_total} passing tasks; found {summary.total_passing_tasks}")
    for family in REQUIRED_PRIMARY_FAMILIES:
        count = summary.family_counts.get(family, 0)
        if count < min_per_family:
            errors.append(
                f"primary family `{family}` needs at least {min_per_family} passing tasks; found {count}"
            )
    if summary.public_repo_count < min_public_repos:
        errors.append(
            "need at least "
            f"{min_public_repos} distinct public GitHub repos with passing edit-task executions; "
            f"found {summary.public_repo_count}"
        )
    if summary.public_execution_count < min_public_tasks:
        errors.append(
            f"need at least {min_public_tasks} passing public GitHub edit-task executions; "
            f"found {summary.public_execution_count}"
        )
    if workflow_count == 0:
        errors.append("task bank must include at least one passing workflow edit-task case")
    if ask_count == 0:
        errors.append("task bank must include at least one passing ask edit-task case")

    return summary, errors


def render_phase16_benchmark_report(
    summary: Phase16BenchmarkSummary,
    *,
    verifier_command: str,
    verifier_output: str,
    errors: list[str],
) -> str:
    status = "PASS" if not errors else "FAIL"
    family_lines = "\n".join(
        f"- `{family}`: `{summary.family_counts.get(family, 0)}`"
        for family in REQUIRED_PRIMARY_FAMILIES
    )
    output_block = verifier_output.rstrip() or "(no output)"
    failure_block = "\n".join(f"- {item}" for item in errors) if errors else "- none"
    return (
        "# Phase 16 Benchmark Gate\n\n"
        f"Result: **{status}**\n\n"
        "## Exact Counts\n\n"
        f"- total passing tasks: `{summary.total_passing_tasks}`\n"
        f"- public GitHub repo count: `{summary.public_repo_count}`\n"
        f"- public GitHub passing edit-task executions: `{summary.public_execution_count}`\n"
        f"- workflow count: `{summary.workflow_count}`\n"
        f"- ask count: `{summary.ask_count}`\n\n"
        "## Primary Family Counts\n\n"
        f"{family_lines}\n\n"
        "## Verifier Command\n\n"
        f"```powershell\n{verifier_command}\n```\n\n"
        "## Verifier Output\n\n"
        f"```text\n{output_block}\n```\n\n"
        "## Failure Reasons\n\n"
        f"{failure_block}\n"
    )


def _validate_task_record(task: dict[str, Any], *, index: int) -> list[str]:
    errors: list[str] = []
    label = f"task[{index}]"
    task_id = _string(task.get("task_id"))
    if not task_id:
        errors.append(f"{label} missing `task_id`")
    primary_family = _string(task.get("primary_family"))
    if primary_family not in REQUIRED_PRIMARY_FAMILIES:
        errors.append(f"{label} has invalid `primary_family`: {primary_family or '<missing>'}")
    status = _string(task.get("status"))
    if status not in VALID_STATUSES:
        errors.append(f"{label} has invalid `status`: {status or '<missing>'}")
    if status not in PASS_STATUSES:
        errors.append(f"{label} is not countable because status is `{status or '<missing>'}`")
    source_type = _string(task.get("source_type"))
    if source_type not in VALID_SOURCE_TYPES:
        errors.append(f"{label} has invalid `source_type`: {source_type or '<missing>'}")
    workflow_or_ask = _string(task.get("workflow_or_ask"))
    if workflow_or_ask not in VALID_PATHS:
        errors.append(f"{label} has invalid `workflow_or_ask`: {workflow_or_ask or '<missing>'}")
    if _string(task.get("runtime_used")) != "claw":
        errors.append(f"{label} must use `runtime_used=claw`")
    if _string(task.get("runtime_fallback")) != "none":
        errors.append(f"{label} must use `runtime_fallback=none`")
    if not _string(task.get("session_id")):
        errors.append(f"{label} missing `session_id`")
    if not _string(task.get("evaluation_path")):
        errors.append(f"{label} missing `evaluation_path`")

    checks_run = _string_list(task.get("checks_run"))
    check_results = _sequence(task.get("check_results"))
    if not checks_run:
        errors.append(f"{label} missing `checks_run` evidence")
    if not check_results:
        errors.append(f"{label} missing `check_results` evidence")

    acceptance_criteria = _sequence(task.get("acceptance_criteria"))
    acceptance_results = _sequence(task.get("acceptance_criteria_results"))
    if not acceptance_criteria:
        errors.append(f"{label} missing `acceptance_criteria`")
    if not acceptance_results:
        errors.append(f"{label} missing `acceptance_criteria_results`")

    expected_change_targets = _string_list(task.get("expected_files_to_change"))
    actual_changed = _string_list(task.get("actual_files_changed"))
    actual_created = _string_list(task.get("actual_files_created"))
    if expected_change_targets and not (actual_changed or actual_created):
        errors.append(
            f"{label} is missing `actual_files_changed` / `actual_files_created` evidence for an edit task"
        )

    if source_type == "public_github_copy":
        if not _string(task.get("repo_name")):
            errors.append(f"{label} missing `repo_name` for public GitHub task")
        if not _string(task.get("repo_url")):
            errors.append(f"{label} missing `repo_url` for public GitHub task")
        if not _string(task.get("commit_hash")):
            errors.append(f"{label} missing `commit_hash` for public GitHub task")
        if not _string(task.get("clean_clone_path")):
            errors.append(f"{label} missing `clean_clone_path` for public GitHub task")

    return errors


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _string_list(value: Any) -> list[str]:
    result: list[str] = []
    for item in _sequence(value):
        text = _string(item)
        if text:
            result.append(text)
    return result


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()

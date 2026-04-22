from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


VALID_CASE_STATUSES = {"pass", "fail", "partial", "blocked", "skipped"}
VALID_SURFACES = {"workflow", "ask"}
VALID_SEQUENCE_FAMILIES = {"edit_to_edit", "cross_workflow", "validator_preflight"}
SYNTAX_ONLY_CHECKS = {"py_compile", "json_validate", "toml_validate"}

MIN_ISOLATION_CASES = 15
MIN_EDIT_TO_EDIT_CASES = 5
MIN_CROSS_WORKFLOW_CASES = 6
MIN_ASK_CASES = 2
MIN_WORKFLOW_EDIT_TASK_CASES = 2
MIN_VALIDATOR_PREFLIGHT_CASES = 2


@dataclass(frozen=True)
class Phase16IsolationSummary:
    total_cases: int
    passing_cases: int
    edit_to_edit_cases: int
    cross_workflow_cases: int
    ask_cases: int
    workflow_edit_task_cases: int
    validator_preflight_cases: int


def load_isolation_eval_jsonl(path: str | Path) -> list[dict[str, Any]]:
    eval_path = Path(path)
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{eval_path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{eval_path}:{line_no}: expected JSON object")
        cases.append(item)
    return cases


def verify_phase16_isolation_cases(
    cases: list[dict[str, Any]],
    *,
    task_isolation_doc: str | Path | None = None,
    contamination_matrix_doc: str | Path | None = None,
    check_scheduler_doc: str | Path | None = None,
) -> tuple[Phase16IsolationSummary, list[str]]:
    errors: list[str] = []
    seen_case_ids: set[str] = set()
    passing_cases: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        case_errors = _validate_isolation_case(case, index=index)
        case_id = _string(case.get("case_id"))
        if case_id:
            if case_id in seen_case_ids:
                case_errors.append(f"case `{case_id}` is duplicated")
            seen_case_ids.add(case_id)
        errors.extend(case_errors)
        if not case_errors and case.get("status") == "pass":
            passing_cases.append(case)

    summary = Phase16IsolationSummary(
        total_cases=len(cases),
        passing_cases=len(passing_cases),
        edit_to_edit_cases=sum(
            1 for case in passing_cases if _string(case.get("sequence_family")) == "edit_to_edit"
        ),
        cross_workflow_cases=sum(
            1 for case in passing_cases if _string(case.get("sequence_family")) == "cross_workflow"
        ),
        ask_cases=sum(1 for case in passing_cases if _string(case.get("workflow_or_ask")) == "ask"),
        workflow_edit_task_cases=sum(
            1 for case in passing_cases if _string(case.get("current_workflow")) == "workflow.edit-task"
        ),
        validator_preflight_cases=sum(
            1
            for case in passing_cases
            if _string(case.get("sequence_family")) == "validator_preflight"
            or _truthy(case.get("validator_preflight_case"))
        ),
    )

    for label, path in (
        ("task-isolation artifact", task_isolation_doc),
        ("contamination matrix artifact", contamination_matrix_doc),
        ("check-scheduler artifact", check_scheduler_doc),
    ):
        if path is None:
            continue
        if not Path(path).is_file():
            errors.append(f"missing required {label}: {path}")

    if summary.total_cases < MIN_ISOLATION_CASES:
        errors.append(
            f"need at least {MIN_ISOLATION_CASES} isolation cases; found {summary.total_cases}"
        )
    if summary.edit_to_edit_cases < MIN_EDIT_TO_EDIT_CASES:
        errors.append(
            f"need at least {MIN_EDIT_TO_EDIT_CASES} passing edit-to-edit isolation cases; "
            f"found {summary.edit_to_edit_cases}"
        )
    if summary.cross_workflow_cases < MIN_CROSS_WORKFLOW_CASES:
        errors.append(
            f"need at least {MIN_CROSS_WORKFLOW_CASES} passing cross-workflow isolation cases; "
            f"found {summary.cross_workflow_cases}"
        )
    if summary.ask_cases < MIN_ASK_CASES:
        errors.append(
            f"need at least {MIN_ASK_CASES} passing direct ask isolation cases; found {summary.ask_cases}"
        )
    if summary.workflow_edit_task_cases < MIN_WORKFLOW_EDIT_TASK_CASES:
        errors.append(
            f"need at least {MIN_WORKFLOW_EDIT_TASK_CASES} passing workflow edit-task isolation cases; "
            f"found {summary.workflow_edit_task_cases}"
        )
    if summary.validator_preflight_cases < MIN_VALIDATOR_PREFLIGHT_CASES:
        errors.append(
            f"need at least {MIN_VALIDATOR_PREFLIGHT_CASES} passing validator-preflight isolation cases; "
            f"found {summary.validator_preflight_cases}"
        )

    return summary, errors


def render_phase16_isolation_report(
    summary: Phase16IsolationSummary,
    *,
    verifier_command: str,
    verifier_output: str,
    errors: list[str],
) -> str:
    status = "PASS" if not errors else "FAIL"
    failure_block = "\n".join(f"- {item}" for item in errors) if errors else "- none"
    output_block = verifier_output.rstrip() or "(no output)"
    return (
        "# Phase 16 Isolation Gate\n\n"
        f"Result: **{status}**\n\n"
        "## Exact Counts\n\n"
        f"- total cases: `{summary.total_cases}`\n"
        f"- passing cases: `{summary.passing_cases}`\n"
        f"- edit-to-edit cases: `{summary.edit_to_edit_cases}`\n"
        f"- cross-workflow cases: `{summary.cross_workflow_cases}`\n"
        f"- ask cases: `{summary.ask_cases}`\n\n"
        f"- workflow edit-task cases: `{summary.workflow_edit_task_cases}`\n"
        f"- validator-preflight cases: `{summary.validator_preflight_cases}`\n\n"
        "## Verifier Command\n\n"
        f"```powershell\n{verifier_command}\n```\n\n"
        "## Verifier Output\n\n"
        f"```text\n{output_block}\n```\n\n"
        "## Failure Reasons\n\n"
        f"{failure_block}\n"
    )


def _validate_isolation_case(case: dict[str, Any], *, index: int) -> list[str]:
    errors: list[str] = []
    label = f"case[{index}]"

    case_id = _string(case.get("case_id"))
    if not case_id:
        errors.append(f"{label} missing `case_id`")
    sequence_family = _string(case.get("sequence_family"))
    if sequence_family not in VALID_SEQUENCE_FAMILIES:
        errors.append(f"{label} has invalid `sequence_family`: {sequence_family or '<missing>'}")
    workflow_or_ask = _string(case.get("workflow_or_ask"))
    if workflow_or_ask not in VALID_SURFACES:
        errors.append(f"{label} has invalid `workflow_or_ask`: {workflow_or_ask or '<missing>'}")
    status = _string(case.get("status"))
    if status not in VALID_CASE_STATUSES:
        errors.append(f"{label} has invalid `status`: {status or '<missing>'}")
    elif status != "pass":
        errors.append(f"{label} is not countable because status is `{status}`")

    if _string(case.get("runtime_used")) != "claw":
        errors.append(f"{label} must use `runtime_used=claw`")
    if _string(case.get("runtime_fallback")) != "none":
        errors.append(f"{label} must use `runtime_fallback=none`")
    if not _string(case.get("session_id")):
        errors.append(f"{label} missing `session_id`")
    if not _string(case.get("prior_workflow")):
        errors.append(f"{label} missing `prior_workflow`")
    if not _string(case.get("current_workflow")):
        errors.append(f"{label} missing `current_workflow`")
    if not _string(case.get("prior_task_workspace")):
        errors.append(f"{label} missing `prior_task_workspace`")
    if not _string(case.get("current_task_workspace")):
        errors.append(f"{label} missing `current_task_workspace`")
    if not _string(case.get("current_prompt")):
        errors.append(f"{label} missing `current_prompt`")
    if not _string(case.get("expected_non_leakage_rule")):
        errors.append(f"{label} missing `expected_non_leakage_rule`")
    if not _string(case.get("actual_plan")):
        errors.append(f"{label} missing `actual_plan`")
    if not _sequence(case.get("actual_planned_checks")):
        errors.append(f"{label} missing `actual_planned_checks`")
    if not _sequence(case.get("checks_run")):
        errors.append(f"{label} missing `checks_run`")
    if not _sequence(case.get("acceptance_criteria_results")):
        errors.append(f"{label} missing `acceptance_criteria_results`")
    if _truthy(case.get("validator_preflight_case")):
        preflight_rejections = _string_list(case.get("preflight_rejections"))
        if not preflight_rejections:
            errors.append(f"{label} is marked as a validator-preflight case but has no `preflight_rejections` evidence")

    leaked_artifacts = _string_list(case.get("prior_task_artifacts_appeared")) + _string_list(
        case.get("leaked_artifacts")
    )
    if leaked_artifacts:
        errors.append(f"{label} leaked prior-task artifacts: {', '.join(leaked_artifacts)}")
    if _truthy(case.get("prior_task_artifacts_detected")) and not leaked_artifacts:
        errors.append(f"{label} flags prior-task artifacts without listing them")

    missing_current_task_files = _string_list(case.get("missing_current_task_files"))
    if missing_current_task_files:
        errors.append(
            f"{label} planned checks reference missing current-task files: "
            + ", ".join(missing_current_task_files)
        )
    prior_task_validators = _string_list(case.get("prior_task_validators_in_checks"))
    if prior_task_validators:
        errors.append(
            f"{label} planned checks reference prior-task validators: "
            + ", ".join(prior_task_validators)
        )

    unchecked_acceptance = _string_list(case.get("unchecked_acceptance_criteria"))
    if unchecked_acceptance:
        errors.append(
            f"{label} has unchecked acceptance criteria: " + ", ".join(unchecked_acceptance)
        )

    if _truthy(case.get("behavioral_task")):
        checks_run = {item.lower() for item in _string_list(case.get("checks_run"))}
        syntax_only_flag = _truthy(case.get("syntax_only_validation"))
        if syntax_only_flag or (checks_run and checks_run.issubset(SYNTAX_ONLY_CHECKS)):
            errors.append(
                f"{label} is a behavioral task but relies only on syntax checks"
            )

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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)

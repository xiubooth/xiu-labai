from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


VALID_STATUSES = {"pass", "fail", "partial", "blocked", "skipped"}
VALID_SURFACES = {"workflow", "ask"}
VALID_SOURCE_TYPES = {"local_user_copy", "generated_fixture", "public_github_copy"}

MIN_DEPENDENCY_FALLBACK_CASES = 2
MIN_DEPENDENCY_FALLBACK_WORKFLOW_CASES = 1
MIN_DEPENDENCY_FALLBACK_ASK_CASES = 1


@dataclass(frozen=True)
class Phase16DependencyFallbackSummary:
    total_cases: int
    passing_cases: int
    workflow_cases: int
    ask_cases: int
    generated_validator_cases: int
    data_processing_cases: int
    task_bank_dependency_cases: int


def load_phase16_dependency_fallback_jsonl(path: str | Path) -> list[dict[str, Any]]:
    eval_path = Path(path)
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(eval_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{eval_path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{eval_path}:{line_no}: expected JSON object")
        records.append(payload)
    return records


def verify_phase16_dependency_fallback(
    records: list[dict[str, Any]],
    *,
    task_bank_records: list[dict[str, Any]] | None = None,
    report_doc: str | Path | None = None,
    data_processing_doc: str | Path | None = None,
    validator_doc: str | Path | None = None,
) -> tuple[Phase16DependencyFallbackSummary, list[str]]:
    errors: list[str] = []
    seen_case_ids: set[str] = set()
    passing_records: list[dict[str, Any]] = []
    task_bank_by_id = {
        _string(record.get("task_id")): record
        for record in (task_bank_records or [])
        if _string(record.get("task_id"))
    }

    for index, record in enumerate(records, start=1):
        case_errors = _validate_dependency_fallback_record(record, index=index)
        case_id = _string(record.get("case_id"))
        if case_id:
            if case_id in seen_case_ids:
                case_errors.append(f"dependency-fallback case `{case_id}` is duplicated")
            seen_case_ids.add(case_id)
        task_bank_task_id = _string(record.get("task_bank_task_id"))
        if task_bank_records is not None:
            if not task_bank_task_id:
                case_errors.append(f"dependency-fallback case `{case_id or index}` missing `task_bank_task_id`")
            else:
                task_bank_record = task_bank_by_id.get(task_bank_task_id)
                if task_bank_record is None:
                    case_errors.append(
                        f"dependency-fallback case `{case_id or index}` references missing task-bank row `{task_bank_task_id}`"
                    )
                else:
                    case_errors.extend(
                        _validate_task_bank_dependency_alignment(
                            record,
                            task_bank_record,
                            label=f"dependency-fallback case `{case_id or index}`",
                        )
                    )
        errors.extend(case_errors)
        if not case_errors and _string(record.get("status")) == "pass":
            passing_records.append(record)

    summary = Phase16DependencyFallbackSummary(
        total_cases=len(records),
        passing_cases=len(passing_records),
        workflow_cases=sum(1 for record in passing_records if _string(record.get("workflow_or_ask")) == "workflow"),
        ask_cases=sum(1 for record in passing_records if _string(record.get("workflow_or_ask")) == "ask"),
        generated_validator_cases=sum(1 for record in passing_records if _truthy(record.get("validator_generated"))),
        data_processing_cases=sum(
            1
            for record in passing_records
            if _string(record.get("scenario_family")) == "complex_data_processing_requirement"
        ),
        task_bank_dependency_cases=sum(
            1
            for record in (task_bank_records or [])
            if _truthy(record.get("dependency_fallback_used"))
            and _string(record.get("status")) == "pass"
        ),
    )

    for label, path in (
        ("dependency-fallback artifact", report_doc),
        ("data-processing regression artifact", data_processing_doc),
        ("validator-quality artifact", validator_doc),
    ):
        if path is None:
            continue
        if not Path(path).is_file():
            errors.append(f"missing required {label}: {path}")

    if summary.total_cases < MIN_DEPENDENCY_FALLBACK_CASES:
        errors.append(
            f"need at least {MIN_DEPENDENCY_FALLBACK_CASES} dependency-fallback cases; found {summary.total_cases}"
        )
    if summary.passing_cases != summary.total_cases:
        errors.append(
            "all dependency-fallback cases must pass; "
            f"found {summary.passing_cases}/{summary.total_cases} passing"
        )
    if summary.workflow_cases < MIN_DEPENDENCY_FALLBACK_WORKFLOW_CASES:
        errors.append("need at least one passing workflow dependency-fallback case")
    if summary.ask_cases < MIN_DEPENDENCY_FALLBACK_ASK_CASES:
        errors.append("need at least one passing ask dependency-fallback case")
    if summary.generated_validator_cases < MIN_DEPENDENCY_FALLBACK_CASES:
        errors.append("dependency-fallback cases must use generated current-run validators")
    if summary.data_processing_cases < MIN_DEPENDENCY_FALLBACK_CASES:
        errors.append("dependency-fallback evidence must cover complex data-processing tasks")
    if summary.task_bank_dependency_cases < MIN_DEPENDENCY_FALLBACK_CASES:
        errors.append(
            "task bank must include at least two passing dependency-fallback rows with auditable evidence"
        )
    return summary, errors


def render_phase16_dependency_fallback_report(
    summary: Phase16DependencyFallbackSummary,
    *,
    verifier_command: str,
    verifier_output: str,
    errors: list[str],
) -> str:
    status = "PASS" if not errors else "FAIL"
    failure_block = "\n".join(f"- {item}" for item in errors) if errors else "- none"
    output_block = verifier_output.rstrip() or "(no output)"
    return (
        "# Phase 16 Dependency Fallback Gate\n\n"
        f"Result: **{status}**\n\n"
        "## Exact Counts\n\n"
        f"- total cases: `{summary.total_cases}`\n"
        f"- passing cases: `{summary.passing_cases}`\n"
        f"- workflow cases: `{summary.workflow_cases}`\n"
        f"- ask cases: `{summary.ask_cases}`\n"
        f"- generated validator cases: `{summary.generated_validator_cases}`\n"
        f"- complex data-processing cases: `{summary.data_processing_cases}`\n"
        f"- dependency-fallback task-bank rows: `{summary.task_bank_dependency_cases}`\n\n"
        "## Verifier Command\n\n"
        f"```powershell\n{verifier_command}\n```\n\n"
        "## Verifier Output\n\n"
        f"```text\n{output_block}\n```\n\n"
        "## Failure Reasons\n\n"
        f"{failure_block}\n"
    )


def _validate_dependency_fallback_record(record: dict[str, Any], *, index: int) -> list[str]:
    errors: list[str] = []
    label = f"dependency_fallback_case[{index}]"
    case_id = _string(record.get("case_id"))
    if not case_id:
        errors.append(f"{label} missing `case_id`")
    status = _string(record.get("status"))
    if status not in VALID_STATUSES:
        errors.append(f"{label} has invalid `status`: {status or '<missing>'}")
    elif status != "pass":
        errors.append(f"{label} is not countable because status is `{status}`")
    workflow_or_ask = _string(record.get("workflow_or_ask"))
    if workflow_or_ask not in VALID_SURFACES:
        errors.append(f"{label} has invalid `workflow_or_ask`: {workflow_or_ask or '<missing>'}")
    source_type = _string(record.get("source_type"))
    if source_type not in VALID_SOURCE_TYPES:
        errors.append(f"{label} has invalid `source_type`: {source_type or '<missing>'}")
    if _string(record.get("runtime_used")) != "claw":
        errors.append(f"{label} must use `runtime_used=claw`")
    if _string(record.get("runtime_fallback")) != "none":
        errors.append(f"{label} must use `runtime_fallback=none`")
    if not _string(record.get("session_id")):
        errors.append(f"{label} missing `session_id`")
    if not _truthy(record.get("behavioral_task")):
        errors.append(f"{label} must record `behavioral_task=true`")
    if _truthy(record.get("syntax_only_validation")):
        errors.append(f"{label} cannot rely on syntax-only validation")
    if not _truthy(record.get("dependency_fallback_used")):
        errors.append(f"{label} missing `dependency_fallback_used=true`")
    if not _string_list(record.get("unavailable_dependencies")):
        errors.append(f"{label} missing `unavailable_dependencies`")
    if not _string(record.get("fallback_validation_mode")):
        errors.append(f"{label} missing `fallback_validation_mode`")
    if not _string(record.get("fallback_validation_tested")):
        errors.append(f"{label} missing `fallback_validation_tested`")
    if not _string(record.get("fallback_validation_untested")):
        errors.append(f"{label} missing `fallback_validation_untested`")
    if _truthy(record.get("live_integration_tested")):
        errors.append(f"{label} must not claim live integration was tested")
    if not _truthy(record.get("local_logic_validation_complete")):
        errors.append(f"{label} must record `local_logic_validation_complete=true`")
    checks_run = _string_list(record.get("checks_run"))
    if not checks_run:
        errors.append(f"{label} missing `checks_run`")
    elif all(item == "py_compile" for item in checks_run):
        errors.append(f"{label} cannot pass on py_compile only")
    acceptance_criteria = _sequence(record.get("acceptance_criteria"))
    acceptance_results = _sequence(record.get("acceptance_criteria_results"))
    unchecked = _string_list(record.get("unchecked_acceptance_criteria"))
    if not acceptance_criteria:
        errors.append(f"{label} missing `acceptance_criteria`")
    if not acceptance_results:
        errors.append(f"{label} missing `acceptance_criteria_results`")
    if unchecked:
        errors.append(f"{label} has unchecked acceptance criteria: {', '.join(unchecked)}")
    for result in acceptance_results:
        if not isinstance(result, dict):
            errors.append(f"{label} has non-object acceptance result entries")
            break
        if _string(result.get("status")) != "pass":
            errors.append(
                f"{label} has non-passing acceptance result for `{_string(result.get('criterion')) or '<unknown>'}`"
            )
    if not _truthy(record.get("validator_generated")):
        errors.append(f"{label} must use a generated validator")
    if not _string(record.get("validator_path")):
        errors.append(f"{label} missing `validator_path`")
    if not _truthy(record.get("validator_task_specific_name")):
        errors.append(f"{label} missing task-specific validator naming evidence")
    if not _truthy(record.get("validator_syntax_preflight_passed")):
        errors.append(f"{label} missing validator syntax preflight evidence")
    if not _string(record.get("validator_task_run_id")):
        errors.append(f"{label} missing `validator_task_run_id`")
    if not _truthy(record.get("validator_executed")):
        errors.append(f"{label} missing `validator_executed=true`")
    if _string(record.get("validator_execution_status")) != "pass":
        errors.append(f"{label} must record `validator_execution_status=pass`")
    if _truthy(record.get("validator_generation_failure")):
        errors.append(f"{label} cannot count a validator-generation failure as passing")
    return errors


def _validate_task_bank_dependency_alignment(
    record: dict[str, Any],
    task_bank_record: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if _string(task_bank_record.get("status")) != "pass":
        errors.append(f"{label} references a task-bank row that is not passing")
    if _string(task_bank_record.get("session_id")) != _string(record.get("session_id")):
        errors.append(f"{label} task-bank session id does not match dependency-fallback evidence")
    if _string(task_bank_record.get("workflow_or_ask")) != _string(record.get("workflow_or_ask")):
        errors.append(f"{label} task-bank workflow surface does not match dependency-fallback evidence")
    if not _truthy(task_bank_record.get("dependency_fallback_used")):
        errors.append(f"{label} task-bank row is missing `dependency_fallback_used=true`")
    if set(_string_list(task_bank_record.get("unavailable_dependencies"))) != set(
        _string_list(record.get("unavailable_dependencies"))
    ):
        errors.append(f"{label} task-bank unavailable dependencies do not match dependency-fallback evidence")
    if _string(task_bank_record.get("fallback_validation_mode")) != _string(record.get("fallback_validation_mode")):
        errors.append(f"{label} task-bank fallback mode does not match dependency-fallback evidence")
    if _truthy(task_bank_record.get("live_integration_tested")):
        errors.append(f"{label} task-bank row falsely claims live integration testing")
    if not _truthy(task_bank_record.get("local_logic_validation_complete")):
        errors.append(f"{label} task-bank row is missing `local_logic_validation_complete=true`")
    if _truthy(task_bank_record.get("syntax_only_validation")):
        errors.append(f"{label} task-bank row cannot rely on syntax-only validation")
    return errors


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    return [item for item in _sequence(value) if isinstance(item, str) and item]


def _truthy(value: Any) -> bool:
    return bool(value)

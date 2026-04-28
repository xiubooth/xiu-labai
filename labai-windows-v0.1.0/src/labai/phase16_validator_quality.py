from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


VALID_STATUSES = {"pass", "fail", "partial", "blocked", "skipped"}
VALID_SURFACES = {"workflow", "ask"}
VALID_SOURCE_TYPES = {"local_user_copy", "generated_fixture", "public_github_copy"}

REQUIRED_SCENARIO_FAMILIES = {
    "complex_data_processing_requirement",
    "multi_file_bugfix",
    "retry_loop",
    "public_github_coding_task",
}

MIN_QUALITY_CASES = 5
MIN_GENERATED_VALIDATOR_CASES = 2
MIN_VALIDATOR_PREFLIGHT_CASES = 2


@dataclass(frozen=True)
class Phase16ValidatorQualitySummary:
    total_cases: int
    passing_cases: int
    generated_validator_cases: int
    syntax_preflighted_cases: int
    ask_cases: int
    public_github_cases: int
    data_processing_cases: int
    data_processing_ask_cases: int
    retry_loop_cases: int
    multi_file_cases: int
    validator_preflight_cases: int


def load_phase16_quality_jsonl(path: str | Path) -> list[dict[str, Any]]:
    eval_path = Path(path)
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), start=1):
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


def verify_phase16_validator_quality(
    records: list[dict[str, Any]],
    *,
    isolation_cases: list[dict[str, Any]] | None = None,
    report_doc: str | Path | None = None,
    workspace_doc: str | Path | None = None,
    data_processing_doc: str | Path | None = None,
) -> tuple[Phase16ValidatorQualitySummary, list[str]]:
    errors: list[str] = []
    seen_case_ids: set[str] = set()
    passing_records: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        case_errors = _validate_quality_record(record, index=index)
        case_id = _string(record.get("case_id"))
        if case_id:
            if case_id in seen_case_ids:
                case_errors.append(f"quality case `{case_id}` is duplicated")
            seen_case_ids.add(case_id)
        errors.extend(case_errors)
        if not case_errors and _string(record.get("status")) == "pass":
            passing_records.append(record)

    summary = Phase16ValidatorQualitySummary(
        total_cases=len(records),
        passing_cases=len(passing_records),
        generated_validator_cases=sum(1 for record in passing_records if _truthy(record.get("validator_generated"))),
        syntax_preflighted_cases=sum(
            1
            for record in passing_records
            if _truthy(record.get("validator_generated"))
            and _truthy(record.get("validator_syntax_preflight_passed"))
        ),
        ask_cases=sum(1 for record in passing_records if _string(record.get("workflow_or_ask")) == "ask"),
        public_github_cases=sum(
            1 for record in passing_records if _string(record.get("source_type")) == "public_github_copy"
        ),
        data_processing_cases=sum(
            1
            for record in passing_records
            if _string(record.get("scenario_family")) == "complex_data_processing_requirement"
        ),
        data_processing_ask_cases=sum(
            1
            for record in passing_records
            if _string(record.get("scenario_family")) == "complex_data_processing_requirement"
            and _string(record.get("workflow_or_ask")) == "ask"
        ),
        retry_loop_cases=sum(
            1 for record in passing_records if _string(record.get("scenario_family")) == "retry_loop"
        ),
        multi_file_cases=sum(
            1 for record in passing_records if _string(record.get("scenario_family")) == "multi_file_bugfix"
        ),
        validator_preflight_cases=sum(
            1
            for case in (isolation_cases or [])
            if _truthy(case.get("validator_preflight_case")) and _string(case.get("status")) == "pass"
        ),
    )

    for label, path in (
        ("validator-quality artifact", report_doc),
        ("workspace-comprehension artifact", workspace_doc),
        ("data-processing regression artifact", data_processing_doc),
    ):
        if path is None:
            continue
        if not Path(path).is_file():
            errors.append(f"missing required {label}: {path}")

    if summary.total_cases < MIN_QUALITY_CASES:
        errors.append(f"need at least {MIN_QUALITY_CASES} validator-quality cases; found {summary.total_cases}")
    if summary.passing_cases != summary.total_cases:
        errors.append(
            f"all validator-quality cases must pass; found {summary.passing_cases}/{summary.total_cases} passing"
        )
    if summary.generated_validator_cases < MIN_GENERATED_VALIDATOR_CASES:
        errors.append(
            f"need at least {MIN_GENERATED_VALIDATOR_CASES} passing cases with generated validators; "
            f"found {summary.generated_validator_cases}"
        )
    if summary.syntax_preflighted_cases < summary.generated_validator_cases:
        errors.append(
            "every passing generated-validator case must record syntax preflight before execution"
        )
    if summary.ask_cases < 1:
        errors.append("need at least one passing ask validator-quality case")
    if summary.public_github_cases < 1:
        errors.append("need at least one passing public GitHub validator-quality case")
    if summary.data_processing_cases < 2:
        errors.append(
            "need both workflow and ask coverage for the complex data-processing regression family"
        )
    if summary.data_processing_ask_cases < 1:
        errors.append("need at least one passing ask case for complex data-processing requirement work")
    if summary.retry_loop_cases < 1:
        errors.append("need at least one passing retry-loop validator-quality case")
    if summary.multi_file_cases < 1:
        errors.append("need at least one passing multi-file bugfix validator-quality case")
    if summary.validator_preflight_cases < MIN_VALIDATOR_PREFLIGHT_CASES:
        errors.append(
            f"need at least {MIN_VALIDATOR_PREFLIGHT_CASES} passing validator-preflight isolation cases; "
            f"found {summary.validator_preflight_cases}"
        )

    scenario_families = {_string(record.get("scenario_family")) for record in passing_records}
    for family in REQUIRED_SCENARIO_FAMILIES:
        if family not in scenario_families:
            errors.append(f"missing required validator-quality scenario family `{family}`")

    return summary, errors


def render_phase16_validator_quality_report(
    summary: Phase16ValidatorQualitySummary,
    *,
    verifier_command: str,
    verifier_output: str,
    errors: list[str],
) -> str:
    status = "PASS" if not errors else "FAIL"
    failure_block = "\n".join(f"- {item}" for item in errors) if errors else "- none"
    output_block = verifier_output.rstrip() or "(no output)"
    return (
        "# Phase 16 Validator Quality Gate\n\n"
        f"Result: **{status}**\n\n"
        "## Exact Counts\n\n"
        f"- total cases: `{summary.total_cases}`\n"
        f"- passing cases: `{summary.passing_cases}`\n"
        f"- generated validator cases: `{summary.generated_validator_cases}`\n"
        f"- syntax-preflighted validator cases: `{summary.syntax_preflighted_cases}`\n"
        f"- ask cases: `{summary.ask_cases}`\n"
        f"- public GitHub cases: `{summary.public_github_cases}`\n"
        f"- complex data-processing cases: `{summary.data_processing_cases}`\n"
        f"- complex data-processing ask cases: `{summary.data_processing_ask_cases}`\n"
        f"- retry-loop cases: `{summary.retry_loop_cases}`\n"
        f"- multi-file bugfix cases: `{summary.multi_file_cases}`\n"
        f"- validator-preflight isolation cases: `{summary.validator_preflight_cases}`\n\n"
        "## Verifier Command\n\n"
        f"```powershell\n{verifier_command}\n```\n\n"
        "## Verifier Output\n\n"
        f"```text\n{output_block}\n```\n\n"
        "## Failure Reasons\n\n"
        f"{failure_block}\n"
    )


def _validate_quality_record(record: dict[str, Any], *, index: int) -> list[str]:
    errors: list[str] = []
    label = f"quality_case[{index}]"
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
    if not _string(record.get("scenario_family")):
        errors.append(f"{label} missing `scenario_family`")
    if _string(record.get("runtime_used")) != "claw":
        errors.append(f"{label} must use `runtime_used=claw`")
    if _string(record.get("runtime_fallback")) != "none":
        errors.append(f"{label} must use `runtime_fallback=none`")
    if not _string(record.get("session_id")):
        errors.append(f"{label} missing `session_id`")

    behavioral_task = _truthy(record.get("behavioral_task"))
    checks_run = _string_list(record.get("checks_run"))
    if not checks_run:
        errors.append(f"{label} missing `checks_run`")
    if behavioral_task and _truthy(record.get("syntax_only_validation")):
        errors.append(f"{label} is a behavioral task but records syntax-only validation")

    acceptance_criteria = _sequence(record.get("acceptance_criteria"))
    acceptance_results = _sequence(record.get("acceptance_criteria_results"))
    unchecked = _string_list(record.get("unchecked_acceptance_criteria"))
    if behavioral_task and not acceptance_criteria:
        errors.append(f"{label} missing `acceptance_criteria` for a behavioral task")
    if behavioral_task and not acceptance_results:
        errors.append(f"{label} missing `acceptance_criteria_results` for a behavioral task")
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

    validator_generated = _truthy(record.get("validator_generated"))
    if validator_generated:
        if not _string(record.get("validator_path")):
            errors.append(f"{label} missing `validator_path` for generated-validator case")
        if not _truthy(record.get("validator_task_specific_name")):
            errors.append(f"{label} validator filename is not marked task-specific")
        if not _truthy(record.get("validator_syntax_preflight_passed")):
            errors.append(f"{label} validator did not pass syntax preflight")
        if not _string(record.get("validator_task_run_id")):
            errors.append(f"{label} missing `validator_task_run_id`")
        if not _truthy(record.get("validator_executed")):
            errors.append(f"{label} validator was not executed after preflight")
        if _string(record.get("validator_execution_status")) != "pass":
            errors.append(f"{label} validator execution did not pass")

    failure_classification = _string(record.get("failure_classification"))
    if failure_classification and failure_classification not in {
        "none",
        "validator_generation_failure",
        "validation_plan_error",
        "check_scheduler_error",
        "source_behavior_failure",
    }:
        errors.append(f"{label} has invalid `failure_classification`: {failure_classification}")
    if _truthy(record.get("validator_generation_failure")) and failure_classification == "user_blocker":
        errors.append(f"{label} mislabeled a validator-generation failure as a user blocker")

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

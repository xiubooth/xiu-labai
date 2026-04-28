from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


VALID_STATUSES = {"pass", "fail", "partial", "blocked", "skipped"}
VALID_SURFACES = {"workflow", "ask"}
VALID_EVIDENCE_KINDS = {"live_task", "regression_test"}

REQUIRED_REGRESSION_FAMILIES = {
    "source_behavior_required_but_only_validator_changed",
    "validation_harness_hang_or_capture_failure",
    "relevant_source_not_read",
}
REQUIRED_LIVE_SCENARIO_FAMILIES = {
    "task1_daily_crsp_script_output",
}
VALID_SCRIPT_OUTPUT_PATHS = {
    "direct_callable",
    "output_file_readback",
    "monkeypatched_sink_capture",
    "direct_helper_validation",
}

MIN_TOTAL_CASES = 5
MIN_LIVE_SOURCE_REQUIRED_CASES = 2
MIN_WORKFLOW_LIVE_CASES = 1
MIN_ASK_LIVE_CASES = 1


@dataclass(frozen=True)
class Phase16SourceAndEvidenceSummary:
    total_cases: int
    passing_cases: int
    live_task_cases: int
    workflow_live_cases: int
    ask_live_cases: int
    task1_script_output_cases: int
    validator_only_prevented_cases: int
    hanging_validator_cases: int
    relevant_source_read_cases: int


def load_phase16_source_and_evidence_jsonl(path: str | Path) -> list[dict[str, Any]]:
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


def verify_phase16_source_and_evidence(
    records: list[dict[str, Any]],
    *,
    task_bank_records: list[dict[str, Any]] | None = None,
    report_doc: str | Path | None = None,
    validator_doc: str | Path | None = None,
    data_processing_doc: str | Path | None = None,
) -> tuple[Phase16SourceAndEvidenceSummary, list[str]]:
    errors: list[str] = []
    seen_case_ids: set[str] = set()
    passing_records: list[dict[str, Any]] = []
    task_bank_by_id = {
        _string(record.get("task_id")): record
        for record in (task_bank_records or [])
        if _string(record.get("task_id"))
    }

    for index, record in enumerate(records, start=1):
        case_errors = _validate_source_and_evidence_record(record, index=index)
        case_id = _string(record.get("case_id"))
        if case_id:
            if case_id in seen_case_ids:
                case_errors.append(f"source/evidence case `{case_id}` is duplicated")
            seen_case_ids.add(case_id)
        if _string(record.get("evidence_kind")) == "live_task":
            task_bank_task_id = _string(record.get("task_bank_task_id"))
            if task_bank_records is not None:
                if not task_bank_task_id:
                    case_errors.append(
                        f"source/evidence case `{case_id or index}` missing `task_bank_task_id`"
                    )
                else:
                    task_bank_record = task_bank_by_id.get(task_bank_task_id)
                    if task_bank_record is None:
                        case_errors.append(
                            f"source/evidence case `{case_id or index}` references missing task-bank row "
                            f"`{task_bank_task_id}`"
                        )
                    else:
                        case_errors.extend(
                            _validate_task_bank_source_and_evidence_alignment(
                                record,
                                task_bank_record,
                                label=f"source/evidence case `{case_id or index}`",
                            )
                        )
        errors.extend(case_errors)
        if not case_errors and _string(record.get("status")) == "pass":
            passing_records.append(record)

    summary = Phase16SourceAndEvidenceSummary(
        total_cases=len(records),
        passing_cases=len(passing_records),
        live_task_cases=sum(
            1 for record in passing_records if _string(record.get("evidence_kind")) == "live_task"
        ),
        workflow_live_cases=sum(
            1
            for record in passing_records
            if _string(record.get("evidence_kind")) == "live_task"
            and _string(record.get("workflow_or_ask")) == "workflow"
        ),
        ask_live_cases=sum(
            1
            for record in passing_records
            if _string(record.get("evidence_kind")) == "live_task"
            and _string(record.get("workflow_or_ask")) == "ask"
        ),
        task1_script_output_cases=sum(
            1
            for record in passing_records
            if _string(record.get("evidence_kind")) == "live_task"
            and _string(record.get("scenario_family")) == "task1_daily_crsp_script_output"
        ),
        validator_only_prevented_cases=sum(
            1
            for record in passing_records
            if _string(record.get("scenario_family"))
            == "source_behavior_required_but_only_validator_changed"
            and _truthy(record.get("validator_only_pass_prevented"))
        ),
        hanging_validator_cases=sum(
            1
            for record in passing_records
            if _string(record.get("scenario_family"))
            == "validation_harness_hang_or_capture_failure"
            and _truthy(record.get("validator_timeout_bounded"))
            and _truthy(record.get("validator_timeout_handled"))
        ),
        relevant_source_read_cases=sum(
            1
            for record in passing_records
            if _string(record.get("scenario_family")) == "relevant_source_not_read"
            and _truthy(record.get("relevant_source_missing_detected"))
        ),
    )

    for label, path in (
        ("validator-quality artifact", validator_doc),
        ("data-processing regression artifact", data_processing_doc),
    ):
        if path is None:
            continue
        if not Path(path).is_file():
            errors.append(f"missing required {label}: {path}")

    if report_doc is not None:
        report_path = Path(report_doc)
        parent = report_path.parent
        if not parent.is_dir():
            errors.append(f"missing parent directory for source-and-evidence artifact: {parent}")

    if summary.total_cases < MIN_TOTAL_CASES:
        errors.append(f"need at least {MIN_TOTAL_CASES} source/evidence cases; found {summary.total_cases}")
    if summary.passing_cases != summary.total_cases:
        errors.append(
            f"all source/evidence cases must pass; found {summary.passing_cases}/{summary.total_cases} passing"
        )
    if summary.live_task_cases < MIN_LIVE_SOURCE_REQUIRED_CASES:
        errors.append(
            f"need at least {MIN_LIVE_SOURCE_REQUIRED_CASES} passing live source-required cases; "
            f"found {summary.live_task_cases}"
        )
    if summary.workflow_live_cases < MIN_WORKFLOW_LIVE_CASES:
        errors.append("need at least one passing workflow live source-required case")
    if summary.ask_live_cases < MIN_ASK_LIVE_CASES:
        errors.append("need at least one passing ask live source-required case")
    live_scenario_families = {
        _string(record.get("scenario_family"))
        for record in passing_records
        if _string(record.get("evidence_kind")) == "live_task"
    }
    for family in REQUIRED_LIVE_SCENARIO_FAMILIES:
        if family not in live_scenario_families:
            errors.append(f"missing required live source/evidence scenario `{family}`")
    if summary.validator_only_prevented_cases < 1:
        errors.append("missing regression coverage for validator-only pass prevention")
    if summary.hanging_validator_cases < 1:
        errors.append("missing regression coverage for validator hang/capture failure handling")
    if summary.relevant_source_read_cases < 1:
        errors.append("missing regression coverage for relevant-source-read enforcement")

    regression_families = {
        _string(record.get("scenario_family"))
        for record in passing_records
        if _string(record.get("evidence_kind")) == "regression_test"
    }
    for family in REQUIRED_REGRESSION_FAMILIES:
        if family not in regression_families:
            errors.append(f"missing required source/evidence regression family `{family}`")

    return summary, errors


def render_phase16_source_and_evidence_report(
    summary: Phase16SourceAndEvidenceSummary,
    *,
    verifier_command: str,
    verifier_output: str,
    errors: list[str],
) -> str:
    status = "PASS" if not errors else "FAIL"
    failure_block = "\n".join(f"- {item}" for item in errors) if errors else "- none"
    output_block = verifier_output.rstrip() or "(no output)"
    return (
        "# Phase 16 Source And Evidence Gate\n\n"
        f"Result: **{status}**\n\n"
        "## Exact Counts\n\n"
        f"- total cases: `{summary.total_cases}`\n"
        f"- passing cases: `{summary.passing_cases}`\n"
        f"- live source-required cases: `{summary.live_task_cases}`\n"
        f"- workflow live cases: `{summary.workflow_live_cases}`\n"
        f"- ask live cases: `{summary.ask_live_cases}`\n"
        f"- Task 1 script-output live cases: `{summary.task1_script_output_cases}`\n"
        f"- validator-only prevention cases: `{summary.validator_only_prevented_cases}`\n"
        f"- hanging-validator handling cases: `{summary.hanging_validator_cases}`\n"
        f"- relevant-source-read cases: `{summary.relevant_source_read_cases}`\n\n"
        "## Verifier Command\n\n"
        f"```powershell\n{verifier_command}\n```\n\n"
        "## Verifier Output\n\n"
        f"```text\n{output_block}\n```\n\n"
        "## Failure Reasons\n\n"
        f"{failure_block}\n"
    )


def _validate_source_and_evidence_record(record: dict[str, Any], *, index: int) -> list[str]:
    errors: list[str] = []
    label = f"source_evidence_case[{index}]"
    case_id = _string(record.get("case_id"))
    if not case_id:
        errors.append(f"{label} missing `case_id`")
    evidence_kind = _string(record.get("evidence_kind"))
    if evidence_kind not in VALID_EVIDENCE_KINDS:
        errors.append(f"{label} has invalid `evidence_kind`: {evidence_kind or '<missing>'}")
        return errors
    status = _string(record.get("status"))
    if status not in VALID_STATUSES:
        errors.append(f"{label} has invalid `status`: {status or '<missing>'}")
    elif status != "pass":
        errors.append(f"{label} is not countable because status is `{status}`")
    if not _string(record.get("scenario_family")):
        errors.append(f"{label} missing `scenario_family`")

    if evidence_kind == "live_task":
        errors.extend(_validate_live_source_and_evidence_record(record, label=label))
    else:
        errors.extend(_validate_regression_source_and_evidence_record(record, label=label))
    return errors


def _validate_live_source_and_evidence_record(record: dict[str, Any], *, label: str) -> list[str]:
    errors: list[str] = []
    workflow_or_ask = _string(record.get("workflow_or_ask"))
    if workflow_or_ask not in VALID_SURFACES:
        errors.append(f"{label} has invalid `workflow_or_ask`: {workflow_or_ask or '<missing>'}")
    if not _string(record.get("session_id")):
        errors.append(f"{label} missing `session_id`")
    if not _truthy(record.get("source_behavior_required")):
        errors.append(f"{label} must record `source_behavior_required=true`")
    if not _truthy(record.get("source_change_needed")) and not _truthy(record.get("source_behavior_preexisted")):
        errors.append(f"{label} must record whether source change was needed or behavior already existed")
    if not _string_list(record.get("required_relevant_source_files")):
        errors.append(f"{label} missing `required_relevant_source_files`")
    if _truthy(record.get("source_change_needed")) and not _truthy(record.get("source_behavior_preexisted")):
        if not _string_list(record.get("changed_source_files")):
            errors.append(f"{label} changed no real source files even though source behavior was required")
    if not _truthy(record.get("direct_validation_evidence")):
        errors.append(f"{label} missing `direct_validation_evidence=true`")
    if not _truthy(record.get("criterion_level_evidence")):
        errors.append(f"{label} missing `criterion_level_evidence=true`")
    if _truthy(record.get("syntax_only_validation")):
        errors.append(f"{label} cannot rely on syntax-only validation")
    if not _truthy(record.get("validator_timeout_bounded")):
        errors.append(f"{label} missing `validator_timeout_bounded=true`")
    if _truthy(record.get("validator_timed_out")):
        errors.append(f"{label} cannot count a timed-out validator as passing evidence")
    if _truthy(record.get("validator_repaired")) and not _truthy(
        record.get("source_revalidated_after_validator_repair")
    ):
        errors.append(f"{label} repaired the validator but did not revalidate source behavior afterward")
    if _string(record.get("scenario_family")) == "task1_daily_crsp_script_output":
        read_strategy = _string(record.get("read_strategy"))
        if not read_strategy or read_strategy == "none":
            errors.append(f"{label} must record a non-`none` `read_strategy`")
        validation_path = _string(record.get("validation_path"))
        if validation_path not in VALID_SCRIPT_OUTPUT_PATHS:
            errors.append(
                f"{label} has invalid `validation_path`: {validation_path or '<missing>'}"
            )
        if validation_path == "direct_module_dataframe":
            errors.append(f"{label} cannot rely on direct module-level DataFrame exposure")
        if not _truthy(record.get("strategy_switched_after_repeated_validator_failure")):
            errors.append(
                f"{label} missing `strategy_switched_after_repeated_validator_failure=true`"
            )
        required_sources = _string_list(record.get("required_relevant_source_files"))
        if not any(Path(item).name.lower().startswith("15_preparedailycrsp_task") for item in required_sources):
            errors.append(
                f"{label} must include at least one concrete `15_PrepareDailyCRSP_task*.py` file in `required_relevant_source_files`"
            )
    if _truthy(record.get("validator_generated")):
        if not _truthy(record.get("validator_task_specific_name")):
            errors.append(f"{label} missing task-specific validator naming evidence")
        if not _truthy(record.get("validator_syntax_preflight_passed")):
            errors.append(f"{label} missing validator syntax preflight evidence")
        if not _truthy(record.get("validator_executed")):
            errors.append(f"{label} missing validator execution evidence")
        if _string(record.get("validator_execution_status")) != "pass":
            errors.append(f"{label} validator execution did not pass")
    return errors


def _validate_regression_source_and_evidence_record(record: dict[str, Any], *, label: str) -> list[str]:
    errors: list[str] = []
    if not _string(record.get("test_name")):
        errors.append(f"{label} missing `test_name`")
    family = _string(record.get("scenario_family"))
    if family not in REQUIRED_REGRESSION_FAMILIES:
        errors.append(f"{label} has invalid regression `scenario_family`: {family or '<missing>'}")
        return errors
    if family == "source_behavior_required_but_only_validator_changed":
        if not _truthy(record.get("validator_only_pass_prevented")):
            errors.append(f"{label} missing `validator_only_pass_prevented=true`")
    elif family == "validation_harness_hang_or_capture_failure":
        if not _truthy(record.get("validator_timeout_bounded")):
            errors.append(f"{label} missing `validator_timeout_bounded=true`")
        if not _truthy(record.get("validator_timeout_handled")):
            errors.append(f"{label} missing `validator_timeout_handled=true`")
    elif family == "relevant_source_not_read":
        if not _truthy(record.get("relevant_source_missing_detected")):
            errors.append(f"{label} missing `relevant_source_missing_detected=true`")
    return errors


def _validate_task_bank_source_and_evidence_alignment(
    record: dict[str, Any],
    task_bank_record: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if _string(task_bank_record.get("status")) != "pass":
        errors.append(f"{label} references a task-bank row that is not passing")
    if _string(task_bank_record.get("session_id")) != _string(record.get("session_id")):
        errors.append(f"{label} task-bank session id does not match source/evidence evidence")
    if _string(task_bank_record.get("workflow_or_ask")) != _string(record.get("workflow_or_ask")):
        errors.append(f"{label} task-bank workflow surface does not match source/evidence evidence")
    actual_inspected = set(_string_list(task_bank_record.get("actual_files_inspected")))
    required_sources = set(_string_list(record.get("required_relevant_source_files")))
    missing_inspected = sorted(required_sources - actual_inspected)
    if missing_inspected:
        errors.append(
            f"{label} is missing inspected relevant source files in the task bank: {', '.join(missing_inspected)}"
        )
    actual_changed = set(_string_list(task_bank_record.get("actual_files_changed")))
    changed_source = set(_string_list(record.get("changed_source_files")))
    if _truthy(record.get("source_change_needed")) and not _truthy(record.get("source_behavior_preexisted")):
        if not changed_source:
            errors.append(f"{label} is missing `changed_source_files` evidence")
        elif not changed_source.issubset(actual_changed):
            errors.append(f"{label} changed source files do not align with the task-bank row")
    if _string(record.get("scenario_family")) == "task1_daily_crsp_script_output":
        task_bank_read_strategy = _string(task_bank_record.get("read_strategy"))
        if not task_bank_read_strategy or task_bank_read_strategy == "none":
            errors.append(f"{label} task-bank row must record a non-`none` read strategy")
        validation_path = _string(task_bank_record.get("validation_path"))
        if validation_path not in VALID_SCRIPT_OUTPUT_PATHS:
            errors.append(f"{label} task-bank row is missing a valid script-output `validation_path`")
        if not _truthy(task_bank_record.get("strategy_switched_after_repeated_validator_failure")):
            errors.append(
                f"{label} task-bank row must record `strategy_switched_after_repeated_validator_failure=true`"
            )
    checks_run = _string_list(task_bank_record.get("checks_run"))
    if not checks_run:
        errors.append(f"{label} task-bank row is missing `checks_run`")
    elif all(item in {"py_compile", "import_check"} for item in checks_run):
        errors.append(f"{label} task-bank row cannot pass on syntax-only checks")
    acceptance_results = _sequence(task_bank_record.get("acceptance_criteria_results"))
    if not acceptance_results:
        errors.append(f"{label} task-bank row is missing criterion-level acceptance results")
    for result in acceptance_results:
        if not isinstance(result, dict):
            errors.append(f"{label} task-bank acceptance results contain non-object entries")
            break
        if _string(result.get("status")) != "pass":
            errors.append(
                f"{label} task-bank row has non-passing acceptance result for "
                f"`{_string(result.get('criterion')) or '<unknown>'}`"
            )
    if _truthy(task_bank_record.get("syntax_only_validation")):
        errors.append(f"{label} task-bank row records syntax-only validation")
    return errors


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    result: list[str] = []
    for item in _sequence(value):
        text = _string(item)
        if text:
            result.append(text)
    return result


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)

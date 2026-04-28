from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / ".planning" / "phases" / "18-route2-workflow-closure"
EVIDENCE_PATH = ARTIFACT_ROOT / "18-WORKFLOW-CLOSURE-EVIDENCE.jsonl"
ARTIFACT_DOCS = (
    ARTIFACT_ROOT / "18-ROUTE2-WORKFLOW-CLOSURE-SUMMARY.md",
    ARTIFACT_ROOT / "18-TASK-SHAPE-ROUTING.md",
    ARTIFACT_ROOT / "18-REPEATED-FAILURE-SWITCHING.md",
    ARTIFACT_ROOT / "18-OWNER-DETECTION.md",
    ARTIFACT_ROOT / "18-CODE-QUALITY-WARNINGS.md",
    ARTIFACT_ROOT / "18-TASK1-CLOSURE-REGRESSION.md",
    ARTIFACT_ROOT / "18-TASK8-CLOSURE-REGRESSION.md",
    ARTIFACT_ROOT / "18-TASK4-CLOSURE-REGRESSION.md",
    ARTIFACT_ROOT / "18-TASK9-CLOSURE-REGRESSION.md",
)
REQUIRED_ROWS = (
    "task_shape_routing",
    "repeated_failure_switching",
    "owner_primary_artifact_enforcement",
    "code_quality_warnings",
    "task1_closure_regression",
    "task8_closure_regression",
    "task4_closure_regression",
    "task9_closure_regression",
    "ask_smoke",
    "base_gates",
)


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _load_session(row: dict[str, object], errors: list[str]) -> dict[str, object]:
    session_path_text = str(row.get("session_path", ""))
    _require(bool(session_path_text), f"missing session_path on `{row.get('name')}`", errors)
    if not session_path_text:
        return {}
    session_path = Path(session_path_text)
    _require(session_path.exists(), f"missing session file for `{row.get('name')}`: {session_path}", errors)
    if not session_path.exists():
        return {}
    session = _read_json(session_path)
    ledger_path_text = str(row.get("evidence_ledger_path", ""))
    if ledger_path_text:
        ledger_path = Path(ledger_path_text)
        _require(ledger_path.exists(), f"missing evidence ledger for `{row.get('name')}`: {ledger_path}", errors)
    return session


def _workspace_trace(session: dict[str, object]) -> dict[str, object]:
    trace = session.get("workspace_trace", {})
    return trace if isinstance(trace, dict) else {}


def _failed_python_validate_count(session: dict[str, object]) -> int:
    details = _workspace_trace(session).get("executed_check_details", ())
    if not isinstance(details, list):
        return 0
    return sum(
        1
        for item in details
        if isinstance(item, dict)
        and str(item.get("result_name")) == "python_validate"
        and str(item.get("status")) != "passed"
    )


def main() -> int:
    errors: list[str] = []

    validator_routing_text = (REPO_ROOT / "src" / "labai" / "validator_routing.py").read_text(encoding="utf-8")
    cli_text = (REPO_ROOT / "src" / "labai" / "cli.py").read_text(encoding="utf-8")
    owner_detection_text = (REPO_ROOT / "src" / "labai" / "owner_detection.py").read_text(encoding="utf-8")

    for path in (*ARTIFACT_DOCS, EVIDENCE_PATH):
        _require(path.exists(), f"missing required Route 2.1 file: {path.relative_to(REPO_ROOT)}", errors)

    _require("def route_task_validation" in validator_routing_text, "validator_routing.py must define route_task_validation", errors)
    _require("TaskShape = Literal[" in validator_routing_text, "validator_routing.py must define TaskShape", errors)
    _require("_strategy_retry_stop_reason" in cli_text, "cli.py must define _strategy_retry_stop_reason", errors)
    _require("route_task_validation(" in cli_text, "cli.py must call route_task_validation", errors)
    _require("download_[A-Za-z0-9_]+" in owner_detection_text, "owner_detection.py must use the generic Task 9 download-function chain check", errors)
    _require("Task 9 interface mismatch" in owner_detection_text, "owner_detection.py must emit the Task 9 interface mismatch blocker", errors)

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    rows = _read_jsonl(EVIDENCE_PATH)
    by_name = {str(row.get("name")): row for row in rows}
    for row_name in REQUIRED_ROWS:
        _require(row_name in by_name, f"missing evidence row `{row_name}`", errors)
        if row_name in by_name:
            _require(str(by_name[row_name].get("status")) == "passed", f"evidence row `{row_name}` is not passed", errors)

    task_shape_row = by_name.get("task_shape_routing", {})
    _require(bool(task_shape_row.get("router_exists")), "task_shape_routing must prove router_exists=true", errors)
    _require(bool(task_shape_row.get("router_used")), "task_shape_routing must prove router_used=true", errors)
    _require(str(task_shape_row.get("task1_shape")) == "script_output_task", "Task 1 must route to script_output_task", errors)
    _require(str(task_shape_row.get("task8_shape")) == "multi_output_pipeline_task", "Task 8 must route to multi_output_pipeline_task", errors)
    _require(str(task_shape_row.get("task4_shape")) == "annual_to_monthly_pipeline_task", "Task 4 must route to annual_to_monthly_pipeline_task", errors)
    _require(str(task_shape_row.get("task9_shape")) == "external_dependency_task", "Task 9 must route to external_dependency_task", errors)
    _require(bool(task_shape_row.get("task1_rejects_direct_dataframe")), "Task 1 must reject direct DataFrame exposure", errors)
    _require(bool(task_shape_row.get("task4_rejects_callable_only")), "Task 4 must reject callable-only validation", errors)

    switching_row = by_name.get("repeated_failure_switching", {})
    _require(bool(switching_row.get("strategy_switch_evidence_exists")), "repeated_failure_switching must prove strategy switch evidence exists", errors)
    _require(bool(switching_row.get("no_unchanged_strategy_repeated_more_than_twice")), "repeated_failure_switching must prove no unchanged strategy repeated more than twice", errors)
    _require(int(switching_row.get("max_failed_python_validate_per_session", 99)) <= 2, "repeated_failure_switching must keep failed python_validate repetitions <= 2", errors)
    _require(int(switching_row.get("max_repair_rounds_observed", 99)) <= 3, "repeated_failure_switching must keep fresh closure sessions at 3 repair rounds or fewer", errors)

    owner_row = by_name.get("owner_primary_artifact_enforcement", {})
    _require(bool(owner_row.get("helper_only_rejected")), "owner_primary_artifact_enforcement must prove helper_only_rejected=true", errors)
    _require(bool(owner_row.get("validator_only_rejected")), "owner_primary_artifact_enforcement must prove validator_only_rejected=true", errors)
    _require(bool(owner_row.get("task9_missing_function_chain_detected")), "owner_primary_artifact_enforcement must prove task9_missing_function_chain_detected=true", errors)
    _require(str(owner_row.get("task4_primary_owner")) == "14_PreAnnualCS.py", "owner_primary_artifact_enforcement must record 14_PreAnnualCS.py as the Task 4 primary owner", errors)

    warnings_row = by_name.get("code_quality_warnings", {})
    _require(bool(warnings_row.get("warning_system_present")), "code_quality_warnings must prove warning_system_present=true", errors)
    _require(bool(warnings_row.get("task1_centralization_resolved")), "code_quality_warnings must prove Task 1 centralization resolved or warned", errors)
    _require(bool(warnings_row.get("task8_owner_boundary_resolved")), "code_quality_warnings must prove Task 8 owner boundary resolved or warned", errors)

    task1_row = by_name.get("task1_closure_regression", {})
    task1_session = _load_session(task1_row, errors) if task1_row else {}
    if task1_session:
        trace = _workspace_trace(task1_session)
        _require(str(task1_session.get("status")) == "ok", "Task 1 session must finish with status=ok", errors)
        _require(str(trace.get("check_status")) == "passed", "Task 1 session must finish with check_status=passed", errors)
        _require(str((trace.get("validator_routing") or {}).get("task_shape")) == "script_output_task", "Task 1 session must record script_output_task", errors)
        _require(_failed_python_validate_count(task1_session) <= 1, "Task 1 must not repeat failed python_validate more than once in the fresh closure session", errors)
        _require("DataFrame output to validate" not in task1_session.get("final_answer", ""), "Task 1 final answer must not rely on direct DataFrame exposure failures", errors)
        _require(bool(task1_row.get("centralization_resolution_recorded")), "Task 1 evidence row must record centralization resolution", errors)

    task8_row = by_name.get("task8_closure_regression", {})
    task8_session = _load_session(task8_row, errors) if task8_row else {}
    if task8_session:
        trace = _workspace_trace(task8_session)
        _require(str(task8_session.get("status")) == "ok", "Task 8 session must finish with status=ok", errors)
        _require(str(trace.get("check_status")) == "passed", "Task 8 session must finish with check_status=passed", errors)
        _require(str((trace.get("validator_routing") or {}).get("task_shape")) == "multi_output_pipeline_task", "Task 8 session must record multi_output_pipeline_task", errors)
        _require(_failed_python_validate_count(task8_session) <= 1, "Task 8 must not repeat failed python_validate more than once in the fresh closure session", errors)
        changed = tuple(trace.get("modified_files", ()) or ())
        _require("13_PrepareRatings.py" in changed, "Task 8 fresh closure session must either resolve or replace the 13_PrepareRatings.py production-path boundary", errors)

    task4_row = by_name.get("task4_closure_regression", {})
    task4_session = _load_session(task4_row, errors) if task4_row else {}
    if task4_session:
        trace = _workspace_trace(task4_session)
        _require(str(task4_session.get("status")) == "ok", "Task 4 session must finish with status=ok", errors)
        _require(str(trace.get("check_status")) == "passed", "Task 4 session must finish with check_status=passed", errors)
        _require(str((trace.get("validator_routing") or {}).get("task_shape")) == "annual_to_monthly_pipeline_task", "Task 4 session must record annual_to_monthly_pipeline_task", errors)
        _require(_failed_python_validate_count(task4_session) <= 1, "Task 4 must not repeat failed python_validate more than once in the fresh closure session", errors)
        session_text = Path(str(task4_row.get("session_path"))).read_text(encoding="utf-8")
        _require("callable download or writable output path" not in session_text, "Task 4 fresh closure session must not fail on the old callable/writable-output-path assumption", errors)

    task9_row = by_name.get("task9_closure_regression", {})
    task9_session = _load_session(task9_row, errors) if task9_row else {}
    if task9_session:
        trace = _workspace_trace(task9_session)
        outcome = str(task9_row.get("outcome", ""))
        _require(outcome in {"true_success", "honest_failure"}, "Task 9 outcome must be true_success or honest_failure", errors)
        if outcome == "honest_failure":
            _require(str(task9_session.get("status")) != "ok", "Task 9 honest failure must not end with status=ok", errors)
            _require(str(trace.get("check_status")) == "failed", "Task 9 honest failure must end with check_status=failed", errors)
            _require(_failed_python_validate_count(task9_session) <= 1, "Task 9 honest failure must stop after at most one failed python_validate in the fresh closure session", errors)
            blocking = tuple((trace.get("owner_detection") or {}).get("blocking_issues", ()) or ())
            _require(any("download_factor_dataset_bundle" in item for item in blocking), "Task 9 honest failure must record the missing-function-chain blocker", errors)
            _require(bool(task9_row.get("no_false_success")), "Task 9 honest failure must record no_false_success=true", errors)

    ask_row = by_name.get("ask_smoke", {})
    ask_session = _load_session(ask_row, errors) if ask_row else {}
    if ask_session:
        _require(str(ask_session.get("status")) == "ok", "ask_smoke session must finish with status=ok", errors)
        _require(str(ask_session.get("command")) == "ask", "ask_smoke session must come from `labai ask`", errors)
        _require((ask_session.get("final_answer") or "").strip() == "HELLO", "ask_smoke final answer must be HELLO", errors)
        _require(bool(ask_row.get("stdout_contains_hello")), "ask_smoke must prove stdout_contains_hello=true", errors)
        _require(not bool(ask_row.get("boolean_only")), "ask_smoke must prove boolean_only=false", errors)

    gates_row = by_name.get("base_gates", {})
    for field in (
        "pytest_passed",
        "doctor_ready",
        "tools_registered",
        "route1_passed",
        "route11_passed",
        "route2_passed",
        "phase16_task_bank_passed",
        "phase16_isolation_passed",
        "phase16_validator_quality_passed",
        "phase16_dependency_fallback_passed",
        "phase16_source_and_evidence_passed",
    ):
        _require(bool(gates_row.get(field)), f"base_gates must prove `{field}=true`", errors)

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print("PASS: Phase 18 Route 2.1 workflow closure verifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / ".planning" / "phases" / "18-route2-mature-main-loop"
EVIDENCE_PATH = ARTIFACT_ROOT / "18-REGRESSION-EVIDENCE.jsonl"
ROUTE2_DOC = REPO_ROOT / "docs" / "external-agent-patterns-route2.md"
CHECK_SPECS = (
    REPO_ROOT / ".continue" / "checks" / "labai-phase18-route2-task-manifest.md",
    REPO_ROOT / ".continue" / "checks" / "labai-phase18-route2-required-reads.md",
    REPO_ROOT / ".continue" / "checks" / "labai-phase18-route2-structured-edits.md",
    REPO_ROOT / ".continue" / "checks" / "labai-phase18-route2-runtime-exec.md",
    REPO_ROOT / ".continue" / "checks" / "labai-phase18-route2-typed-validation.md",
    REPO_ROOT / ".continue" / "checks" / "labai-phase18-route2-evidence-ledger.md",
)
ARTIFACT_DOCS = (
    ARTIFACT_ROOT / "18-ROUTE2-MATURE-SUMMARY.md",
    ARTIFACT_ROOT / "18-MATURE-COMPONENTS.md",
    ARTIFACT_ROOT / "18-SUBMODULES.md",
    ARTIFACT_ROOT / "18-TASK-MANIFEST.md",
    ARTIFACT_ROOT / "18-REPO-MAP-GREP-AST.md",
    ARTIFACT_ROOT / "18-REQUIRED-READS.md",
    ARTIFACT_ROOT / "18-OWNER-DETECTION.md",
    ARTIFACT_ROOT / "18-STRUCTURED-EDIT-OPS.md",
    ARTIFACT_ROOT / "18-RUNTIME-EXECUTION.md",
    ARTIFACT_ROOT / "18-TYPED-VALIDATION.md",
    ARTIFACT_ROOT / "18-STRATEGY-SWITCHING.md",
    ARTIFACT_ROOT / "18-EVIDENCE-LEDGER.md",
    ARTIFACT_ROOT / "18-ASK-CODING-DISCIPLINE.md",
    ARTIFACT_ROOT / "18-TASK9-REGRESSION.md",
    ARTIFACT_ROOT / "18-TASK1-REGRESSION.md",
    ARTIFACT_ROOT / "18-TASK8-REGRESSION.md",
    ARTIFACT_ROOT / "18-NOTEBOOK-REGRESSION.md",
)
REQUIRED_ROWS = (
    "unidiff_runtime",
    "grep_ast_adapter",
    "submodule_status",
    "ask_coding_discipline",
    "task9_regression",
    "task1_regression",
    "task8_regression",
    "notebook_regression",
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
    _require(bool(session_path_text), f"missing session_path on evidence row `{row.get('name')}`", errors)
    if not session_path_text:
        return {}
    session_path = Path(session_path_text)
    _require(session_path.exists(), f"missing session file for `{row.get('name')}`: {session_path_text}", errors)
    if not session_path.exists():
        return {}
    session = _read_json(session_path)
    _require(str(session.get("status")) == "ok", f"session `{session_path.name}` did not finish with status=ok", errors)
    workspace_trace = session.get("workspace_trace", {})
    if isinstance(workspace_trace, dict):
        _require(
            str(workspace_trace.get("check_status")) == "passed",
            f"session `{session_path.name}` did not finish with check_status=passed",
            errors,
        )
    ledger_path = Path(str(row.get("evidence_ledger_path", "")))
    _require(ledger_path.exists(), f"missing evidence ledger for `{row.get('name')}`: {ledger_path}", errors)
    if ledger_path.exists():
        _require(any(True for _ in ledger_path.open(encoding="utf-8")), f"empty evidence ledger for `{row.get('name')}`", errors)
    return session


def _session_field(session: dict[str, object], key: str):
    if key in session:
        return session.get(key)
    workspace_trace = session.get("workspace_trace", {})
    if isinstance(workspace_trace, dict) and key in workspace_trace:
        return workspace_trace.get(key)
    return None


def _require_route2_session_structures(session: dict[str, object], row_name: str, errors: list[str]) -> None:
    for key in ("task_manifest", "owner_detection", "required_read_evidence", "structured_edit_ops", "typed_validation_results"):
        _require(_session_field(session, key) is not None, f"session for `{row_name}` missing `{key}`", errors)
    manifest = _session_field(session, "task_manifest") or {}
    if isinstance(manifest, dict):
        _require(bool(manifest.get("required_read_files")), f"`{row_name}` has empty task_manifest.required_read_files", errors)
        _require(bool(manifest.get("primary_target_artifacts")), f"`{row_name}` has empty task_manifest.primary_target_artifacts", errors)
    owner = _session_field(session, "owner_detection") or {}
    if isinstance(owner, dict):
        _require(bool(owner.get("primary_owner_files")), f"`{row_name}` has no owner_detection.primary_owner_files", errors)
    reads = _session_field(session, "required_read_evidence") or []
    _require(isinstance(reads, list) and len(reads) > 0, f"`{row_name}` has no required_read_evidence", errors)
    edit_ops = _session_field(session, "structured_edit_ops") or []
    _require(isinstance(edit_ops, list) and len(edit_ops) > 0, f"`{row_name}` has no structured_edit_ops", errors)
    typed = _session_field(session, "typed_validation_results") or []
    _require(isinstance(typed, list) and len(typed) > 0, f"`{row_name}` has no typed_validation_results", errors)


def _latest_passing_typed_validation(session: dict[str, object]) -> dict[str, object]:
    typed = _session_field(session, "typed_validation_results") or []
    if not isinstance(typed, list):
        return {}
    for item in reversed(typed):
        if isinstance(item, dict) and str(item.get("status")) == "pass":
            return item
    return {}


def main() -> int:
    errors: list[str] = []

    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    structured_text = (REPO_ROOT / "src" / "labai" / "structured_edits.py").read_text(encoding="utf-8")
    repo_map_text = (REPO_ROOT / "src" / "labai" / "repo_map.py").read_text(encoding="utf-8")
    grep_adapter_text = (REPO_ROOT / "src" / "labai" / "external" / "grep_ast_adapter.py").read_text(encoding="utf-8")
    notebook_text = (REPO_ROOT / "src" / "labai" / "notebook_io.py").read_text(encoding="utf-8")
    cli_text = (REPO_ROOT / "src" / "labai" / "cli.py").read_text(encoding="utf-8")

    for required_fragment in (
        "click>=8.1,<9",
        "nbformat>=5.10,<6.0",
        "nbclient>=0.10,<1.0",
        "ipykernel>=6,<7",
        "unidiff>=0.7,<1.0",
        'route2 = [',
        "grep-ast>=0.9,<1",
    ):
        _require(required_fragment in pyproject_text, f"pyproject.toml missing `{required_fragment}`", errors)

    _require("from unidiff import PatchSet" in structured_text, "structured_edits.py must import PatchSet", errors)
    _require("PatchSet(" in structured_text, "structured_edits.py must use PatchSet", errors)
    _require("summarize_python_file" in repo_map_text, "repo_map.py must call summarize_python_file", errors)
    _require("detect_grep_ast" in repo_map_text, "repo_map.py must call detect_grep_ast", errors)
    _require("def summarize_python_file" in grep_adapter_text, "grep_ast_adapter.py must define summarize_python_file", errors)
    _require("def detect_grep_ast" in grep_adapter_text, "grep_ast_adapter.py must define detect_grep_ast", errors)
    _require("import nbformat" in notebook_text, "notebook_io.py must import nbformat", errors)
    _require("from nbclient import NotebookClient" in notebook_text, "notebook_io.py must import NotebookClient", errors)
    _require("import click" in cli_text and "click.echo" in cli_text, "cli.py must use click.echo", errors)

    for path in CHECK_SPECS + ARTIFACT_DOCS + (ROUTE2_DOC, EVIDENCE_PATH):
        _require(path.exists(), f"missing required Route 2 file: {path.relative_to(REPO_ROOT)}", errors)

    if not EVIDENCE_PATH.exists():
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    rows = _read_jsonl(EVIDENCE_PATH)
    by_name = {str(row.get("name")): row for row in rows}
    for row_name in REQUIRED_ROWS:
        _require(row_name in by_name, f"missing evidence row `{row_name}`", errors)
        if row_name in by_name:
            _require(str(by_name[row_name].get("status")) == "passed", f"evidence row `{row_name}` is not passed", errors)

    unidiff_row = by_name.get("unidiff_runtime", {})
    _require(bool(unidiff_row.get("patchset_imported")), "unidiff_runtime must prove PatchSet import", errors)
    _require(bool(unidiff_row.get("parser_used")), "unidiff_runtime must prove the parser is used", errors)

    grep_row = by_name.get("grep_ast_adapter", {})
    _require(bool(grep_row.get("adapter_exists")), "grep_ast_adapter must prove adapter_exists=true", errors)
    _require(bool(grep_row.get("repo_map_calls_adapter")), "grep_ast_adapter must prove repo_map_calls_adapter=true", errors)
    _require(str(grep_row.get("backend")) in {"grep_ast", "python_ast_fallback"}, "grep_ast_adapter must record a valid backend", errors)

    submodule_row = by_name.get("submodule_status", {})
    _require(bool(submodule_row.get("documented_unavailable")), "submodule_status must document unavailable submodules", errors)
    _require(submodule_row.get("git_root_present") is False, "submodule_status must record git_root_present=false in this workspace", errors)
    _require(int(submodule_row.get("attempted_commands", 0)) >= 5, "submodule_status must record the five attempted submodule commands", errors)

    notebook_row = by_name.get("notebook_regression", {})
    notebook_session = _load_session(notebook_row, errors) if notebook_row else {}
    if notebook_session:
        _require_route2_session_structures(notebook_session, "notebook_regression", errors)
        manifest = _session_field(notebook_session, "task_manifest") or {}
        if isinstance(manifest, dict):
            _require(str(manifest.get("task_kind")) == "notebook_deliverable", "notebook_regression must use notebook_deliverable manifest", errors)
            _require("bank_loan_model_analysis.ipynb" in tuple(manifest.get("primary_target_artifacts", ())), "notebook_regression must target the notebook as the primary artifact", errors)
        _require(bool(notebook_row.get("outputs_embedded")), "notebook_regression must prove embedded outputs", errors)
        _require(bool(notebook_row.get("workspace_safe")), "notebook_regression must prove workspace-safe paths", errors)
        latest = _latest_passing_typed_validation(notebook_session)
        _require(str(latest.get("validation_type")) == "notebook_execution", "notebook_regression final typed validation must be notebook_execution", errors)

    task9_row = by_name.get("task9_regression", {})
    task9_session = _load_session(task9_row, errors) if task9_row else {}
    if task9_session:
        _require_route2_session_structures(task9_session, "task9_regression", errors)
        _require(bool(task9_row.get("true_success")), "task9_regression must be true success", errors)
        _require(bool(task9_row.get("daily_uses_time_d")), "task9_regression must prove daily_uses_time_d", errors)
        _require(bool(task9_row.get("monthly_uses_time_avail_m")), "task9_regression must prove monthly_uses_time_avail_m", errors)
        _require(bool(task9_row.get("coverage_2010_plus")), "task9_regression must prove 2010+ coverage", errors)
        _require(bool(task9_row.get("naming_consistent")), "task9_regression must prove consistent naming", errors)
        _require(bool(task9_row.get("real_code_path_validated")), "task9_regression must prove real_code_path_validated", errors)
        latest = _latest_passing_typed_validation(task9_session)
        _require(str(latest.get("validation_type")) in {"behavior", "data_contract"}, "task9_regression final typed validation must be behavior or data_contract", errors)

    task1_row = by_name.get("task1_regression", {})
    task1_session = _load_session(task1_row, errors) if task1_row else {}
    if task1_session:
        _require_route2_session_structures(task1_session, "task1_regression", errors)
        _require(bool(task1_row.get("earliest_date_2010_plus")), "task1_regression must prove earliest_date_2010_plus", errors)
        _require(bool(task1_row.get("field_contract_ok")), "task1_regression must prove field_contract_ok", errors)
        _require(bool(task1_row.get("no_null_permno_date")), "task1_regression must prove no_null_permno_date", errors)
        _require(bool(task1_row.get("owning_source_changed")), "task1_regression must prove owning_source_changed", errors)
        _require(bool(task1_row.get("strategy_switch_recorded")), "task1_regression must prove strategy_switch_recorded", errors)

    task8_row = by_name.get("task8_regression", {})
    task8_session = _load_session(task8_row, errors) if task8_row else {}
    if task8_session:
        _require_route2_session_structures(task8_session, "task8_regression", errors)
        _require(bool(task8_row.get("date_parseable")), "task8_regression must prove date_parseable", errors)
        _require(bool(task8_row.get("duplicate_detection_checked")), "task8_regression must prove duplicate_detection_checked", errors)
        _require(bool(task8_row.get("scope_2010_plus")), "task8_regression must prove scope_2010_plus", errors)
        _require(bool(task8_row.get("as_of_usable")), "task8_regression must prove as_of_usable", errors)
        _require(bool(task8_row.get("real_owner_validated")), "task8_regression must prove real_owner_validated", errors)
        latest = _latest_passing_typed_validation(task8_session)
        _require(str(latest.get("validation_type")) == "data_contract", "task8_regression final typed validation must be data_contract", errors)

    ask_row = by_name.get("ask_coding_discipline", {})
    ask_session = _load_session(ask_row, errors) if ask_row else {}
    if ask_session:
        _require_route2_session_structures(ask_session, "ask_coding_discipline", errors)
        _require(str(ask_session.get("command")) == "ask", "ask_coding_discipline session must come from `labai ask`", errors)
        _require(bool(ask_row.get("human_readable_answer")), "ask_coding_discipline must prove human_readable_answer", errors)
        _require(bool(ask_row.get("owning_source_changed")), "ask_coding_discipline must prove owning_source_changed", errors)
        _require(bool(ask_row.get("behavioral_validation_passed")), "ask_coding_discipline must prove behavioral_validation_passed", errors)

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print("PASS: Phase 18 Route 2 mature loop verifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

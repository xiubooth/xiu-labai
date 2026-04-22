from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / ".planning" / "phases" / "18-route1-hardening-ask-notebook-validator"
EVIDENCE_PATH = ARTIFACT_ROOT / "18-REGRESSION-EVIDENCE.jsonl"
CHECK_SPECS = (
    REPO_ROOT / ".continue" / "checks" / "labai-phase18-ask-output.md",
    REPO_ROOT / ".continue" / "checks" / "labai-phase18-notebook-deliverable.md",
    REPO_ROOT / ".continue" / "checks" / "labai-phase18-source-evidence.md",
    REPO_ROOT / ".continue" / "checks" / "labai-phase18-validator-output.md",
)
DOCS = (
    REPO_ROOT / "docs" / "external-agent-patterns-route1.md",
    ARTIFACT_ROOT / "18-ROUTE1-SUMMARY.md",
    ARTIFACT_ROOT / "18-ASK-STDOUT.md",
    ARTIFACT_ROOT / "18-NOTEBOOK-DELIVERABLE.md",
    ARTIFACT_ROOT / "18-VALIDATOR-CHECKS.md",
    ARTIFACT_ROOT / "18-WINDOWS-PATH-ENCODING.md",
    ARTIFACT_ROOT / "18-CONTINUE-REFERENCE.md",
)
REQUIRED_EVIDENCE = (
    "ask_smoke",
    "notebook_workflow_regression",
    "source_required_regression",
    "task1_regression",
    "task8_regression",
    "windows_safe_output_regression",
)


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    errors: list[str] = []

    for module_name in ("click", "nbformat", "nbclient"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"missing importable runtime dependency `{module_name}`: {exc}")

    notebook_io_text = (REPO_ROOT / "src" / "labai" / "notebook_io.py").read_text(encoding="utf-8")
    cli_text = (REPO_ROOT / "src" / "labai" / "cli.py").read_text(encoding="utf-8")
    _require("import nbformat" in notebook_io_text, "notebook_io.py must import nbformat", errors)
    _require("from nbclient import NotebookClient" in notebook_io_text, "notebook_io.py must import NotebookClient", errors)
    _require("import click" in cli_text and "click.echo" in cli_text, "cli.py must use click.echo", errors)

    for path in CHECK_SPECS + DOCS:
        _require(path.exists(), f"missing required Route 1 artifact: {path.relative_to(REPO_ROOT)}", errors)

    continue_reference = (ARTIFACT_ROOT / "18-CONTINUE-REFERENCE.md").read_text(encoding="utf-8")
    continue_ready = (REPO_ROOT / "third_party" / "continue").exists() or "submodule add" in continue_reference
    _require(continue_ready, "Continue reference must exist as a submodule or documented failed attempt", errors)

    _require(EVIDENCE_PATH.exists(), "missing Phase 18 regression evidence JSONL", errors)
    if EVIDENCE_PATH.exists():
        rows = _read_jsonl(EVIDENCE_PATH)
        by_name = {str(row.get("name")): row for row in rows}
        for key in REQUIRED_EVIDENCE:
            _require(key in by_name, f"missing evidence row `{key}`", errors)
            if key in by_name:
                _require(str(by_name[key].get("status")) == "passed", f"evidence row `{key}` is not passed", errors)
        ask_row = by_name.get("ask_smoke", {})
        _require(bool(ask_row.get("stdout_not_boolean_only")), "ask smoke must prove stdout is not boolean-only", errors)
        notebook_row = by_name.get("notebook_workflow_regression", {})
        for field in (
            "workspace_escape_blocked",
            "primary_artifact_notebook",
            "parsed_with_nbformat",
            "executed_with_nbclient",
            "outputs_embedded",
        ):
            _require(bool(notebook_row.get(field)), f"notebook regression missing `{field}=true`", errors)
        source_row = by_name.get("source_required_regression", {})
        for field in ("prompt_named_files_inspected", "validator_only_rejected", "helper_only_rejected"):
            _require(bool(source_row.get(field)), f"source regression missing `{field}=true`", errors)

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print("PASS: Phase 18 Route 1 verifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

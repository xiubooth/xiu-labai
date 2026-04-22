from __future__ import annotations

from dataclasses import dataclass
import asyncio
import os
from pathlib import Path
import tempfile
from typing import Iterable

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode


@dataclass(frozen=True)
class NotebookExecutionResult:
    success: bool
    notebook_path: Path
    workspace_root: Path
    executed_cell_count: int
    output_cell_count: int
    outputs_embedded: bool
    error_name: str = ""
    error_value: str = ""
    failing_cell_index: int = -1


@dataclass(frozen=True)
class CriterionEvidence:
    criterion: str
    passed: bool
    detail: str = ""


_NOTEBOOK_RUNTIME_ENV_KEYS = (
    "IPYTHONDIR",
    "JUPYTER_RUNTIME_DIR",
    "MPLCONFIGDIR",
)


def resolve_workspace_path(workspace_root: Path | str, requested_path: Path | str) -> Path:
    resolved_root = Path(workspace_root).expanduser().resolve()
    candidate = Path(requested_path).expanduser()
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Requested notebook path escapes the active workspace: {requested_path}"
        ) from exc
    return resolved_candidate


def read_notebook_bom_safe(path: Path | str) -> NotebookNode:
    notebook_path = Path(path)
    try:
        with notebook_path.open("r", encoding="utf-8-sig") as handle:
            return nbformat.read(handle, as_version=4)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse notebook `{notebook_path}`: {exc}") from exc


def write_notebook_utf8(path: Path | str, notebook: NotebookNode) -> None:
    notebook_path = Path(path)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(notebook_path.parent),
        delete=False,
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        nbformat.write(notebook, handle)
    temp_path.replace(notebook_path)


def create_minimal_notebook(title: str) -> NotebookNode:
    return nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(f"# {title}"),
            nbformat.v4.new_code_cell("print('ready')"),
        ]
    )


def _set_notebook_runtime_env(workspace_root: Path) -> dict[str, str | None]:
    runtime_root = workspace_root / ".labai_notebook_runtime"
    overrides = {
        "IPYTHONDIR": runtime_root / "ipython",
        "JUPYTER_RUNTIME_DIR": runtime_root / "jupyter_runtime",
        "MPLCONFIGDIR": runtime_root / "mpl",
    }
    previous: dict[str, str | None] = {}
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        import jupyter_core.paths as jupyter_paths

        jupyter_paths.allow_insecure_writes = True
    except Exception:
        pass
    for key, value in overrides.items():
        value.mkdir(parents=True, exist_ok=True)
        previous[key] = os.environ.get(key)
        os.environ[key] = str(value)
    return previous


def _restore_notebook_runtime_env(previous: dict[str, str | None]) -> None:
    for key in _NOTEBOOK_RUNTIME_ENV_KEYS:
        original = previous.get(key)
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def _failing_cell_index(notebook: NotebookNode) -> int:
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []) or []:
            if output.get("output_type") == "error":
                return index
    return -1


def execute_notebook_in_workspace(
    notebook_path: Path | str,
    workspace_root: Path | str,
    timeout: int = 600,
    kernel_name: str = "python3",
) -> NotebookExecutionResult:
    if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    resolved_workspace = Path(workspace_root).expanduser().resolve()
    resolved_notebook = resolve_workspace_path(resolved_workspace, notebook_path)
    notebook = read_notebook_bom_safe(resolved_notebook)
    previous_env = _set_notebook_runtime_env(resolved_workspace)
    success = False
    error_name = ""
    error_value = ""
    try:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(resolved_workspace)}},
        )
        client.execute()
        success = True
    except Exception as exc:
        error_name = type(exc).__name__
        error_value = str(exc)
    finally:
        write_notebook_utf8(resolved_notebook, notebook)
        _restore_notebook_runtime_env(previous_env)
    executed_cell_count = sum(
        1
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and cell.get("execution_count") is not None
    )
    output_cell_count = sum(
        1
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and bool(cell.get("outputs"))
    )
    return NotebookExecutionResult(
        success=success,
        notebook_path=resolved_notebook,
        workspace_root=resolved_workspace,
        executed_cell_count=executed_cell_count,
        output_cell_count=output_cell_count,
        outputs_embedded=output_cell_count > 0,
        error_name=error_name,
        error_value=error_value,
        failing_cell_index=_failing_cell_index(notebook),
    )


def _iter_notebook_text_segments(notebook: NotebookNode) -> Iterable[str]:
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            yield "".join(str(item) for item in source)
        else:
            yield str(source)
        for output in cell.get("outputs", []) or []:
            text = output.get("text", "")
            if isinstance(text, list):
                yield "".join(str(item) for item in text)
            elif text:
                yield str(text)
            data = output.get("data", {}) or {}
            for key in ("text/plain", "text/html"):
                value = data.get(key, "")
                if isinstance(value, list):
                    yield "".join(str(item) for item in value)
                elif value:
                    yield str(value)


def notebook_has_embedded_outputs(path: Path | str) -> bool:
    notebook = read_notebook_bom_safe(path)
    return any(
        bool(cell.get("outputs"))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def notebook_contains_terms(path: Path | str, terms: Iterable[str]) -> dict[str, tuple[str, ...] | bool]:
    notebook = read_notebook_bom_safe(path)
    text = "\n".join(segment.lower() for segment in _iter_notebook_text_segments(notebook))
    missing = tuple(
        str(term)
        for term in terms
        if str(term).strip() and str(term).lower() not in text
    )
    return {
        "all_present": not missing,
        "missing_terms": missing,
    }


def validate_notebook_deliverable(
    path: Path | str,
    workspace_root: Path | str,
    required_terms: Iterable[str],
    require_outputs: bool = True,
    require_execution: bool = True,
) -> list[CriterionEvidence]:
    evidence: list[CriterionEvidence] = []
    resolved_notebook = resolve_workspace_path(workspace_root, path)
    evidence.append(
        CriterionEvidence(
            criterion="notebook_exists_in_workspace",
            passed=resolved_notebook.is_file(),
            detail=str(resolved_notebook),
        )
    )
    notebook = read_notebook_bom_safe(resolved_notebook)
    evidence.append(
        CriterionEvidence(
            criterion="notebook_parseable_with_nbformat",
            passed=True,
            detail=f"cell_count={len(notebook.get('cells', []))}",
        )
    )
    outputs_present = notebook_has_embedded_outputs(resolved_notebook)
    if require_outputs:
        evidence.append(
            CriterionEvidence(
                criterion="notebook_has_embedded_outputs",
                passed=outputs_present,
                detail=f"outputs_embedded={outputs_present}",
            )
        )
    if require_execution:
        evidence.append(
            CriterionEvidence(
                criterion="notebook_execution_evidence_present",
                passed=outputs_present,
                detail="embedded code-cell outputs required for executed deliverables",
            )
        )
    term_result = notebook_contains_terms(resolved_notebook, required_terms)
    evidence.append(
        CriterionEvidence(
            criterion="required_terms_present",
            passed=bool(term_result["all_present"]),
            detail=(
                "all required terms present"
                if term_result["all_present"]
                else f"missing_terms={list(term_result['missing_terms'])}"
            ),
        )
    )
    return evidence

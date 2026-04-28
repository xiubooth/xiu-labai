from __future__ import annotations

import ast
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from shutil import which


@dataclass(frozen=True)
class GrepAstStatus:
    available: bool
    backend: str
    detail: str = ""


@dataclass(frozen=True)
class GrepAstFileSummary:
    path: str
    backend: str
    top_level_functions: tuple[str, ...] = ()
    top_level_classes: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    keyword_hits: tuple[str, ...] = ()
    detail: str = ""
    parse_error: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


def summarize_python_file(path: Path, *, keywords: tuple[str, ...] = ()) -> GrepAstFileSummary:
    status = detect_grep_ast()
    text = path.read_text(encoding="utf-8")
    try:
        module = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return GrepAstFileSummary(
            path=path.as_posix(),
            backend="python_ast_fallback",
            detail=status.detail or "grep-ast unavailable; used Python AST fallback.",
            parse_error=str(exc),
        )

    functions: list[str] = []
    classes: list[str] = []
    imports: list[str] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name:
                imports.append(module_name)
    lowered_text = text.lower()
    keyword_hits = tuple(
        keyword
        for keyword in keywords
        if keyword and keyword.lower() in lowered_text
    )
    return GrepAstFileSummary(
        path=path.as_posix(),
        backend="grep_ast" if status.available else "python_ast_fallback",
        top_level_functions=tuple(dict.fromkeys(functions)),
        top_level_classes=tuple(dict.fromkeys(classes)),
        imports=tuple(dict.fromkeys(imports)),
        keyword_hits=tuple(dict.fromkeys(keyword_hits)),
        detail=status.detail or "Used Python AST fallback for symbol extraction.",
        metadata={
            "grep_ast_available": status.available,
            "grep_ast_backend": status.backend,
        },
    )


def detect_grep_ast() -> GrepAstStatus:
    if importlib.util.find_spec("grep_ast") is not None:
        return GrepAstStatus(
            available=True,
            backend="python_import",
            detail="grep-ast package is installed; LabAI still uses the stable Python AST fallback until a tighter adapter is needed.",
        )
    if which("grep-ast"):
        return GrepAstStatus(
            available=True,
            backend="cli",
            detail="grep-ast CLI is on PATH; LabAI still uses the stable Python AST fallback until a tighter adapter is needed.",
        )
    return GrepAstStatus(
        available=False,
        backend="python_ast_fallback",
        detail="grep-ast is not installed in this environment; Python AST fallback is active.",
    )

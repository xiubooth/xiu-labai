from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from labai.external import detect_grep_ast, summarize_python_file
from labai.task_manifest import TaskManifest


_IGNORED_DIRS = {
    ".git",
    ".labai",
    ".planning",
    ".pytest-tmp",
    ".pytest-work",
    "__pycache__",
    "node_modules",
}
_TRACKED_SUFFIXES = {".py", ".ipynb", ".toml", ".json", ".yaml", ".yml", ".md", ".txt"}


@dataclass(frozen=True)
class RepoMapEntry:
    relative_path: str
    file_type: str
    size_bytes: int
    top_level_functions: tuple[str, ...]
    top_level_classes: tuple[str, ...]
    imports: tuple[str, ...]
    key_symbol_names: tuple[str, ...]
    keyword_hits: tuple[str, ...]
    prompt_named: bool
    wildcard_match: bool
    reference_file: bool
    likely_owner_score: int
    last_modified_in_current_run: bool
    analysis_backend: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RepoMapResult:
    entries: tuple[RepoMapEntry, ...]
    grep_ast_backend: str
    grep_ast_detail: str

    def to_record(self) -> dict[str, object]:
        return {
            "grep_ast_backend": self.grep_ast_backend,
            "grep_ast_detail": self.grep_ast_detail,
            "entries": [entry.to_record() for entry in self.entries],
        }


def build_repo_map(
    workspace_root: Path,
    manifest: TaskManifest,
    *,
    modified_in_run: tuple[str, ...] = (),
) -> RepoMapResult:
    status = detect_grep_ast()
    prompt_keywords = _manifest_keywords(manifest)
    entries: list[RepoMapEntry] = []
    for path in _iter_repo_files(workspace_root):
        relative = path.relative_to(workspace_root).as_posix()
        if path.suffix.lower() == ".py":
            summary = summarize_python_file(path, keywords=prompt_keywords)
            top_level_functions = summary.top_level_functions
            top_level_classes = summary.top_level_classes
            imports = summary.imports
            keyword_hits = summary.keyword_hits
            backend = summary.backend
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            top_level_functions = ()
            top_level_classes = ()
            imports = ()
            keyword_hits = tuple(
                keyword
                for keyword in prompt_keywords
                if keyword and keyword.lower() in text.lower()
            )
            backend = status.backend
        key_symbols = tuple(dict.fromkeys((*top_level_functions, *top_level_classes)))[:12]
        entry = RepoMapEntry(
            relative_path=relative,
            file_type=path.suffix.lower().lstrip(".") or "no_ext",
            size_bytes=path.stat().st_size,
            top_level_functions=top_level_functions,
            top_level_classes=top_level_classes,
            imports=imports,
            key_symbol_names=key_symbols,
            keyword_hits=keyword_hits,
            prompt_named=relative in manifest.prompt_named_files,
            wildcard_match=relative in manifest.wildcard_matches,
            reference_file=relative in manifest.reference_files,
            likely_owner_score=_likely_owner_score(relative, manifest, key_symbols, keyword_hits),
            last_modified_in_current_run=relative in modified_in_run,
            analysis_backend=backend,
        )
        entries.append(entry)
    entries.sort(key=lambda item: (-item.likely_owner_score, item.relative_path))
    return RepoMapResult(
        entries=tuple(entries),
        grep_ast_backend=status.backend,
        grep_ast_detail=status.detail,
    )


def _iter_repo_files(workspace_root: Path):
    for path in workspace_root.rglob("*"):
        try:
            relative_parts = path.relative_to(workspace_root).parts
        except ValueError:
            relative_parts = path.parts
        if any(part in _IGNORED_DIRS for part in relative_parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in _TRACKED_SUFFIXES:
            continue
        yield path


def _manifest_keywords(manifest: TaskManifest) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", manifest.user_instruction)
    keywords: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in {
            "implement",
            "primary",
            "required",
            "behavior",
            "validate",
            "validation",
            "workspace",
            "source",
            "files",
            "inspect",
        }:
            continue
        if lowered not in keywords:
            keywords.append(lowered)
    return tuple(keywords[:24])


def _likely_owner_score(
    relative_path: str,
    manifest: TaskManifest,
    key_symbols: tuple[str, ...],
    keyword_hits: tuple[str, ...],
) -> int:
    score = 0
    lowered = relative_path.lower()
    if relative_path in manifest.primary_target_artifacts:
        score += 60
    if relative_path in manifest.prompt_named_files:
        score += 50
    if relative_path in manifest.candidate_owner_files:
        score += 35
    if relative_path in manifest.reference_files:
        score -= 15
    if relative_path in manifest.allowed_support_files:
        score -= 20
    if lowered.startswith("validation/") or "/validation/" in lowered or lowered.startswith("tests/"):
        score -= 30
    if lowered.endswith(".ipynb"):
        score += 20
    if lowered.endswith(".py"):
        score += 15
    score += min(len(keyword_hits) * 8, 32)
    if any(symbol.lower() in manifest.user_instruction.lower() for symbol in key_symbols):
        score += 10
    return score

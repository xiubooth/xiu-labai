from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

from labai.config import LabaiConfig


_PATH_QUOTE_PATTERN = re.compile(r'(?P<quote>["\'])(?P<value>.+?)(?P=quote)')
_WINDOWS_ABSOLUTE_FILE_PATTERN = re.compile(
    r"(?P<path>[A-Za-z]:\\[^\r\n\"<>|?*]*?\.(?:pdf|py|ipynb|md|toml|json|ya?ml|ps1|txt))",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_DIR_PATTERN = re.compile(
    r"(?P<path>[A-Za-z]:\\[^\s\r\n\"<>|?*]+)",
    re.IGNORECASE,
)
_RELATIVE_PATH_PATTERN = re.compile(
    r"(?P<token>(?:[A-Za-z0-9_\-.]+[\\/])+[A-Za-z0-9_\-.]+|[A-Za-z0-9_\-.]+\.(?:pdf|py|ipynb|md|toml|json|ya?ml|ps1|txt))"
)
_LOCKED_TARGETS_PATTERN = re.compile(
    r"^Locked target files:\s*$\n(?P<body>(?:- .+\n)+)",
    re.IGNORECASE | re.MULTILINE,
)
_FOCUSED_TARGETS_PATTERN = re.compile(
    r"^Focused files to revisit first:\s*$\n(?P<body>(?:- .+\n)+)",
    re.IGNORECASE | re.MULTILINE,
)
_DEFAULT_DENIED_PATHS = (
    r"%SystemRoot%",
    r"%ProgramFiles%",
    r"%ProgramFiles(x86)%",
    r"%ProgramData%",
)
_WORKSPACE_ROOT_MARKERS = (
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "README.md",
    "src",
    "tests",
)


class WorkspaceAccessError(RuntimeError):
    """Raised when a path is outside the configured workspace allowlist."""


@dataclass(frozen=True)
class WorkspaceAccessDecision:
    allowed: bool
    scope: str
    reason: str
    path: str


@dataclass(frozen=True)
class WorkspaceWriteResult:
    status: str
    path: str = ""
    operation: str = ""
    collision_suffix: int = 0
    fallback_reason: str = ""
    error: str = ""


class WorkspaceAccessManager:
    def __init__(self, config: LabaiConfig) -> None:
        self.config = config
        self.app_repo_root = config.project_root.resolve()
        self.active_workspace_root = config.workspace.active_workspace_root.resolve()
        self.allowed_workspace_roots = tuple(
            path.resolve() for path in config.workspace.allowed_workspace_roots
        )
        self.allowed_paper_roots = tuple(
            path.resolve() for path in config.workspace.allowed_paper_roots
        )
        configured_denied = tuple(path.resolve() for path in config.workspace.deny_roots)
        default_denied = tuple(_resolve_denied_path(raw_value) for raw_value in _DEFAULT_DENIED_PATHS)
        self.deny_roots = tuple(
            path
            for path in dict.fromkeys((*default_denied, *configured_denied))
            if path is not None
        )

    def read_roots(self) -> tuple[Path, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.active_workspace_root,
                    *self.allowed_workspace_roots,
                    *self.allowed_paper_roots,
                    self.app_repo_root,
                )
            )
        )

    def write_roots(self) -> tuple[Path, ...]:
        paper_write_roots: tuple[Path, ...] = (
            self.allowed_paper_roots if self.config.workspace.same_folder_deliverables else ()
        )
        return tuple(
            dict.fromkeys(
                (
                    self.active_workspace_root,
                    *self.allowed_workspace_roots,
                    *paper_write_roots,
                )
            )
        )

    def prompt_paths(self, prompt: str) -> tuple[str, ...]:
        discovered: list[str] = []
        seen: set[str] = set()
        for raw_path in _iter_prompt_candidates(prompt):
            resolved = self.resolve_prompt_path(raw_path, must_exist=False)
            if resolved is None:
                continue
            display = self.display_path(resolved)
            if display in seen:
                continue
            seen.add(display)
            discovered.append(display)
        return tuple(discovered[:8])

    def resolve_prompt_path(self, raw_path: str, *, must_exist: bool) -> Path | None:
        normalized = _normalize_input_path(raw_path)
        if not normalized:
            return None

        if _looks_like_absolute_path(normalized):
            if not self.config.workspace.allow_absolute_paths:
                return None
            candidate = Path(normalized).resolve()
            if must_exist and not candidate.exists():
                return None
            if self.is_allowed(candidate, for_write=False):
                return candidate
            return None

        for root in (*self.read_roots(),):
            candidate = (root / normalized).resolve()
            if must_exist and not candidate.exists():
                continue
            if self.is_allowed(candidate, for_write=False):
                return candidate
        return None

    def resolve_user_path(
        self,
        raw_path: str,
        *,
        for_write: bool,
        must_exist: bool | None = None,
    ) -> Path:
        normalized = _normalize_input_path(raw_path)
        if not normalized:
            raise WorkspaceAccessError("Path must not be empty.")

        if must_exist is None:
            must_exist = not for_write

        if _looks_like_absolute_path(normalized):
            if not self.config.workspace.allow_absolute_paths:
                raise WorkspaceAccessError("Absolute paths are disabled by workspace policy.")
            candidate = Path(normalized)
        else:
            candidate = self.active_workspace_root / normalized

        resolved = candidate.resolve()
        if must_exist and not resolved.exists():
            raise WorkspaceAccessError(f"Path does not exist: {resolved}")
        if not self.is_allowed(resolved, for_write=for_write):
            decision = self.describe_path(resolved, for_write=for_write)
            raise WorkspaceAccessError(decision.reason)
        return resolved

    def resolve_explicit_read_root(self, raw_path: str) -> Path | None:
        normalized = _normalize_input_path(raw_path)
        if not normalized or not _looks_like_absolute_path(normalized):
            return None

        candidate = Path(normalized).resolve()
        if not candidate.exists():
            return None
        if self._is_denied(candidate):
            raise WorkspaceAccessError(f"Path is blocked by the deny-root policy: {candidate}")

        if candidate.is_file():
            if candidate.suffix.lower() == ".pdf":
                return None
            candidate = candidate.parent.resolve()

        return self._infer_workspace_root(candidate)

    def is_allowed(self, path: Path, *, for_write: bool) -> bool:
        if self._is_denied(path):
            return False
        roots = self.write_roots() if for_write else self.read_roots()
        return any(_is_within(path, root) for root in roots)

    def describe_path(self, path: Path, *, for_write: bool) -> WorkspaceAccessDecision:
        resolved = path.resolve()
        if self._is_denied(resolved):
            return WorkspaceAccessDecision(
                allowed=False,
                scope="denied",
                reason=f"Path is blocked by the deny-root policy: {resolved}",
                path=str(resolved),
            )
        roots = self.write_roots() if for_write else self.read_roots()
        for root in roots:
            if _is_within(resolved, root):
                return WorkspaceAccessDecision(
                    allowed=True,
                    scope=self._scope_name(root),
                    reason=(
                        f"Path is within the {'write' if for_write else 'read'} allowlist root {root}"
                    ),
                    path=self.display_path(resolved),
                )
        return WorkspaceAccessDecision(
            allowed=False,
            scope="blocked",
            reason=(
                f"Path stays outside the configured {'write' if for_write else 'read'} roots: {resolved}"
            ),
            path=str(resolved),
        )

    def display_path(self, path: Path) -> str:
        resolved = path.resolve()
        if _is_within(resolved, self.active_workspace_root):
            relative = resolved.relative_to(self.active_workspace_root)
            return relative.as_posix() or "."
        if _is_within(resolved, self.app_repo_root):
            relative = resolved.relative_to(self.app_repo_root)
            return relative.as_posix() or "."
        return str(resolved)

    def origin(self, path: Path) -> str:
        resolved = path.resolve()
        if _is_within(resolved, self.active_workspace_root):
            return "active_workspace"
        if any(_is_within(resolved, root) for root in self.allowed_workspace_roots):
            return "allowed_workspace_root"
        if any(_is_within(resolved, root) for root in self.allowed_paper_roots):
            return "allowed_paper_root"
        if _is_within(resolved, self.app_repo_root):
            return "app_repo_root"
        return "external"

    def write_text_file(
        self,
        raw_path: str,
        content: str,
        *,
        overwrite: bool,
    ) -> WorkspaceWriteResult:
        target = self.resolve_user_path(raw_path, for_write=True, must_exist=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            final_target, collision_suffix = _resolve_collision(target)
        else:
            final_target = target
            collision_suffix = 0

        try:
            final_target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return WorkspaceWriteResult(
                status="error",
                path=self.display_path(final_target),
                operation="write_text_file",
                collision_suffix=collision_suffix,
                error=str(exc),
            )

        return WorkspaceWriteResult(
            status="ok",
            path=self.display_path(final_target),
            operation="write_text_file",
            collision_suffix=collision_suffix,
        )

    def _is_denied(self, path: Path) -> bool:
        resolved = path.resolve()
        return any(_is_within(resolved, denied_root) for denied_root in self.deny_roots)

    def _scope_name(self, root: Path) -> str:
        if root == self.active_workspace_root:
            return "active_workspace_root"
        if root == self.app_repo_root:
            return "app_repo_root"
        if root in self.allowed_paper_roots:
            return "allowed_paper_root"
        return "allowed_workspace_root"

    def _infer_workspace_root(self, candidate: Path) -> Path:
        current = candidate.resolve()
        while True:
            if self._looks_like_workspace_root(current):
                return current
            parent = current.parent
            if parent == current or self._is_denied(parent):
                return current
            current = parent

    def _looks_like_workspace_root(self, candidate: Path) -> bool:
        if not candidate.is_dir():
            return False
        return any((candidate / marker).exists() for marker in _WORKSPACE_ROOT_MARKERS)


def _resolve_collision(path: Path) -> tuple[Path, int]:
    suffix = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate, suffix
        suffix += 1


def _iter_prompt_candidates(prompt: str) -> tuple[str, ...]:
    required_file_targets = _extract_required_file_block_candidates(prompt)
    if required_file_targets:
        return required_file_targets
    locked_targets = _extract_locked_prompt_candidates(prompt)
    if locked_targets:
        return locked_targets
    focused_targets = _extract_focused_prompt_candidates(prompt)
    if focused_targets:
        return focused_targets

    candidates: list[str] = []
    for match in _PATH_QUOTE_PATTERN.finditer(prompt):
        value = _sanitize_relative_prompt_candidate(match.group("value"))
        if _looks_like_path_candidate(value):
            _append_candidate(candidates, value)
    for match in _WINDOWS_ABSOLUTE_FILE_PATTERN.finditer(prompt):
        _append_candidate(candidates, _sanitize_relative_prompt_candidate(match.group("path")))
    for match in _WINDOWS_ABSOLUTE_DIR_PATTERN.finditer(prompt):
        _append_candidate(candidates, _sanitize_relative_prompt_candidate(match.group("path")))
    for match in _RELATIVE_PATH_PATTERN.finditer(prompt):
        token = _sanitize_relative_prompt_candidate(match.group("token"))
        if _is_nested_relative_candidate(token, candidates):
            continue
        if _looks_like_path_candidate(token):
            _append_candidate(candidates, token)
    return tuple(dict.fromkeys(item for item in candidates if item))


def _extract_required_file_block_candidates(prompt: str) -> tuple[str, ...]:
    if not re.search(r"^Required FILE blocks? this round:\s*$", prompt, re.IGNORECASE | re.MULTILINE):
        return ()

    candidates: list[str] = []
    for match in re.finditer(
        r"^(?:===|#{1,6})\s*FILE:\s*(?P<path>.+?)(?:\s*===)?\s*$",
        prompt,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        token = _sanitize_relative_prompt_candidate(match.group("path"))
        if not token or not _looks_like_path_candidate(token):
            continue
        _append_candidate(candidates, token)
    return tuple(candidates)


def _extract_locked_prompt_candidates(prompt: str) -> tuple[str, ...]:
    match = _LOCKED_TARGETS_PATTERN.search(prompt)
    if match is None:
        return ()

    return _extract_bulleted_prompt_candidates(match.group("body"))


def _extract_focused_prompt_candidates(prompt: str) -> tuple[str, ...]:
    match = _FOCUSED_TARGETS_PATTERN.search(prompt)
    if match is None:
        return ()

    return _extract_bulleted_prompt_candidates(match.group("body"))


def _extract_bulleted_prompt_candidates(body: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for raw_line in body.splitlines():
        if not raw_line.startswith("- "):
            continue
        token = _sanitize_relative_prompt_candidate(raw_line[2:])
        if not token or not _looks_like_path_candidate(token):
            continue
        _append_candidate(candidates, token)
    return tuple(candidates)


def _looks_like_path_candidate(value: str) -> bool:
    lowered = value.lower()
    if any(
        lowered.endswith(suffix)
        for suffix in (".pdf", ".py", ".ipynb", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".ps1")
    ):
        return True
    return "\\" in value or "/" in value


def _sanitize_prompt_candidate(value: str) -> str:
    sanitized = value.strip().strip("`'\"()[]{}<>.,:;!?，。；：！？、）】》」』")
    file_match = re.match(
        r"^(?P<path>.+?\.(?:pdf|py|ipynb|md|toml|json|ya?ml|ps1|txt))(?=$|[^\w./\\-])",
        sanitized,
        re.IGNORECASE,
    )
    if file_match:
        return file_match.group("path")
    return sanitized


def _append_candidate(candidates: list[str], candidate: str) -> None:
    if candidate and candidate not in candidates:
        candidates.append(candidate)


def _sanitize_relative_prompt_candidate(value: str) -> str:
    sanitized = value.strip().strip("`'\"()[]{}<>,:;!?锛屻€傦紱锛氾紒锛熴€侊級銆戙€嬨€嶃€?")
    file_match = re.match(
        r"^(?P<path>.+?\.(?:pdf|py|ipynb|md|toml|json|ya?ml|ps1|txt))(?=$|[^\w./\\-])",
        sanitized,
        re.IGNORECASE,
    )
    if file_match:
        return file_match.group("path")
    return sanitized


def _is_nested_relative_candidate(token: str, candidates: list[str]) -> bool:
    token_normalized = token.replace("\\", "/").lstrip("./").lower()
    if not token_normalized:
        return False
    return any(
        re.match(r"^[a-z]:[\\/]", existing, re.IGNORECASE)
        and existing.replace("\\", "/").lower().endswith(token_normalized)
        for existing in candidates
    )


def _normalize_input_path(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value.strip().strip("\"'")))


def _looks_like_absolute_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or Path(value).is_absolute()


def _resolve_denied_path(raw_value: str) -> Path | None:
    normalized = _normalize_input_path(raw_value)
    if not normalized:
        return None
    candidate = Path(normalized)
    if not candidate.is_absolute():
        return None
    return candidate.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False

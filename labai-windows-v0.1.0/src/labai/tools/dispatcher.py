from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

from labai.workspace import WorkspaceAccessManager

from .registry import ToolSpec, get_tool_spec


IGNORED_PATH_PARTS = frozenset({".git", ".venv", "__pycache__"})
IGNORED_PATH_PREFIXES = ("pytest-cache-files-",)
IGNORED_RELATIVE_ROOTS = (
    Path(".labai") / "audit",
    Path(".labai") / "sessions",
    Path(".pip-tmp"),
    Path(".pytest-workspaces"),
    Path(".pytest_cache"),
)


class ToolExecutionError(RuntimeError):
    """Raised when tool execution fails validation or cannot complete."""


class ToolValidationError(ToolExecutionError):
    """Raised when tool inputs are invalid or outside the safe repo scope."""


class ToolExecutionNotReadyError(RuntimeError):
    """Backward-compatible alias for the old scaffold-only dispatcher error."""


class ToolDispatcher:
    def __init__(self, access_root: Path | WorkspaceAccessManager) -> None:
        if isinstance(access_root, WorkspaceAccessManager):
            self.access_manager = access_root
            self.default_root = access_root.active_workspace_root
            self.app_repo_root = access_root.app_repo_root
        else:
            self.access_manager = None
            self.default_root = Path(access_root).resolve()
            self.app_repo_root = self.default_root

    def execute(self, tool_name: str, **kwargs: object) -> object:
        spec = get_tool_spec(tool_name)
        if not spec.stub_callable:
            raise ToolValidationError(f"Tool '{tool_name}' is not stub-callable.")

        if tool_name == "read_text_file":
            return self._read_text_file(
                spec,
                path=self._coerce_string(kwargs.get("path"), "path"),
            )
        if tool_name == "list_directory":
            return self._list_directory(
                spec,
                path=self._coerce_string(kwargs.get("path", "."), "path"),
            )
        if tool_name == "find_files":
            return self._find_files(
                spec,
                pattern=self._coerce_string(kwargs.get("pattern"), "pattern"),
                path=self._coerce_string(kwargs.get("path", "."), "path"),
            )
        raise ToolValidationError(f"Unsupported tool '{tool_name}'.")

    def _read_text_file(self, spec: ToolSpec, *, path: str) -> dict[str, Any]:
        target = self._resolve_path(path)
        if not target.is_file():
            raise ToolValidationError(f"Expected a file path, got: {target}")
        return {
            "tool": spec.name,
            "path": self._display_path(target),
            "text": target.read_text(encoding="utf-8"),
        }

    def _list_directory(self, spec: ToolSpec, *, path: str) -> dict[str, Any]:
        target = self._resolve_path(path)
        if not target.is_dir():
            raise ToolValidationError(f"Expected a directory path, got: {target}")

        entries: list[dict[str, str]] = []
        for child in sorted(
            target.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.lower()),
        ):
            if self._is_ignored_path(child):
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": self._display_path(child),
                    "type": "directory" if child.is_dir() else "file",
                }
            )

        return {
            "tool": spec.name,
            "path": self._display_path(target),
            "entries": entries,
        }

    def _find_files(self, spec: ToolSpec, *, pattern: str, path: str) -> dict[str, Any]:
        start = self._resolve_path(path)
        if not start.is_dir():
            raise ToolValidationError(f"Expected a directory path, got: {start}")

        matches: list[str] = []
        for current_root, dirnames, filenames in os.walk(start):
            current_path = Path(current_root)
            dirnames[:] = [
                name
                for name in sorted(dirnames, key=str.lower)
                if not self._is_ignored_path(current_path / name)
            ]
            for filename in sorted(filenames, key=str.lower):
                candidate = current_path / filename
                if self._is_ignored_path(candidate):
                    continue
                relative_path = self._display_path(candidate)
                if PurePosixPath(relative_path).match(pattern):
                    matches.append(relative_path)

        return {
            "tool": spec.name,
            "path": self._display_path(start),
            "pattern": pattern,
            "matches": matches,
        }

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.default_root / candidate
        resolved = candidate.resolve()
        self._ensure_allowed(resolved)
        if resolved != self.default_root and self._is_ignored_path(resolved):
            raise ToolValidationError(
                f"Path is excluded from tool access: {self._display_path(resolved)}"
            )
        return resolved

    def _ensure_allowed(self, path: Path) -> None:
        if self.access_manager is not None:
            if not self.access_manager.is_allowed(path, for_write=False):
                raise ToolValidationError(
                    f"Path must stay within the configured workspace read roots: {path}"
                )
            return
        try:
            path.relative_to(self.default_root)
        except ValueError as exc:
            raise ToolValidationError(
                f"Path must stay within the repository root: {path}"
            ) from exc

    def _display_path(self, path: Path) -> str:
        if self.access_manager is not None:
            return self.access_manager.display_path(path)
        relative = path.relative_to(self.default_root)
        return relative.as_posix() or "."

    def _is_ignored_path(self, path: Path) -> bool:
        if path == self.default_root:
            return False

        base_root = self.default_root if self._is_within(path, self.default_root) else self.app_repo_root
        try:
            relative = path.relative_to(base_root)
        except ValueError:
            return False
        if any(part in IGNORED_PATH_PARTS for part in relative.parts):
            return True
        if any(part.startswith(prefix) for part in relative.parts for prefix in IGNORED_PATH_PREFIXES):
            return True

        return any(
            relative == ignored_root or ignored_root in relative.parents
            for ignored_root in IGNORED_RELATIVE_ROOTS
        )

    def _is_within(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _coerce_string(self, value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise ToolValidationError(f"'{field_name}' must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ToolValidationError(f"'{field_name}' must not be empty.")
        return normalized

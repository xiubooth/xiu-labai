from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    read_only: bool = True
    stub_callable: bool = True
    safe_scope: str = "workspace"
    parameters: tuple[str, ...] = field(default_factory=tuple)


_TOOL_SPECS = (
    ToolSpec(
        name="read_text_file",
        description="Read a UTF-8 text file from the active workspace or another allowlisted root.",
        parameters=("path",),
    ),
    ToolSpec(
        name="list_directory",
        description="List files and directories under the active workspace or another allowlisted root.",
        parameters=("path",),
    ),
    ToolSpec(
        name="find_files",
        description="Find files by glob pattern under the active workspace or another allowlisted root.",
        parameters=("pattern", "path"),
    ),
)


def list_tool_specs() -> tuple[ToolSpec, ...]:
    return _TOOL_SPECS


def get_tool_spec(name: str) -> ToolSpec:
    for spec in _TOOL_SPECS:
        if spec.name == name:
            return spec
    available = ", ".join(spec.name for spec in _TOOL_SPECS)
    raise KeyError(f"Unknown tool '{name}'. Available tools: {available}")

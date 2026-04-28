from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Literal


LabaiPlatform = Literal["windows", "macos", "linux", "unknown"]


@dataclass(frozen=True)
class LabaiPlatformPaths:
    platform: LabaiPlatform
    app_support_dir: Path
    runtime_dir: Path
    managed_claw_path: Path
    launcher_dir: Path
    venv_python_path: Path
    temp_root: Path


def detect_platform(raw_platform: str | None = None) -> LabaiPlatform:
    value = (raw_platform or sys.platform).lower()
    if value.startswith("win"):
        return "windows"
    if value == "darwin" or value.startswith("mac"):
        return "macos"
    if value.startswith("linux"):
        return "linux"
    return "unknown"


def get_platform_paths(
    *,
    project_root: Path | str = ".",
    platform: LabaiPlatform | str | None = None,
    home: Path | str | None = None,
    local_appdata: Path | str | None = None,
) -> LabaiPlatformPaths:
    resolved_platform = detect_platform(str(platform)) if platform is not None else detect_platform()
    root = Path(project_root).resolve()
    user_home = Path(home).expanduser() if home is not None else Path.home()

    if resolved_platform == "windows":
        app_support = Path(
            local_appdata
            or os.environ.get("LOCALAPPDATA")
            or (user_home / "AppData" / "Local")
        ) / "LabAI"
        managed_claw = app_support / "runtime" / "claw" / "claw.exe"
        venv_python = root / ".venv" / "Scripts" / "python.exe"
    elif resolved_platform == "macos":
        app_support = user_home / "Library" / "Application Support" / "LabAI"
        managed_claw = app_support / "runtime" / "claw" / "claw"
        venv_python = root / ".venv" / "bin" / "python"
    else:
        app_support = user_home / ".local" / "share" / "LabAI"
        managed_claw = app_support / "runtime" / "claw" / "claw"
        venv_python = root / ".venv" / "bin" / "python"

    return LabaiPlatformPaths(
        platform=resolved_platform,
        app_support_dir=app_support,
        runtime_dir=app_support / "runtime",
        managed_claw_path=managed_claw,
        launcher_dir=app_support / "bin",
        venv_python_path=venv_python,
        temp_root=root / ".labai" / "temp",
    )


def format_path_for_config(path: Path) -> str:
    return path.as_posix()

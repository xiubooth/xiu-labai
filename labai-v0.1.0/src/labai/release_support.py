from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile


RELEASE_VERSION = "0.1.0"
RELEASE_ARCHIVE_NAME = f"labai-v{RELEASE_VERSION}.zip"
DEFAULT_RELEASE_IGNORE_PATH = Path(".releaseignore")

ALLOWED_RELEASE_TOP_LEVEL_FILES = frozenset(
    {
        "Launch-LabAI-Setup.cmd",
        "README.md",
        ".env.example",
        "pyproject.toml",
        "RELEASE_MANIFEST.md",
        "RELEASE_CHECKLIST.md",
    }
)
ALLOWED_RELEASE_TOP_LEVEL_DIRS = frozenset(
    {
        ".continue",
        "docs",
        "runtime-assets",
        "scripts",
        "src",
        "templates",
    }
)

ALWAYS_EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
}
ALWAYS_EXCLUDED_FILE_NAMES = {
    ".env",
    ".env.local",
}
ALWAYS_EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
}

DEFAULT_CLEANUP_RULES: tuple[tuple[str, str], ...] = (
    (".labai/sessions", "generated session traces"),
    (".labai/audit", "generated audit logs"),
    (".labai/outputs", "generated output artifacts"),
    (".labai/library", "generated paper-library manifests, extracts, chunks, and indexes"),
    (".labai/runtime", "generated runtime bridge state"),
    (".labai/temp", "temporary runtime smoke workspaces"),
    (".pytest-workspaces", "pytest-created isolated workspaces"),
    (".pytest_cache", "pytest cache"),
    (".pip-tmp", "temporary pip build directories"),
    ("build", "Python build output"),
    ("dist", "release build output"),
    (".tmp*", "temporary debug workspaces"),
    ("pytest-cache-files-*", "pytest temporary cache shards"),
    ("**/__pycache__", "Python bytecode cache directories"),
    (".planning/phases/*/evaluation-copies", "generated phase evaluation copies"),
    (".planning/phases/*/generated-fixtures", "generated phase fixtures"),
    (".planning/phases/*/live-logs", "generated live logs"),
)

STATIC_REQUIRED_RELEASE_FILES: tuple[str, ...] = (
    "Launch-LabAI-Setup.cmd",
    "README.md",
    ".env.example",
    "pyproject.toml",
    "RELEASE_MANIFEST.md",
    "RELEASE_CHECKLIST.md",
    "docs/INSTALL_WINDOWS.md",
    "docs/FIRST_RUN.md",
    "docs/PROFILES.md",
    "docs/API_PROVIDERS.md",
    "docs/TROUBLESHOOTING_INSTALL.md",
    "docs/external-agent-patterns-route1.md",
    "docs/external-agent-patterns-route2.md",
    "scripts/windows/bootstrap-windows.ps1",
    "scripts/windows/install-labai.ps1",
    "scripts/windows/setup-local-ollama.ps1",
    "scripts/windows/setup-api-provider.ps1",
    "scripts/windows/switch-profile.ps1",
    "scripts/windows/verify-install.ps1",
    "runtime-assets/claw/windows-x64/claw.exe",
    "runtime-assets/claw/windows-x64/README.md",
    "templates/profiles/local.toml",
    "templates/profiles/api-deepseek.toml",
    "templates/profiles/fallback.toml",
)


@dataclass(frozen=True)
class CleanupTarget:
    relative_path: str
    absolute_path: Path
    reason: str
    is_dir: bool


@dataclass(frozen=True)
class ReleaseBuildResult:
    archive_path: Path
    staging_root: Path
    included_files: tuple[str, ...]


def normalize_repo_root(path: Path | str) -> Path:
    return Path(path).resolve()


def load_release_ignore_patterns(
    repo_root: Path,
    *,
    release_ignore_path: Path | None = None,
) -> tuple[str, ...]:
    path = (repo_root / (release_ignore_path or DEFAULT_RELEASE_IGNORE_PATH)).resolve()
    if not path.is_file():
        return ()
    patterns: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line.replace("\\", "/"))
    return tuple(patterns)


def build_cleanup_targets(repo_root: Path) -> list[CleanupTarget]:
    resolved_root = normalize_repo_root(repo_root)
    targets: dict[str, CleanupTarget] = {}
    for pattern, reason in DEFAULT_CLEANUP_RULES:
        for matched_path in sorted(_iter_rule_matches(resolved_root, pattern)):
            if not matched_path.exists():
                continue
            relative_path = matched_path.relative_to(resolved_root).as_posix()
            targets[relative_path] = CleanupTarget(
                relative_path=relative_path,
                absolute_path=matched_path,
                reason=reason,
                is_dir=matched_path.is_dir(),
            )
    return sorted(targets.values(), key=lambda item: item.relative_path)


def apply_cleanup_targets(targets: list[CleanupTarget]) -> list[str]:
    removed: list[str] = []
    for target in sorted(targets, key=lambda item: (len(item.relative_path.split("/")), item.relative_path), reverse=True):
        if not target.absolute_path.exists():
            continue
        if target.absolute_path.is_dir():
            shutil.rmtree(target.absolute_path)
        else:
            target.absolute_path.unlink()
        removed.append(target.relative_path)
    return removed


def collect_release_files(
    repo_root: Path,
    *,
    release_ignore_path: Path | None = None,
) -> list[Path]:
    resolved_root = normalize_repo_root(repo_root)
    patterns = load_release_ignore_patterns(resolved_root, release_ignore_path=release_ignore_path)
    included: list[Path] = []

    for current_root, dirnames, filenames in os.walk(resolved_root):
        root_path = Path(current_root)
        relative_root = root_path.relative_to(resolved_root)
        filtered_dirs: list[str] = []
        for dirname in dirnames:
            relative_dir = _build_relative_posix_path(relative_root, dirname)
            if should_exclude_release_path(relative_dir, is_dir=True, patterns=patterns):
                continue
            filtered_dirs.append(dirname)
        dirnames[:] = filtered_dirs

        for filename in filenames:
            relative_file = _build_relative_posix_path(relative_root, filename)
            if should_exclude_release_path(relative_file, is_dir=False, patterns=patterns):
                continue
            included.append((resolved_root / Path(relative_file)).resolve())

    return sorted(included, key=lambda path: path.relative_to(resolved_root).as_posix())


def validate_release_file_set(
    repo_root: Path,
    files: list[Path],
    *,
    release_ignore_path: Path | None = None,
) -> None:
    resolved_root = normalize_repo_root(repo_root)
    patterns = load_release_ignore_patterns(resolved_root, release_ignore_path=release_ignore_path)
    relative_files = {path.relative_to(resolved_root).as_posix() for path in files}
    missing = [path for path in required_release_files(resolved_root) if path not in relative_files]
    if missing:
        raise RuntimeError(
            "Release package is missing required files: " + ", ".join(missing)
        )

    forbidden = [
        relative_path
        for relative_path in sorted(relative_files)
        if should_exclude_release_path(relative_path, is_dir=False, patterns=patterns)
    ]
    if forbidden:
        raise RuntimeError(
            "Release package still includes excluded paths: " + ", ".join(forbidden)
        )


def stage_release_tree(
    repo_root: Path,
    staging_root: Path,
    *,
    release_ignore_path: Path | None = None,
) -> tuple[str, ...]:
    resolved_root = normalize_repo_root(repo_root)
    resolved_staging = normalize_repo_root(staging_root)
    if resolved_staging.exists():
        shutil.rmtree(resolved_staging)
    resolved_staging.mkdir(parents=True, exist_ok=True)

    files = collect_release_files(resolved_root, release_ignore_path=release_ignore_path)
    validate_release_file_set(
        resolved_root,
        files,
        release_ignore_path=release_ignore_path,
    )

    copied: list[str] = []
    for source_path in files:
        relative_path = source_path.relative_to(resolved_root)
        destination_path = resolved_staging / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied.append(relative_path.as_posix())
    return tuple(copied)


def create_release_archive(
    repo_root: Path,
    archive_path: Path,
    *,
    staging_root: Path,
    release_ignore_path: Path | None = None,
) -> ReleaseBuildResult:
    resolved_root = normalize_repo_root(repo_root)
    resolved_archive = normalize_repo_root(archive_path)
    resolved_archive.parent.mkdir(parents=True, exist_ok=True)
    copied = stage_release_tree(
        resolved_root,
        staging_root,
        release_ignore_path=release_ignore_path,
    )
    if resolved_archive.exists():
        resolved_archive.unlink()
    with zipfile.ZipFile(resolved_archive, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_path in copied:
            archive.write((normalize_repo_root(staging_root) / relative_path), arcname=relative_path)
    validate_release_archive(
        resolved_archive,
        repo_root=resolved_root,
        release_ignore_path=release_ignore_path,
    )
    return ReleaseBuildResult(
        archive_path=resolved_archive,
        staging_root=normalize_repo_root(staging_root),
        included_files=copied,
    )


def validate_release_archive(
    archive_path: Path,
    *,
    release_ignore_path: Path | None = None,
    repo_root: Path | None = None,
) -> None:
    resolved_archive = normalize_repo_root(archive_path)
    if not resolved_archive.is_file():
        raise RuntimeError(f"Release archive does not exist: {resolved_archive}")
    patterns = ()
    if repo_root is not None:
        patterns = load_release_ignore_patterns(
            normalize_repo_root(repo_root),
            release_ignore_path=release_ignore_path,
        )

    with zipfile.ZipFile(resolved_archive, mode="r") as archive:
        names = tuple(sorted(name.rstrip("/") for name in archive.namelist() if name and not name.endswith("/")))
    required_files = required_release_files(normalize_repo_root(repo_root)) if repo_root is not None else STATIC_REQUIRED_RELEASE_FILES
    missing = [path for path in required_files if path not in names]
    if missing:
        raise RuntimeError(
            "Release archive is missing required files: " + ", ".join(missing)
        )
    if patterns:
        forbidden = [
            name
            for name in names
            if should_exclude_release_path(name, is_dir=False, patterns=patterns)
        ]
        if forbidden:
            raise RuntimeError(
                "Release archive includes forbidden paths: " + ", ".join(forbidden)
            )


def should_exclude_release_path(
    relative_path: str | PurePosixPath,
    *,
    is_dir: bool,
    patterns: tuple[str, ...],
) -> bool:
    normalized = str(relative_path).replace("\\", "/").strip("/")
    if not normalized:
        return False
    candidate = PurePosixPath(normalized)

    if not _is_allowed_release_root(candidate):
        return True
    if not _is_allowed_continue_path(candidate, is_dir=is_dir):
        return True

    if any(part in ALWAYS_EXCLUDED_DIR_NAMES for part in candidate.parts):
        return True
    if candidate.name in ALWAYS_EXCLUDED_FILE_NAMES:
        return True
    if candidate.suffix in ALWAYS_EXCLUDED_SUFFIXES:
        return True

    for pattern in patterns:
        if _matches_release_pattern(candidate, pattern, is_dir=is_dir):
            return True
    return False


def _matches_release_pattern(candidate: PurePosixPath, pattern: str, *, is_dir: bool) -> bool:
    normalized_pattern = pattern.replace("\\", "/").strip()
    if not normalized_pattern:
        return False

    candidate_text = candidate.as_posix()
    if normalized_pattern.endswith("/"):
        base_pattern = normalized_pattern.rstrip("/")
        if fnmatch(candidate_text, base_pattern):
            return True
        if candidate_text == base_pattern or candidate_text.startswith(base_pattern + "/"):
            return True
        return any(fnmatch(part, base_pattern) for part in candidate.parts)

    if fnmatch(candidate_text, normalized_pattern):
        return True
    if fnmatch(candidate.name, normalized_pattern):
        return True
    if is_dir and (candidate_text == normalized_pattern or candidate_text.startswith(normalized_pattern + "/")):
        return True
    return False


def _iter_rule_matches(repo_root: Path, pattern: str) -> list[Path]:
    normalized_pattern = pattern.replace("\\", "/")
    if normalized_pattern.startswith("**/"):
        subpattern = normalized_pattern.split("**/", 1)[1]
        return list(repo_root.rglob(subpattern))
    if any(token in normalized_pattern for token in ("*", "?", "[")):
        return list(repo_root.glob(normalized_pattern))
    candidate = repo_root / Path(normalized_pattern)
    return [candidate] if candidate.exists() else []


def _build_relative_posix_path(relative_root: Path, leaf_name: str) -> str:
    if relative_root == Path("."):
        return leaf_name.replace("\\", "/")
    return PurePosixPath(relative_root.as_posix(), leaf_name).as_posix()


def discover_release_source_files(repo_root: Path) -> tuple[str, ...]:
    resolved_root = normalize_repo_root(repo_root)
    source_root = resolved_root / "src" / "labai"
    if not source_root.is_dir():
        return ()
    source_files = sorted(
        path.relative_to(resolved_root).as_posix()
        for path in source_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    return tuple(source_files)


def discover_release_continue_checks(repo_root: Path) -> tuple[str, ...]:
    resolved_root = normalize_repo_root(repo_root)
    checks_root = resolved_root / ".continue" / "checks"
    if not checks_root.is_dir():
        return ()
    check_files = sorted(
        path.relative_to(resolved_root).as_posix()
        for path in checks_root.rglob("*.md")
    )
    return tuple(check_files)


def required_release_files(repo_root: Path) -> tuple[str, ...]:
    resolved_root = normalize_repo_root(repo_root)
    required = set(STATIC_REQUIRED_RELEASE_FILES)
    required.update(discover_release_source_files(resolved_root))
    required.update(discover_release_continue_checks(resolved_root))
    return tuple(sorted(required))


def _is_allowed_release_root(candidate: PurePosixPath) -> bool:
    root = candidate.parts[0]
    if root in ALLOWED_RELEASE_TOP_LEVEL_DIRS:
        return True
    return len(candidate.parts) == 1 and root in ALLOWED_RELEASE_TOP_LEVEL_FILES


def _is_allowed_continue_path(candidate: PurePosixPath, *, is_dir: bool) -> bool:
    if candidate.parts[0] != ".continue":
        return True
    if len(candidate.parts) == 1:
        return True
    if candidate.parts[1] != "checks":
        return False
    if is_dir or len(candidate.parts) == 2:
        return True
    return candidate.suffix == ".md"

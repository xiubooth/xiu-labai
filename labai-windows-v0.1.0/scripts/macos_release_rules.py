from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import tomllib
import zipfile


ROOT_FILES: tuple[str, ...] = (
    ".env.example",
    "Launch-LabAI-Setup.command",
    "README.md",
    "RELEASE_MACOS_CHECKLIST.md",
    "RELEASE_MACOS_MANIFEST.md",
    "pyproject.toml",
)
MACOS_RELEASE_README_SOURCE = "docs/release/README_MACOS.md"

MACOS_DOCS: tuple[str, ...] = (
    "docs/API_PROVIDERS.md",
    "docs/FIRST_RUN.md",
    "docs/INSTALL_MAC.md",
    "docs/MAC_SMOKE_TEST.md",
    "docs/PROFILES.md",
    "docs/TROUBLESHOOTING_MAC.md",
)

MACOS_SCRIPTS: tuple[str, ...] = (
    "scripts/mac/build-claw-macos.sh",
    "scripts/mac/bootstrap-mac.sh",
    "scripts/mac/install-labai.sh",
    "scripts/mac/setup-api-provider.sh",
    "scripts/mac/setup-local-ollama.sh",
    "scripts/mac/verify-install.sh",
)

MACOS_TEMPLATES: tuple[str, ...] = (
    "templates/profiles/api-deepseek-mac.toml",
    "templates/profiles/fallback-mac.toml",
    "templates/profiles/local-mac.toml",
)

MACOS_CLAW_READMES: tuple[str, ...] = (
    "runtime-assets/claw/macos-arm64/README.md",
    "runtime-assets/claw/macos-x64/README.md",
)

MACOS_CLAW_BINARIES: tuple[str, ...] = (
    "runtime-assets/claw/macos-arm64/claw",
    "runtime-assets/claw/macos-x64/claw",
)

MACOS_CLAW_SHA256S: tuple[str, ...] = (
    "runtime-assets/claw/macos-arm64/claw.sha256",
    "runtime-assets/claw/macos-x64/claw.sha256",
)

PHASE18_19_REQUIRED_MODULES: tuple[str, ...] = (
    "src/labai/aci.py",
    "src/labai/data_contracts.py",
    "src/labai/evidence_ledger.py",
    "src/labai/external/__init__.py",
    "src/labai/external/grep_ast_adapter.py",
    "src/labai/notebook_io.py",
    "src/labai/owner_detection.py",
    "src/labai/repo_map.py",
    "src/labai/runtime/platform.py",
    "src/labai/runtime/progress.py",
    "src/labai/runtime_exec.py",
    "src/labai/structured_edits.py",
    "src/labai/task_manifest.py",
    "src/labai/typed_validation.py",
    "src/labai/validator_routing.py",
)

REQUIRED_DEPENDENCIES: tuple[str, ...] = (
    "click",
    "typer",
    "nbformat",
    "nbclient",
    "ipykernel",
    "pymupdf",
    "pypdf",
    "unidiff",
    "pandas",
    "numpy",
)

FORBIDDEN_EXACT_PATHS: tuple[str, ...] = (
    "Launch-LabAI-Setup.cmd",
    "RELEASE_CHECKLIST.md",
    "RELEASE_MANIFEST.md",
)

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    ".claw/",
    ".codex/",
    ".labai/",
    ".planning/",
    ".pytest",
    ".release-staging/",
    "dist/",
    "examples/",
    "runtime-assets/claw/windows-x64/",
    "scripts/windows/",
    "tests/",
    "__pycache__/",
)

FORBIDDEN_PARTS: tuple[str, ...] = (
    "__pycache__",
)

FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    ".exe",
    ".log",
    ".pdf",
    ".ps1",
    ".pyc",
    ".pyd",
    ".pyo",
    ".zip",
)

MAC_SCRIPT_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "%LOCALAPPDATA%",
    ".exe",
    ".ps1",
    "PowerShell",
    "cmd.exe",
    "winget",
)


@dataclass(frozen=True)
class MacosReleasePlan:
    version: str
    archive_name: str
    package_kind: str
    real_claw_binaries: tuple[str, ...]


@dataclass(frozen=True)
class MacosReleaseResult:
    archive_path: Path
    staging_root: Path
    included_files: tuple[str, ...]
    plan: MacosReleasePlan


def read_project_version(repo_root: Path) -> str:
    pyproject_path = repo_root / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def build_macos_release_plan(repo_root: Path) -> MacosReleasePlan:
    version = read_project_version(repo_root)
    real_binaries = tuple(
        relative_path
        for relative_path in MACOS_CLAW_BINARIES
        if (repo_root / relative_path).is_file()
    )
    package_kind = "macos" if set(MACOS_CLAW_BINARIES).issubset(real_binaries) else "macos-scaffold"
    archive_name = f"labai-{package_kind}-v{version}.zip"
    return MacosReleasePlan(
        version=version,
        archive_name=archive_name,
        package_kind=package_kind,
        real_claw_binaries=real_binaries,
    )


def build_macos_file_set(repo_root: Path) -> tuple[str, ...]:
    plan = build_macos_release_plan(repo_root)
    files: set[str] = set()
    files.update(ROOT_FILES)
    files.update(MACOS_DOCS)
    files.update(MACOS_SCRIPTS)
    files.update(MACOS_TEMPLATES)
    files.update(MACOS_CLAW_READMES)
    files.update(plan.real_claw_binaries)
    for binary_path in plan.real_claw_binaries:
        sha_path = str(Path(binary_path).with_name("claw.sha256")).replace("\\", "/")
        files.add(sha_path)
    files.update(_discover_labi_source_files(repo_root))
    files.update(_discover_continue_checks(repo_root))

    missing = sorted(path for path in files if not (repo_root / path).is_file())
    if missing:
        raise RuntimeError("macOS release source is missing required files: " + ", ".join(missing))

    forbidden = sorted(path for path in files if is_forbidden_macos_archive_path(path))
    if forbidden:
        raise RuntimeError("macOS release file set includes forbidden paths: " + ", ".join(forbidden))

    return tuple(sorted(files))


def create_macos_release_archive(
    repo_root: Path,
    archive_path: Path | None = None,
    *,
    staging_root: Path,
) -> MacosReleaseResult:
    resolved_root = repo_root.resolve()
    plan = build_macos_release_plan(resolved_root)
    resolved_archive = (archive_path or (resolved_root / "dist" / plan.archive_name)).resolve()
    resolved_staging = staging_root.resolve()

    if resolved_staging.exists():
        shutil.rmtree(resolved_staging)
    resolved_staging.mkdir(parents=True, exist_ok=True)
    resolved_archive.parent.mkdir(parents=True, exist_ok=True)

    included_files = build_macos_file_set(resolved_root)
    for relative_path in included_files:
        source_path = resolved_root / relative_path
        destination_path = resolved_staging / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    shutil.copy2(resolved_root / MACOS_RELEASE_README_SOURCE, resolved_staging / "README.md")

    if resolved_archive.exists():
        resolved_archive.unlink()
    with zipfile.ZipFile(resolved_archive, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive_root = resolved_archive.stem
        for relative_path in included_files:
            source_path = resolved_staging / relative_path
            info = zipfile.ZipInfo(str(Path(archive_root, relative_path)).replace("\\", "/"))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = _archive_unix_mode(relative_path) << 16
            archive.writestr(info, source_path.read_bytes())

    validate_macos_archive(resolved_archive, repo_root=resolved_root)
    return MacosReleaseResult(
        archive_path=resolved_archive,
        staging_root=resolved_staging,
        included_files=included_files,
        plan=plan,
    )


def validate_macos_archive(archive_path: Path, *, repo_root: Path) -> tuple[str, ...]:
    resolved_archive = archive_path.resolve()
    if not resolved_archive.is_file():
        raise RuntimeError(f"macOS archive does not exist: {resolved_archive}")

    with zipfile.ZipFile(resolved_archive, mode="r") as archive:
        raw_infos = {
            info.filename.rstrip("/"): info
            for info in archive.infolist()
            if info.filename and not info.filename.endswith("/")
        }
        names, info_name_map = _strip_archive_root(raw_infos, expected_root=resolved_archive.stem)
        infos = {name: raw_infos[info_name_map[name]] for name in names}
        payloads = {name: archive.read(info_name_map[name]) for name in names}

    expected = set(build_macos_file_set(repo_root.resolve()))
    missing = sorted(expected.difference(names))
    if missing:
        raise RuntimeError("macOS archive is missing required files: " + ", ".join(missing))

    forbidden = sorted(name for name in names if is_forbidden_macos_archive_path(name))
    if forbidden:
        raise RuntimeError("macOS archive includes forbidden paths: " + ", ".join(forbidden))

    unexpected_continue = sorted(
        name
        for name in names
        if name.startswith(".continue/") and not (name.startswith(".continue/checks/") and name.endswith(".md"))
    )
    if unexpected_continue:
        raise RuntimeError("macOS archive includes non-check .continue content: " + ", ".join(unexpected_continue))

    missing_modules = sorted(path for path in PHASE18_19_REQUIRED_MODULES if path not in names)
    if missing_modules:
        raise RuntimeError("macOS archive is missing Phase 18/19 modules: " + ", ".join(missing_modules))

    _validate_python_source_syntax(payloads)
    _validate_pyproject_dependencies(payloads["pyproject.toml"])
    _validate_macos_script_payloads(payloads)
    _validate_macos_claw_payloads(payloads, infos)
    _validate_macos_profile_payloads(payloads)
    _validate_launcher_payload(payloads["Launch-LabAI-Setup.command"])
    return names


def _strip_archive_root(
    infos: dict[str, zipfile.ZipInfo],
    *,
    expected_root: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    names = tuple(sorted(infos))
    prefix = expected_root.rstrip("/") + "/"
    if names and all(name.startswith(prefix) for name in names):
        stripped = tuple(sorted(name[len(prefix) :] for name in names))
        return stripped, {name[len(prefix) :]: name for name in names}
    roots = {name.split("/", 1)[0] for name in names if "/" in name}
    if len(roots) == 1 and all("/" in name for name in names):
        root = next(iter(roots))
        common_prefix = root + "/"
        stripped = tuple(sorted(name[len(common_prefix) :] for name in names))
        return stripped, {name[len(common_prefix) :]: name for name in names}
    return names, {name: name for name in names}


def is_forbidden_macos_archive_path(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/").strip("/")
    if not normalized:
        return True
    if normalized in FORBIDDEN_EXACT_PATHS:
        return True
    if any(normalized.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return True
    if any(part in FORBIDDEN_PARTS for part in normalized.split("/")):
        return True
    return Path(normalized).suffix.lower() in FORBIDDEN_SUFFIXES


def _discover_labi_source_files(repo_root: Path) -> tuple[str, ...]:
    source_root = repo_root / "src" / "labai"
    return tuple(
        sorted(
            path.relative_to(repo_root).as_posix()
            for path in source_root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    )


def _discover_continue_checks(repo_root: Path) -> tuple[str, ...]:
    checks_root = repo_root / ".continue" / "checks"
    if not checks_root.is_dir():
        return ()
    return tuple(sorted(path.relative_to(repo_root).as_posix() for path in checks_root.rglob("*.md")))


def _validate_python_source_syntax(payloads: dict[str, bytes]) -> None:
    errors: list[str] = []
    for relative_path, payload in sorted(payloads.items()):
        if not (relative_path.startswith("src/labai/") and relative_path.endswith(".py")):
            continue
        try:
            source = payload.decode("utf-8")
            compile(source, relative_path, "exec")
        except SyntaxError as exc:
            errors.append(f"{relative_path}:{exc.lineno}: {exc.msg}")
        except UnicodeDecodeError as exc:
            errors.append(f"{relative_path}: cannot decode as UTF-8: {exc}")
    if errors:
        raise RuntimeError("macOS archive contains Python source syntax errors: " + "; ".join(errors))


def _validate_pyproject_dependencies(payload: bytes) -> None:
    data = tomllib.loads(payload.decode("utf-8"))
    dependency_names = {
        dependency.split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].strip().lower()
        for dependency in data["project"]["dependencies"]
    }
    missing = sorted(set(REQUIRED_DEPENDENCIES).difference(dependency_names))
    if missing:
        raise RuntimeError("pyproject.toml is missing runtime dependencies: " + ", ".join(missing))


def _validate_macos_script_payloads(payloads: dict[str, bytes]) -> None:
    for relative_path in MACOS_SCRIPTS:
        text = payloads[relative_path].decode("utf-8")
        if b"\r\n" in payloads[relative_path]:
            raise RuntimeError(f"{relative_path} uses CRLF line endings")
        if not text.startswith("#!/usr/bin/env bash\n"):
            raise RuntimeError(f"{relative_path} is missing the bash shebang")
        if "set -euo pipefail" not in text:
            raise RuntimeError(f"{relative_path} is missing set -euo pipefail")
        token_hits = [token for token in MAC_SCRIPT_FORBIDDEN_TOKENS if token in text]
        if token_hits:
            raise RuntimeError(f"{relative_path} contains Windows-only tokens: {', '.join(token_hits)}")
    setup_ollama = payloads["scripts/mac/setup-local-ollama.sh"].decode("utf-8")
    build_claw = payloads["scripts/mac/build-claw-macos.sh"].decode("utf-8")
    for token in (
        "Claw source not found",
        "uname -m",
        "cargo build --release --bin claw",
        "runtime-assets/claw/macos-arm64",
        "runtime-assets/claw/macos-x64",
        "--version",
        "Normal RA setup must not require Rust or Cargo",
    ):
        if token not in build_claw:
            raise RuntimeError(f"scripts/mac/build-claw-macos.sh is missing required build token: {token}")

    for token in (
        "MIN_PYTHON_MINOR=11",
        "found python3 but version",
        "Homebrew not found; attempting automatic Homebrew install",
        "configure_homebrew_shellenv",
        "brew shellenv",
        ".zprofile",
        "HOMEBREW_INSTALL_URL",
        "Homebrew detected; attempting Python install",
        "--python-only",
        "brew install",
        "selected Python:",
        "Python version ok:",
        "python_path=",
        "venv_path=",
        "Python interpreter:",
        'export PATH="$HOME/Library/Application Support/LabAI/bin:$PATH"',
        "ensure_launcher_path_current",
        "ensure_launcher_path_zprofile",
        "grep -Fqx",
    ):
        install_labi = payloads["scripts/mac/install-labai.sh"].decode("utf-8")
        if token not in install_labi:
            raise RuntimeError(f"scripts/mac/install-labai.sh is missing required Python bootstrap token: {token}")

    for token in (
        "https://ollama.com/download/Ollama-darwin.zip",
        "Ollama not found; attempting macOS install",
        "ensuring ollama CLI link",
        "waiting for Ollama API",
        "pulling missing model:",
        "qwen_models_status: ok",
    ):
        if token not in setup_ollama:
            raise RuntimeError(f"scripts/mac/setup-local-ollama.sh is missing required auto-install token: {token}")

    verify_install = payloads["scripts/mac/verify-install.sh"].decode("utf-8")
    for token in (
        'ollama run "$MODEL" "Say exactly 2 and nothing else."',
        '"$LABAI_CMD" ask -- "Say exactly 2 and nothing else."',
        "launcher_path_status:",
        "shell_path_status:",
        "zprofile_path_status:",
        'export PATH="$HOME/Library/Application Support/LabAI/bin:$PATH"',
        'LABAI_CMD="$LAUNCHER_PATH"',
        "claw_model_syntax_failed",
        "fallback_or_mock",
        "runtime_used",
        "provider_used",
        "mock-static",
        "local_performance_direct_ollama_status:",
        "local_performance_labai_ask_status:",
        "local_performance_labai_ask_status: blocked_by_claw",
        "local_performance_classification:",
        "blocked_by_claw",
        'encoding="utf-8"',
        'errors="replace"',
        "isinstance(value, bytes)",
    ):
        if token not in verify_install:
            raise RuntimeError(f"scripts/mac/verify-install.sh is missing required smoke token: {token}")

    bootstrap = payloads["scripts/mac/bootstrap-mac.sh"].decode("utf-8")
    for token in (
        "[labai-setup] Final summary",
        "detecting mac architecture",
        "runtime-assets/claw/macos-arm64/claw",
        "runtime-assets/claw/macos-x64/claw",
        "configuring LabAI Claw binary path",
        "Python version:",
        "Python path:",
        "Venv path:",
        "Ollama install:",
        "Ollama API:",
        "Qwen models:",
        "Claw:",
        "blocked_by_claw",
        "Local performance:",
        "Launcher path:",
        "Current process PATH:",
        "~/.zprofile PATH:",
        "Next command for this Terminal:",
        "source ~/.zprofile",
        "rehash",
        "Alternative direct command:",
        "claw_model_syntax_failed",
        'export PATH="$HOME/Library/Application Support/LabAI/bin:$PATH"',
        "ensure_launcher_path_current",
        "ensure_launcher_path_zprofile",
    ):
        if token not in bootstrap:
            raise RuntimeError(f"scripts/mac/bootstrap-mac.sh is missing required summary token: {token}")


def _validate_macos_claw_payloads(payloads: dict[str, bytes], infos: dict[str, zipfile.ZipInfo]) -> None:
    for binary_path in MACOS_CLAW_BINARIES:
        if binary_path not in payloads:
            continue
        payload = payloads[binary_path]
        if not payload:
            raise RuntimeError(f"{binary_path} is empty")
        if payload.startswith(b"MZ"):
            raise RuntimeError(f"{binary_path} is a Windows PE executable, not a macOS Claw binary")
        if not (
            payload.startswith(b"\xcf\xfa\xed\xfe")
            or payload.startswith(b"\xfe\xed\xfa\xcf")
            or payload.startswith(b"\xca\xfe\xba\xbe")
            or payload.startswith(b"\xbe\xba\xfe\xca")
        ):
            raise RuntimeError(f"{binary_path} does not look like a Mach-O macOS executable")
        mode = (infos[binary_path].external_attr >> 16) & 0o777
        if not (mode & 0o111):
            raise RuntimeError(f"{binary_path} is not marked executable in the archive")
        sha_path = str(Path(binary_path).with_name("claw.sha256")).replace("\\", "/")
        if sha_path not in payloads:
            raise RuntimeError(f"{binary_path} is missing its claw.sha256 sidecar")
        recorded_hash = _extract_sha256(payloads[sha_path].decode("utf-8"))
        actual_hash = hashlib.sha256(payload).hexdigest()
        if recorded_hash != actual_hash:
            raise RuntimeError(f"{sha_path} does not match {binary_path}")
        readme_path = str(Path(binary_path).with_name("README.md")).replace("\\", "/")
        readme = payloads.get(readme_path, b"").decode("utf-8")
        for token in ("Source commit:", "Build date:", "Smoke command:", "Smoke output:", "one-click"):
            if token not in readme:
                raise RuntimeError(f"{readme_path} is missing required Claw smoke metadata: {token}")


def _extract_sha256(text: str) -> str:
    match = re.search(r"\b[0-9a-fA-F]{64}\b", text)
    if not match:
        raise RuntimeError("claw.sha256 does not contain a SHA256 digest")
    return match.group(0).lower()


def _archive_unix_mode(relative_path: str) -> int:
    if relative_path in MACOS_SCRIPTS or relative_path in MACOS_CLAW_BINARIES or relative_path == "Launch-LabAI-Setup.command":
        return 0o100755
    return 0o100644


def _validate_macos_profile_payloads(payloads: dict[str, bytes]) -> None:
    forbidden_tokens = ("%LOCALAPPDATA%", "claw.exe", "Scripts\\python.exe")
    for relative_path in MACOS_TEMPLATES:
        text = payloads[relative_path].decode("utf-8")
        token_hits = [token for token in forbidden_tokens if token in text]
        if token_hits:
            raise RuntimeError(f"{relative_path} contains Windows-only profile tokens: {', '.join(token_hits)}")


def _validate_launcher_payload(payload: bytes) -> None:
    text = payload.decode("utf-8")
    if "scripts/mac/bootstrap-mac.sh" not in text:
        raise RuntimeError("Launch-LabAI-Setup.command does not call scripts/mac/bootstrap-mac.sh")

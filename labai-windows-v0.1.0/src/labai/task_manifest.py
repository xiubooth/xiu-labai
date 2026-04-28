from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Literal

from labai.workspace import _iter_prompt_candidates


TaskKind = Literal[
    "source_edit",
    "notebook_deliverable",
    "config_edit",
    "docs_code_sync",
    "deliverable_creation",
    "validation_only",
    "unknown",
]


@dataclass(frozen=True)
class TaskManifest:
    task_run_id: str
    workspace_root: str
    user_instruction: str
    task_kind: TaskKind
    prompt_named_files: tuple[str, ...]
    prompt_named_wildcards: tuple[str, ...]
    wildcard_matches: tuple[str, ...]
    required_read_files: tuple[str, ...]
    reference_files: tuple[str, ...]
    candidate_owner_files: tuple[str, ...]
    primary_target_artifacts: tuple[str, ...]
    allowed_support_files: tuple[str, ...]
    disallowed_success_modes: tuple[str, ...]
    external_dependencies_detected: tuple[str, ...]
    expected_validation_strategy: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    true_blocker_conditions: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return asdict(self)


_WILDCARD_PATTERN = re.compile(
    r"\b[A-Za-z0-9_.\-\\/]*\*[A-Za-z0-9_.\-\\/]*\.(?:py|ipynb|toml|json|ya?ml|md|txt)\b",
    re.IGNORECASE,
)
_UPDATE_TOKENS = ("fix", "modify", "edit", "update", "change", "implement", "repair")
_CREATE_TOKENS = ("create", "write", "generate", "save", "export")
_REFERENCE_PATTERNS = (
    r"reference implementations?\s+(?:are|is)\s+(?P<body>[^\n]+)",
    r"reference files?\s+inspected if present\s*:\s*(?P<body>(?:\n\s*-\s*.+)+)",
)
_PRIMARY_PATTERNS = (
    r"primary task files?\s+(?:are|is)\s+(?P<body>[^\n]+)",
    r"existing primary deliverable file\s+(?:is|are)\s*:\s*(?P<body>(?:\n\s*-\s*.+)+)",
)


def build_task_manifest(
    user_instruction: str,
    workspace_root: Path,
    *,
    task_run_id: str,
    planned_modifications: tuple[str, ...] = (),
    planned_creations: tuple[str, ...] = (),
    referenced_paths: tuple[str, ...] = (),
    acceptance_criteria: tuple[str, ...] = (),
) -> TaskManifest:
    prompt = _extract_current_task_prompt(user_instruction)
    named_files = _discover_existing_prompt_files(prompt, workspace_root)
    wildcards = tuple(dict.fromkeys(match.group(0) for match in _WILDCARD_PATTERN.finditer(prompt)))
    wildcard_matches = _expand_wildcards(workspace_root, wildcards)
    reference_files = _discover_section_targets(prompt, workspace_root, _REFERENCE_PATTERNS)
    primary_artifacts = _discover_primary_artifacts(
        prompt,
        workspace_root,
        named_files=named_files,
        planned_modifications=planned_modifications,
    )
    candidate_owner_files = tuple(
        dict.fromkeys(
            (
                *primary_artifacts,
                *named_files,
                *wildcard_matches,
                *planned_modifications,
            )
        )
    )
    required_read_files = tuple(
        dict.fromkeys(
            (
                *named_files,
                *wildcard_matches,
                *primary_artifacts,
                *candidate_owner_files,
                *reference_files,
                *referenced_paths,
            )
        )
    )
    allowed_support_files = tuple(
        dict.fromkeys(
            item
            for item in (*planned_creations, *referenced_paths)
            if item not in primary_artifacts
        )
    )
    task_kind = _classify_task_kind(prompt, named_files, primary_artifacts)
    manifest_acceptance = acceptance_criteria or _extract_acceptance_criteria(prompt)
    external_dependencies = _detect_external_dependencies(prompt)
    return TaskManifest(
        task_run_id=task_run_id,
        workspace_root=str(workspace_root.resolve()),
        user_instruction=user_instruction,
        task_kind=task_kind,
        prompt_named_files=named_files,
        prompt_named_wildcards=wildcards,
        wildcard_matches=wildcard_matches,
        required_read_files=required_read_files,
        reference_files=reference_files,
        candidate_owner_files=candidate_owner_files,
        primary_target_artifacts=primary_artifacts,
        allowed_support_files=allowed_support_files,
        disallowed_success_modes=_extract_disallowed_success_modes(prompt, task_kind),
        external_dependencies_detected=external_dependencies,
        expected_validation_strategy=_expected_validation_strategy(task_kind, external_dependencies),
        acceptance_criteria=manifest_acceptance,
        true_blocker_conditions=_true_blocker_conditions(external_dependencies, named_files, workspace_root),
    )


def _extract_current_task_prompt(prompt: str) -> str:
    marker = "Original instruction:\n"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def _discover_existing_prompt_files(prompt: str, workspace_root: Path) -> tuple[str, ...]:
    discovered: list[str] = []
    seen: set[str] = set()
    for candidate in _iter_prompt_candidates(prompt):
        normalized = _normalize_relative(candidate)
        if not normalized:
            continue
        if "*" in normalized:
            continue
        resolved = (workspace_root / normalized).resolve()
        if not resolved.is_file():
            continue
        relative = resolved.relative_to(workspace_root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        discovered.append(relative)
    return tuple(discovered)


def _expand_wildcards(workspace_root: Path, wildcards: tuple[str, ...]) -> tuple[str, ...]:
    matches: list[str] = []
    for wildcard in wildcards:
        for path in sorted(workspace_root.glob(wildcard)):
            if not path.is_file():
                continue
            matches.append(path.relative_to(workspace_root).as_posix())
    return tuple(dict.fromkeys(matches))


def _discover_section_targets(
    prompt: str,
    workspace_root: Path,
    patterns: tuple[str, ...],
) -> tuple[str, ...]:
    discovered: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE | re.MULTILINE):
            body = match.group("body")
            for candidate in _iter_prompt_candidates(body):
                normalized = _normalize_relative(candidate)
                if not normalized or "*" in normalized:
                    continue
                resolved = (workspace_root / normalized).resolve()
                if resolved.is_file():
                    discovered.append(resolved.relative_to(workspace_root).as_posix())
    return tuple(dict.fromkeys(discovered))


def _discover_primary_artifacts(
    prompt: str,
    workspace_root: Path,
    *,
    named_files: tuple[str, ...],
    planned_modifications: tuple[str, ...],
) -> tuple[str, ...]:
    primary = list(_discover_section_targets(prompt, workspace_root, _PRIMARY_PATTERNS))
    if any(item.endswith(".ipynb") for item in named_files):
        primary.extend(item for item in named_files if item.endswith(".ipynb"))
    if not primary and len(planned_modifications) == 1:
        primary.extend(planned_modifications)
    return tuple(dict.fromkeys(primary))


def _classify_task_kind(
    prompt: str,
    named_files: tuple[str, ...],
    primary_artifacts: tuple[str, ...],
) -> TaskKind:
    lowered = prompt.lower()
    if any(item.endswith(".ipynb") for item in (*primary_artifacts, *named_files)):
        return "notebook_deliverable"
    if "validation only" in lowered or ("validate" in lowered and not any(token in lowered for token in _UPDATE_TOKENS)):
        return "validation_only"
    if any(Path(item).suffix.lower() in {".toml", ".json", ".yaml", ".yml"} for item in named_files):
        return "config_edit"
    if "readme" in lowered and any(token in lowered for token in _UPDATE_TOKENS):
        return "docs_code_sync"
    if any(token in lowered for token in _CREATE_TOKENS) and not any(token in lowered for token in _UPDATE_TOKENS):
        return "deliverable_creation"
    if any(token in lowered for token in _UPDATE_TOKENS) or any(
        Path(item).suffix.lower() in {".py", ".ipynb"} for item in named_files
    ):
        return "source_edit"
    return "unknown"


def _extract_acceptance_criteria(prompt: str) -> tuple[str, ...]:
    if "Required behavior:" in prompt:
        body = prompt.split("Required behavior:", 1)[1].split("\n", 1)[0]
        items = [item.strip(" .") for item in re.split(r";|,", body) if item.strip()]
        return tuple(dict.fromkeys(items))
    criteria: list[str] = []
    for match in re.finditer(r"must\s+(?P<body>[^.\n]+)", prompt, flags=re.IGNORECASE):
        body = match.group("body").strip(" .")
        if body and body not in criteria:
            criteria.append(body)
    return tuple(criteria[:12])


def _extract_disallowed_success_modes(prompt: str, task_kind: TaskKind) -> tuple[str, ...]:
    modes: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip("- ").strip()
        lowered = stripped.lower()
        if lowered.startswith("do not declare success") or lowered.startswith("do not rely on") or lowered.startswith("do not "):
            modes.append(stripped)
    if task_kind == "source_edit":
        modes.append("validator-only or helper-only changes cannot count as source-edit success")
    if task_kind == "notebook_deliverable":
        modes.append("helper-only changes cannot count as notebook-deliverable success")
    return tuple(dict.fromkeys(modes))


def _detect_external_dependencies(prompt: str) -> tuple[str, ...]:
    lowered = prompt.lower()
    dependencies: list[str] = []
    for token in ("wrds", "ollama", "deepseek", "docker"):
        if token in lowered:
            dependencies.append(token)
    return tuple(dict.fromkeys(dependencies))


def _expected_validation_strategy(
    task_kind: TaskKind,
    external_dependencies: tuple[str, ...],
) -> tuple[str, ...]:
    strategies: list[str] = []
    if task_kind == "notebook_deliverable":
        strategies.extend(
            (
                "Use notebook parse validation.",
                "Execute the notebook and require embedded outputs.",
            )
        )
    elif task_kind == "source_edit":
        strategies.extend(
            (
                "Require source-evidence validation on the owning file.",
                "Prefer behavioral or data-contract validation over syntax-only checks.",
            )
        )
    elif task_kind == "config_edit":
        strategies.append("Validate syntax plus explicit config literals.")
    if "wrds" in external_dependencies:
        strategies.append("If WRDS is unavailable, switch to stubbed connector or monkeypatched sink validation.")
    return tuple(dict.fromkeys(strategies))


def _true_blocker_conditions(
    external_dependencies: tuple[str, ...],
    named_files: tuple[str, ...],
    workspace_root: Path,
) -> tuple[str, ...]:
    conditions = [
        "A required owning source file is missing from the active workspace.",
        "A required external dependency is unavailable and the behavior cannot be validated with a local stub or monkeypatch.",
    ]
    if not named_files:
        conditions.insert(
            0,
            f"No prompt-named source files were found under {workspace_root.resolve()}; owner detection must rely on repo-map evidence.",
        )
    if "wrds" in external_dependencies:
        conditions.append("Missing live WRDS alone is not a blocker if the core data path can be validated locally.")
    return tuple(dict.fromkeys(conditions))


def _normalize_relative(candidate: str) -> str:
    normalized = str(Path(candidate.replace("\\", "/"))).replace("\\", "/")
    normalized = normalized.lstrip("/").strip()
    return normalized

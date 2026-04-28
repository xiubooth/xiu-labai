from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from labai.repo_map import RepoMapEntry, RepoMapResult
from labai.task_manifest import TaskManifest


@dataclass(frozen=True)
class OwnerDetectionResult:
    primary_owner_files: tuple[str, ...]
    primary_artifacts: tuple[str, ...]
    support_files: tuple[str, ...]
    validator_files: tuple[str, ...]
    reference_only_files: tuple[str, ...]
    helper_only_files: tuple[str, ...]
    stale_files: tuple[str, ...]
    generated_artifacts: tuple[str, ...]
    output_artifacts: tuple[str, ...]
    owner_confidence: dict[str, float]
    selected_reasons: dict[str, str]
    code_quality_warnings: tuple[str, ...]
    owner_boundary_warnings: tuple[str, ...]
    stale_file_warnings: tuple[str, ...]
    blocking_issues: tuple[str, ...]
    reasoning: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def detect_owner_files(
    manifest: TaskManifest,
    repo_map: RepoMapResult,
    workspace_root: Path | None = None,
) -> OwnerDetectionResult:
    entries = repo_map.entries
    prompt_lower = manifest.user_instruction.lower()
    primary_artifacts = tuple(dict.fromkeys(manifest.primary_target_artifacts))
    primary: list[str] = []
    validators = tuple(
        dict.fromkeys(
            entry.relative_path
            for entry in entries
            if _looks_like_validator(entry.relative_path)
        )
    )
    generated_artifacts = tuple(item for item in validators if item.startswith("validation/"))
    helper_only = tuple(
        dict.fromkeys(
            entry.relative_path
            for entry in entries
            if _looks_like_helper(entry.relative_path)
        )
    )
    entry_by_path = {entry.relative_path: entry for entry in entries}
    prompt_named = tuple(manifest.prompt_named_files)
    stale_files, stale_reasons = _classify_stale_files(
        prompt_named=prompt_named,
        workspace_root=workspace_root,
        user_instruction=manifest.user_instruction,
    )
    if manifest.task_kind == "notebook_deliverable":
        primary.extend(primary_artifacts or tuple(item for item in prompt_named if item.endswith(".ipynb")))
    elif _looks_like_annual_pipeline(prompt_lower, prompt_named):
        annual_order = ("14_PreAnnualCS.py", "01_download_datasets.py", "11_PrepareLinkingTables.py")
        for candidate in annual_order:
            if candidate in entry_by_path or candidate in manifest.required_read_files:
                primary.append(candidate)
    elif _looks_like_task8_family(prompt_lower):
        if "12_PrepareOtherData_daily.py" in entry_by_path:
            primary.append("12_PrepareOtherData_daily.py")
        if "13_PrepareRatings.py" in entry_by_path and "13_PrepareRatings.py" not in stale_files:
            primary.append("13_PrepareRatings.py")
    elif _looks_like_task1_family(prompt_lower):
        if "01_download_datasets.py" in entry_by_path:
            primary.append("01_download_datasets.py")
        primary.extend(
            item
            for item in manifest.wildcard_matches
            if Path(item).name.lower().startswith("15_preparedailycrsp_task")
        )
    else:
        primary.extend(
            item
            for item in (*primary_artifacts, *prompt_named)
            if item not in stale_files and _is_owner_candidate(item)
        )
    if not primary:
        for entry in entries:
            if entry.relative_path in manifest.candidate_owner_files and _is_owner_candidate(entry.relative_path):
                primary.append(entry.relative_path)
        if not primary and entries:
            first = next((entry.relative_path for entry in entries if _is_owner_candidate(entry.relative_path)), "")
            if first:
                primary.append(first)
    primary = list(dict.fromkeys(item for item in primary if item and item not in stale_files))
    support_files = tuple(
        dict.fromkeys(
            item
            for item in (*manifest.allowed_support_files, *(entry.relative_path for entry in entries))
            if item not in primary
            and item not in validators
            and item not in manifest.reference_files
            and item not in stale_files
            and _is_support_file(item)
        )
    )
    selected_reasons: dict[str, str] = {}
    owner_confidence: dict[str, float] = {}
    for item in primary:
        selected_reasons[item] = _owner_reason(item, prompt_lower)
        owner_confidence[item] = _owner_confidence(item, entry_by_path.get(item))
    for item, reason in stale_reasons.items():
        selected_reasons[item] = reason
        owner_confidence[item] = 0.2
    code_quality_warnings = list(
        _code_quality_warnings(
            manifest=manifest,
            workspace_root=workspace_root,
            primary=tuple(primary),
            stale_files=stale_files,
            stale_reasons=stale_reasons,
        )
    )
    owner_boundary_warnings = list(
        _owner_boundary_warnings(
            manifest=manifest,
            workspace_root=workspace_root,
            primary=tuple(primary),
            stale_files=stale_files,
        )
    )
    stale_file_warnings = tuple(
        f"{path}: {reason}"
        for path, reason in stale_reasons.items()
    )
    blocking_issues = tuple(
        _blocking_issues(
            manifest=manifest,
            workspace_root=workspace_root,
            primary=tuple(primary),
            stale_files=stale_files,
        )
    )
    reasoning = [
        "Primary owners are selected from prompt-named or route-shaped production files, not from validators or support files.",
        "Validator, helper, stale, and reference files are tracked separately so helper-only or validator-only edits cannot satisfy owner success.",
    ]
    if stale_reasons:
        reasoning.append("Prompt-named files that appear broken or unused are classified as stale/reference unless production-path evidence proves otherwise.")
    return OwnerDetectionResult(
        primary_owner_files=tuple(primary),
        primary_artifacts=primary_artifacts,
        support_files=support_files,
        validator_files=validators,
        reference_only_files=manifest.reference_files,
        helper_only_files=helper_only,
        stale_files=stale_files,
        generated_artifacts=generated_artifacts,
        output_artifacts=manifest.primary_target_artifacts,
        owner_confidence=owner_confidence,
        selected_reasons=selected_reasons,
        code_quality_warnings=tuple(dict.fromkeys(code_quality_warnings)),
        owner_boundary_warnings=tuple(dict.fromkeys(owner_boundary_warnings)),
        stale_file_warnings=tuple(dict.fromkeys(stale_file_warnings)),
        blocking_issues=tuple(dict.fromkeys(blocking_issues)),
        reasoning=tuple(reasoning),
    )


def owner_requirement_satisfied(
    owner_detection: OwnerDetectionResult,
    *,
    modified_files: tuple[str, ...],
    created_files: tuple[str, ...] = (),
) -> bool:
    touched = set((*modified_files, *created_files))
    required = set(
        (
            *owner_detection.primary_owner_files,
            *owner_detection.primary_artifacts,
            *owner_detection.output_artifacts,
        )
    )
    if not required:
        return False
    return bool(required & touched)


def _is_owner_candidate(relative_path: str) -> bool:
    lowered = relative_path.lower()
    if lowered.endswith(".ipynb"):
        return True
    if lowered.endswith(".py") and not _looks_like_validator(lowered):
        return True
    return False


def _looks_like_validator(relative_path: str) -> bool:
    lowered = relative_path.lower()
    return lowered.startswith("validation/") or lowered.startswith("tests/") or "validate" in Path(lowered).stem


def _is_support_file(relative_path: str) -> bool:
    lowered = relative_path.lower()
    return any(token in lowered for token in ("helper", "support", "handoff", "notes")) or lowered.endswith(".md")


def _looks_like_helper(relative_path: str) -> bool:
    lowered = relative_path.lower()
    return any(token in lowered for token in ("helper", "support", "wrapper", "handoff"))


def _looks_like_task1_family(prompt_lower: str) -> bool:
    return "task 1" in prompt_lower or "daily crsp" in prompt_lower


def _looks_like_task8_family(prompt_lower: str) -> bool:
    return all(token in prompt_lower for token in ("ratings", "pensions")) or "short interest" in prompt_lower


def _looks_like_annual_pipeline(prompt_lower: str, prompt_named: tuple[str, ...]) -> bool:
    annual_tokens = (
        "task 4",
        "annual",
        "6 month",
        "6-month",
        "time_avail_m",
        "datadate",
        "cusip",
        "ccm link",
    )
    return any(token in prompt_lower for token in annual_tokens) or any(
        Path(item).name.lower() == "14_preannualcs.py" for item in prompt_named
    )


def _classify_stale_files(
    *,
    prompt_named: tuple[str, ...],
    workspace_root: Path | None,
    user_instruction: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    stale: list[str] = []
    reasons: dict[str, str] = {}
    if workspace_root is None:
        return (), {}
    for relative_path in prompt_named:
        path = (workspace_root / relative_path).resolve()
        if not path.is_file() or path.suffix.lower() != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        if _contains_placeholder_markers(lowered):
            stale.append(relative_path)
            reasons[relative_path] = "prompt-named file still looks like a placeholder or legacy stub"
            continue
        if _contains_production_path_placeholder(lowered):
            stale.append(relative_path)
            reasons[relative_path] = "prompt-named file still contains placeholder paths that do not look production-ready"
            continue
        if (
            "task 8" in user_instruction.lower()
            and Path(relative_path).name.lower() == "13_prepareratings.py"
            and "to_pickle(" not in lowered
            and "def " not in lowered
        ):
            stale.append(relative_path)
            reasons[relative_path] = "ratings side file does not currently expose a clear production path"
    return tuple(dict.fromkeys(stale)), reasons


def _contains_placeholder_markers(lowered: str) -> bool:
    return any(
        token in lowered
        for token in (
            "your implementation here",
            "todo",
            "pass\n",
            "pass\r\n",
            "placeholder",
            "fake path",
            "undefined helper",
        )
    )


def _contains_production_path_placeholder(lowered: str) -> bool:
    return any(
        token in lowered
        for token in (
            "/path/to/raw_data",
            "c:/path/to/",
            "raw_data_path = '/path/",
            "clean_path = '/path/",
        )
    )


def _owner_reason(relative_path: str, prompt_lower: str) -> str:
    lowered = Path(relative_path).name.lower()
    if lowered.endswith(".ipynb"):
        return "Selected as the primary notebook artifact."
    if lowered == "14_preannualcs.py":
        return "Selected as the primary annual-to-monthly pipeline owner."
    if lowered == "12_prepareotherdata_daily.py":
        return "Selected as the main production owner for other-data cleaning outputs."
    if lowered == "01_download_datasets.py" and ("task 1" in prompt_lower or "task 9" in prompt_lower):
        return "Selected as the main download/output owner named by the task."
    return "Selected as a likely owning source based on prompt naming and repo-map evidence."


def _owner_confidence(relative_path: str, entry: RepoMapEntry | None) -> float:
    if relative_path.endswith(".ipynb"):
        return 1.0
    if entry is None:
        return 0.7
    score = max(entry.likely_owner_score, 0)
    return round(min(1.0, 0.45 + (score / 100.0)), 2)


def _code_quality_warnings(
    *,
    manifest: TaskManifest,
    workspace_root: Path | None,
    primary: tuple[str, ...],
    stale_files: tuple[str, ...],
    stale_reasons: dict[str, str],
) -> tuple[str, ...]:
    warnings: list[str] = []
    if workspace_root is None:
        return ()
    prompt_lower = manifest.user_instruction.lower()
    if _looks_like_task1_family(prompt_lower):
        downstream = [
            item
            for item in manifest.wildcard_matches
            if Path(item).name.lower().startswith("15_preparedailycrsp_task")
        ]
        hardcoded = [
            item
            for item in downstream
            if (workspace_root / item).is_file()
            and "2010-01-01" in (workspace_root / item).read_text(encoding="utf-8", errors="ignore")
        ]
        if len(hardcoded) >= 2:
            warnings.append(
                "centralized_rule_patchy: multiple downstream daily CRSP files still contain hardcoded 2010 date filters"
            )
    for relative_path in stale_files:
        warnings.append(f"stale_prompt_named_file: {relative_path} :: {stale_reasons.get(relative_path, '')}".strip())
    if _looks_like_task8_family(prompt_lower) and "13_PrepareRatings.py" in stale_files:
        warnings.append("task8_owner_boundary: 13_PrepareRatings.py was classified as stale/reference instead of a clean production owner")
    return tuple(dict.fromkeys(warnings))


def _owner_boundary_warnings(
    *,
    manifest: TaskManifest,
    workspace_root: Path | None,
    primary: tuple[str, ...],
    stale_files: tuple[str, ...],
) -> tuple[str, ...]:
    warnings: list[str] = []
    prompt_lower = manifest.user_instruction.lower()
    if _looks_like_task8_family(prompt_lower) and "12_PrepareOtherData_daily.py" in primary and "13_PrepareRatings.py" in stale_files:
        warnings.append(
            "task8_owner_boundary_warning: 12_PrepareOtherData_daily.py appears to own the live path while 13_PrepareRatings.py remains stale/reference"
        )
    if _looks_like_task1_family(prompt_lower) and not any(
        Path(item).name.lower().startswith("15_preparedailycrsp_task") for item in primary
    ):
        warnings.append(
            "task1_owner_boundary_warning: the download owner was selected but downstream daily task files were not promoted as owners"
        )
    return tuple(dict.fromkeys(warnings))


def _blocking_issues(
    *,
    manifest: TaskManifest,
    workspace_root: Path | None,
    primary: tuple[str, ...],
    stale_files: tuple[str, ...],
) -> tuple[str, ...]:
    issues: list[str] = []
    if workspace_root is None:
        return ()
    prompt_lower = manifest.user_instruction.lower()
    if _looks_like_task9_interface(prompt_lower):
        interface_issue = _task9_missing_function_chain(workspace_root)
        if interface_issue:
            issues.append(interface_issue)
    for relative_path in primary:
        path = (workspace_root / relative_path).resolve()
        if not path.is_file():
            continue
        lowered = path.read_text(encoding="utf-8", errors="ignore").lower()
        if _contains_production_path_placeholder(lowered):
            issues.append(f"{relative_path} still contains placeholder production paths.")
        if _contains_obviously_broken_helper_call(lowered):
            issues.append(f"{relative_path} still contains an undefined helper or placeholder production call.")
    if _looks_like_task8_family(prompt_lower) and "13_PrepareRatings.py" in primary and "13_PrepareRatings.py" in stale_files:
        issues.append("13_PrepareRatings.py is still on the production path but remains broken or placeholder-like.")
    return tuple(dict.fromkeys(issues))


def _looks_like_task9_interface(prompt_lower: str) -> bool:
    return "task 9" in prompt_lower or ("fama-french" in prompt_lower and "liquidity" in prompt_lower)


def _task9_missing_function_chain(workspace_root: Path) -> str:
    producer = workspace_root / "01_download_datasets.py"
    consumer = workspace_root / "12_PrepareOtherData_daily.py"
    if not producer.is_file() or not consumer.is_file():
        return ""
    producer_text = producer.read_text(encoding="utf-8", errors="ignore")
    consumer_text = consumer.read_text(encoding="utf-8", errors="ignore")
    function_names = tuple(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(r"\b(download_[A-Za-z0-9_]+)\s*\(", consumer_text)
        )
    )
    for function_name in function_names:
        if re.search(rf"\bdef\s+{re.escape(function_name)}\s*\(", producer_text):
            continue
        return (
            f"Task 9 interface mismatch: 12_PrepareOtherData_daily.py calls `{function_name}` "
            "but 01_download_datasets.py does not define it."
        )
    return ""


def _contains_obviously_broken_helper_call(lowered: str) -> bool:
    return any(
        token in lowered
        for token in (
            "undefined_helper(",
            "missing_function(",
            "wrapper_to_missing_function(",
        )
    )

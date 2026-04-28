from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from io import StringIO
from pathlib import Path
import re
from typing import Literal

from unidiff import PatchSet


EditOpType = Literal[
    "replace_file",
    "apply_unified_diff",
    "insert_after",
    "replace_range",
    "create_file",
    "modify_notebook",
    "execute_notebook",
    "delete_file",
]


@dataclass(frozen=True)
class StructuredEditOp:
    operation_id: str
    op_type: EditOpType
    target_path: str
    target_role: str
    reason: str
    preconditions: tuple[str, ...]
    expected_postconditions: tuple[str, ...]
    source_read_evidence_ids: tuple[str, ...]
    acceptance_criteria_supported: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LandedEditEvidence:
    operation_id: str
    target_path: str
    target_role: str
    before_hash: str
    after_hash: str
    added_lines: int
    removed_lines: int
    status: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredFileBlock:
    path: str
    content: str
    language: str


@dataclass(frozen=True)
class UnifiedDiffResult:
    normalized_content: str
    touched_files: tuple[str, ...]
    added_lines: int
    removed_lines: int
    hunks: int


def build_structured_edit_ops(
    *,
    task_run_id: str,
    planned_modifications: tuple[str, ...],
    planned_creations: tuple[str, ...],
    primary_targets: tuple[str, ...],
    acceptance_criteria: tuple[str, ...],
    owner_detection=None,
) -> tuple[StructuredEditOp, ...]:
    ops: list[StructuredEditOp] = []
    counter = 0
    primary_owners = set(getattr(owner_detection, "primary_owner_files", ()) or ())
    primary_artifacts = set(primary_targets or getattr(owner_detection, "primary_artifacts", ()) or ())
    validators = set(getattr(owner_detection, "validator_files", ()) or ())
    support_files = set(getattr(owner_detection, "support_files", ()) or ())
    stale_files = set(getattr(owner_detection, "stale_files", ()) or ())
    for target in planned_modifications:
        counter += 1
        if target in primary_artifacts:
            target_role = "primary_artifact"
        elif target in primary_owners:
            target_role = "owning_source"
        elif target in stale_files:
            target_role = "stale_file"
        elif target in validators:
            target_role = "validator"
        elif target in support_files:
            target_role = "support_file"
        else:
            target_role = "owning_source"
        ops.append(
            StructuredEditOp(
                operation_id=f"{task_run_id}-op-{counter}",
                op_type="modify_notebook" if target.endswith(".ipynb") else "replace_file",
                target_path=target,
                target_role=target_role,
                reason="Apply the requested in-place workspace edit.",
                preconditions=("target must exist inside the active workspace",),
                expected_postconditions=("content changes land on the target file",),
                source_read_evidence_ids=(),
                acceptance_criteria_supported=acceptance_criteria,
            )
        )
    for target in planned_creations:
        counter += 1
        if target in validators:
            target_role = "validator"
        elif target in support_files:
            target_role = "support_file"
        else:
            target_role = "support_file"
        ops.append(
            StructuredEditOp(
                operation_id=f"{task_run_id}-op-{counter}",
                op_type="create_file",
                target_path=target,
                target_role=target_role,
                reason="Create the focused support or validation file for the current task.",
                preconditions=("target must stay inside the active workspace",),
                expected_postconditions=("created file exists after apply",),
                source_read_evidence_ids=(),
                acceptance_criteria_supported=acceptance_criteria,
            )
        )
    return tuple(ops)


def extract_structured_file_block_specs(answer_text: str) -> dict[str, StructuredFileBlock]:
    if not answer_text.strip():
        return {}
    blocks: dict[str, StructuredFileBlock] = {}
    patterns = (
        r"^(?:===|#{1,6})\s*FILE:\s*(?P<path>.+?)(?:\s*===)?\s*\n```(?P<lang>[^\n]*)\n(?P<content>.*?)\n```",
        r"^\*\*(?:\d+\.\s*)?(?:file:\s*)?(?P<path>[^*\n]+)\*\*\s*\n```(?P<lang>[^\n]*)\n(?P<content>.*?)\n```",
        r"^#{1,6}\s*FILE BLOCK\s*-\s*(?P<path>.+?)\s*\n```(?P<lang>[^\n]*)\n(?P<content>.*?)\n```",
    )
    for pattern in patterns:
        matches = re.finditer(pattern, answer_text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        for match in matches:
            raw_path = _normalize_block_path(match.group("path"))
            if not raw_path:
                continue
            blocks.setdefault(
                raw_path,
                StructuredFileBlock(
                    path=raw_path,
                    content=match.group("content").rstrip(),
                    language=(match.groupdict().get("lang", "") or "").strip().lower(),
                ),
            )
    return blocks


def apply_unified_diff_text(
    *,
    diff_text: str,
    original_content: str,
    workspace_root: Path,
    expected_target: str,
) -> UnifiedDiffResult:
    patch_set = PatchSet(StringIO(diff_text))
    if not patch_set:
        raise ValueError("Unified diff did not contain any file patches.")
    touched_files: list[str] = []
    added_lines = 0
    removed_lines = 0
    hunks = 0
    patched_content = original_content
    for patch in patch_set:
        target = _normalize_patch_path(patch.target_file or patch.source_file)
        if not target:
            raise ValueError("Unified diff is missing a valid target path.")
        if target.startswith("../") or "/../" in target:
            raise ValueError("Unified diff targets a parent-directory path, which is not allowed.")
        if expected_target and target != _normalize_block_path(expected_target):
            raise ValueError(f"Unified diff targeted `{target}` instead of `{expected_target}`.")
        if not (workspace_root / target).resolve().is_relative_to(workspace_root.resolve()):
            raise ValueError("Unified diff targets a path outside the active workspace.")
        touched_files.append(target)
        added_lines += int(patch.added)
        removed_lines += int(patch.removed)
        hunks += len(patch)
        patched_content = _apply_patch_to_text(patch, patched_content)
    return UnifiedDiffResult(
        normalized_content=patched_content,
        touched_files=tuple(dict.fromkeys(touched_files)),
        added_lines=added_lines,
        removed_lines=removed_lines,
        hunks=hunks,
    )


def landed_edit_evidence(
    *,
    operation: StructuredEditOp,
    before_content: str | None,
    after_content: str,
    added_lines: int = 0,
    removed_lines: int = 0,
    status: str = "landed",
) -> LandedEditEvidence:
    before_hash = _sha256_text(before_content or "")
    after_hash = _sha256_text(after_content)
    return LandedEditEvidence(
        operation_id=operation.operation_id,
        target_path=operation.target_path,
        target_role=operation.target_role,
        before_hash=before_hash,
        after_hash=after_hash,
        added_lines=added_lines,
        removed_lines=removed_lines,
        status=status,
    )


def _apply_patch_to_text(patch, original_content: str) -> str:
    source_lines = original_content.splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    for hunk in patch:
        hunk_start = max(int(hunk.source_start) - 1, 0)
        output.extend(source_lines[cursor:hunk_start])
        cursor = hunk_start
        for line in hunk:
            value = line.value
            if line.is_context:
                if cursor >= len(source_lines):
                    raise ValueError("Unified diff context exceeded the source file length.")
                output.append(source_lines[cursor])
                cursor += 1
            elif line.is_removed:
                if cursor >= len(source_lines):
                    raise ValueError("Unified diff removal exceeded the source file length.")
                cursor += 1
            elif line.is_added:
                output.append(value)
    output.extend(source_lines[cursor:])
    return "".join(output)


def _normalize_patch_path(path_text: str) -> str:
    normalized = path_text.strip()
    normalized = re.sub(r"^[ab]/", "", normalized)
    return _normalize_block_path(normalized)


def _normalize_block_path(path_text: str) -> str:
    candidate = path_text.strip().replace("\\", "/")
    candidate = re.sub(r"^(?:file:\s*)", "", candidate, flags=re.IGNORECASE)
    candidate = str(Path(candidate)).replace("\\", "/")
    candidate = candidate.lstrip("/").strip().lower()
    return candidate


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

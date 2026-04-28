from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from labai.owner_detection import OwnerDetectionResult
from labai.task_manifest import TaskManifest


@dataclass(frozen=True)
class InspectionEvidence:
    file_path: str
    line_range: str
    chars_exposed: int
    timestamp: str
    task_run_id: str
    reason: str
    excerpt: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def build_required_read_set(
    manifest: TaskManifest,
    owner_detection: OwnerDetectionResult,
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for item in manifest.prompt_named_files:
        pairs.append((item, "prompt_named"))
    for item in manifest.wildcard_matches:
        pairs.append((item, "wildcard_match"))
    for item in manifest.primary_target_artifacts:
        pairs.append((item, "primary_artifact"))
    for item in owner_detection.primary_owner_files:
        pairs.append((item, "owner_candidate"))
    for item in manifest.reference_files:
        pairs.append((item, "reference_implementation"))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path, reason in pairs:
        key = (path, reason)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((path, reason))
    return tuple(deduped)


def inspect_required_reads(
    workspace_root: Path,
    manifest: TaskManifest,
    owner_detection: OwnerDetectionResult,
    *,
    timestamp: str,
    max_chars_per_file: int = 2400,
) -> tuple[InspectionEvidence, ...]:
    evidence: list[InspectionEvidence] = []
    for relative_path, reason in build_required_read_set(manifest, owner_detection):
        path = (workspace_root / relative_path).resolve()
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        start_line, end_line, excerpt = _select_excerpt(lines, manifest.user_instruction, max_chars=max_chars_per_file)
        evidence.append(
            InspectionEvidence(
                file_path=relative_path,
                line_range=f"{start_line}:{end_line}",
                chars_exposed=len(excerpt),
                timestamp=timestamp,
                task_run_id=manifest.task_run_id,
                reason=reason,
                excerpt=excerpt,
            )
        )
    return tuple(evidence)


def view_file_window(path: Path, *, start_line: int = 1, end_line: int = 80) -> str:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = max(start_line, 1)
    end = max(end_line, start)
    return "\n".join(lines[start - 1 : end])


def search_workspace(workspace_root: Path, pattern: str) -> tuple[str, ...]:
    matches: list[str] = []
    compiled = re.compile(re.escape(pattern), re.IGNORECASE)
    for path in workspace_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if compiled.search(text):
            matches.append(path.relative_to(workspace_root).as_posix())
    return tuple(matches[:20])


def _select_excerpt(lines: list[str], prompt: str, *, max_chars: int) -> tuple[int, int, str]:
    prompt_tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", prompt)]
    anchor_index = 0
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(token in lowered for token in prompt_tokens):
            anchor_index = index
            break
    start = max(anchor_index - 8, 0)
    end = min(start + 40, len(lines))
    excerpt_lines = lines[start:end]
    excerpt = "\n".join(excerpt_lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip()
    return start + 1, end, excerpt

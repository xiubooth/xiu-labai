from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import unicodedata

from labai.config import format_project_path
from labai.workspace import WorkspaceAccessManager

from .session import RuntimePersistenceError


@dataclass(frozen=True)
class AnswerArtifact:
    status: str
    path: str = ""
    format: str = "markdown"
    auto_generated: bool = True
    error: str = ""
    policy: str = ""
    intent: str = ""
    decision_reason: str = ""
    file_name: str = ""
    naming_key: str = ""
    collision_suffix: int = 0
    destination_policy: str = "repo_outputs"
    destination_fallback_reason: str = ""


class MarkdownArtifactWriter:
    def __init__(
        self,
        outputs_dir: Path,
        project_root: Path,
        access_manager: WorkspaceAccessManager | None = None,
    ) -> None:
        self.outputs_dir = Path(outputs_dir).resolve()
        self.project_root = Path(project_root).resolve()
        self.access_manager = access_manager

    def write_answer(
        self,
        *,
        session_id: str,
        prompt: str,
        answer: str,
        mode: str,
        completed_at: str,
        response_language: str = "en",
        target_paths: tuple[str, ...] = (),
        include_metadata_comment: bool = False,
    ) -> AnswerArtifact:
        body = answer.strip()
        if not body:
            return AnswerArtifact(
                status="skipped",
                error="empty_answer",
            )

        naming_key = _build_naming_key(
            prompt=prompt,
            mode=mode,
            response_language=response_language,
            target_paths=target_paths,
        )
        destination_dir, destination_policy, fallback_reason = self._choose_destination(
            target_paths=target_paths,
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        file_name, collision_suffix = _resolve_collision(destination_dir, naming_key)
        output_path = destination_dir / file_name
        rendered = body.rstrip() + "\n"
        if include_metadata_comment:
            rendered += (
                f"<!-- session_id: {session_id}; mode: {mode}; naming_key: {naming_key}; "
                f"collision_suffix: {collision_suffix}; destination_policy: {destination_policy}; "
                f"destination_fallback_reason: {fallback_reason or 'none'} -->\n"
            )

        try:
            output_path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            if destination_policy == "source_adjacent":
                fallback_reason = fallback_reason or "source_adjacent_write_failed"
                destination_dir = self.outputs_dir
                destination_policy = "repo_outputs"
                destination_dir.mkdir(parents=True, exist_ok=True)
                file_name, collision_suffix = _resolve_collision(destination_dir, naming_key)
                output_path = destination_dir / file_name
                try:
                    output_path.write_text(rendered, encoding="utf-8")
                except OSError as fallback_exc:
                    raise RuntimePersistenceError(
                        f"Could not write Markdown artifact to {output_path}"
                    ) from fallback_exc
            else:
                raise RuntimePersistenceError(
                    f"Could not write Markdown artifact to {output_path}"
                ) from exc

        return AnswerArtifact(
            status="generated",
            path=self._display_path(output_path),
            format="markdown",
            policy="always",
            intent="answer_only",
            decision_reason="generated_from_answer_body",
            file_name=file_name,
            naming_key=naming_key,
            collision_suffix=collision_suffix,
            destination_policy=destination_policy,
            destination_fallback_reason=fallback_reason,
        )

    def describe_existing_output(
        self,
        *,
        output_path: str,
        format_name: str = "markdown",
        policy: str = "explicit_only",
        intent: str = "deliverable_requested",
        decision_reason: str = "explicit_deliverable_request",
        destination_policy: str = "workspace_target",
        collision_suffix: int = 0,
        destination_fallback_reason: str = "",
    ) -> AnswerArtifact:
        resolved = self._resolve_target_path(output_path)
        display_path = output_path
        if resolved is not None:
            display_path = self._display_path(resolved)
        output_name = Path(output_path).name
        return AnswerArtifact(
            status="generated",
            path=display_path,
            format=format_name,
            auto_generated=False,
            policy=policy,
            intent=intent,
            decision_reason=decision_reason,
            file_name=output_name,
            naming_key=Path(output_name).stem,
            collision_suffix=collision_suffix,
            destination_policy=destination_policy,
            destination_fallback_reason=destination_fallback_reason,
        )

    def _choose_destination(self, *, target_paths: tuple[str, ...]) -> tuple[Path, str, str]:
        pdf_paths = [
            self._resolve_target_path(path_text)
            for path_text in target_paths
            if path_text.lower().endswith(".pdf")
        ]
        if not pdf_paths:
            return self.outputs_dir, "repo_outputs", ""

        if any(path is None for path in pdf_paths):
            return self.outputs_dir, "repo_outputs", "source_outside_project_root"

        resolved_pdf_paths = [path for path in pdf_paths if path is not None]
        if not all(path.is_file() for path in resolved_pdf_paths):
            return self.outputs_dir, "repo_outputs", "source_pdf_missing"

        parent_dirs = {path.parent.resolve() for path in resolved_pdf_paths}
        if len(parent_dirs) != 1:
            return self.outputs_dir, "repo_outputs", "mixed_source_directories"

        parent_dir = next(iter(parent_dirs))
        if not parent_dir.exists() or not parent_dir.is_dir():
            return self.outputs_dir, "repo_outputs", "source_directory_missing"
        if not self._can_write_to(parent_dir):
            return self.outputs_dir, "repo_outputs", "source_directory_not_writable"
        if not os.access(parent_dir, os.W_OK):
            return self.outputs_dir, "repo_outputs", "source_directory_not_writable"

        return parent_dir, "source_adjacent", ""

    def _resolve_target_path(self, path_text: str) -> Path | None:
        if self.access_manager is not None:
            return self.access_manager.resolve_prompt_path(path_text, must_exist=True)
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        resolved = candidate.resolve()
        if not _is_within(resolved, self.project_root):
            return None
        return resolved

    def _display_path(self, path: Path) -> str:
        if self.access_manager is not None:
            return self.access_manager.display_path(path)
        return format_project_path(path, self.project_root)

    def _can_write_to(self, path: Path) -> bool:
        if self.access_manager is None:
            return _is_within(path, self.project_root)
        return self.access_manager.is_allowed(path, for_write=True)


def _build_naming_key(
    *,
    prompt: str,
    mode: str,
    response_language: str,
    target_paths: tuple[str, ...],
) -> str:
    pdf_labels = [_target_label(path) for path in target_paths if path.lower().endswith(".pdf")]
    language_suffix = "_zh" if response_language == "zh-CN" else ""
    prompt_lower = prompt.lower()

    if mode == "paper_summary":
        base = f"{pdf_labels[0]}_summary" if pdf_labels else "paper_summary"
        return f"{base}{language_suffix}"

    if mode == "paper_compare":
        if len(pdf_labels) >= 2:
            base = f"{pdf_labels[0]}_vs_{pdf_labels[1]}_compare"
        elif len(pdf_labels) == 1:
            base = f"{pdf_labels[0]}_compare"
        else:
            base = "paper_compare"
        return f"{base}{language_suffix}"

    if mode == "paper_grounded_qa":
        prefix = pdf_labels[0] if len(pdf_labels) == 1 else "paper_library"
        return f"{prefix}_{_question_intent_slug(prompt)}{language_suffix}"

    if mode == "architecture_review":
        if "claw" in prompt_lower and "native" in prompt_lower and _contains_compare_intent(prompt_lower):
            return f"claw_vs_native_compare{language_suffix}"
        if "architecture" in prompt_lower or "\u67b6\u6784" in prompt:
            return f"repo_architecture_summary{language_suffix}"
        return f"architecture_review{language_suffix}"

    if mode == "compare_options":
        if "claw" in prompt_lower and "native" in prompt_lower:
            return f"claw_vs_native_compare{language_suffix}"
        return f"compare_options{language_suffix}"

    if mode == "implementation_plan":
        if "pdf" in prompt_lower and "ingest" in prompt_lower:
            return f"pdf_ingest_implementation_plan{language_suffix}"
        return f"implementation_plan{language_suffix}"

    if mode == "prompt_compiler":
        return f"compiled_prompt{language_suffix}"

    if mode == "repo_overview":
        if "architecture" in prompt_lower or "\u67b6\u6784" in prompt:
            return f"repo_architecture_summary{language_suffix}"
        return f"repo_overview_summary{language_suffix}"
    if mode == "workspace_verification":
        return f"workspace_verification{language_suffix}"

    if mode == "file_explain":
        return f"file_explain{language_suffix}"

    return f"answer{language_suffix}"


def _resolve_collision(destination_dir: Path, naming_key: str) -> tuple[str, int]:
    candidate = f"{naming_key}.md"
    if not (destination_dir / candidate).exists():
        return candidate, 0

    suffix = 2
    while True:
        candidate = f"{naming_key}_{suffix}.md"
        if not (destination_dir / candidate).exists():
            return candidate, suffix
        suffix += 1


def _question_intent_slug(prompt: str) -> str:
    prompt_lower = prompt.lower()
    patterns = (
        (("recurring limitations", "main recurring limitations"), "recurring_limitations"),
        (("main limitation", "main limitations"), "main_limitation"),
        (("limitation", "limitations"), "limitations"),
        (("core method", "core methods"), "core_method"),
        (("method", "methods"), "method_summary"),
        (("architecture",), "architecture_summary"),
        (("compile prompt", "compiled prompt", "prompt compiler"), "compiled_prompt"),
        (("implementation plan", "plan"), "implementation_plan"),
        (("summary", "summarize"), "summary"),
    )
    chinese_patterns = (
        (("\u6700\u4e3b\u8981\u7684\u5c40\u9650", "\u4e3b\u8981\u5c40\u9650"), "main_limitation"),
        (("\u5c40\u9650",), "limitations"),
        (("\u6838\u5fc3\u65b9\u6cd5",), "core_method"),
        (("\u65b9\u6cd5",), "method_summary"),
        (("\u67b6\u6784",), "architecture_summary"),
        (("\u63d0\u793a\u8bcd", "\u7f16\u8bd1\u63d0\u793a", "\u91cd\u5199\u63d0\u793a"), "compiled_prompt"),
        (("\u603b\u7ed3",), "summary"),
    )

    for tokens, label in patterns:
        if any(token in prompt_lower for token in tokens):
            return label
    for tokens, label in chinese_patterns:
        if any(token in prompt for token in tokens):
            return label

    meaningful_words = [
        _sanitize_token(word, lowercase=True)
        for word in re.findall(r"[A-Za-z0-9]+", prompt)
        if word.lower()
        not in {
            "the",
            "a",
            "an",
            "what",
            "is",
            "are",
            "in",
            "of",
            "on",
            "and",
            "to",
            "based",
            "paper",
            "papers",
            "pdf",
            "examples",
            "under",
            "this",
        }
    ]
    meaningful_words = [word for word in meaningful_words if word]
    if meaningful_words:
        return "_".join(meaningful_words[:3])
    return "question"


def _contains_compare_intent(prompt_lower: str) -> bool:
    return any(token in prompt_lower for token in ("compare", "comparison", "vs", "versus", "tradeoff", "trade-off"))


def _target_label(path_text: str) -> str:
    stem = Path(path_text).stem
    return _sanitize_token(stem, lowercase=False) or "paper"


def _sanitize_token(text: str, *, lowercase: bool) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    ascii_text = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    if lowercase:
        ascii_text = ascii_text.lower()
    return ascii_text[:48] or ""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

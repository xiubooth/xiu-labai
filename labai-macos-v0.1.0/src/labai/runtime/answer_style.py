from __future__ import annotations

import re

_EVIDENCE_SECTION_HEADINGS = (
    "Evidence/files consulted",
    "Evidence refs",
    "Grounded supporting evidence",
    "Retrieved chunk excerpts",
)
_SCHEMA_HEADINGS = {
    "Document identity",
    "Main contribution / purpose",
    "Method / structure",
    "Important findings / conclusion",
    "Key caveats",
    "Whole-document coverage",
    "Direct answer",
    "Grounded supporting evidence",
    "Uncertainty",
    "Retrieved chunk excerpts",
    "Relevant components",
    "Data/control flow",
    "Runtime path and fallback path",
    "Interaction points",
    "Risks and hidden assumptions",
    "Goal",
    "Proposed steps",
    "Likely files/modules to change",
    "Risks",
    "Validation plan",
    "Documents compared",
    "Commonalities",
    "Differences",
    "Strengths / weaknesses / limitations",
    "Recommendation or synthesis",
    "Purpose",
    "Main directories/modules",
    "Important entry points",
    "Current runtime path",
    "Key risks or caveats",
    "File purpose",
    "Key functions/classes",
    "Inputs and outputs",
    "Dependencies",
    "Risks or confusing spots",
    "Options being compared",
    "Strengths",
    "Weaknesses",
    "Tradeoffs",
    "Recommendation",
}
_OPERATIONAL_PREFIXES = (
    "selected_mode:",
    "mode_reason:",
    "answer_schema:",
    "read_strategy:",
    "read_strategy_reason:",
    "response_style:",
    "response_language:",
    "selected_model:",
    "selected_embedding_model:",
    "operational_status:",
    "requested_runtime:",
    "runtime_used:",
    "runtime_fallback:",
    "requested_provider:",
    "provider_used:",
    "provider_fallback:",
    "tools_used:",
    "tool_count:",
    "evidence_count:",
    "paper_target_count:",
    "paper_document_count:",
    "paper_read_strategy:",
    "paper_window_count:",
    "paper_retrieval_count:",
    "paper_indexed_documents:",
    "status:",
    "session_id:",
    "artifact_status:",
    "artifact_path:",
    "artifact_file:",
    "session_file:",
    "audit_log:",
    "answer:",
)


def needs_style_repair(
    text: str,
    *,
    prompt: str = "",
    response_language: str,
    response_style: str,
    include_explicit_evidence_refs: bool,
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if response_language == "zh-CN" and not looks_like_chinese_response(stripped):
        return True
    if response_style == "continuous_prose" and looks_like_structured_output(stripped):
        return True
    if not _prefers_machine_readable_output(prompt) and _looks_like_machine_readable_output(stripped):
        return True
    if not include_explicit_evidence_refs and _contains_hidden_metadata(stripped):
        return True
    return False


def normalize_answer_text(
    text: str,
    *,
    response_language: str,
    response_style: str,
    include_explicit_evidence_refs: bool,
) -> str:
    normalized = text.strip()
    if not normalized:
        return ""

    normalized = _strip_operational_lines(normalized)
    if not include_explicit_evidence_refs:
        normalized = _strip_evidence_sections(normalized)
    if response_style == "continuous_prose":
        normalized = _flatten_to_continuous_prose(normalized, response_language=response_language)
    return normalized.strip()


def looks_like_structured_output(text: str) -> bool:
    if re.search(r"(?m)^\s*(?:[-*]|\d+\.)\s+", text):
        return True
    if re.search(r"(?m)^#{1,6}\s+\S+", text):
        return True
    if re.search(r"(?m)^[A-Z][A-Za-z /_-]+:$", text):
        return True
    for line in text.splitlines():
        if _normalize_heading_candidate(line) in _SCHEMA_HEADINGS:
            return True
    return _contains_hidden_metadata(text)


def looks_like_chinese_response(text: str) -> bool:
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_letters = sum(1 for char in text if "a" <= char.lower() <= "z")
    if chinese_chars >= 24:
        return True
    return chinese_chars > 0 and chinese_chars >= max(8, ascii_letters // 3)


def build_rewrite_requirements(
    *,
    response_language: str,
    response_style: str,
    include_explicit_evidence_refs: bool,
) -> list[str]:
    requirements: list[str] = []
    if response_language == "zh-CN":
        requirements.append("The final answer MUST be in Simplified Chinese.")
    if response_style == "continuous_prose":
        requirements.append(
            "The final answer MUST be one continuous prose passage with no bullets, no numbering, no outline, and no section headings."
        )
    if not include_explicit_evidence_refs:
        requirements.append("Do not add an evidence appendix, evidence section, files-consulted footer, or source list.")
    requirements.append("Do not add new facts; preserve only facts already present in the draft answer and the original prompt.")
    requirements.append("Do not mention runtime metadata, internal mode names, session ids, or operational traces.")
    requirements.append("Return readable natural-language output, not JSON, YAML, or key-value objects, unless the user explicitly asked for a machine-readable format.")
    requirements.append("Return only the rewritten answer body with no prefatory label.")
    return requirements


def _flatten_to_continuous_prose(text: str, *, response_language: str) -> str:
    segments: list[str] = []
    for raw_line in text.splitlines():
        cleaned = _clean_prose_segment(raw_line)
        if not cleaned:
            continue
        heading = _normalize_heading_candidate(cleaned)
        if heading in _SCHEMA_HEADINGS or heading in _EVIDENCE_SECTION_HEADINGS:
            continue
        segments.append(cleaned)

    if not segments:
        return text.strip()

    if response_language == "zh-CN":
        return _join_chinese_segments(segments)
    return " ".join(segments)


def _clean_prose_segment(raw_line: str) -> str:
    cleaned = raw_line.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^\s*(?:[-*+]\s+|\d+\.\s+)", "", cleaned)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _join_chinese_segments(segments: list[str]) -> str:
    text = ""
    for segment in segments:
        if not text:
            text = segment
            continue
        if text.endswith(("\uff1a", ":")):
            text += segment
            continue
        if not text.endswith(("\u3002", "\uff01", "\uff1f", "\uff1b")):
            text += "\uff1b"
        text += segment
    if text and text[-1] not in "\u3002\uff01\uff1f":
        text += "\u3002"
    return text


def _strip_evidence_sections(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    skip_evidence_block = False

    for raw_line in lines:
        heading = _normalize_heading_candidate(raw_line)
        if heading in _EVIDENCE_SECTION_HEADINGS:
            skip_evidence_block = True
            continue
        if skip_evidence_block:
            if heading in _SCHEMA_HEADINGS:
                skip_evidence_block = False
            elif raw_line.strip():
                continue
            else:
                continue
        cleaned.append(raw_line)
    return "\n".join(cleaned).strip()


def _strip_operational_lines(text: str) -> str:
    lines = [
        line
        for line in text.splitlines()
        if not any(line.strip().lower().startswith(prefix) for prefix in _OPERATIONAL_PREFIXES)
    ]
    return "\n".join(lines).strip()


def _contains_hidden_metadata(text: str) -> bool:
    if any(marker in text for marker in _EVIDENCE_SECTION_HEADINGS):
        return True
    lowered_lines = [line.strip().lower() for line in text.splitlines()]
    return any(any(line.startswith(prefix) for prefix in _OPERATIONAL_PREFIXES) for line in lowered_lines)


def _looks_like_machine_readable_output(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("{") and stripped.endswith("}"):
        return True
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    return False


def _prefers_machine_readable_output(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(token in lowered for token in ("json", "yaml", "toml", "csv", "tsv", "xml"))


def _normalize_heading_candidate(text: str) -> str:
    candidate = _clean_prose_segment(text).rstrip(":\uff1a").strip()
    return candidate

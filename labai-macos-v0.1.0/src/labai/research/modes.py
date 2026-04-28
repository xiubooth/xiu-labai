from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from labai.config import LabaiConfig
from labai.workspace import WorkspaceAccessManager

ResearchMode = Literal[
    "general_chat",
    "repo_overview",
    "workspace_verification",
    "project_onboarding",
    "file_explain",
    "architecture_review",
    "implementation_plan",
    "workspace_edit",
    "prompt_compiler",
    "compare_options",
    "paper_summary",
    "paper_compare",
    "paper_grounded_qa",
]
ReadStrategy = Literal["none", "full_document", "retrieval", "hybrid"]
ResponseStyle = Literal["structured", "continuous_prose"]
PaperOutputProfile = Literal["none", "quick_summary", "detailed_paper_note"]

GENERAL_MODEL_MODES = frozenset(
    {
        "general_chat",
        "repo_overview",
        "workspace_verification",
        "project_onboarding",
        "architecture_review",
        "paper_summary",
        "paper_compare",
        "paper_grounded_qa",
    }
)
_PATH_TOKEN_PATTERN = re.compile(
    r"(?P<token>(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+\.(?:pdf|py|md|toml|json|ya?ml|ps1|txt))"
)
_COMPARE_KEYWORDS = (
    "compare",
    "comparison",
    "versus",
    "vs",
    "tradeoff",
    "trade-off",
    "pros and cons",
    "recommend when",
    "\u6bd4\u8f83",
    "\u5bf9\u6bd4",
    "\u53d6\u820d",
)
_PLAN_KEYWORDS = (
    "implementation plan",
    "draft a plan",
    "plan for",
    "next phase",
    "proposed steps",
    "how would you add",
    "implementation steps",
    "validate",
    "pdf ingest",
    "\u8ba1\u5212",
    "\u65b9\u6848",
    "\u6b65\u9aa4",
    "\u4e0b\u4e00\u9636\u6bb5",
    "\u5b9e\u73b0\u8ba1\u5212",
)
_ARCHITECTURE_KEYWORDS = (
    "architecture",
    "runtime path",
    "fallback path",
    "fit together",
    "control flow",
    "data flow",
    "interaction",
    "\u67b6\u6784",
    "\u6d41\u7a0b",
    "\u56de\u9000\u8def\u5f84",
    "\u8fd0\u884c\u65f6\u8def\u5f84",
    "\u7ec4\u4ef6",
)
_FILE_KEYWORDS = (
    "explain file",
    "this file",
    "what does",
    "module",
    "class",
    "function",
    "explain",
    "\u6587\u4ef6",
    "\u6a21\u5757",
)
_WORKSPACE_WRITE_KEYWORDS = (
    "create ",
    "generate ",
    "write ",
    "open ",
    "fix ",
    "modify ",
    "edit ",
    "update ",
    "change ",
    "repair ",
    "refactor ",
    "implement ",
    "synchronize ",
    "sync ",
    "\u751f\u6210",
    "\u521b\u5efa",
    "\u65b0\u5efa",
    "\u4fee\u590d",
    "\u4fee\u6539",
    "\u7f16\u8f91",
    "\u66f4\u65b0",
    "\u91cd\u6784",
)
_WORKSPACE_EDIT_STRUCTURAL_KEYWORDS = (
    "bug",
    "pytest",
    "test ",
    "tests ",
    "check ",
    "checks ",
    "code",
    "config",
    "entrypoint",
    "runner",
    "readme",
    "handoff",
    "next_steps.md",
    "handoff_notes.md",
    "src/",
    "tests/",
    ".py",
    ".md",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    "\u6d4b\u8bd5",
    "\u4ee3\u7801",
    "\u914d\u7f6e",
    "\u5165\u53e3",
)
_OVERVIEW_KEYWORDS = (
    "overview",
    "summarize the repo",
    "summarize this repository",
    "new ra",
    "onboarding",
    "project structure",
    "repository",
    "repo",
    "\u9879\u76ee",
    "\u603b\u7ed3",
    "\u6982\u89c8",
)
_VERIFY_WORKSPACE_KEYWORDS = (
    "verify workspace",
    "verify-workspace",
    "workspace ready",
    "workspace readiness",
    "readiness state",
    "ready for me to work in",
    "ready for work",
    "ready to work in",
    "take over today",
    "what is missing",
    "what looks risky",
    "what looks blocked",
    "what should i fix first",
    "first three practical next steps",
    "can i meaningfully start working",
    "should i start working",
    "is this workspace ready",
    "day-one",
    "day one",
    "\u5de5\u4f5c\u533a\u662f\u5426 ready",
    "\u5de5\u4f5c\u533a\u662f\u5426\u5c31\u7eea",
    "\u662f\u5426 ready",
    "\u662f\u5426\u53ef\u4ee5\u63a5\u624b",
    "\u80fd\u4e0d\u80fd\u5f00\u59cb\u5de5\u4f5c",
    "\u7f3a\u4ec0\u4e48",
    "\u6709\u54ea\u4e9b\u98ce\u9669",
    "\u6700\u53ef\u80fd\u7684\u5165\u53e3",
    "\u7b2c\u4e00\u6b65\u5e94\u8be5\u505a\u4ec0\u4e48",
)
_ONBOARDING_KEYWORDS = (
    "onboard",
    "onboarding",
    "onboard-project",
    "new ra",
    "read first",
    "start reading",
    "what should i read first",
    "inherit this project",
    "project purpose",
    "main modules",
    "entry points",
    "next steps for a new ra",
    "\u63a5\u624b",
    "\u5165\u95e8",
    "\u9879\u76ee\u76ee\u7684",
    "\u4e3b\u8981\u6587\u4ef6",
    "\u4e3b\u8981\u6a21\u5757",
    "\u5165\u53e3",
    "\u7b2c\u4e00\u6b65\u8be5\u770b\u4ec0\u4e48",
    "\u65b0 ra",
)
_PAPER_KEYWORDS = (
    "pdf",
    "paper",
    "papers",
    "document",
    "documents",
    "\u8bba\u6587",
    "\u6587\u6863",
)
_SUMMARY_KEYWORDS = (
    "summarize",
    "summary",
    "summarise",
    "overview",
    "\u603b\u7ed3",
    "\u6982\u8ff0",
    "\u6458\u8981",
)
_QUESTION_KEYWORDS = (
    "based on",
    "according to",
    "what are",
    "what is",
    "which",
    "why",
    "how",
    "where",
    "\u4f9d\u636e",
    "\u6839\u636e",
    "\u54ea\u4e9b",
    "\u4ec0\u4e48",
    "\u4e3a\u4ec0\u4e48",
    "\u5982\u4f55",
    "\u5728\u54ea",
    "\u54ea\u91cc",
)
_CODE_HEAVY_KEYWORDS = (
    "code",
    "coder",
    "implementation",
    "implement",
    "api",
    "function",
    "class",
    "module",
    "pipeline",
    "plan",
    "\u4ee3\u7801",
    "\u5b9e\u73b0",
    "\u6a21\u5757",
    "\u63a5\u53e3",
)
_GREETING_PATTERNS = (
    "hello",
    "hi",
    "hey",
    "\u4f60\u597d",
    "\u55e8",
    "\u60a8\u597d",
)
_NO_BULLET_KEYWORDS = (
    "no bullets",
    "don't use bullets",
    "do not use bullets",
    "no outline",
    "continuous prose",
    "continuous summary",
    "paragraph form",
    "\u4e0d\u8981\u5206\u70b9",
    "\u4e0d\u8981\u5217\u63d0\u7eb2",
    "\u8fde\u7eed\u603b\u7ed3",
    "\u8fde\u7eed\u6bb5\u843d",
)
_FULL_DOCUMENT_KEYWORDS = (
    "whole paper",
    "entire paper",
    "full paper",
    "read the whole paper",
    "read the full paper",
    "full document",
    "complete read",
    "continuous summary",
    "cover background",
    "cover the whole paper",
    "\u5168\u6587",
    "\u6574\u7bc7",
    "\u5b8c\u6574\u901a\u8bfb",
    "\u5f53\u4f5c\u4e00\u7bc7\u9700\u8981\u5b8c\u6574\u901a\u8bfb\u7684\u8bba\u6587\u6765\u5904\u7406",
    "\u8fde\u7eed\u603b\u7ed3",
    "\u7814\u7a76\u80cc\u666f",
    "\u6837\u672c\u4e0e\u6570\u636e",
    "\u603b\u4f53\u7ed3\u8bba",
)
_DETAIL_DIMENSION_KEYWORDS = (
    "background",
    "problem",
    "sample",
    "samples",
    "data",
    "dataset",
    "method",
    "methods",
    "model",
    "models",
    "finding",
    "findings",
    "result",
    "results",
    "conclusion",
    "conclusions",
    "limitation",
    "limitations",
    "investment",
    "\u7814\u7a76\u80cc\u666f",
    "\u95ee\u9898",
    "\u6837\u672c",
    "\u6570\u636e",
    "\u65b9\u6cd5",
    "\u6a21\u578b",
    "\u53d1\u73b0",
    "\u5b9e\u8bc1",
    "\u6295\u8d44\u542b\u4e49",
    "\u7ed3\u8bba",
    "\u5c40\u9650",
)
_DETAILED_PAPER_OUTPUT_KEYWORDS = (
    "detailed summary",
    "detailed paper note",
    "preserve concrete details",
    "concrete details",
    "concrete numbers",
    "exact methods",
    "exact findings",
    "include sample/data details",
    "include sample details",
    "include data details",
    "date range",
    "date ranges",
    "time range",
    "sample definition",
    "sample setup",
    "train/validation/test",
    "training/validation/test",
    "training split",
    "validation split",
    "test split",
    "method family",
    "method families",
    "do not omit important details",
    "do not omit key details",
    "\u8be6\u7ec6",
    "\u8be6\u7ec6\u603b\u7ed3",
    "\u4fdd\u7559\u5177\u4f53\u7ec6\u8282",
    "\u5177\u4f53\u7ec6\u8282",
    "\u5177\u4f53\u6570\u5b57",
    "\u65f6\u95f4\u533a\u95f4",
    "\u6837\u672c\u8bbe\u5b9a",
    "\u8bad\u7ec3/\u9a8c\u8bc1/\u6d4b\u8bd5",
    "\u8bad\u7ec3/\u9a8c\u8bc1/\u6d4b\u8bd5\u5212\u5206",
    "\u660e\u786e\u63d0\u5230\u7684\u65b9\u6cd5\u65cf",
    "\u65b9\u6cd5\u65cf",
    "\u4e0d\u8981\u4e3a\u4e86\u7b80\u6d01\u7701\u7565\u5173\u952e\u5185\u5bb9",
)
_SAMPLE_DATA_BACKSTOP_KEYWORDS = (
    "sample",
    "samples",
    "data",
    "dataset",
    "datasets",
    "sample period",
    "sample size",
    "asset universe",
    "data source",
    "\u6837\u672c",
    "\u6570\u636e",
)
_EVIDENCE_REQUEST_KEYWORDS = (
    "grounded",
    "cite",
    "citation",
    "evidence",
    "page",
    "pages",
    "chunk",
    "where is",
    "where does",
    "\u8bc1\u636e",
    "\u5f15\u7528",
    "\u9875",
    "\u9875\u7801",
)
_RECURRING_LIMITATION_KEYWORDS = (
    "recurring limitations",
    "main recurring limitations",
    "limitations across papers",
    "limitations across pdfs",
    "common limitations",
    "cross-paper limitations",
    "\u5171\u540c\u5c40\u9650",
    "\u91cd\u590d\u51fa\u73b0\u7684\u5c40\u9650",
    "\u4e3b\u8981\u5c40\u9650",
)
_SAME_DOCUMENT_COMPARE_KEYWORDS = (
    "within the paper",
    "inside this paper",
    "same paper",
    "compare the methods in this paper",
    "compare the models in this paper",
    "\u6587\u4e2d\u5185\u90e8\u6bd4\u8f83",
    "\u6587\u4e2d\u6bd4\u8f83",
)
_ENGLISH_LANGUAGE_OVERRIDE_PATTERNS = (
    re.compile(r"\b(?:answer|respond|reply)\s+(?:entirely\s+)?in\s+english\b", re.IGNORECASE),
    re.compile(r"\buse\s+english\b", re.IGNORECASE),
    re.compile(r"\btake\s+english\s+answer\b", re.IGNORECASE),
    re.compile(r"\benglish\s+only\b", re.IGNORECASE),
    re.compile(r"\banswer\s+in\s+english\b", re.IGNORECASE),
    re.compile(r"\brespond\s+in\s+english\b", re.IGNORECASE),
    re.compile(r"\bplease\s+answer\s+in\s+english\b", re.IGNORECASE),
    re.compile(r"\u8bf7\u7528\u82f1\u6587\u56de\u7b54"),
    re.compile(r"\u62ff\u82f1\u6587\u56de\u7b54\u6211"),
    re.compile(r"\u7528\u82f1\u6587"),
)
_CHINESE_LANGUAGE_OVERRIDE_PATTERNS = (
    re.compile(r"\b(?:answer|respond|reply)\s+(?:entirely\s+)?in\s+(?:simplified\s+)?chinese\b", re.IGNORECASE),
    re.compile(r"\buse\s+(?:simplified\s+)?chinese\b", re.IGNORECASE),
    re.compile(r"\bchinese\s+only\b", re.IGNORECASE),
    re.compile(r"\u8bf7\u7528\u4e2d\u6587\u56de\u7b54"),
    re.compile(r"\u62ff\u4e2d\u6587\u56de\u7b54\u6211"),
    re.compile(r"\u7528\u4e2d\u6587"),
)
_BILINGUAL_OUTPUT_PATTERNS = (
    re.compile(r"\bbilingual\b", re.IGNORECASE),
    re.compile(r"\bboth\s+english\s+and\s+chinese\b", re.IGNORECASE),
    re.compile(r"\u4e2d\u82f1\u53cc\u8bed"),
    re.compile(r"\u82f1\u4e2d\u53cc\u8bed"),
)


@dataclass(frozen=True)
class ModeSelection:
    mode: ResearchMode
    reason: str
    answer_schema: str
    response_language: str
    selected_model: str
    read_strategy: ReadStrategy
    read_strategy_reason: str
    response_style: ResponseStyle
    include_explicit_evidence_refs: bool
    paper_output_profile: PaperOutputProfile
    paper_output_profile_reason: str
    matched_paths: tuple[str, ...] = ()
    response_language_reason: str = ""
    response_language_explicit_override: bool = False
    explicit_override: bool = False


@dataclass(frozen=True)
class AskRoutingDecision:
    mode_selection: ModeSelection
    answer_override: str = ""


def select_mode(config: LabaiConfig, prompt: str) -> ModeSelection:
    matched_paths = extract_prompt_paths(prompt, config)
    language, language_reason, language_explicit_override = detect_response_language(prompt)
    normalized = prompt.strip().lower()
    explicit_paper_targets = _has_explicit_paper_targets(matched_paths)
    workspace_edit_request = _looks_like_workspace_edit_request(normalized, matched_paths)

    if config.research.mode_override:
        mode = config.research.mode_override
        return _build_selection(
            config,
            mode,  # type: ignore[arg-type]
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Configured internal mode override is active.",
            explicit_override=True,
        )

    if _contains_any(normalized, _PLAN_KEYWORDS) and not explicit_paper_targets and not workspace_edit_request:
        return _build_selection(
            config,
            "implementation_plan",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt asks for a concrete plan or next steps.",
        )

    if workspace_edit_request and not explicit_paper_targets:
        return _build_selection(
            config,
            "workspace_edit",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt asks for concrete code or workspace changes, plus task-completion behavior such as checks or retries.",
        )

    if _looks_like_paper_prompt(normalized, matched_paths):
        return _select_paper_mode(
            config,
            prompt,
            normalized,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
        )

    if workspace_edit_request:
        return _build_selection(
            config,
            "workspace_edit",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt asks for concrete code or workspace changes, plus task-completion behavior such as checks or retries.",
        )
    if _contains_any(normalized, _COMPARE_KEYWORDS):
        return _build_selection(
            config,
            "compare_options",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt asks for a comparison or tradeoff.",
        )
    if _contains_any(normalized, _PLAN_KEYWORDS):
        return _build_selection(
            config,
            "implementation_plan",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt asks for a concrete plan or next steps.",
        )
    if _contains_any(normalized, _ARCHITECTURE_KEYWORDS):
        return _build_selection(
            config,
            "architecture_review",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt asks about architecture, flow, or runtime interactions.",
        )
    if _contains_any(normalized, _VERIFY_WORKSPACE_KEYWORDS):
        return _build_selection(
            config,
            "workspace_verification",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt asks whether a workspace is ready for practical RA work, including gaps, risks, or next steps.",
        )
    if _contains_any(normalized, _ONBOARDING_KEYWORDS):
        return _build_selection(
            config,
            "project_onboarding",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt asks for practical project onboarding for a new RA.",
        )
    if _contains_any(normalized, _OVERVIEW_KEYWORDS):
        return _build_selection(
            config,
            "repo_overview",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt asks for a repository overview or onboarding summary.",
        )
    if (matched_paths or _looks_like_file_explain(normalized)) and not _contains_any(normalized, _WORKSPACE_WRITE_KEYWORDS):
        return _build_selection(
            config,
            "file_explain",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt references a specific file or file-level explanation request.",
        )
    if _is_brief_greeting(normalized):
        return _build_selection(
            config,
            "repo_overview",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Brief generic greeting; defaulting to a lightweight overview-capable response.",
        )
    return _build_selection(
        config,
        "repo_overview",
        prompt,
        language,
        language_reason,
        language_explicit_override,
        matched_paths,
        "No stronger signal matched; defaulting to the general repo overview mode.",
    )


def route_ask_prompt(config: LabaiConfig, prompt: str) -> AskRoutingDecision:
    language, language_reason, language_explicit_override = detect_response_language(prompt)
    normalized = prompt.strip().lower()
    direct_answer_override = _build_ask_direct_answer_override(prompt)
    answer_override = direct_answer_override or _build_ask_answer_override(prompt)
    reason = (
        "labai ask is the lightweight direct-model answer surface. "
        "It answers from the prompt only and does not execute repo, workspace, PDF, or edit workflows automatically."
    )
    if answer_override and not direct_answer_override:
        reason = (
            "labai ask is the lightweight direct-model answer surface. "
            "This prompt requests real file, PDF, repository, or workspace operations, so ask returns explicit workflow guidance instead of executing a workflow."
        )

    mode_selection = ModeSelection(
        mode="general_chat",
        reason=reason,
        answer_schema="general_chat_response",
        response_language=language,
        selected_model=select_mode_model(config, "general_chat", prompt),
        read_strategy="none",
        read_strategy_reason="labai ask is a lightweight direct-answer surface and does not inspect files or documents automatically.",
        response_style=_select_response_style(normalized, "general_chat", "none"),
        include_explicit_evidence_refs=False,
        paper_output_profile="none",
        paper_output_profile_reason="Paper output profiles do not apply to direct lightweight ask routing.",
        matched_paths=(),
        response_language_reason=language_reason,
        response_language_explicit_override=language_explicit_override,
        explicit_override=False,
    )
    return AskRoutingDecision(
        mode_selection=mode_selection,
        answer_override=answer_override,
    )


def _build_ask_direct_answer_override(prompt: str) -> str:
    if not _looks_like_answer_only_prompt(prompt):
        return ""
    match = re.search(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)", prompt)
    if match is None:
        return ""
    left = int(match.group(1))
    operator = match.group(2)
    right = int(match.group(3))
    if operator == "+":
        return str(left + right)
    if operator == "-":
        return str(left - right)
    if operator == "*":
        return str(left * right)
    if operator == "/":
        if right == 0:
            return ""
        quotient = left / right
        if quotient.is_integer():
            return str(int(quotient))
        return format(quotient, "g")
    return ""


def _looks_like_answer_only_prompt(prompt: str) -> bool:
    normalized = prompt.strip().lower()
    answer_only_markers = (
        "only output",
        "only answer",
        "without saying other things",
        "without explanation",
        "without explaining",
        "just the answer",
        "只输出",
        "不要解释",
        "不要说别的",
    )
    return any(marker in normalized for marker in answer_only_markers)


def select_mode_model(config: LabaiConfig, mode: ResearchMode, prompt: str = "") -> str:
    normalized = prompt.lower()
    if mode.startswith("paper_") and _contains_any(normalized, _CODE_HEAVY_KEYWORDS):
        return config.models.code_model
    if mode in GENERAL_MODEL_MODES:
        return config.models.general_model
    return config.models.code_model


def detect_response_language(prompt: str) -> tuple[str, str, bool]:
    override = _detect_explicit_language_override(prompt)
    if override is not None:
        return override
    for character in prompt:
        if "\u4e00" <= character <= "\u9fff":
            return (
                "zh-CN",
                "Prompt contains Chinese text and no later explicit language override was detected.",
                False,
            )
    return (
        "en",
        "Defaulting to English because no Chinese prompt text or explicit language override was detected.",
        False,
    )


def extract_prompt_paths(prompt: str, config: LabaiConfig) -> tuple[str, ...]:
    access_manager = WorkspaceAccessManager(config)
    matches = list(access_manager.prompt_paths(prompt))
    seen = set(matches)
    if any(token in prompt for token in ("/", "\\")) or re.search(r"[A-Za-z]:\\", prompt):
        return tuple(matches[:8])
    for raw_match in _PATH_TOKEN_PATTERN.finditer(prompt):
        token = raw_match.group("token").strip("`'\"()[]{}<>.,:;")
        if not token or "/" in token or "\\" in token:
            continue
        for discovered in _find_by_basename(access_manager, token):
            if discovered in seen:
                continue
            seen.add(discovered)
            matches.append(discovered)
    return tuple(matches[:8])


def mode_router_summary() -> str:
    return (
        "ready | "
        "modes=general_chat,repo_overview,workspace_verification,project_onboarding,file_explain,architecture_review,implementation_plan,prompt_compiler,"
        "compare_options,workspace_edit,paper_summary,paper_compare,paper_grounded_qa | "
        "read_strategies=full_document,retrieval,hybrid"
    )


def model_selector_summary(config: LabaiConfig) -> str:
    return (
        "ready | "
        f"general_chat={config.models.general_model} | "
        f"repo_overview={config.models.general_model} | "
        f"workspace_verification={config.models.general_model} | "
        f"project_onboarding={config.models.general_model} | "
        f"architecture_review={config.models.general_model} | "
        f"file_explain={config.models.code_model} | "
        f"implementation_plan={config.models.code_model} | "
        f"workspace_edit={config.models.code_model} | "
        f"prompt_compiler={config.models.code_model} | "
        f"compare_options={config.models.code_model} | "
        f"paper_summary={config.models.general_model} | "
        f"paper_compare={config.models.general_model} | "
        f"paper_grounded_qa={config.models.general_model} or {config.models.code_model} when paper prompts are implementation-heavy"
    )


def _detect_explicit_language_override(prompt: str) -> tuple[str, str, bool] | None:
    if _prompt_requests_bilingual_output(prompt):
        return None

    latest_match: tuple[int, str, str] | None = None
    pattern_sets = (
        ("en", _ENGLISH_LANGUAGE_OVERRIDE_PATTERNS, "Most recent explicit language instruction requested English output."),
        ("zh-CN", _CHINESE_LANGUAGE_OVERRIDE_PATTERNS, "Most recent explicit language instruction requested Chinese output."),
    )
    for language, patterns, reason in pattern_sets:
        for pattern in patterns:
            for match in pattern.finditer(prompt):
                candidate = (match.end(), language, reason)
                if latest_match is None or candidate[0] >= latest_match[0]:
                    latest_match = candidate
    if latest_match is None:
        return None
    return latest_match[1], latest_match[2], True


def _prompt_requests_bilingual_output(prompt: str) -> bool:
    return any(pattern.search(prompt) for pattern in _BILINGUAL_OUTPUT_PATTERNS)


def _build_selection(
    config: LabaiConfig,
    mode: ResearchMode,
    prompt: str,
    language: str,
    language_reason: str,
    language_explicit_override: bool,
    matched_paths: tuple[str, ...],
    reason: str,
    *,
    explicit_override: bool = False,
) -> ModeSelection:
    normalized = prompt.strip().lower()
    read_strategy, read_strategy_reason = _select_read_strategy(mode, normalized, matched_paths)
    response_style = _select_response_style(normalized, mode, read_strategy)
    include_explicit_evidence_refs = _should_include_explicit_evidence_refs(
        normalized,
        mode,
        read_strategy,
    )
    paper_output_profile, paper_output_profile_reason = _select_paper_output_profile(
        mode,
        normalized,
    )
    return ModeSelection(
        mode=mode,
        reason=reason,
        answer_schema=_select_answer_schema(mode, read_strategy, response_style, prompt),
        response_language=language,
        selected_model=select_mode_model(config, mode, prompt),
        read_strategy=read_strategy,
        read_strategy_reason=read_strategy_reason,
        response_style=response_style,
        include_explicit_evidence_refs=include_explicit_evidence_refs,
        paper_output_profile=paper_output_profile,
        paper_output_profile_reason=paper_output_profile_reason,
        matched_paths=matched_paths,
        response_language_reason=language_reason,
        response_language_explicit_override=language_explicit_override,
        explicit_override=explicit_override,
    )


def _select_paper_mode(
    config: LabaiConfig,
    prompt: str,
    normalized: str,
    language: str,
    language_reason: str,
    language_explicit_override: bool,
    matched_paths: tuple[str, ...],
) -> ModeSelection:
    pdf_count = _count_explicit_pdf_paths(matched_paths)
    paper_target_count = pdf_count or (1 if _has_explicit_paper_targets(matched_paths) else 0)
    compare_requested = _contains_any(normalized, _COMPARE_KEYWORDS)
    same_document_compare = _contains_any(normalized, _SAME_DOCUMENT_COMPARE_KEYWORDS)
    summary_requested = _contains_any(normalized, _SUMMARY_KEYWORDS)
    full_document_requested = _looks_like_full_document_request(normalized)
    specific_question = _looks_like_specific_paper_question(normalized)

    if compare_requested and (paper_target_count >= 2 or same_document_compare):
        return _build_selection(
            config,
            "paper_compare",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt targets multiple PDFs or explicitly requests a comparison within paper evidence.",
        )

    if full_document_requested or (summary_requested and paper_target_count == 1):
        return _build_selection(
            config,
            "paper_summary",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt targets a paper summary path rather than a narrow retrieval question.",
        )

    if specific_question or paper_target_count >= 1:
        return _build_selection(
            config,
            "paper_grounded_qa",
            prompt,
            language,
            language_reason,
            language_explicit_override,
            matched_paths,
            "Prompt asks a grounded question over one or more local documents.",
        )

    return _build_selection(
        config,
        "paper_summary",
        prompt,
        language,
        language_reason,
        language_explicit_override,
        matched_paths,
        "Prompt looks paper-oriented, so defaulting to the paper summary path.",
    )


def _select_read_strategy(
    mode: ResearchMode,
    normalized_prompt: str,
    matched_paths: tuple[str, ...],
) -> tuple[ReadStrategy, str]:
    if not mode.startswith("paper_"):
        return "none", "Read-strategy routing is only active for paper-aware modes."

    if mode == "paper_compare":
        return "hybrid", "Paper comparisons need both broad document coverage and targeted evidence."

    if mode == "paper_grounded_qa":
        if _contains_any(normalized_prompt, _RECURRING_LIMITATION_KEYWORDS):
            return "hybrid", "Recurring cross-paper limitation synthesis needs document-level coverage plus targeted evidence."
        if _contains_any(normalized_prompt, _SAMPLE_DATA_BACKSTOP_KEYWORDS):
            return "hybrid", "Sample/data questions often need slot-level document coverage plus targeted evidence."
        return "retrieval", "Grounded QA is narrow by default and should anchor on targeted evidence retrieval."

    full_document_requested = _looks_like_full_document_request(normalized_prompt)
    detail_heavy = _requests_multiple_dimensions(normalized_prompt)
    if full_document_requested and detail_heavy:
        return "hybrid", "Prompt asks for a whole-paper read plus several specific dimensions, so use broad coverage with targeted detail checks."
    if full_document_requested:
        return "full_document", "Prompt clearly asks for a whole-document integrated read."
    if _contains_any(normalized_prompt, _QUESTION_KEYWORDS):
        return "retrieval", "Prompt is phrased as a narrow grounded question."
    if _count_explicit_pdf_paths(matched_paths) == 1:
        return "hybrid", "Single-document summaries benefit from broad coverage plus selective evidence support."
    return "retrieval", "Defaulting to targeted retrieval for paper prompts without whole-document cues."


def _select_response_style(
    normalized_prompt: str,
    mode: ResearchMode,
    read_strategy: ReadStrategy,
) -> ResponseStyle:
    if _contains_any(normalized_prompt, _NO_BULLET_KEYWORDS):
        return "continuous_prose"
    if mode == "paper_summary" and read_strategy == "full_document":
        return "continuous_prose"
    return "structured"


def _should_include_explicit_evidence_refs(
    normalized_prompt: str,
    mode: ResearchMode,
    read_strategy: ReadStrategy,
) -> bool:
    if _contains_any(normalized_prompt, _EVIDENCE_REQUEST_KEYWORDS):
        return True
    if mode == "paper_grounded_qa":
        return True
    if mode == "paper_compare" and read_strategy == "hybrid":
        return True
    return False


def _select_paper_output_profile(
    mode: ResearchMode,
    normalized_prompt: str,
) -> tuple[PaperOutputProfile, str]:
    if not mode.startswith("paper_"):
        return "none", "Paper output profiles apply only to paper-aware modes."

    detailed_requested = _contains_any(normalized_prompt, _DETAILED_PAPER_OUTPUT_KEYWORDS)
    recurring_limitations = _contains_any(normalized_prompt, _RECURRING_LIMITATION_KEYWORDS)
    narrow_grounded_qa = _looks_like_narrow_grounded_paper_question(normalized_prompt)

    if mode == "paper_summary":
        if detailed_requested:
            return (
                "detailed_paper_note",
                "Prompt explicitly requests detailed supported paper-specific content.",
            )
        return (
            "quick_summary",
            "Ordinary paper summaries default to concise quick-summary output.",
        )

    if mode == "paper_compare":
        return (
            "detailed_paper_note",
            "Paper comparisons default to detailed slot-grounded paper-note output.",
        )

    if recurring_limitations:
        return (
            "detailed_paper_note",
            "Recurring cross-paper limitations synthesis defaults to detailed grounded note output.",
        )
    if detailed_requested:
        return (
            "detailed_paper_note",
            "Prompt explicitly requests detailed supported paper-specific content.",
        )
    if narrow_grounded_qa:
        return (
            "quick_summary",
            "Narrow grounded paper QA stays concise unless the prompt explicitly asks for more detail.",
        )
    return (
        "quick_summary",
        "Paper grounded QA defaults to concise output unless the prompt clearly asks for a detailed note.",
    )


def _select_answer_schema(
    mode: ResearchMode,
    read_strategy: ReadStrategy,
    response_style: ResponseStyle,
    prompt: str,
) -> str:
    if mode == "general_chat":
        return "general_chat_response"
    if _is_brief_greeting(prompt.strip().lower()):
        return "brief_response"
    if response_style == "continuous_prose":
        return f"{mode}_{read_strategy}_continuous_prose"
    if read_strategy == "none":
        return f"{mode}_sections"
    return f"{mode}_{read_strategy}_sections"


def _contains_any(prompt: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in prompt for keyword in keywords)


def _looks_like_file_explain(prompt: str) -> bool:
    return _contains_any(prompt, _FILE_KEYWORDS) and any(
        token in prompt
        for token in ("file", ".py", ".md", "class", "function", "\u6587\u4ef6", "\u6a21\u5757")
    )


def _looks_like_workspace_edit_request(prompt: str, matched_paths: tuple[str, ...]) -> bool:
    if not _contains_any(prompt, _WORKSPACE_WRITE_KEYWORDS):
        return False
    if matched_paths:
        return True
    return any(token in prompt for token in _WORKSPACE_EDIT_STRUCTURAL_KEYWORDS)


def _looks_like_paper_prompt(prompt: str, matched_paths: tuple[str, ...]) -> bool:
    if any(path.lower().endswith(".pdf") for path in matched_paths):
        return True
    if _contains_any(prompt, _PAPER_KEYWORDS):
        return True
    if any("papers" in path.lower() or "paper" in path.lower() for path in matched_paths):
        return True
    return False


def _looks_like_full_document_request(prompt: str) -> bool:
    return _contains_any(prompt, _FULL_DOCUMENT_KEYWORDS)


def _requests_multiple_dimensions(prompt: str) -> bool:
    hits = sum(1 for keyword in _DETAIL_DIMENSION_KEYWORDS if keyword in prompt)
    return hits >= 2


def _looks_like_specific_paper_question(prompt: str) -> bool:
    if _contains_any(prompt, _SUMMARY_KEYWORDS) or _looks_like_full_document_request(prompt):
        return False
    return _contains_any(prompt, _QUESTION_KEYWORDS)


def _looks_like_narrow_grounded_paper_question(prompt: str) -> bool:
    if not _looks_like_specific_paper_question(prompt):
        return False
    if _contains_any(prompt, _RECURRING_LIMITATION_KEYWORDS):
        return False
    return not _requests_multiple_dimensions(prompt)


def _count_explicit_pdf_paths(matched_paths: tuple[str, ...]) -> int:
    return sum(1 for item in matched_paths if item.lower().endswith(".pdf"))


def _has_explicit_paper_targets(matched_paths: tuple[str, ...]) -> bool:
    if _count_explicit_pdf_paths(matched_paths) > 0:
        return True
    return any("/papers" in item.lower() or item.lower().endswith("papers") for item in matched_paths)


def _is_brief_greeting(prompt: str) -> bool:
    compact = prompt.strip()
    if len(compact) > 24:
        return False
    return any(compact.startswith(greeting) for greeting in _GREETING_PATTERNS)


def _build_ask_answer_override(prompt: str) -> str:
    normalized = prompt.strip().lower()
    raw_path_tokens = _extract_ask_path_tokens(prompt)
    if (
        raw_path_tokens
        and not _extract_ask_pdf_tokens(prompt)
        and _contains_any(normalized, _PAPER_KEYWORDS)
    ):
        return (
            'Use `labai workflow read-paper "<pdf>"` for a single PDF or '
            '`labai workflow compare-papers "<pdf_a>" "<pdf_b>"` for explicit multi-PDF analysis.'
        )
    workflow_command = _suggest_ask_workflow_command(prompt)
    if not workflow_command:
        return ""
    if workflow_command.startswith("labai workflow edit-task"):
        return f"Use `{workflow_command}` for actual file edits."
    if workflow_command.startswith("labai workflow read-paper"):
        return f"To summarize that PDF directly, use: `{workflow_command}`"
    if workflow_command.startswith("labai workflow compare-papers"):
        return f"To compare those PDFs directly, use: `{workflow_command}`"
    if workflow_command == "labai workflow onboard-project":
        return "Use `labai workflow onboard-project` for a repo-based onboarding summary."
    if workflow_command == "labai workflow verify-workspace":
        return "Use `labai workflow verify-workspace` for actual workspace readiness checks."
    return f"Use `{workflow_command}` for explicit workflow execution."


def _suggest_ask_workflow_command(prompt: str) -> str:
    normalized = prompt.strip().lower()
    pdf_tokens = _extract_ask_pdf_tokens(prompt)
    raw_path_tokens = _extract_ask_path_tokens(prompt)

    if len(pdf_tokens) >= 2 and _contains_any(normalized, _COMPARE_KEYWORDS):
        quoted_paths = " ".join(_quote_workflow_argument(token) for token in pdf_tokens[:4])
        return f"labai workflow compare-papers {quoted_paths}"

    if pdf_tokens:
        return f"labai workflow read-paper {_quote_workflow_argument(pdf_tokens[0])}"

    if _contains_any(normalized, _VERIFY_WORKSPACE_KEYWORDS):
        return "labai workflow verify-workspace"

    if _contains_any(normalized, _ONBOARDING_KEYWORDS) or _contains_any(normalized, _OVERVIEW_KEYWORDS):
        if any(token in normalized for token in ("repo", "repository", "project", "workspace", "new ra")):
            return "labai workflow onboard-project"

    if _looks_like_workspace_edit_request(normalized, raw_path_tokens):
        escaped_prompt = prompt.strip().replace('"', '\\"')
        return f'labai workflow edit-task "{escaped_prompt}"'

    return ""


def _extract_ask_path_tokens(prompt: str) -> tuple[str, ...]:
    matches: list[str] = []
    seen: set[str] = set()
    for raw_match in re.finditer(r"[A-Za-z]:\\[^\r\n\"'<>]+", prompt):
        token = raw_match.group(0).strip("`'\"()[]{}<>.,;")
        if not token or token in seen:
            continue
        seen.add(token)
        matches.append(token)
    for raw_match in _PATH_TOKEN_PATTERN.finditer(prompt):
        token = raw_match.group("token").strip("`'\"()[]{}<>.,:;")
        if not token or token in seen:
            continue
        seen.add(token)
        matches.append(token)
    return tuple(matches[:8])


def _extract_ask_pdf_tokens(prompt: str) -> tuple[str, ...]:
    return tuple(token for token in _extract_ask_path_tokens(prompt) if token.lower().endswith(".pdf"))


def _quote_workflow_argument(value: str) -> str:
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def _find_by_basename(access_manager: WorkspaceAccessManager, basename: str) -> tuple[str, ...]:
    discovered: list[str] = []
    seen: set[str] = set()
    for root in access_manager.read_roots():
        if not root.exists() or not root.is_dir():
            continue
        for candidate in root.rglob(basename):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if not access_manager.is_allowed(resolved, for_write=False):
                continue
            display_path = access_manager.display_path(resolved)
            if _is_ignored_relative(display_path):
                continue
            if display_path in seen:
                continue
            seen.add(display_path)
            discovered.append(display_path)
            if len(discovered) >= 3:
                return tuple(discovered)
    return tuple(discovered)


def _is_repo_path(candidate: Path, repo_root: Path) -> bool:
    try:
        relative = candidate.relative_to(repo_root)
    except ValueError:
        return False
    return not _is_ignored_relative(relative.as_posix())


def _is_ignored_relative(relative_path: str) -> bool:
    ignored_parts = {".git", ".venv", "__pycache__"}
    parts = tuple(part for part in relative_path.split("/") if part)
    if any(part in ignored_parts for part in parts):
        return True
    return relative_path.startswith(".labai/audit") or relative_path.startswith(".labai/sessions")

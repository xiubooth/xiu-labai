from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from labai.runtime.answer_style import (
    build_rewrite_requirements,
    needs_style_repair,
    normalize_answer_text,
)

from .base import (
    ProviderError,
    ProviderHealth,
    ProviderRequest,
    ProviderResponse,
)

if TYPE_CHECKING:
    from labai.config import LabaiConfig


class OllamaProvider:
    name = "ollama"

    def healthcheck(self, config: LabaiConfig) -> ProviderHealth:
        validation_error = _validate_ollama_config(config)
        if validation_error is not None:
            return ProviderHealth(
                status="invalid_config",
                detail=validation_error,
                available=False,
                model=config.ollama.model,
                metadata={"base_url": config.ollama.base_url},
            )

        request = Request(
            _ollama_url(config.ollama.base_url, "/api/version"),
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=config.ollama.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            return ProviderHealth(
                status="unreachable",
                detail=(
                    "Local Ollama is not reachable at "
                    f"{config.ollama.base_url}: {exc}"
                ),
                available=False,
                model=config.ollama.model,
                metadata={"base_url": config.ollama.base_url},
            )

        version = str(payload.get("version", "unknown"))
        return ProviderHealth(
            status="ready",
            detail=f"Local Ollama is reachable at {config.ollama.base_url}.",
            available=True,
            model=config.ollama.model,
            metadata={
                "base_url": config.ollama.base_url,
                "version": version,
            },
        )

    def ask(self, config: LabaiConfig, request: ProviderRequest) -> ProviderResponse:
        validation_error = _validate_ollama_config(config)
        if validation_error is not None:
            raise ProviderError(validation_error)

        selected_model = request.preferred_model or config.ollama.model
        prompt = _compose_prompt(request)
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = Request(
            _ollama_url(config.ollama.base_url, "/api/generate"),
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=config.ollama.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            raise ProviderError(
                f"Local Ollama request failed at {config.ollama.base_url}: {exc}"
            ) from exc

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ProviderError("Local Ollama returned invalid JSON.") from exc

        text = str(data.get("response", "")).strip()
        if not text:
            raise ProviderError("Local Ollama returned an empty response.")
        text = normalize_answer_text(
            text,
            response_language=request.response_language,
            response_style=request.response_style,
            include_explicit_evidence_refs=request.include_explicit_evidence_refs,
        )
        if needs_style_repair(
            text,
            prompt=request.prompt,
            response_language=request.response_language,
            response_style=request.response_style,
            include_explicit_evidence_refs=request.include_explicit_evidence_refs,
        ):
            text = _repair_text_response(
                config,
                request=request,
                answer_text=text,
                selected_model=selected_model,
            )
        text = normalize_answer_text(
            text,
            response_language=request.response_language,
            response_style=request.response_style,
            include_explicit_evidence_refs=request.include_explicit_evidence_refs,
        )

        return ProviderResponse(
            text=text,
            provider_name=self.name,
            model=str(data.get("model", selected_model)),
            metadata=_response_metadata(data),
        )


def _validate_ollama_config(config: LabaiConfig) -> str | None:
    parsed = urlparse(config.ollama.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "Ollama base_url must be an absolute local URL."

    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return (
            "Ollama base_url must stay on a local loopback host in this phase."
        )

    model_name = config.ollama.model.strip()
    if not model_name:
        return "Ollama model must not be empty."
    if model_name.endswith("-cloud") or ":cloud" in model_name:
        return "Cloud-backed Ollama models are out of scope for this phase."

    return None


def _compose_prompt(request: ProviderRequest) -> str:
    if request.answer_schema == "brief_response" and not request.observations:
        return request.prompt

    evidence_lines = [f"- {item}" for item in request.evidence_refs] or ["- None yet"]
    observation_lines = [f"- {observation}" for observation in request.observations] or ["- None yet"]
    section_lines = [f"- {item}" for item in _mode_outline(request.mode)]
    constraint_lines = [f"- {item}" for item in _mode_constraints(request.mode)]
    draft_lines = request.grounded_draft.splitlines() if request.grounded_draft else []
    if request.response_style == "continuous_prose":
        section_intro = "Cover these points in continuous prose when they are relevant:"
        style_lines = [
            "Return continuous prose rather than bullets, outline blocks, or rigid section headings.",
            "Do not print runtime metadata, internal labels, or evidence appendices unless the user explicitly asked for them.",
            "Do not return JSON, YAML, code fences, or key-value objects.",
        ]
    else:
        section_intro = "Use natural section headings and cover these points when they are relevant:"
        style_lines = [
            "Use a readable structured answer shape when it helps the user.",
            "Prefer natural-language sections or paragraphs instead of JSON, YAML, or key-value objects unless the user explicitly asked for them.",
        ]
    evidence_output_line = (
        "Include compact evidence references in the answer when they materially support the claim."
        if request.include_explicit_evidence_refs
        else "Do not add a dedicated evidence/files consulted section unless the user explicitly asked for citations or grounded support."
    )
    sections = [
        "You are answering a single-turn local repository research request.",
        f"Internal mode: {request.mode}",
        f"Mode rationale: {request.mode_reason}",
        f"Answer schema: {request.answer_schema}",
        f"Read strategy: {request.read_strategy}",
        f"Read strategy rationale: {request.read_strategy_reason}",
        f"Requested response style: {request.response_style}",
        f"Target response language: {request.response_language}",
        "Preserve the user's language in the answer.",
        "If the target language is zh-CN, translate all prose and section headings into Simplified Chinese. Keep only code identifiers, file paths, and CLI literals in English.",
        "Do not alter or invent file paths, function names, class names, config keys, or CLI literals. Reproduce identifiers exactly as they appear in the evidence or grounded draft.",
        "In this project, `claw` means the local external runtime CLI integrated by the repository. Do not reinterpret it as a cloud acronym or SaaS product.",
        "Use the provided evidence first and do not ask the user to provide repository paths or glob patterns.",
        "Base concrete claims on the consulted evidence. If something is not confirmed by the consulted files, say that it is not confirmed.",
        "Do not print raw schema identifiers.",
        *style_lines,
        evidence_output_line,
        section_intro,
        *section_lines,
        "Respect these project constraints:",
        *constraint_lines,
        "",
        "User prompt:",
        request.prompt,
        "",
        "Evidence consulted so far:",
        *evidence_lines,
        "",
        "Repository observations:",
        *observation_lines,
        "",
        "Grounded draft (treat this as the factual scaffold; rewrite it for clarity, but do not add new facts):",
        *(draft_lines or ["- No grounded draft provided."]),
        "",
        "Do not mention session ids, audit paths, runtime names, selected_mode, or other operational metadata in the answer body.",
    ]
    return "\n".join(sections)


def _mode_outline(mode: str) -> tuple[str, ...]:
    if mode == "repo_overview":
        return (
            "Purpose",
            "Main directories/modules",
            "Important entry points",
            "Current runtime path",
            "Key risks or caveats",
            "Evidence/files consulted",
        )
    if mode == "workspace_verification":
        return (
            "Readiness status",
            "Why this status was chosen",
            "What is clearly present",
            "Likely entry points or run surfaces",
            "Config/env and dependency signals",
            "Missing pieces or blockers",
            "Risks or uncertainty",
            "What to read first",
            "First three practical next steps",
            "Evidence/files consulted",
        )
    if mode == "file_explain":
        return (
            "File purpose",
            "Key functions/classes",
            "Inputs and outputs",
            "Dependencies",
            "Risks or confusing spots",
            "Evidence/files consulted",
        )
    if mode == "architecture_review":
        return (
            "Relevant components",
            "Data/control flow",
            "Runtime path and fallback path",
            "Interaction points",
            "Risks and hidden assumptions",
            "Evidence/files consulted",
        )
    if mode == "implementation_plan":
        return (
            "Goal",
            "Proposed steps",
            "Likely files/modules to change",
            "Risks",
            "Validation plan",
            "Evidence/files consulted",
        )
    if mode == "workspace_edit":
        return (
            "Short plan",
            "Structured file blocks",
            "Summary",
        )
    if mode == "prompt_compiler":
        return (
            "Strong prompt",
            "Constraints",
            "Acceptance criteria",
            "Missing assumptions or open questions",
            "Recommendation",
            "Compact variant",
            "Strict executable variant",
        )
    if mode == "paper_summary":
        return (
            "Document identity",
            "Main contribution / purpose",
            "Method / structure",
            "Key caveats",
            "Evidence refs with file + page + chunk",
        )
    if mode == "paper_compare":
        return (
            "Documents compared",
            "Commonalities",
            "Differences",
            "Strengths / weaknesses / limitations",
            "Recommendation or synthesis",
            "Evidence refs with file + page + chunk",
        )
    if mode == "paper_grounded_qa":
        return (
            "Direct answer",
            "Grounded supporting evidence",
            "Uncertainty when evidence is weak",
            "Evidence refs with file + page + chunk",
        )
    return (
        "Options being compared",
        "Strengths",
        "Weaknesses",
        "Tradeoffs",
        "Recommendation",
        "Evidence/files consulted",
    )


def _mode_constraints(mode: str) -> tuple[str, ...]:
    common = (
        "Keep the public CLI unchanged: labai doctor, labai tools, labai ask <prompt>.",
        "Stay local-first. Do not introduce remote cloud model APIs, MCP, shell execution tools, or write-file tools.",
        "Do not mention remote providers, shell features, databases, or extra commands unless the consulted evidence explicitly supports them.",
    )
    if mode == "workspace_verification":
        return common + (
            "Treat this as a readiness decision for a new RA who may need to work in the project today, not as a generic repo summary.",
            "Lead with a practical readiness classification and explain why that status was chosen from visible evidence.",
            "Prioritize concrete run surfaces, config or dependency assumptions, missing pieces, risks, and the first practical next steps.",
            "If the grounded draft or observations mention workspace coverage, reflect that broader coverage instead of sounding like only a few files were sampled.",
            "Be explicit about what is not confirmed; do not invent a clean setup story when the workspace is under-documented.",
        )
    if mode == "architecture_review":
        return common + (
            "Discuss the implemented local runtime path, native fallback path, and repo-local tracing only.",
            "Name concrete modules or functions from the evidence when describing control flow.",
            "Avoid generic remote-service or performance commentary unless the evidence explicitly mentions it.",
        )
    if mode == "implementation_plan":
        return common + (
            "When planning future work, do not propose databases, Docker, or new public CLI commands unless the evidence explicitly allows them.",
            "Prefer filesystem-backed and repo-local changes when discussing the next phase.",
            "Prefer modifying existing modules from the consulted evidence over inventing brand-new top-level modules.",
            "Do not recommend third-party libraries unless they are already mentioned in the consulted evidence.",
        )
    if mode == "workspace_edit":
        return common + (
            "This is a coding-task workflow, not a roadmap-planning task.",
            "Follow the structured file-block format from the user prompt for every file that should change.",
            "Do not return only a plan or prose memo when the task requires concrete file edits.",
            "Keep unrelated files untouched and keep the change set scoped to the requested task.",
            "When the prompt names checks or tests, prioritize getting those checks to pass.",
            "Emit one FILE block for each locked target that should change, and do not invent collision-suffixed filenames or placeholder paths.",
        )
    if mode == "prompt_compiler":
        return common + (
            "Do not solve the task itself. Rewrite it into a stronger handoff prompt for another agent.",
            "Do not inspect or rely on repository evidence unless the user explicitly included repo-specific context in the rough need.",
            "Do not invent file paths, commands, modules, workflows, or deliverables that the user did not ask for.",
            "Keep the final section headings exactly as requested for the prompt-compilation output contract.",
            "Prefer practical agent instructions over abstract prompt-engineering commentary.",
        )
    if mode == "compare_options":
        return common + (
            "Interpret Claw as the local runtime adapter implemented in this repository, not as a cloud product.",
            "Compare the concrete code paths and tradeoffs visible in the consulted files.",
            "Avoid generic vendor, pricing, or cloud-service tradeoffs unless the evidence explicitly mentions them.",
        )
    if mode.startswith("paper_"):
        return common + (
            "For paper prompts, answer only from the provided grounded draft, whole-document coverage notes, aggregated slot summaries, and retrieved PDF evidence.",
            "Treat the grounded draft as a semantic-slot scaffold rather than a loose summary impression.",
            "If the read strategy is full_document or hybrid, whole-document coverage notes and slot summaries are part of the factual evidence and should not be ignored.",
            "Preserve exact file/page/chunk or file/pages/window evidence references in the answer when citing support.",
            "If a document is extraction-poor or OCR is required, say so explicitly instead of inferring missing text.",
            "Only state a method, limitation, contribution, or comparison point if it appears in the provided evidence or grounded draft.",
            "If a requested detail is weakly supported, use restrained wording instead of broad generalization.",
            "If a requested detail is not supported by the PDF evidence, say that it is not confirmed from the PDF evidence or that it is not clearly stated in the paper.",
            "Do not invent manual workflows, evaluation results, or external context unless those exact ideas appear in the retrieved evidence.",
            "Do not add broad textbook commentary about machine learning, finance, or investment implications unless those exact ideas appear in the provided evidence.",
        )
    return common
def _repair_text_response(
    config: LabaiConfig,
    *,
    request: ProviderRequest,
    answer_text: str,
    selected_model: str,
) -> str:
    payload = {
        "model": selected_model,
        "prompt": _compose_rewrite_prompt(request, answer_text),
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = Request(
        _ollama_url(config.ollama.base_url, "/api/generate"),
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=config.ollama.timeout_seconds) as response:
            raw_response = response.read().decode("utf-8")
        data = json.loads(raw_response)
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError):
        return answer_text

    repaired = str(data.get("response", "")).strip()
    return repaired or answer_text


def _compose_rewrite_prompt(request: ProviderRequest, answer_text: str) -> str:
    paper_requirements: list[str] = []
    if request.mode.startswith("paper_"):
        paper_requirements = [
            "- Use only the grounded slot scaffold and retrieved PDF evidence as factual support.",
            "- Remove generic domain commentary that is not clearly supported by the paper evidence.",
            "- If a requested detail is weak or missing, say it is not clearly stated in the paper instead of guessing.",
        ]
    sections = [
        "The previous draft answer violated the requested presentation rules. Rewrite it so it strictly obeys them.",
        f"Original user prompt: {request.prompt}",
        "Hard requirements:",
        *[
            f"- {item}"
            for item in build_rewrite_requirements(
                response_language=request.response_language,
                response_style=request.response_style,
                include_explicit_evidence_refs=request.include_explicit_evidence_refs,
            )
        ],
        *paper_requirements,
        "",
        "Draft answer to rewrite:",
        answer_text,
        "",
        "Return only the rewritten final answer body.",
    ]
    return "\n".join(sections)


def _ollama_url(base_url: str, suffix: str) -> str:
    return f"{base_url.rstrip('/')}{suffix}"


def _response_metadata(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "created_at",
        "done",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )
    return {
        key: data[key]
        for key in keys
        if key in data
    }

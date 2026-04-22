from __future__ import annotations

from dataclasses import dataclass

from labai.config import LabaiConfig, format_project_path

from .library import count_indexed_documents, count_library_pdfs, resolve_embedding_model
from .parsing import parser_readiness

PaperDoctorStatus = str


@dataclass(frozen=True)
class PaperLibraryReport:
    status: PaperDoctorStatus
    summary: str
    parser_status: str
    parser_detail: str
    embedding_status: str
    embedding_detail: str
    active_embedding_model: str | None
    indexed_documents: int
    ocr_required_documents: int
    discovered_library_documents: int
    library_roots: tuple[str, ...]
    next_steps: tuple[str, ...]


def build_paper_library_report(config: LabaiConfig) -> PaperLibraryReport:
    parser_state = parser_readiness()
    parser_status = "ready" if parser_state.pymupdf_available or parser_state.pypdf_available else "not_ready"
    parser_detail = (
        f"pymupdf={parser_state.pymupdf_detail} | pypdf={parser_state.pypdf_detail}"
    )

    embedding = resolve_embedding_model(config)
    indexed_documents, ocr_required_documents = count_indexed_documents(config)
    discovered_library_documents = count_library_pdfs(config)

    if parser_status != "ready" or embedding.active_model is None:
        status = "not_ready"
        summary = "PDF retrieval is not ready because parsers or the local embedding model are unavailable."
    elif indexed_documents > 0:
        status = "ready"
        summary = "PDF retrieval is ready and the local paper index contains retrievable documents."
    elif discovered_library_documents > 0 and ocr_required_documents > 0:
        status = "partially_ready"
        summary = "PDF retrieval is partially ready, but the current library only contains extraction-poor PDFs."
    else:
        status = "ready_with_empty_library"
        summary = "PDF retrieval is ready, but the local paper library has not been indexed yet."

    next_steps: list[str] = []
    if parser_status != "ready":
        next_steps.append('Run `python -m pip install -e ".[dev]"` to ensure PyMuPDF and pypdf are available.')
    if embedding.active_model is None:
        next_steps.append(
            f'Pull the local embedding model with `ollama pull {config.papers.embedding_model}` '
            f"or the configured fallback {config.papers.embedding_fallback_model}."
        )
    if discovered_library_documents == 0:
        roots = ", ".join(format_project_path(path, config.project_root) for path in config.papers.library_roots)
        next_steps.append(
            f"Add one or more PDFs under {roots} or reference a PDF path directly in `labai ask`."
        )
    elif indexed_documents == 0 and ocr_required_documents == 0:
        next_steps.append(
            "Run a PDF-focused `labai ask` prompt so the current library is ingested and indexed."
        )
    elif indexed_documents == 0 and ocr_required_documents > 0:
        next_steps.append(
            "Replace or augment extraction-poor PDFs with text-based PDFs; OCR is not implemented in this phase."
        )

    return PaperLibraryReport(
        status=status,
        summary=summary,
        parser_status=parser_status,
        parser_detail=parser_detail,
        embedding_status=embedding.status,
        embedding_detail=embedding.detail,
        active_embedding_model=embedding.active_model,
        indexed_documents=indexed_documents,
        ocr_required_documents=ocr_required_documents,
        discovered_library_documents=discovered_library_documents,
        library_roots=tuple(format_project_path(path, config.project_root) for path in config.papers.library_roots),
        next_steps=tuple(dict.fromkeys(step for step in next_steps if step)),
    )

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import re
from typing import Any


PARSER_PYMUPDF = "pymupdf"
PARSER_PYPDF = "pypdf"
EXTRACTION_READY = "ready"
EXTRACTION_OCR_REQUIRED = "ocr_required"
EXTRACTION_ERROR = "error"


class PdfParseError(RuntimeError):
    """Raised when a PDF cannot be parsed locally."""


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    char_count: int
    text_available: bool


@dataclass(frozen=True)
class ParsedPdf:
    source_path: str
    parser_used: str
    parser_attempts: tuple[str, ...]
    extraction_status: str
    page_count: int
    title: str
    metadata: dict[str, str]
    pages: tuple[PageText, ...]
    total_text_chars: int
    error: str = ""


@dataclass(frozen=True)
class ParserReadiness:
    pymupdf_available: bool
    pypdf_available: bool
    pymupdf_detail: str
    pypdf_detail: str


def parser_readiness() -> ParserReadiness:
    pymupdf_module, pymupdf_error = _load_module("fitz")
    pypdf_module, pypdf_error = _load_module("pypdf")
    return ParserReadiness(
        pymupdf_available=pymupdf_module is not None,
        pypdf_available=pypdf_module is not None,
        pymupdf_detail="available" if pymupdf_module is not None else str(pymupdf_error or "not installed"),
        pypdf_detail="available" if pypdf_module is not None else str(pypdf_error or "not installed"),
    )


def parse_pdf(
    path: Path,
    *,
    parser_preference: str,
    min_page_text_chars: int,
) -> ParsedPdf:
    parser_order = _ordered_parsers(parser_preference)
    successes: list[ParsedPdf] = []
    attempt_notes: list[str] = []

    for parser_name in parser_order:
        try:
            parsed = _parse_with_parser(path, parser_name, min_page_text_chars)
        except PdfParseError as exc:
            attempt_notes.append(f"{parser_name}:error:{exc}")
            continue

        attempt_notes.append(f"{parser_name}:{parsed.extraction_status}")
        successes.append(parsed)
        if parsed.extraction_status == EXTRACTION_READY:
            return ParsedPdf(
                source_path=parsed.source_path,
                parser_used=parsed.parser_used,
                parser_attempts=tuple(attempt_notes),
                extraction_status=parsed.extraction_status,
                page_count=parsed.page_count,
                title=parsed.title,
                metadata=parsed.metadata,
                pages=parsed.pages,
                total_text_chars=parsed.total_text_chars,
                error=parsed.error,
            )

    if successes:
        best = max(successes, key=lambda item: item.total_text_chars)
        return ParsedPdf(
            source_path=best.source_path,
            parser_used=best.parser_used,
            parser_attempts=tuple(attempt_notes),
            extraction_status=best.extraction_status,
            page_count=best.page_count,
            title=best.title,
            metadata=best.metadata,
            pages=best.pages,
            total_text_chars=best.total_text_chars,
            error=best.error,
        )

    return ParsedPdf(
        source_path=path.as_posix(),
        parser_used="none",
        parser_attempts=tuple(attempt_notes) or ("no_parser_attempted",),
        extraction_status=EXTRACTION_ERROR,
        page_count=0,
        title=path.stem,
        metadata={},
        pages=(),
        total_text_chars=0,
        error="No local PDF parser succeeded.",
    )


def _ordered_parsers(parser_preference: str) -> tuple[str, ...]:
    if parser_preference == "pypdf_then_pymupdf":
        return (PARSER_PYPDF, PARSER_PYMUPDF)
    return (PARSER_PYMUPDF, PARSER_PYPDF)


def _parse_with_parser(path: Path, parser_name: str, min_page_text_chars: int) -> ParsedPdf:
    if parser_name == PARSER_PYMUPDF:
        return _parse_with_pymupdf(path, min_page_text_chars)
    if parser_name == PARSER_PYPDF:
        return _parse_with_pypdf(path, min_page_text_chars)
    raise PdfParseError(f"Unsupported parser {parser_name}")


def _parse_with_pymupdf(path: Path, min_page_text_chars: int) -> ParsedPdf:
    fitz_module, error = _load_module("fitz")
    if fitz_module is None:
        raise PdfParseError(f"PyMuPDF unavailable: {error}")

    try:
        with fitz_module.open(str(path)) as document:
            metadata = _normalize_metadata(getattr(document, "metadata", {}) or {})
            title = metadata.get("title") or path.stem
            pages = tuple(
                _page_text(page_number + 1, _normalize_text(document.load_page(page_number).get_text("text")), min_page_text_chars)
                for page_number in range(document.page_count)
            )
            return _build_parsed_result(
                path,
                parser_used=PARSER_PYMUPDF,
                title=title,
                metadata=metadata,
                pages=pages,
                min_page_text_chars=min_page_text_chars,
            )
    except Exception as exc:  # pragma: no cover - library-specific failure surface
        raise PdfParseError(str(exc)) from exc


def _parse_with_pypdf(path: Path, min_page_text_chars: int) -> ParsedPdf:
    pypdf_module, error = _load_module("pypdf")
    if pypdf_module is None:
        raise PdfParseError(f"pypdf unavailable: {error}")

    try:
        reader = pypdf_module.PdfReader(str(path))
        metadata = _normalize_metadata(dict(getattr(reader, "metadata", {}) or {}))
        title = metadata.get("title") or path.stem
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            extracted = ""
            try:
                extracted = page.extract_text() or ""
            except Exception:
                extracted = ""
            pages.append(_page_text(page_number, _normalize_text(extracted), min_page_text_chars))
        return _build_parsed_result(
            path,
            parser_used=PARSER_PYPDF,
            title=title,
            metadata=metadata,
            pages=tuple(pages),
            min_page_text_chars=min_page_text_chars,
        )
    except Exception as exc:  # pragma: no cover - library-specific failure surface
        raise PdfParseError(str(exc)) from exc


def _build_parsed_result(
    path: Path,
    *,
    parser_used: str,
    title: str,
    metadata: dict[str, str],
    pages: tuple[PageText, ...],
    min_page_text_chars: int,
) -> ParsedPdf:
    total_text_chars = sum(page.char_count for page in pages)
    pages_with_text = sum(1 for page in pages if page.text_available)
    extraction_status = EXTRACTION_READY
    if not pages or pages_with_text == 0 or total_text_chars < max(min_page_text_chars, 1):
        extraction_status = EXTRACTION_OCR_REQUIRED
    elif pages_with_text < max(1, len(pages) // 2):
        extraction_status = EXTRACTION_OCR_REQUIRED

    return ParsedPdf(
        source_path=path.as_posix(),
        parser_used=parser_used,
        parser_attempts=(f"{parser_used}:{extraction_status}",),
        extraction_status=extraction_status,
        page_count=len(pages),
        title=title,
        metadata=metadata,
        pages=pages,
        total_text_chars=total_text_chars,
    )


def _page_text(page_number: int, text: str, min_page_text_chars: int) -> PageText:
    char_count = len(text)
    return PageText(
        page_number=page_number,
        text=text,
        char_count=char_count,
        text_available=char_count >= min_page_text_chars,
    )


def _normalize_text(text: str) -> str:
    collapsed = re.sub(r"[ \t]+", " ", text.replace("\x00", " "))
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def _normalize_metadata(raw_metadata: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in raw_metadata.items():
        normalized_key = str(key).strip().lstrip("/")
        if not normalized_key:
            continue
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        normalized[normalized_key.lower()] = text
    return normalized


def _load_module(module_name: str) -> tuple[Any | None, Exception | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # pragma: no cover - import failures are environment-specific
        return None, exc

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from labai.config import LabaiConfig, format_project_path
from labai.workspace import WorkspaceAccessManager
from .notes import (
    PaperDocumentNote,
    PaperSlotNote,
    WindowInput,
    build_semantic_document_notes,
)

from .parsing import (
    EXTRACTION_OCR_REQUIRED,
    EXTRACTION_READY,
    ParsedPdf,
    parse_pdf,
)

EMBED_STATUS_READY = "ready"
EMBED_STATUS_FALLBACK = "fallback"
EMBED_STATUS_UNAVAILABLE = "unavailable"


class PaperLibraryError(RuntimeError):
    """Raised when local paper ingest or retrieval cannot complete safely."""


@dataclass(frozen=True)
class PaperChunk:
    chunk_id: str
    document_id: str
    source_path: str
    title: str
    page_numbers: tuple[int, ...]
    text: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class PaperIngestAction:
    source_path: str
    document_id: str
    status: str
    parser_used: str
    extraction_status: str
    chunk_count: int
    detail: str = ""


@dataclass(frozen=True)
class PaperRetrieveHit:
    document_id: str
    source_path: str
    title: str
    chunk_id: str
    page_numbers: tuple[int, ...]
    score: float
    text: str
    evidence_ref: str


@dataclass(frozen=True)
class PaperWindow:
    document_id: str
    source_path: str
    title: str
    window_id: str
    page_numbers: tuple[int, ...]
    char_count: int
    note: str
    evidence_ref: str


@dataclass(frozen=True)
class PaperContext:
    target_paths: tuple[str, ...]
    discovered_documents: tuple[str, ...]
    selected_embedding_model: str | None
    embedding_status: str
    fallback_embedding_model: str | None
    read_strategy: str
    ingest_actions: tuple[PaperIngestAction, ...]
    document_windows: tuple[PaperWindow, ...]
    slot_notes: tuple[PaperSlotNote, ...]
    document_notes: tuple[PaperDocumentNote, ...]
    retrieved_chunks: tuple[PaperRetrieveHit, ...]
    observations: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    indexed_document_count: int
    ocr_required_paths: tuple[str, ...] = ()
    index_updated: bool = False


@dataclass(frozen=True)
class EmbeddingResolution:
    requested_model: str
    active_model: str | None
    status: str
    detail: str
    available_models: tuple[str, ...]
    fallback_model: str | None = None


def discover_paper_targets(
    config: LabaiConfig,
    matched_paths: tuple[str, ...],
) -> tuple[Path, ...]:
    access_manager = WorkspaceAccessManager(config)
    discovered: list[Path] = []
    seen: set[Path] = set()

    for path_text in matched_paths:
        candidate = access_manager.resolve_prompt_path(path_text, must_exist=True)
        if candidate is None:
            continue
        if candidate.is_file() and candidate.suffix.lower() == ".pdf":
            if candidate not in seen:
                seen.add(candidate)
                discovered.append(candidate)
            continue
        if candidate.is_dir():
            if candidate not in seen:
                seen.add(candidate)
                discovered.append(candidate)

    if discovered:
        return tuple(discovered)

    for library_root in (*config.papers.library_roots, *config.workspace.allowed_paper_roots):
        if library_root.exists() and library_root not in seen:
            seen.add(library_root)
            discovered.append(library_root)

    return tuple(discovered)


def prepare_paper_context(
    config: LabaiConfig,
    prompt: str,
    *,
    target_paths: tuple[Path, ...],
    retrieval_top_k: int | None = None,
    read_strategy: str = "retrieval",
) -> PaperContext:
    resolved_targets = _expand_pdf_targets(config, target_paths)
    relative_targets = tuple(
        _tracked_source_path(config, path)
        for path in target_paths
    )

    if not resolved_targets:
        return PaperContext(
            target_paths=relative_targets,
            discovered_documents=(),
            selected_embedding_model=None,
            embedding_status=EMBED_STATUS_UNAVAILABLE,
            fallback_embedding_model=None,
            read_strategy=read_strategy,
            ingest_actions=(),
            document_windows=(),
            slot_notes=(),
            document_notes=(),
            retrieved_chunks=(),
            observations=(
                "No PDF files were found in the requested targets or configured paper library roots.",
            ),
            evidence_refs=(),
            indexed_document_count=0,
            ocr_required_paths=(),
            index_updated=False,
        )

    _ensure_runtime_dirs(config)
    embedding = resolve_embedding_model(config)
    if embedding.active_model is None:
        raise PaperLibraryError(embedding.detail)

    ingest_actions: list[PaperIngestAction] = []
    indexed_documents: list[dict[str, Any]] = []
    ocr_required_paths: list[str] = []
    index_updated = False

    for pdf_path in resolved_targets:
        indexed_document, action = _ingest_pdf(config, pdf_path, embedding.active_model)
        ingest_actions.append(action)
        if indexed_document is not None:
            indexed_documents.append(indexed_document)
            if action.status == "indexed":
                index_updated = True
        if action.extraction_status == EXTRACTION_OCR_REQUIRED:
            ocr_required_paths.append(action.source_path)

    document_windows = ()
    slot_notes = ()
    document_notes = ()
    if read_strategy in {"full_document", "hybrid"}:
        document_windows, slot_notes, document_notes = _build_document_windows(
            config,
            indexed_documents,
        )

    retrieved_chunks = ()
    if read_strategy in {"retrieval", "hybrid"}:
        retrieved_chunks = _retrieve_chunks(
            config,
            prompt,
            indexed_documents,
            embedding.active_model,
            top_k=retrieval_top_k or config.papers.retrieval_top_k,
        )

    evidence_refs = tuple(hit.evidence_ref for hit in retrieved_chunks) + tuple(
        window.evidence_ref for window in document_windows
    )
    observations = _paper_observations(
        embedding,
        read_strategy,
        ingest_actions,
        document_windows,
        slot_notes,
        document_notes,
        retrieved_chunks,
        ocr_required_paths,
    )

    return PaperContext(
        target_paths=relative_targets,
        discovered_documents=tuple(
            _tracked_source_path(config, path)
            for path in resolved_targets
        ),
        selected_embedding_model=embedding.active_model,
        embedding_status=embedding.status,
        fallback_embedding_model=embedding.fallback_model,
        read_strategy=read_strategy,
        ingest_actions=tuple(ingest_actions),
        document_windows=document_windows,
        slot_notes=slot_notes,
        document_notes=document_notes,
        retrieved_chunks=retrieved_chunks,
        observations=observations,
        evidence_refs=evidence_refs,
        indexed_document_count=sum(1 for item in indexed_documents if item["extraction_status"] == EXTRACTION_READY),
        ocr_required_paths=tuple(dict.fromkeys(ocr_required_paths)),
        index_updated=index_updated,
    )


def resolve_embedding_model(config: LabaiConfig) -> EmbeddingResolution:
    available_models = fetch_local_ollama_models(config)
    requested = config.papers.embedding_model
    fallback_model = config.papers.embedding_fallback_model

    if _model_available(requested, available_models):
        return EmbeddingResolution(
            requested_model=requested,
            active_model=requested,
            status=EMBED_STATUS_READY,
            detail=f"Embedding model {requested} is available locally.",
            available_models=available_models,
        )

    if fallback_model and _model_available(fallback_model, available_models):
        return EmbeddingResolution(
            requested_model=requested,
            active_model=fallback_model,
            status=EMBED_STATUS_FALLBACK,
            detail=(
                f"Preferred embedding model {requested} is unavailable; "
                f"falling back to {fallback_model}."
            ),
            available_models=available_models,
            fallback_model=fallback_model,
        )

    return EmbeddingResolution(
        requested_model=requested,
        active_model=None,
        status=EMBED_STATUS_UNAVAILABLE,
        detail=(
            "No usable local embedding model is available. Pull "
            f"{requested} or the configured fallback {fallback_model} through Ollama."
        ),
        available_models=available_models,
        fallback_model=fallback_model,
    )


def fetch_local_ollama_models(config: LabaiConfig) -> tuple[str, ...]:
    request = Request(
        f"{config.ollama.base_url.rstrip('/')}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=config.ollama.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError):
        return ()

    discovered: list[str] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        model_name = str(item.get("model") or item.get("name") or "").strip()
        if not model_name:
            continue
        discovered.append(model_name)
        discovered.append(model_name.split(":", maxsplit=1)[0])
    return tuple(dict.fromkeys(discovered))


def count_indexed_documents(config: LabaiConfig) -> tuple[int, int]:
    ready_count = 0
    ocr_required_count = 0
    for manifest_path in sorted(config.papers.manifests_dir.glob("*.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status = str(payload.get("extraction_status", ""))
        if status == EXTRACTION_READY:
            ready_count += 1
        elif status == EXTRACTION_OCR_REQUIRED:
            ocr_required_count += 1
    return ready_count, ocr_required_count


def count_library_pdfs(config: LabaiConfig) -> int:
    total = 0
    seen: set[Path] = set()
    for root in (*config.papers.library_roots, *config.workspace.allowed_paper_roots):
        if root.is_file() and root.suffix.lower() == ".pdf":
            if root not in seen:
                seen.add(root)
                total += 1
            continue
        if not root.is_dir():
            continue
        for pdf_path in root.rglob("*.pdf"):
            if pdf_path in seen:
                continue
            seen.add(pdf_path)
            total += 1
    return total


def _ensure_runtime_dirs(config: LabaiConfig) -> None:
    for path in (
        config.papers.runtime_root,
        config.papers.manifests_dir,
        config.papers.extracted_dir,
        config.papers.chunks_dir,
        config.papers.index_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _expand_pdf_targets(
    config: LabaiConfig,
    target_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for target in target_paths:
        resolved = target.resolve()
        if not resolved.exists():
            continue
        if resolved.is_file() and resolved.suffix.lower() == ".pdf":
            if resolved not in seen:
                seen.add(resolved)
                expanded.append(resolved)
            continue
        if resolved.is_dir():
            for pdf_path in sorted(resolved.rglob("*.pdf")):
                if pdf_path not in seen:
                    seen.add(pdf_path)
                    expanded.append(pdf_path)
    return tuple(expanded)


def _ingest_pdf(
    config: LabaiConfig,
    pdf_path: Path,
    embedding_model: str,
) -> tuple[dict[str, Any] | None, PaperIngestAction]:
    source_path = _tracked_source_path(config, pdf_path)
    document_id = _document_id(source_path)
    manifest_path = config.papers.manifests_dir / f"{document_id}.json"
    extracted_path = config.papers.extracted_dir / f"{document_id}.json"
    chunks_path = config.papers.chunks_dir / f"{document_id}.json"
    index_path = config.papers.index_dir / f"{document_id}.json"
    file_hash = _sha256_path(pdf_path)

    existing_manifest = _read_json(manifest_path)
    should_skip = (
        config.papers.reingest_policy == "if_changed"
        and existing_manifest is not None
        and existing_manifest.get("file_hash") == file_hash
        and existing_manifest.get("embedding_model") == embedding_model
        and existing_manifest.get("chunk_size") == config.papers.chunk_size
        and existing_manifest.get("chunk_overlap") == config.papers.chunk_overlap
        and existing_manifest.get("parser_preference") == config.papers.parser_preference
        and extracted_path.is_file()
        and chunks_path.is_file()
        and index_path.is_file()
    )
    if should_skip:
        indexed_document = _read_json(index_path)
        if indexed_document is None:
            raise PaperLibraryError(f"Index file missing or unreadable for {source_path}.")
        extraction_status = str(existing_manifest.get("extraction_status", "ready"))
        return indexed_document, PaperIngestAction(
            source_path=source_path,
            document_id=document_id,
            status="skipped_unchanged",
            parser_used=str(existing_manifest.get("parser_used", "")),
            extraction_status=extraction_status,
            chunk_count=int(existing_manifest.get("chunk_count", 0)),
            detail="Hash, parser, and embedding settings are unchanged.",
        )

    parsed = parse_pdf(
        pdf_path,
        parser_preference=config.papers.parser_preference,
        min_page_text_chars=config.papers.min_page_text_chars,
    )
    manifest = _build_manifest(
        config,
        pdf_path,
        document_id=document_id,
        file_hash=file_hash,
        parsed=parsed,
        embedding_model=embedding_model,
    )
    _write_json(extracted_path, _extracted_payload(config, parsed))

    if parsed.extraction_status != EXTRACTION_READY:
        low_text_status = (
            "skipped_low_text"
            if config.papers.low_text_policy == "skip_document"
            else "ocr_required"
        )
        _write_json(manifest_path, manifest)
        _write_json(chunks_path, {"document_id": document_id, "chunks": []})
        index_payload = _index_payload(config, manifest, [])
        _write_json(index_path, index_payload)
        return index_payload, PaperIngestAction(
            source_path=source_path,
            document_id=document_id,
            status=low_text_status,
            parser_used=parsed.parser_used,
            extraction_status=parsed.extraction_status,
            chunk_count=0,
            detail=(
                "Text extraction was too weak for grounded retrieval and the document was skipped by policy."
                if low_text_status == "skipped_low_text"
                else "Text extraction was too weak for grounded retrieval; OCR is required."
            ),
        )

    chunk_inputs = _chunk_parsed_pdf(parsed, config.papers.chunk_size, config.papers.chunk_overlap)
    embeddings = _embed_texts(config, embedding_model, tuple(item["text"] for item in chunk_inputs))
    chunks = [
        PaperChunk(
            chunk_id=item["chunk_id"],
            document_id=document_id,
            source_path=source_path,
            title=parsed.title,
            page_numbers=tuple(item["page_numbers"]),
            text=item["text"],
            embedding=tuple(embedding),
        )
        for item, embedding in zip(chunk_inputs, embeddings, strict=False)
    ]
    manifest["chunk_count"] = len(chunks)
    manifest["chunk_ids"] = [chunk.chunk_id for chunk in chunks]
    _write_json(manifest_path, manifest)
    _write_json(chunks_path, {"document_id": document_id, "chunks": [_chunk_to_payload(chunk) for chunk in chunks]})
    index_payload = _index_payload(config, manifest, chunks)
    _write_json(index_path, index_payload)
    return index_payload, PaperIngestAction(
        source_path=source_path,
        document_id=document_id,
        status="indexed",
        parser_used=parsed.parser_used,
        extraction_status=parsed.extraction_status,
        chunk_count=len(chunks),
        detail=f"Indexed {len(chunks)} chunk(s) with {embedding_model}.",
    )


def _build_manifest(
    config: LabaiConfig,
    pdf_path: Path,
    *,
    document_id: str,
    file_hash: str,
    parsed: ParsedPdf,
    embedding_model: str,
) -> dict[str, Any]:
    relative_path = _tracked_source_path(config, pdf_path)
    page_text_availability = [
        {
            "page_number": page.page_number,
            "char_count": page.char_count,
            "text_available": page.text_available,
        }
        for page in parsed.pages
    ]
    return {
        "document_id": document_id,
        "source_path": relative_path,
        "source_absolute_path": str(pdf_path.resolve()),
        "source_file_name": pdf_path.name,
        "file_hash": file_hash,
        "file_size_bytes": pdf_path.stat().st_size,
        "title": parsed.title,
        "page_count": parsed.page_count,
        "metadata": parsed.metadata,
        "page_text_availability": page_text_availability,
        "parser_used": parsed.parser_used,
        "parser_attempts": list(parsed.parser_attempts),
        "parser_preference": config.papers.parser_preference,
        "extraction_status": parsed.extraction_status,
        "extraction_error": parsed.error,
        "total_text_chars": parsed.total_text_chars,
        "chunk_size": config.papers.chunk_size,
        "chunk_overlap": config.papers.chunk_overlap,
        "chunk_count": 0,
        "chunk_ids": [],
        "embedding_model": embedding_model,
        "updated_at": _utc_timestamp(),
    }


def _extracted_payload(config: LabaiConfig, parsed: ParsedPdf) -> dict[str, Any]:
    return {
        "source_path": parsed.source_path,
        "parser_used": parsed.parser_used,
        "parser_attempts": list(parsed.parser_attempts),
        "extraction_status": parsed.extraction_status,
        "page_count": parsed.page_count,
        "title": parsed.title,
        "metadata": parsed.metadata,
        "pages": [asdict(page) for page in parsed.pages],
        "total_text_chars": parsed.total_text_chars,
        "error": parsed.error,
        "updated_at": _utc_timestamp(),
    }


def _chunk_parsed_pdf(parsed: ParsedPdf, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page in parsed.pages:
        if not page.text_available or not page.text:
            continue
        start = 0
        chunk_index = 1
        while start < len(page.text):
            end = min(len(page.text), start + chunk_size)
            chunk_text = page.text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"{parsed.title.lower().replace(' ', '-')}-{page.page_number:02d}-{chunk_index:02d}",
                        "page_numbers": [page.page_number],
                        "text": chunk_text,
                    }
                )
            if end >= len(page.text):
                break
            start = max(end - chunk_overlap, start + 1)
            chunk_index += 1
    return chunks


def _embed_texts(
    config: LabaiConfig,
    model_name: str,
    texts: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    if not texts:
        return ()

    payload = json.dumps(
        {
            "model": model_name,
            "input": list(texts),
        }
    ).encode("utf-8")
    request = Request(
        f"{config.ollama.base_url.rstrip('/')}/api/embed",
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.ollama.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise PaperLibraryError(
            f"Local Ollama embeddings request failed for {model_name}: {exc}"
        ) from exc

    raw_embeddings = body.get("embeddings")
    if not isinstance(raw_embeddings, list) or not raw_embeddings:
        raw_single = body.get("embedding")
        if isinstance(raw_single, list) and raw_single:
            raw_embeddings = [raw_single]
        else:
            raise PaperLibraryError(
                f"Local Ollama embeddings response for {model_name} did not include embeddings."
            )

    embeddings: list[tuple[float, ...]] = []
    for vector in raw_embeddings:
        if not isinstance(vector, list):
            raise PaperLibraryError("Local Ollama embeddings response included an invalid vector.")
        embeddings.append(tuple(float(item) for item in vector))
    return tuple(embeddings)


def _index_payload(
    config: LabaiConfig,
    manifest: dict[str, Any],
    chunks: list[PaperChunk],
) -> dict[str, Any]:
    return {
        "document_id": manifest["document_id"],
        "source_path": manifest["source_path"],
        "title": manifest["title"],
        "file_hash": manifest["file_hash"],
        "page_count": manifest["page_count"],
        "metadata": manifest["metadata"],
        "parser_used": manifest["parser_used"],
        "extraction_status": manifest["extraction_status"],
        "embedding_model": manifest["embedding_model"],
        "updated_at": _utc_timestamp(),
        "chunk_size": config.papers.chunk_size,
        "chunk_overlap": config.papers.chunk_overlap,
        "chunks": [_chunk_to_payload(chunk) for chunk in chunks],
    }


def _retrieve_chunks(
    config: LabaiConfig,
    prompt: str,
    indexed_documents: list[dict[str, Any]],
    embedding_model: str,
    *,
    top_k: int,
) -> tuple[PaperRetrieveHit, ...]:
    all_chunks: list[PaperChunk] = []
    for document in indexed_documents:
        if document.get("extraction_status") != EXTRACTION_READY:
            continue
        for chunk_payload in document.get("chunks", []):
            chunk = _payload_to_chunk(chunk_payload, document)
            if _looks_like_reference_chunk(chunk.text):
                continue
            all_chunks.append(chunk)

    if not all_chunks:
        return ()

    query_embedding = _embed_texts(config, embedding_model, (prompt,))
    if not query_embedding:
        return ()

    ranked = sorted(
        (
            (
                _cosine_similarity(query_embedding[0], chunk.embedding),
                chunk,
            )
            for chunk in all_chunks
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    hits: list[PaperRetrieveHit] = []
    for score, chunk in ranked[:top_k]:
        hits.append(
            PaperRetrieveHit(
                document_id=chunk.document_id,
                source_path=chunk.source_path,
                title=chunk.title,
                chunk_id=chunk.chunk_id,
                page_numbers=chunk.page_numbers,
                score=round(score, 4),
                text=chunk.text,
                evidence_ref=_evidence_ref(chunk),
            )
        )
    return tuple(hits)


def _paper_observations(
    embedding: EmbeddingResolution,
    read_strategy: str,
    ingest_actions: list[PaperIngestAction],
    document_windows: tuple[PaperWindow, ...],
    slot_notes: tuple[PaperSlotNote, ...],
    document_notes: tuple[PaperDocumentNote, ...],
    retrieved_chunks: tuple[PaperRetrieveHit, ...],
    ocr_required_paths: list[str],
) -> tuple[str, ...]:
    observations = [
        f"Paper ingest used embedding model {embedding.active_model} with status {embedding.status}.",
        f"Paper ingest processed {len(ingest_actions)} PDF target(s).",
        f"Paper read strategy for this ask: {read_strategy}.",
    ]
    indexed_count = sum(1 for action in ingest_actions if action.status in {"indexed", "skipped_unchanged"})
    observations.append(f"Paper index has {indexed_count} retrievable document(s) in scope for this ask.")
    if document_windows:
        observations.append(
            f"Whole-document coverage prepared {len(document_windows)} reading window(s)."
        )
    if slot_notes:
        observations.append(
            f"Semantic slot extraction produced {len(slot_notes)} slot note(s) across {len(document_notes)} document note object(s)."
        )
    if ocr_required_paths:
        observations.append(
            "Some PDFs have extraction-poor text and were marked as OCR-required: "
            + ", ".join(ocr_required_paths)
        )
    if retrieved_chunks:
        sample_refs = ", ".join(hit.evidence_ref for hit in retrieved_chunks[:4])
        observations.append(
            f"Retrieved {len(retrieved_chunks)} relevant chunk(s): {sample_refs}"
        )
    else:
        observations.append("No retrievable PDF chunks were available for this ask.")
    return tuple(observations)


def _build_document_windows(
    config: LabaiConfig,
    indexed_documents: list[dict[str, Any]],
) -> tuple[tuple[PaperWindow, ...], tuple[PaperSlotNote, ...], tuple[PaperDocumentNote, ...]]:
    windows: list[PaperWindow] = []
    slot_notes: list[PaperSlotNote] = []
    document_notes: list[PaperDocumentNote] = []
    max_chars = max(config.papers.chunk_size * 2, 1800)

    for document in indexed_documents:
        if document.get("extraction_status") != EXTRACTION_READY:
            continue
        document_id = str(document.get("document_id", ""))
        extracted_path = config.papers.extracted_dir / f"{document_id}.json"
        extracted = _read_json(extracted_path)
        if extracted is None:
            continue
        source_path = str(document.get("source_path", ""))
        title = str(document.get("title", ""))
        pages = [
            page
            for page in extracted.get("pages", [])
            if isinstance(page, dict) and bool(page.get("text_available")) and str(page.get("text", "")).strip()
        ]
        if not pages:
            continue

        page_numbers: list[int] = []
        text_parts: list[str] = []
        char_count = 0
        window_index = 1
        raw_windows: list[WindowInput] = []

        for page in pages:
            page_number = int(page.get("page_number", 0))
            page_text = str(page.get("text", "")).strip()
            if not page_text:
                continue
            projected = char_count + len(page_text)
            if text_parts and projected > max_chars:
                raw_windows.append(
                    _window_input_from_parts(
                        document_id=document_id,
                        source_path=source_path,
                        page_numbers=tuple(page_numbers),
                        text="\n".join(text_parts),
                        window_index=window_index,
                    )
                )
                window_index += 1
                page_numbers = []
                text_parts = []
                char_count = 0

            page_numbers.append(page_number)
            text_parts.append(page_text)
            char_count += len(page_text)

        if text_parts:
            raw_windows.append(
                _window_input_from_parts(
                    document_id=document_id,
                    source_path=source_path,
                    page_numbers=tuple(page_numbers),
                    text="\n".join(text_parts),
                    window_index=window_index,
                )
            )

        window_summaries, document_slot_notes, document_note = build_semantic_document_notes(
            document_id=document_id,
            source_path=source_path,
            title=title,
            windows=tuple(raw_windows),
        )
        for raw_window, summary in zip(raw_windows, window_summaries, strict=True):
            windows.append(
                PaperWindow(
                    document_id=document_id,
                    source_path=source_path,
                    title=title,
                    window_id=raw_window.window_id,
                    page_numbers=raw_window.page_numbers,
                    char_count=len(raw_window.text),
                    note=summary.note,
                    evidence_ref=raw_window.evidence_ref,
                )
            )
        slot_notes.extend(document_slot_notes)
        document_notes.append(document_note)

    return tuple(windows), tuple(slot_notes), tuple(document_notes)


def _window_input_from_parts(
    *,
    document_id: str,
    source_path: str,
    page_numbers: tuple[int, ...],
    text: str,
    window_index: int,
) -> WindowInput:
    page_label = (
        str(page_numbers[0])
        if len(page_numbers) == 1
        else f"{page_numbers[0]}-{page_numbers[-1]}"
    )
    evidence_ref = f"{source_path}#pages={page_label}#window={window_index:02d}"
    return WindowInput(
        window_id=f"{document_id}-{window_index:02d}",
        page_numbers=page_numbers,
        evidence_ref=evidence_ref,
        text=text,
    )


def _chunk_to_payload(chunk: PaperChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_path": chunk.source_path,
        "title": chunk.title,
        "page_numbers": list(chunk.page_numbers),
        "text": chunk.text,
        "embedding": list(chunk.embedding),
    }


def _payload_to_chunk(chunk_payload: dict[str, Any], document_payload: dict[str, Any]) -> PaperChunk:
    return PaperChunk(
        chunk_id=str(chunk_payload["chunk_id"]),
        document_id=str(document_payload["document_id"]),
        source_path=str(document_payload["source_path"]),
        title=str(document_payload["title"]),
        page_numbers=tuple(int(value) for value in chunk_payload.get("page_numbers", [])),
        text=str(chunk_payload.get("text", "")),
        embedding=tuple(float(value) for value in chunk_payload.get("embedding", [])),
    )


def _evidence_ref(chunk: PaperChunk) -> str:
    pages = ",".join(str(page) for page in chunk.page_numbers) or "?"
    return f"{chunk.source_path}#page={pages}#chunk={chunk.chunk_id}"


def _document_id(source_path: str) -> str:
    return hashlib.sha256(source_path.lower().encode("utf-8")).hexdigest()[:16]


def _tracked_source_path(config: LabaiConfig, path: Path) -> str:
    resolved = path.resolve()
    try:
        resolved.relative_to(config.project_root.resolve())
        return format_project_path(resolved, config.project_root)
    except ValueError:
        return str(resolved)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(left)
    right_values = tuple(right)
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0
    numerator = sum(a * b for a, b in zip(left_values, right_values, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _model_available(model_name: str, available_models: tuple[str, ...]) -> bool:
    if not model_name:
        return False
    base_name = model_name.split(":", maxsplit=1)[0]
    return model_name in available_models or base_name in available_models


def _looks_like_reference_chunk(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"(?im)^\s*(references|bibliography)\s*$", text):
        return True
    year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    journal_hits = sum(
        lowered.count(token)
        for token in ("journal of", "review of", "working paper", "conference", "proceedings")
    )
    citation_hits = len(re.findall(r"\d+:\d+(?:[–-]\d+)?", text))
    return year_hits >= 4 and (journal_hits >= 1 or citation_hits >= 2)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

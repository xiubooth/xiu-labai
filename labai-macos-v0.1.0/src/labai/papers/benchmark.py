from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
from typing import Literal
import unicodedata

from labai.config import LabaiConfig

from .notes import WindowInput, build_semantic_document_notes
from .parsing import EXTRACTION_READY, ParsedPdf, parse_pdf

StructureType = Literal["strongly_sectioned", "moderately_sectioned", "dense_weak_section_cue"]

_BENCHMARK_SLOT_ORDER: tuple[str, ...] = (
    "research_question",
    "sample_or_data",
    "method",
    "main_findings",
    "limitations",
    "conclusion",
)


@dataclass(frozen=True)
class BenchmarkManifestEntry:
    filename: str
    full_path: str
    page_count: int
    parser_used: str
    extraction_status: str
    structure_type: StructureType
    language: str
    section_cue_count: int
    slot_support: dict[str, str]
    recoverable_slots: tuple[str, ...]
    vulnerabilities: tuple[str, ...]
    note_count: int
    window_count: int


@dataclass(frozen=True)
class BenchmarkSplit:
    dev: tuple[str, ...]
    validation: tuple[str, ...]
    holdout: tuple[str, ...]


def build_benchmark_manifest(
    config: LabaiConfig,
    pdf_root: Path,
) -> tuple[BenchmarkManifestEntry, ...]:
    entries: list[BenchmarkManifestEntry] = []
    for path in sorted(
        (item for item in Path(pdf_root).rglob("*.pdf") if item.is_file()),
        key=lambda item: item.relative_to(pdf_root).as_posix().lower(),
    ):
        entries.append(build_manifest_entry(config, path))
    return tuple(entries)


def build_manifest_entry(
    config: LabaiConfig,
    pdf_path: Path,
) -> BenchmarkManifestEntry:
    parsed = parse_pdf(
        pdf_path,
        parser_preference=config.papers.parser_preference,
        min_page_text_chars=config.papers.min_page_text_chars,
    )
    windows = _page_windows(parsed)
    _, slot_notes, document_note = build_semantic_document_notes(
        document_id=pdf_path.stem,
        source_path=str(pdf_path),
        title=parsed.title or pdf_path.stem,
        windows=windows,
    )
    structure_type, section_cue_count = classify_structure_type(parsed)
    language = detect_document_language(parsed)
    slot_support = {
        item.slot_name: item.support_status
        for item in document_note.cleaned_slots
        if item.slot_name in _BENCHMARK_SLOT_ORDER
    }
    recoverable_slots = tuple(
        slot_name
        for slot_name in _BENCHMARK_SLOT_ORDER
        if slot_support.get(slot_name, "not_clearly_stated") != "not_clearly_stated"
    )
    vulnerabilities = classify_manifest_vulnerabilities(
        parsed,
        document_note=document_note,
        structure_type=structure_type,
    )
    return BenchmarkManifestEntry(
        filename=pdf_path.name,
        full_path=str(pdf_path),
        page_count=parsed.page_count,
        parser_used=parsed.parser_used,
        extraction_status=parsed.extraction_status,
        structure_type=structure_type,
        language=language,
        section_cue_count=section_cue_count,
        slot_support=slot_support,
        recoverable_slots=recoverable_slots,
        vulnerabilities=vulnerabilities,
        note_count=len(slot_notes),
        window_count=len(windows),
    )


def split_benchmark_manifest(entries: tuple[BenchmarkManifestEntry, ...]) -> BenchmarkSplit:
    buckets = {"dev": [], "validation": [], "holdout": []}
    structure_groups: dict[StructureType, list[BenchmarkManifestEntry]] = {
        "strongly_sectioned": [],
        "moderately_sectioned": [],
        "dense_weak_section_cue": [],
    }
    for entry in entries:
        structure_groups[entry.structure_type].append(entry)

    bucket_names = ("dev", "validation", "holdout")
    for structure_type, group_entries in structure_groups.items():
        ordered = sorted(
            group_entries,
            key=lambda item: (
                -len(item.vulnerabilities),
                -item.page_count,
                item.filename.lower(),
            ),
        )
        for index, entry in enumerate(ordered):
            buckets[bucket_names[index % len(bucket_names)]].append(entry.filename)

    if len(entries) >= 3:
        for bucket_name in bucket_names:
            if buckets[bucket_name]:
                continue
            donor_name = max(bucket_names, key=lambda name: len(buckets[name]))
            if len(buckets[donor_name]) <= 1:
                continue
            buckets[bucket_name].append(buckets[donor_name].pop())

    return BenchmarkSplit(
        dev=tuple(sorted(buckets["dev"])),
        validation=tuple(sorted(buckets["validation"])),
        holdout=tuple(sorted(buckets["holdout"])),
    )


def serialize_manifest(entries: tuple[BenchmarkManifestEntry, ...]) -> list[dict[str, object]]:
    return [asdict(entry) for entry in entries]


def serialize_split(split: BenchmarkSplit) -> dict[str, list[str]]:
    return {
        "dev": list(split.dev),
        "validation": list(split.validation),
        "holdout": list(split.holdout),
    }


def write_manifest_artifacts(
    output_dir: Path,
    *,
    entries: tuple[BenchmarkManifestEntry, ...],
    split: BenchmarkSplit,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(serialize_manifest(entries), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "split.json").write_text(
        json.dumps(serialize_split(split), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "manifest.md").write_text(
        _manifest_markdown(entries, split),
        encoding="utf-8",
    )


def classify_structure_type(parsed: ParsedPdf) -> tuple[StructureType, int]:
    texts = [page.text for page in parsed.pages if page.text_available and page.text.strip()]
    cue_count = _section_cue_count(texts)
    if parsed.page_count >= 45 and cue_count <= 5:
        return "dense_weak_section_cue", cue_count
    if cue_count >= 8:
        return "strongly_sectioned", cue_count
    if cue_count >= 4:
        return "moderately_sectioned", cue_count
    return "dense_weak_section_cue", cue_count


def detect_document_language(parsed: ParsedPdf) -> str:
    corpus = " ".join(page.text for page in parsed.pages if page.text_available)
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", corpus))
    latin_chars = len(re.findall(r"[A-Za-z]", corpus))
    if chinese_chars and latin_chars and chinese_chars >= 80:
        return "mixed"
    if chinese_chars >= 80:
        return "zh-CN"
    return "en"


def classify_manifest_vulnerabilities(
    parsed: ParsedPdf,
    *,
    document_note,
    structure_type: StructureType,
) -> tuple[str, ...]:
    full_text = unicodedata.normalize(
        "NFKC",
        " ".join(page.text for page in parsed.pages if page.text_available),
    ).lower()
    late_pages = [
        page.text
        for page in parsed.pages[max(0, len(parsed.pages) - 6) :]
        if page.text_available
    ]
    late_text = unicodedata.normalize("NFKC", " ".join(late_pages)).lower()
    early_pages = [
        page.text
        for page in parsed.pages[: min(8, len(parsed.pages))]
        if page.text_available
    ]
    early_text = unicodedata.normalize("NFKC", " ".join(early_pages)).lower()
    slot_support = {
        item.slot_name: item.support_status
        for item in document_note.cleaned_slots
    }
    vulnerabilities: list[str] = []

    if slot_support.get("sample_or_data") == "not_clearly_stated" and _has_sample_data_signal(
        early_text or full_text
    ):
        vulnerabilities.append("false_missing_sample_data")
        vulnerabilities.append("dense_paper_sample_data_aggregation_failure")
    if slot_support.get("conclusion") == "not_clearly_stated" and _has_conclusion_signal(
        late_text or full_text
    ):
        vulnerabilities.append("false_missing_conclusion")
        vulnerabilities.append("dense_paper_conclusion_aggregation_failure")
    if slot_support.get("limitations") == "not_clearly_stated" and _has_limitation_signal(
        late_text or full_text
    ):
        vulnerabilities.append("false_missing_limitations")
    if slot_support.get("research_question") == "not_clearly_stated" and _has_research_question_signal(
        early_text or full_text
    ):
        vulnerabilities.append("false_missing_research_question")
    if parsed.page_count >= 40 and len(document_note.cleaned_slots) >= 6:
        supported_slots = sum(
            1
            for slot_name in _BENCHMARK_SLOT_ORDER
            if slot_support.get(slot_name, "not_clearly_stated") != "not_clearly_stated"
        )
        if structure_type == "dense_weak_section_cue" or supported_slots <= 4:
            vulnerabilities.append("over_compression")
            vulnerabilities.append("weak_detail_density")
    if structure_type == "dense_weak_section_cue":
        vulnerabilities.append("section_overdependence")

    return tuple(dict.fromkeys(vulnerabilities))


def _page_windows(parsed: ParsedPdf) -> tuple[WindowInput, ...]:
    windows: list[WindowInput] = []
    for page in parsed.pages:
        text = (page.text or "").strip()
        if not text:
            continue
        windows.append(
            WindowInput(
                window_id=f"{Path(parsed.source_path).stem}:page-{page.page_number}",
                page_numbers=(page.page_number,),
                text=text,
                evidence_ref=f"{Path(parsed.source_path).name}#page={page.page_number}",
            )
        )
    return tuple(windows)


def _section_cue_count(page_texts: list[str]) -> int:
    patterns = (
        r"(?im)^\s*(abstract|introduction|background|literature review|data|data and sample|dataset|sample|method|methods|methodology|empirical strategy|results|findings|discussion|discussion and conclusions|conclusion|conclusions|limitations|references)\b",
        r"(?m)^\s*(?:[0-9]+|[IVX]+)\.\s+[A-Z]",
    )
    cues: set[str] = set()
    for text in page_texts:
        normalized = unicodedata.normalize("NFKC", text)
        for pattern in patterns:
            for match in re.findall(pattern, normalized):
                if isinstance(match, tuple):
                    cues.add(" ".join(part for part in match if part))
                else:
                    cues.add(str(match).lower())
    return len(cues)


def _has_sample_data_signal(text: str) -> bool:
    markers = (
        "dataset",
        "sample period",
        "equity panel",
        "observations",
        "training sample",
        "validation sample",
        "test sample",
        "out-of-sample",
        "trading days",
        "tickers",
        "constituents",
        "data source",
        "downloaded",
        "monthly data",
    )
    has_range = bool(re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", text))
    return has_range and any(marker in text for marker in markers)


def _has_conclusion_signal(text: str) -> bool:
    markers = (
        "in conclusion",
        "discussion and conclusions",
        "conclusion",
        "conclusions",
        "overall,",
        "overall ",
        "these findings",
        "we conclude",
    )
    return any(marker in text for marker in markers)


def _has_limitation_signal(text: str) -> bool:
    markers = (
        "limitation",
        "limitations",
        "future work",
        "using monthly data",
        "only ",
        "we only",
        "lack of",
        "is limited to",
    )
    return any(marker in text for marker in markers)


def _has_research_question_signal(text: str) -> bool:
    markers = (
        "this paper",
        "we study",
        "we examine",
        "we show",
        "the objective",
        "the goal",
        "the question",
        "this study aims",
    )
    return any(marker in text for marker in markers)


def _manifest_markdown(entries: tuple[BenchmarkManifestEntry, ...], split: BenchmarkSplit) -> str:
    lines = [
        "# Full PDF Benchmark Manifest",
        "",
        f"- pdf_count: {len(entries)}",
        f"- dev: {len(split.dev)}",
        f"- validation: {len(split.validation)}",
        f"- holdout: {len(split.holdout)}",
        "",
        "| PDF | Pages | Structure | Language | Recoverable Slots | Vulnerabilities |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    entry.filename,
                    str(entry.page_count),
                    entry.structure_type,
                    entry.language,
                    ", ".join(entry.recoverable_slots) or "(none)",
                    ", ".join(entry.vulnerabilities) or "(none)",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Split",
            "",
            f"- dev: {', '.join(split.dev)}",
            f"- validation: {', '.join(split.validation)}",
            f"- holdout: {', '.join(split.holdout)}",
        ]
    )
    return "\n".join(lines) + "\n"

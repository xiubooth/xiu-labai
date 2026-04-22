from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal
import unicodedata

PaperSlotName = Literal[
    "research_question",
    "background_or_motivation",
    "sample_or_data",
    "method",
    "main_findings",
    "limitations",
    "conclusion",
    "practical_or_investment_implications",
    "other",
]
SupportStrength = Literal["strong", "moderate", "weak"]
SlotSupportStatus = Literal["well_supported", "weakly_supported", "not_clearly_stated"]
CleanedSlotSupportStatus = Literal["explicit_supported", "weakly_supported", "not_clearly_stated"]

PRIMARY_SLOT_ORDER: tuple[PaperSlotName, ...] = (
    "research_question",
    "background_or_motivation",
    "sample_or_data",
    "method",
    "main_findings",
    "limitations",
    "conclusion",
    "practical_or_investment_implications",
)
ALL_SLOT_ORDER: tuple[PaperSlotName, ...] = PRIMARY_SLOT_ORDER + ("other",)

_SLOT_LABELS: dict[PaperSlotName, str] = {
    "research_question": "research question",
    "background_or_motivation": "background",
    "sample_or_data": "data",
    "method": "method",
    "main_findings": "findings",
    "limitations": "limitations",
    "conclusion": "conclusion",
    "practical_or_investment_implications": "implications",
    "other": "other",
}

_STRONG_PATTERNS: dict[PaperSlotName, tuple[str, ...]] = {
    "research_question": (
        "this paper studies",
        "this paper examines",
        "we study",
        "we examine",
        "paper studies",
        "paper examines",
        " studies ",
        " examines ",
        "research question",
        "asks whether",
        "investigates whether",
        "本文研究",
        "本文考察",
        "研究问题",
        "探讨",
        "考察",
    ),
    "background_or_motivation": (
        "motivation",
        "motivated by",
        "background",
        "important because",
        "challenge",
        "problem",
        "gap in the literature",
        "研究背景",
        "动机",
        "背景",
        "问题",
        "挑战",
    ),
    "sample_or_data": (
        "dataset",
        "datasets",
        "sample",
        "samples",
        "we use data",
        "the data",
        "observations",
        "panel data",
        "monthly returns",
        "firm-level",
        "stock returns",
        "sample period",
        "equity panel",
        "adjusted closing prices",
        "nasdaq-100 constituents",
        "downloaded via yfinance",
        "trading days",
        "tickers",
        "vix features",
        "5-day rolling mean",
        "22-day rolling mean",
        "training window",
        "out-of-sample evaluation",
        "shock events",
        "样本",
        "数据",
        "数据集",
        "样本期",
    ),
    "method": (
        "the method uses",
        "our method uses",
        "we use",
        "we apply",
        "method",
        "methods",
        "approach",
        "model",
        "models",
        "algorithm",
        "machine learning",
        "random forest",
        "gradient boosting",
        "neural network",
        "lasso",
        "ridge",
        "svm",
        "方法",
        "模型",
        "算法",
        "机器学习",
    ),
    "main_findings": (
        "we find",
        "our results show",
        "the results show",
        "results indicate",
        "findings",
        "outperform",
        "predictive",
        "significant",
        "evidence shows",
        "发现",
        "结果表明",
        "实证结果",
        "主要发现",
    ),
    "limitations": (
        "limitation",
        "limitations",
        "caveat",
        "future work",
        "we do not",
        "cannot",
        "is limited to",
        "lack of",
        "not available",
        "局限",
        "限制",
        "不足",
        "未来研究",
        "未能",
        "无法",
    ),
    "conclusion": (
        "in conclusion",
        "we conclude",
        "overall",
        "to conclude",
        "the conclusion",
        "结论",
        "总体而言",
        "总之",
        "本文结论",
    ),
    "practical_or_investment_implications": (
        "investment implication",
        "investment implications",
        "for investors",
        "portfolio",
        "trading",
        "practical implication",
        "practical implications",
        "policy implication",
        "投资含义",
        "投资者",
        "组合",
        "实践含义",
        "实际应用",
    ),
    "other": (),
}

_WEAK_PATTERNS: dict[PaperSlotName, tuple[str, ...]] = {
    "research_question": ("question", "objective", "goal", "aim", "purpose", "whether", "问题", "目标", "目的"),
    "background_or_motivation": ("because", "context", "literature", "motivate", "motivation", "背景", "文献"),
    "sample_or_data": (
        "data",
        "train",
        "test",
        "corpus",
        "years",
        "firms",
        "returns",
        "observed",
        "vix",
        "tickers",
        "trading days",
        "constituents",
        "adjusted closing prices",
        "yfinance",
        "hubs",
        "数据",
        "训练",
        "测试",
    ),
    "method": ("estimate", "regression", "classifier", "forecast", "specification", "estimator", "回归", "预测"),
    "main_findings": ("result", "results", "finding", "findings", "show", "evidence", "performance", "结果", "发现", "表明"),
    "limitations": ("however", "only", "small", "short", "uncertain", "sensitive", "however,", "但", "然而", "仅", "有限"),
    "conclusion": ("conclude", "conclusion", "summary", "overall", "总结", "结论"),
    "practical_or_investment_implications": ("implication", "implications", "investor", "application", "useful", "启示", "含义"),
    "other": (),
}


@dataclass(frozen=True)
class WindowInput:
    window_id: str
    page_numbers: tuple[int, ...]
    text: str
    evidence_ref: str


@dataclass(frozen=True)
class PaperSlotNote:
    slot_name: PaperSlotName
    document_id: str
    source_path: str
    title: str
    window_id: str
    page_numbers: tuple[int, ...]
    evidence_ref: str
    extracted_content: str
    support_strength: SupportStrength
    explicit: bool


@dataclass(frozen=True)
class AggregatedPaperSlot:
    slot_name: PaperSlotName
    merged_note_text: str
    evidence_refs: tuple[str, ...]
    support_status: SlotSupportStatus
    strongest_support: SupportStrength
    explicit_note_count: int
    inferred_note_count: int
    note_count: int


@dataclass(frozen=True)
class CleanedPaperSlot:
    slot_name: PaperSlotName
    summary_text: str
    evidence_refs: tuple[str, ...]
    support_status: CleanedSlotSupportStatus
    strongest_support: SupportStrength
    explicit_note_count: int
    inferred_note_count: int
    note_count: int


@dataclass(frozen=True)
class WindowSemanticSummary:
    window_id: str
    page_numbers: tuple[int, ...]
    note: str
    slot_notes: tuple[PaperSlotNote, ...]


@dataclass(frozen=True)
class PaperDocumentNote:
    document_id: str
    source_path: str
    title: str
    window_count_processed: int
    aggregated_slots: tuple[AggregatedPaperSlot, ...]
    cleaned_slots: tuple[CleanedPaperSlot, ...]
    missing_slots: tuple[PaperSlotName, ...]


def build_semantic_document_notes(
    *,
    document_id: str,
    source_path: str,
    title: str,
    windows: tuple[WindowInput, ...],
) -> tuple[tuple[WindowSemanticSummary, ...], tuple[PaperSlotNote, ...], PaperDocumentNote]:
    summaries: list[WindowSemanticSummary] = []
    collected_notes: list[PaperSlotNote] = []
    total_windows = len(windows)

    for index, window in enumerate(windows, start=1):
        if _looks_like_reference_window(window.text):
            summaries.append(
                WindowSemanticSummary(
                    window_id=window.window_id,
                    page_numbers=window.page_numbers,
                    note="Reference-style pages were processed for coverage but excluded from semantic slot extraction.",
                    slot_notes=(),
                )
            )
            continue
        slot_notes = _extract_window_slot_notes(
            document_id=document_id,
            source_path=source_path,
            title=title,
            window=window,
            window_index=index,
            total_windows=total_windows,
        )
        summaries.append(
            WindowSemanticSummary(
                window_id=window.window_id,
                page_numbers=window.page_numbers,
                note=_build_window_note(slot_notes, fallback_text=window.text),
                slot_notes=slot_notes,
            )
        )
        collected_notes.extend(slot_notes)

    aggregated_slots = _aggregate_slot_notes(tuple(collected_notes))
    cleaned_slots = _clean_slot_notes(tuple(collected_notes))
    missing_slots = tuple(
        item.slot_name
        for item in cleaned_slots
        if item.slot_name != "other" and item.support_status == "not_clearly_stated"
    )
    return (
        tuple(summaries),
        tuple(collected_notes),
        PaperDocumentNote(
            document_id=document_id,
            source_path=source_path,
            title=title,
            window_count_processed=total_windows,
            aggregated_slots=aggregated_slots,
            cleaned_slots=cleaned_slots,
            missing_slots=missing_slots,
        ),
    )


def slot_label(slot_name: PaperSlotName) -> str:
    return _SLOT_LABELS[slot_name]


def _looks_like_explicit_sample_data_sentence(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).lower()
    strong_markers = (
        "our sample begins",
        "in our sample",
        "we conduct a large-scale empirical analysis",
        "individual stocks",
        "sample period",
        "sample begins in",
        "panel of stocks",
        "balanced panel of stocks",
        "missing data",
        "data source",
        "18 years of training sample",
        "12 years of validation sample",
        "out-of-sample testing",
        "daily adjusted closing prices",
        "nasdaq-100 constituents",
        "downloaded via yfinance",
        "trading days",
        "tickers",
        "equity panel",
        "vix features",
        "5-day rolling mean",
        "22-day rolling mean",
        "training window",
        "out-of-sample evaluation",
        "shock events",
        "msft",
        "adbe",
        "nvda",
        "payx",
    )
    methodish_markers = (
        "tuning parameter",
        "tuning parameters",
        "adaptively",
        "optimized via validation",
        "optimize the tuning",
        "select tuning parameters",
        "objective function",
        "loss associated",
        "gradient",
        "sgd",
        "predictive performance",
        "forecast performance",
        "performance evaluation",
        "out-of-sample r2",
        "diebold-mariano",
        "test statistic",
        "variable importance",
        "model typically chooses",
        "best performing",
        "trading strategy",
        "sharpe ratio",
        "bootstrap samples",
        "sas code",
        "web site",
        "website",
    )
    hard_negative_markers = (
        "sas code",
        "web site",
        "website",
        "variable importance",
        "diebold-mariano",
        "test statistic",
        "sharpe ratio",
        "subsamples that include only",
        "top-1,000 stocks",
        "bottom-1,000 stocks",
        "out-of-sample period",
    )
    weak_markers = (
        "sample",
        "data",
        "dataset",
        "datasets",
        "observations",
        "stock returns",
        "individual stocks",
        "stocks",
        "dates",
        "years",
        "training",
        "validation",
        "testing",
        "panel",
        "predictor variables",
        "predictor set",
        "vix",
        "tickers",
        "trading days",
        "constituents",
        "adjusted closing prices",
        "yfinance",
        "hubs",
    )
    has_strong_marker = any(marker in lowered for marker in strong_markers)
    has_methodish_marker = any(marker in lowered for marker in methodish_markers)
    weak_hits = sum(1 for marker in weak_markers if marker in lowered)
    has_year_or_count = bool(
        re.search(r"\b(19[5-9]\d|20[0-1]\d)\b", lowered)
        or re.search(r"\b\d{1,3},\d{3}\b", lowered)
        or "60 years" in lowered
        or "30 years" in lowered
    )

    if any(marker in lowered for marker in hard_negative_markers):
        return False
    if has_methodish_marker and not (has_strong_marker or has_year_or_count):
        return False
    if has_strong_marker:
        return True
    return weak_hits >= 3 and has_year_or_count


def _extract_window_slot_notes(
    *,
    document_id: str,
    source_path: str,
    title: str,
    window: WindowInput,
    window_index: int,
    total_windows: int,
) -> tuple[PaperSlotNote, ...]:
    sentences = _split_sentences(window.text)
    if not sentences:
        return ()

    candidates: list[tuple[float, PaperSlotNote]] = []
    seen: set[tuple[PaperSlotName, str]] = set()

    for sentence in sentences:
        normalized_sentence = _normalize_sentence(sentence)
        if not normalized_sentence:
            continue
        if _looks_like_reference_sentence(normalized_sentence):
            continue
        slot_scores = _score_sentence(normalized_sentence, window_index=window_index, total_windows=total_windows)
        matched_slots = _select_slots(slot_scores)
        if matched_slots:
            matched_slots = tuple(
                slot
                for slot in matched_slots
                if slot != "sample_or_data" or _looks_like_explicit_sample_data_sentence(normalized_sentence)
            )
        if not matched_slots:
            matched_slots = ("other",)
        best_score = max(slot_scores.values(), default=0.0)
        support_strength = _score_to_strength(best_score)
        explicit_slots = {
            slot
            for slot in matched_slots
            if _is_explicit_slot_match(normalized_sentence.lower(), slot)
        }
        for slot in matched_slots:
            key = (slot, normalized_sentence.lower())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                (
                    slot_scores.get(slot, 0.0),
                    PaperSlotNote(
                        slot_name=slot,
                        document_id=document_id,
                        source_path=source_path,
                        title=title,
                        window_id=window.window_id,
                        page_numbers=window.page_numbers,
                        evidence_ref=window.evidence_ref,
                        extracted_content=_truncate_text(normalized_sentence, limit=280),
                        support_strength=support_strength,
                        explicit=slot in explicit_slots,
                    ),
                )
            )

    if not candidates:
        return ()

    selected: list[PaperSlotNote] = []
    per_slot_counts: dict[PaperSlotName, int] = {}
    for _score, note in sorted(candidates, key=lambda item: (_support_rank(item[1].support_strength), item[0]), reverse=True):
        if per_slot_counts.get(note.slot_name, 0) >= 2:
            continue
        per_slot_counts[note.slot_name] = per_slot_counts.get(note.slot_name, 0) + 1
        selected.append(note)
    return tuple(selected)


def _aggregate_slot_notes(notes: tuple[PaperSlotNote, ...]) -> tuple[AggregatedPaperSlot, ...]:
    grouped: dict[PaperSlotName, list[PaperSlotNote]] = {slot: [] for slot in ALL_SLOT_ORDER}
    for note in notes:
        grouped[note.slot_name].append(note)

    aggregated: list[AggregatedPaperSlot] = []
    for slot_name in ALL_SLOT_ORDER:
        slot_notes = sorted(
            grouped[slot_name],
            key=lambda item: (_support_rank(item.support_strength), item.explicit, item.window_id),
            reverse=True,
        )
        if not slot_notes:
            aggregated.append(
                AggregatedPaperSlot(
                    slot_name=slot_name,
                    merged_note_text="Not clearly stated in the processed document windows.",
                    evidence_refs=(),
                    support_status="not_clearly_stated",
                    strongest_support="weak",
                    explicit_note_count=0,
                    inferred_note_count=0,
                    note_count=0,
                )
            )
            continue

        strongest_support = max(slot_notes, key=lambda item: _support_rank(item.support_strength)).support_strength
        explicit_note_count = sum(1 for item in slot_notes if item.explicit)
        inferred_note_count = len(slot_notes) - explicit_note_count
        if any(item.explicit and item.support_strength in {"strong", "moderate"} for item in slot_notes):
            support_status: SlotSupportStatus = "well_supported"
        elif slot_notes:
            support_status = "weakly_supported"
        else:
            support_status = "not_clearly_stated"

        merged_note_text = "; ".join(
            _dedupe_texts(item.extracted_content for item in slot_notes)[:3]
        )
        evidence_refs = tuple(
            _dedupe_texts(item.evidence_ref for item in slot_notes)[:4]
        )
        aggregated.append(
            AggregatedPaperSlot(
                slot_name=slot_name,
                merged_note_text=merged_note_text or "Not clearly stated in the processed document windows.",
                evidence_refs=evidence_refs,
                support_status=support_status,
                strongest_support=strongest_support,
                explicit_note_count=explicit_note_count,
                inferred_note_count=inferred_note_count,
                note_count=len(slot_notes),
            )
        )
    return tuple(aggregated)


def _clean_slot_notes(notes: tuple[PaperSlotNote, ...]) -> tuple[CleanedPaperSlot, ...]:
    grouped: dict[PaperSlotName, list[PaperSlotNote]] = {slot: [] for slot in ALL_SLOT_ORDER}
    for note in notes:
        grouped[note.slot_name].append(note)

    cleaned: list[CleanedPaperSlot] = []
    for slot_name in ALL_SLOT_ORDER:
        slot_notes = sorted(
            grouped[slot_name],
            key=lambda item: (
                item.explicit,
                _support_rank(item.support_strength),
                _slot_cleanup_score(slot_name, item.extracted_content, page_numbers=item.page_numbers),
            ),
            reverse=True,
        )
        if not slot_notes:
            cleaned.append(
                CleanedPaperSlot(
                    slot_name=slot_name,
                    summary_text="Not clearly stated in the paper.",
                    evidence_refs=(),
                    support_status="not_clearly_stated",
                    strongest_support="weak",
                    explicit_note_count=0,
                    inferred_note_count=0,
                    note_count=0,
                )
            )
            continue

        cleaned_fragments: list[str] = []
        evidence_refs: list[str] = []
        seen_signatures: set[str] = set()
        explicit_note_count = sum(1 for item in slot_notes if item.explicit)
        inferred_note_count = len(slot_notes) - explicit_note_count
        strongest_support = max(slot_notes, key=lambda item: _support_rank(item.support_strength)).support_strength

        for note in slot_notes:
            fragment = _clean_slot_fragment(note.extracted_content, slot_name)
            if not fragment:
                continue
            if not _slot_fragment_fits_slot(slot_name, fragment):
                continue
            signature = _fragment_signature(fragment)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            cleaned_fragments.append(fragment)
            evidence_refs.append(note.evidence_ref)
            if len(cleaned_fragments) >= _slot_fragment_limit(slot_name):
                break

        if cleaned_fragments and explicit_note_count and strongest_support in {"strong", "moderate"}:
            support_status: CleanedSlotSupportStatus = "explicit_supported"
        elif cleaned_fragments:
            support_status = "weakly_supported"
        else:
            support_status = "not_clearly_stated"

        cleaned.append(
            CleanedPaperSlot(
                slot_name=slot_name,
                summary_text=_compose_cleaned_slot_summary(
                    slot_name,
                    cleaned_fragments,
                    support_status=support_status,
                ),
                evidence_refs=tuple(_dedupe_texts(evidence_refs)[:4]),
                support_status=support_status,
                strongest_support=strongest_support,
                explicit_note_count=explicit_note_count,
                inferred_note_count=inferred_note_count,
                note_count=len(slot_notes),
            )
        )
    return tuple(cleaned)


def _build_window_note(slot_notes: tuple[PaperSlotNote, ...], *, fallback_text: str) -> str:
    if not slot_notes:
        return _truncate_text(_normalize_sentence(fallback_text), limit=200) or "No readable text was available in this coverage window."
    parts: list[str] = []
    seen_slots: set[PaperSlotName] = set()
    ordered_notes = sorted(
        slot_notes,
        key=lambda item: (_support_rank(item.support_strength), item.explicit),
        reverse=True,
    )
    for note in ordered_notes:
        if note.slot_name in seen_slots or note.slot_name == "other":
            continue
        seen_slots.add(note.slot_name)
        parts.append(f"{slot_label(note.slot_name)}: {note.extracted_content}")
        if len(parts) >= 3:
            break
    if not parts:
        parts.append(_truncate_text(slot_notes[0].extracted_content, limit=200))
    return " | ".join(parts)


def _slot_cleanup_score(
    slot_name: PaperSlotName,
    text: str,
    *,
    page_numbers: tuple[int, ...] = (),
) -> int:
    lowered = unicodedata.normalize("NFKC", text).lower()
    score = 0
    first_page = page_numbers[0] if page_numbers else 0

    positive_patterns: dict[PaperSlotName, tuple[str, ...]] = {
        "research_question": (
            "focus is on",
            "we aim to",
            "the goal",
            "the objective",
            "the question",
            "predict expected returns",
            "conditional expected stock returns",
            "asset pricing",
        ),
        "background_or_motivation": (
            "the challenge is",
            "motivation",
            "background",
            "multiple comparisons",
            "false discovery",
            "signal-to-noise",
        ),
        "sample_or_data": (
            "in our sample",
            "our sample begins",
            "we conduct a large-scale empirical analysis",
            "individual stocks",
            "from 1957 to 2016",
            "march 1957",
            "december 2016",
            "60 years",
            "30,000",
            "training sample",
            "validation sample",
            "out-of-sample testing",
            "dates and stocks",
            "balanced panel of stocks",
            "missing data",
            "equity panel",
            "daily adjusted closing prices",
            "nasdaq-100 constituents",
            "downloaded via yfinance",
            "training window",
            "out-of-sample evaluation",
            "tickers",
            "trading days",
            "vix features",
            "5-day rolling mean",
            "22-day rolling mean",
            "shock events",
            "msft",
            "adbe",
            "nvda",
            "payx",
        ),
        "method": (
            "comparative analysis of machine learning methods",
            "generalized linear models",
            "regression trees",
            "neural networks",
            "principal components regression",
            "partial least squares",
            "sample splitting and tuning via validation",
            "regularization",
            "methodology",
        ),
        "main_findings": (
            "best performing",
            "outperform",
            "predictive r2",
            "positive predictive performance",
            "shallow learning outperforms",
            "clear statistical rejections",
            "improve models' out-of-sample predictive performance",
            "improve models’ out-of-sample predictive performance",
        ),
        "limitations": (
            "higher propensity of overfitting",
            "overfit",
            "overﬁt",
            "susceptible to in-sample overfit",
            "susceptible to in-sample overﬁt",
            "small data sets",
            "lack of regularization",
            "computationally intensive",
            "limitations of linear models",
            "their flexibility is also their limitation",
            "their ﬂexibility is also their limitation",
        ),
        "conclusion": (
            "we conclude",
            "overall",
            "best performing nonlinear method",
            "best predictor overall",
            "highest overall panel r2",
        ),
        "practical_or_investment_implications": (
            "market timing trading strategy",
            "market timing",
            "portfolio choice",
            "risk management",
            "held by investors",
            "portfolio-level forecasts",
            "economic magnitude of portfolio predictability",
        ),
        "other": (),
    }
    negative_patterns: dict[PaperSlotName, tuple[str, ...]] = {
        "research_question": (
            "review of financial studies",
            "sharpe ratio",
            "portfolio turnover",
            "hidden layers",
            "dropout",
            "spline series expansion",
            "regression trees",
            "neural networks",
            "table ",
            "figure ",
        ),
        "background_or_motivation": ("figure ", "table ", "portfolio turnover"),
        "sample_or_data": (
            "predictive power",
            "portfolio level versus the stock level",
            "portfolio level versus stock level",
            "market timing trading strategy",
            "turnover",
            "sharpe ratio",
            "best performing",
            "sas code",
            "web site",
            "website",
            "gradient",
            "optimization",
            "standard descent",
            "sgd",
            "variable importance",
            "tuning parameter",
            "tuning parameters",
            "select tuning parameters",
            "optimized via validation",
            "objective function",
            "performance evaluation",
            "forecast performance",
            "diebold-mariano",
            "test statistic",
            "model typically chooses",
            "out-of-sample period",
            "subsamples that include only",
            "top-1,000 stocks",
            "bottom-1,000 stocks",
        ),
        "method": (
            "best performing",
            "outperform",
            "predictive r2",
            "trading strategy",
            "portfolio timing",
        ),
        "main_findings": (
            "sample splitting",
            "validation sample",
            "methodology",
            "regularization",
            "objective function",
        ),
        "limitations": (
            "figure ",
            "table ",
            "visualizing",
            "first-order impact",
            "rankings of characteristics",
            "portfolio turnover",
            "white (1980)",
            "first-order approximations",
            "trading strategy",
            "leverage constraint",
        ),
        "conclusion": (
            "figure ",
            "table ",
            "rankings of characteristics",
            "characteristics for all models",
        ),
        "practical_or_investment_implications": (
            "common factor portfolios s&p 500",
            "big value",
            "subcomponents of factor portfolios",
            "sharpe ratio of a buy-and-hold investor",
        ),
        "other": (),
    }

    for pattern in positive_patterns.get(slot_name, ()):
        if pattern in lowered:
            score += 4
    for pattern in negative_patterns.get(slot_name, ()):
        if pattern in lowered:
            score -= 4

    if slot_name == "limitations":
        if any(token in lowered for token in ("no ", "lack", "without", "limited", "small sample", "ocr", "external search")):
            score += 3
    if slot_name == "conclusion":
        if any(token in lowered for token in ("we conclude", "in conclusion", "overall", "results show", "we find")):
            score += 3
    if slot_name == "sample_or_data":
        if any(token in lowered for token in ("sample", "data", "dataset", "stock", "return", "firm", "monthly")):
            score += 2
        if _looks_like_explicit_sample_data_sentence(text):
            score += 5
        if any(token in lowered for token in ("subsamples that include only", "top-1,000 stocks", "bottom-1,000 stocks", "out-of-sample period")):
            score -= 5
    if slot_name == "method":
        if any(token in lowered for token in ("method", "model", "neural", "tree", "lasso", "regression", "algorithm")):
            score += 2
    if slot_name == "main_findings":
        if any(token in lowered for token in ("we find", "results", "outperform", "predictive", "evidence")):
            score += 2
    if slot_name == "research_question":
        if any(token in lowered for token in ("study", "examine", "question", "goal", "aim", "predict")):
            score += 2
    if any(token in lowered for token in ("figure ", "table ", "appendix", "acknowledge", "conference", "review of ", "journal of ")):
        score -= 4
    if re.match(r"^\d+\s+see\b", lowered):
        score -= 4
    if slot_name == "sample_or_data" and re.search(r"\b(19[5-9]\d|20[0-1]\d)\b", lowered):
        score += 3
    if slot_name == "sample_or_data" and re.search(r"\b\d{1,3},\d{3}\b", lowered):
        score += 2
    if slot_name in {"research_question", "background_or_motivation", "sample_or_data", "method"}:
        if 1 <= first_page <= 12:
            score += 2
        elif first_page >= 30:
            score -= 1
    if slot_name in {"main_findings", "limitations", "conclusion", "practical_or_investment_implications"}:
        if first_page >= 30:
            score += 2
        elif 1 <= first_page <= 6:
            score -= 1
    return score


def _clean_slot_fragment(text: str, slot_name: PaperSlotName) -> str:
    normalized = unicodedata.normalize("NFKC", text.replace("\n", " ")).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"^\d+\s+See\s+", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip(" ;,.-")
    if not normalized:
        return ""
    lowered = normalized.lower()
    if _looks_like_reference_sentence(normalized):
        return ""
    if any(token in lowered for token in ("review of financial studies", "we gratefully acknowledge", "conference", "proceedings")):
        return ""
    if lowered.startswith(("figure ", "table ", "panel ", "appendix ")):
        return ""
    if len(re.findall(r"[A-Za-z]", normalized)) < 8:
        return ""
    if re.search(r"\br2\b|\boos\b", lowered):
        return ""
    if (
        normalized.count("=") >= 1
        and len(re.findall(r"\d", normalized)) >= 4
        and not (slot_name == "sample_or_data" and _looks_like_explicit_sample_data_sentence(normalized))
    ):
        return ""
    if (
        sum(1 for char in normalized if char.isdigit()) >= 12
        and not (slot_name == "sample_or_data" and _looks_like_explicit_sample_data_sentence(normalized))
    ):
        return ""
    if slot_name in {"limitations", "conclusion"} and any(
        token in lowered for token in ("figure ", "table ", "plot ", "visualizing", "rankings of characteristics")
    ):
        return ""
    if slot_name == "limitations" and "despite obvious limitations" in lowered:
        return ""
    if slot_name == "conclusion" and normalized.count("(") > 1:
        return ""
    if slot_name == "sample_or_data" and any(
        token in lowered
        for token in (
            "tuning parameter",
            "tuning parameters",
            "select tuning parameters",
            "optimized via validation",
            "objective function",
            "forecast performance",
            "predictive performance",
            "performance evaluation",
            "variable importance",
            "diebold-mariano",
            "test statistic",
            "trading strategy",
            "sharpe ratio",
            "model typically chooses",
        )
    ):
        return ""
    if slot_name in {"limitations", "conclusion"} and re.match(r"^[A-Z][a-z]+ \(\d{4}\)", normalized):
        return ""
    if normalized.endswith(("(", "[", "{", "/", " g", " g(", "0.", "0;", "0,")):
        return ""
    if normalized.count(";") >= 2:
        normalized = normalized.split(";", 1)[0].strip()
    if len(normalized) > 180:
        candidate = re.split(r"(?<=[.!?])\s+", normalized)[0].strip()
        if 40 <= len(candidate) <= 180:
            normalized = candidate
    return _truncate_text(normalized, limit=170)


def _fragment_signature(text: str) -> str:
    lowered = unicodedata.normalize("NFKC", text).lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    tokens = [token for token in lowered.split() if token and token not in {"the", "and", "that", "with", "from", "this", "paper"}]
    return " ".join(tokens[:16])


def _slot_fragment_limit(slot_name: PaperSlotName) -> int:
    if slot_name == "sample_or_data":
        return 2
    return 1


def _slot_fragment_fits_slot(slot_name: PaperSlotName, text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).lower()
    if slot_name == "sample_or_data":
        if any(
            marker in lowered
            for marker in (
                "compare p with t",
                "incremental information",
                "principal components",
                "partial least squares",
                "regression trees",
                "neural networks",
                "lasso",
                "regularization",
            )
        ):
            return False
        return _looks_like_explicit_sample_data_sentence(text)

    required_map: dict[PaperSlotName, tuple[str, ...]] = {
        "research_question": (
            "focus is on",
            "goal",
            "aim",
            "question",
            "predict expected returns",
            "expected stock returns",
            "risk premiums",
            "forecasting returns",
        ),
        "background_or_motivation": ("challenge", "motivation", "background", "false discovery", "signal-to-noise"),
        "sample_or_data": (),
        "method": (
            "method",
            "methods",
            "generalized linear",
            "partial least squares",
            "principal components",
            "regression trees",
            "neural networks",
            "regularization",
            "validation",
        ),
        "main_findings": (
            "outperform",
            "best performing",
            "predictive performance",
            "predictive r2",
            "we find",
            "results",
            "positive predictive performance",
        ),
        "limitations": (
            "limitation",
            "limitations",
            "overfit",
            "overﬁt",
            "lack of regularization",
            "small data",
            "computationally intensive",
            "susceptible",
            "constraint",
        ),
        "conclusion": ("conclude", "overall", "best predictor overall", "best performing nonlinear method", "highest overall"),
        "practical_or_investment_implications": (
            "market timing",
            "trading strategy",
            "portfolio choice",
            "risk management",
            "investor",
            "economic magnitude",
        ),
        "other": (),
    }
    forbidden_map: dict[PaperSlotName, tuple[str, ...]] = {
        "research_question": ("value-weight portfolios", "market timing", "trading strategy", "leverage constraint"),
        "sample_or_data": (),
        "method": ("best performing", "highest overall", "trading strategy"),
        "main_findings": ("sample splitting", "validation sample", "methodology"),
        "limitations": ("trading strategy", "leverage constraint", "value-weight portfolios", "equal weights"),
        "conclusion": ("figure ", "table ", "rankings of characteristics"),
        "practical_or_investment_implications": ("small data", "overfit", "limitation"),
        "background_or_motivation": (),
        "other": (),
    }
    if any(pattern in lowered for pattern in forbidden_map.get(slot_name, ())):
        return False
    required_patterns = required_map.get(slot_name, ())
    if not required_patterns:
        return True
    return any(pattern in lowered for pattern in required_patterns)


def _compose_cleaned_slot_summary(
    slot_name: PaperSlotName,
    fragments: list[str],
    *,
    support_status: CleanedSlotSupportStatus,
) -> str:
    if not fragments or support_status == "not_clearly_stated":
        return "Not clearly stated in the paper."
    if slot_name == "sample_or_data":
        normalized = _dedupe_texts(fragment.rstrip(". ") for fragment in fragments)
        if len(normalized) == 1:
            return normalized[0]
        return f"{normalized[0]}. {normalized[1]}."
    return fragments[0]


def _score_sentence(
    sentence: str,
    *,
    window_index: int,
    total_windows: int,
) -> dict[PaperSlotName, float]:
    lowered = sentence.lower()
    scores: dict[PaperSlotName, float] = {slot: 0.0 for slot in ALL_SLOT_ORDER}
    for slot_name in PRIMARY_SLOT_ORDER:
        for pattern in _STRONG_PATTERNS[slot_name]:
            if pattern in lowered:
                scores[slot_name] += 2.0
        for pattern in _WEAK_PATTERNS[slot_name]:
            if pattern in lowered:
                scores[slot_name] += 1.0

    if window_index <= 2:
        scores["research_question"] += 0.5
        scores["background_or_motivation"] += 0.5
    if total_windows and window_index >= max(1, total_windows - 1):
        scores["conclusion"] += 0.5
        scores["limitations"] += 0.5
        scores["practical_or_investment_implications"] += 0.5
    return scores


def _select_slots(slot_scores: dict[PaperSlotName, float]) -> tuple[PaperSlotName, ...]:
    primary_scores = {slot: score for slot, score in slot_scores.items() if slot != "other"}
    best_score = max(primary_scores.values(), default=0.0)
    if best_score < 1.0:
        return ()
    selected = tuple(
        slot
        for slot, score in primary_scores.items()
        if score >= 2.0 and score >= best_score - 1.0
    )
    if selected:
        return selected[:3]
    strongest = max(primary_scores, key=primary_scores.get)
    return (strongest,)


def _is_explicit_slot_match(sentence: str, slot_name: PaperSlotName) -> bool:
    return any(pattern in sentence for pattern in _STRONG_PATTERNS[slot_name])


def _score_to_strength(score: float) -> SupportStrength:
    if score >= 4.0:
        return "strong"
    if score >= 2.0:
        return "moderate"
    return "weak"


def _support_rank(strength: SupportStrength) -> int:
    if strength == "strong":
        return 3
    if strength == "moderate":
        return 2
    return 1


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[。！？.!?;；])\s*|(?<=:)\s+(?=[A-Z])", normalized)
    return [part.strip() for part in parts if part.strip()]


def _normalize_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    cleaned = cleaned.strip(" -•*")
    return cleaned


def _looks_like_reference_window(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"(?im)^\s*(references|bibliography)\s*$", text):
        return True
    year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    journal_hits = sum(
        lowered.count(token)
        for token in ("journal of", "review of", "working paper", "conference", "proceedings")
    )
    citation_hits = len(re.findall(r"\d+:\d+(?:[–-]\d+)?", text))
    return year_hits >= 6 and (journal_hits >= 2 or citation_hits >= 3)


def _looks_like_reference_sentence(text: str) -> bool:
    lowered = text.lower()
    if lowered.startswith(("references", "bibliography", "review of ", "journal of ", "working paper", "proceedings of ")):
        return True
    if len(re.findall(r"\b(?:19|20)\d{2}\b", text)) >= 1 and (
        re.search(r"\b[A-Z][a-z]+,\s+[A-Z]\.", text) or re.search(r"\d+:\d+(?:[–-]\d+)?", text)
    ):
        return True
    if re.search(
        r"\b[A-Z][a-z]+(?:,\s+[A-Z][a-z]+){1,4}.*\b(?:19|20)\d{2}\b",
        text,
    ) and sum(
        lowered.count(token)
        for token in (
            "journal of",
            "review of",
            "econometrica",
            "technical report",
            "working paper",
        )
    ):
        return True
    if lowered.count(" et al") >= 1:
        return True
    return False


def _truncate_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _dedupe_texts(items) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = str(item).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _looks_like_reference_window(text: str) -> bool:
    lowered = text.lower()
    non_reference_section_markers = (
        "discussion and conclusions",
        "discussion",
        "conclusion",
        "conclusions",
        "data",
        "equity panel",
        "vix features",
        "hubs, events, and split",
        "methodology",
        "results",
        "spectral radius and stability",
        "there are several limitations",
        "overall, the evidence supports",
    )
    if re.search(r"(?im)^\s*(references|bibliography)\s*$", text):
        reference_index = lowered.find("references")
        leading_text = lowered[:reference_index] if reference_index > 0 else lowered
        if any(marker in leading_text for marker in non_reference_section_markers):
            return False
        return True
    year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    journal_hits = sum(
        lowered.count(token)
        for token in ("journal of", "review of", "working paper", "conference", "proceedings")
    )
    citation_hits = len(re.findall(r"\d+:\d+(?:[–-]\d+)?", text))
    if any(marker in lowered for marker in non_reference_section_markers) and not lowered.lstrip().startswith(
        ("references", "bibliography")
    ):
        return False
    return year_hits >= 6 and (journal_hits >= 2 or citation_hits >= 3)

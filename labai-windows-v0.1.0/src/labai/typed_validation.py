from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


ValidationType = Literal[
    "syntax",
    "behavior",
    "notebook_execution",
    "data_contract",
    "source_evidence",
    "dependency_fallback",
]
ValidationStatus = Literal["pass", "fail", "blocked", "skipped"]


@dataclass(frozen=True)
class CriterionResult:
    criterion_id: str
    criterion_text: str
    status: ValidationStatus
    evidence_type: str
    evidence_detail: str
    source_file_or_artifact: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TypedValidationResult:
    validator_id: str
    task_run_id: str
    validation_type: ValidationType
    command_or_method: str
    status: ValidationStatus
    criterion_results: tuple[CriterionResult, ...]
    external_dependencies: tuple[str, ...]
    limitations: tuple[str, ...]
    raw_output_path: str

    def to_record(self) -> dict[str, object]:
        return {
            **asdict(self),
            "criterion_results": [item.to_record() for item in self.criterion_results],
        }


def classify_validation_type(check_name: str, *, source_target: str = "") -> ValidationType:
    lowered = check_name.lower()
    if check_name == "python_validate":
        if source_target.endswith(".ipynb"):
            return "notebook_execution"
        if any(token in source_target.lower() for token in ("daily", "ratings", "segments", "pensions", "liquidity", "factor")):
            return "data_contract"
        return "behavior"
    if check_name in {"py_compile", "json_validate", "toml_validate"}:
        return "syntax"
    if check_name == "source_evidence":
        return "source_evidence"
    if check_name == "dependency_fallback":
        return "dependency_fallback"
    return "behavior"


def build_typed_validation_result(
    *,
    validator_id: str,
    task_run_id: str,
    check_name: str,
    command: tuple[str, ...],
    status: str,
    acceptance_criteria: tuple[str, ...],
    criterion_evidence,
    output_excerpt: str,
    source_target: str,
    unavailable_dependencies: tuple[str, ...] = (),
    raw_output_path: str = "",
) -> TypedValidationResult:
    validation_type = classify_validation_type(check_name, source_target=source_target)
    criterion_results: list[CriterionResult] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(tuple(criterion_evidence or ()), start=1):
        criterion_id = f"{validator_id}-criterion-{index}"
        seen_ids.add(getattr(item, "criterion_text", "").strip().lower())
        criterion_results.append(
            CriterionResult(
                criterion_id=criterion_id,
                criterion_text=getattr(item, "criterion_text", "").strip() or getattr(item, "raw_line", "").strip(),
                status="pass" if getattr(item, "status", "") == "pass" else "fail",
                evidence_type=getattr(item, "source", "") or "marker",
                evidence_detail=getattr(item, "evidence", "") or output_excerpt,
                source_file_or_artifact=source_target,
            )
        )
    if acceptance_criteria:
        for index, criterion in enumerate(acceptance_criteria, start=len(criterion_results) + 1):
            normalized = criterion.strip().lower()
            if normalized in seen_ids:
                continue
            criterion_results.append(
                CriterionResult(
                    criterion_id=f"{validator_id}-criterion-{index}",
                    criterion_text=criterion,
                    status="fail",
                    evidence_type="missing_direct_evidence",
                    evidence_detail="No direct criterion evidence recorded in the final validator output.",
                    source_file_or_artifact=source_target,
                )
            )
    limitations: list[str] = []
    if validation_type == "syntax" and acceptance_criteria:
        limitations.append("Syntax-only validation cannot prove a behavioral task.")
    if unavailable_dependencies:
        limitations.append("Validation used dependency fallback assumptions for unavailable external components.")
    if validation_type == "syntax" and acceptance_criteria:
        normalized_status: ValidationStatus = "fail"
    elif any(item.status == "fail" for item in criterion_results):
        normalized_status: ValidationStatus = "fail"
    elif status == "blocked":
        normalized_status = "blocked"
    elif status == "passed":
        normalized_status = "pass"
    else:
        normalized_status = "fail"
    return TypedValidationResult(
        validator_id=validator_id,
        task_run_id=task_run_id,
        validation_type=validation_type,
        command_or_method=" ".join(command),
        status=normalized_status,
        criterion_results=tuple(criterion_results),
        external_dependencies=tuple(unavailable_dependencies),
        limitations=tuple(limitations),
        raw_output_path=raw_output_path,
    )

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from labai.owner_detection import OwnerDetectionResult
from labai.repo_map import RepoMapResult
from labai.task_manifest import TaskManifest


TaskShape = Literal[
    "direct_function_task",
    "script_output_task",
    "multi_output_pipeline_task",
    "annual_to_monthly_pipeline_task",
    "external_dependency_task",
    "notebook_deliverable_task",
    "data_contract_repair_task",
]


@dataclass(frozen=True)
class ValidatorRoutingDecision:
    task_shape: TaskShape
    task_shape_tags: tuple[str, ...]
    selected_validation_strategy: str
    preferred_validator_kind: str
    fallback_validator_kinds: tuple[str, ...]
    why_selected: tuple[str, ...]
    rejected_strategies: tuple[str, ...]
    required_outputs: tuple[str, ...]
    required_source_or_artifact: tuple[str, ...]
    external_dependency_handling: str
    expected_criterion_checks: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FailureSwitchRecommendation:
    failure_signature: str
    root_cause_category: str
    previous_strategy: str
    new_strategy: str
    preferred_validator_kind: str
    reason: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def route_task_validation(
    manifest: TaskManifest,
    repo_map: RepoMapResult,
    owner_detection: OwnerDetectionResult,
) -> ValidatorRoutingDecision:
    prompt_lower = manifest.user_instruction.lower()
    acceptance_lower = " ".join(manifest.acceptance_criteria).lower()
    combined = " ".join((prompt_lower, acceptance_lower))
    prompt_named = {Path(item).name.lower() for item in manifest.prompt_named_files}
    owners = tuple(owner_detection.primary_owner_files or manifest.primary_target_artifacts or manifest.candidate_owner_files)
    likely_python = tuple(
        entry.relative_path
        for entry in repo_map.entries
        if entry.relative_path in manifest.candidate_owner_files and entry.relative_path.endswith((".py", ".ipynb"))
    )
    required_source = owners or likely_python or manifest.prompt_named_files
    if manifest.task_kind == "notebook_deliverable" or ".ipynb" in combined:
        return ValidatorRoutingDecision(
            task_shape="notebook_deliverable_task",
            task_shape_tags=("notebook_deliverable_task",),
            selected_validation_strategy="nbformat_parse_plus_nbclient_execution",
            preferred_validator_kind="notebook",
            fallback_validator_kinds=("notebook",),
            why_selected=(
                "The prompt names or targets an existing notebook artifact.",
                "Notebook tasks must use the Route 1 notebook IO and execution path.",
            ),
            rejected_strategies=(
                "Reject direct-function validation because notebooks are artifact-driven deliverables.",
                "Reject helper-only validation because the notebook itself is the primary artifact.",
            ),
            required_outputs=("executed_notebook", "embedded_outputs"),
            required_source_or_artifact=required_source,
            external_dependency_handling="Use workspace-scoped notebook execution and persist embedded outputs even on partial failure.",
            expected_criterion_checks=(
                "notebook parse succeeds",
                "notebook stays inside workspace",
                "notebook is the primary durable artifact",
                "embedded outputs exist",
            ),
        )

    annual_tokens = (
        "annual-to-monthly",
        "annual to monthly",
        "annual record",
        "monthly availability",
        "ccm link",
        "link validity",
        "link start",
        "link end",
        "datadate",
        "cusip",
        "14_preannualcs",
    )
    annual_lag_tokens = ("6 month", "6-month", "6 months", "time_avail_m")
    annual_hint = "annual" in combined and any(token in combined for token in annual_lag_tokens)
    if any(token in combined for token in annual_tokens) or annual_hint or "14_preannualcs.py" in prompt_named:
        annual_owner = tuple(
            item
            for item in (
                *owners,
                *manifest.prompt_named_files,
                *manifest.required_read_files,
            )
            if Path(item).name.lower() in {"14_preannualcs.py", "01_download_datasets.py", "11_preparelinkingtables.py"}
        )
        return ValidatorRoutingDecision(
            task_shape="annual_to_monthly_pipeline_task",
            task_shape_tags=("annual_to_monthly_pipeline_task", "external_dependency_task"),
            selected_validation_strategy="synthetic_annual_fixture_and_monthly_output_capture",
            preferred_validator_kind="annual_pipeline",
            fallback_validator_kinds=("annual_pipeline", "pipeline_output", "dataframe"),
            why_selected=(
                "The task contract describes annual records transformed into monthly availability outputs.",
                "Annual pipeline tasks need fixture-driven output validation instead of callable-only entry assumptions.",
            ),
            rejected_strategies=(
                "Reject direct-function validation because annual pipelines may be script-driven.",
                "Reject callable-download-only validation because the monthly expansion path may live outside the download entrypoint.",
                "Reject module-level DataFrame exposure as the default validator surface.",
            ),
            required_outputs=("annual_compustat_output", "monthly_compustat_output"),
            required_source_or_artifact=annual_owner or required_source,
            external_dependency_handling="If WRDS or live data access is unavailable, use synthetic annual and CCM-link fixtures plus monkeypatched sinks.",
            expected_criterion_checks=(
                "OSAP filters enforced",
                "time_avail_m equals datadate plus six months",
                "annual records expand into monthly availability windows",
                "link validity windows enforced",
                "CUSIP first six digits handled correctly",
                "2010+ scope preserved",
            ),
        )

    multi_output_tokens = ("ratings", "short interest", "pensions", "segments", "multi output", "multiple cleaned outputs")
    if any(token in combined for token in multi_output_tokens):
        return ValidatorRoutingDecision(
            task_shape="multi_output_pipeline_task",
            task_shape_tags=("multi_output_pipeline_task", "data_contract_repair_task"),
            selected_validation_strategy="captured_multi_output_contract_validation",
            preferred_validator_kind="pipeline_output",
            fallback_validator_kinds=("pipeline_output", "dataframe"),
            why_selected=(
                "The task contract names multiple cleaned output families that must all be validated together.",
                "A multi-output pipeline needs captured outputs plus concrete data-contract diagnostics.",
            ),
            rejected_strategies=(
                "Reject direct-function validation because the task is output-family oriented.",
                "Reject module-level DataFrame exposure as the default validator surface.",
            ),
            required_outputs=("ratings", "short_interest", "pensions", "segments"),
            required_source_or_artifact=required_source,
            external_dependency_handling="Prefer synthetic fixtures and monkeypatched output capture if live dependencies are unavailable.",
            expected_criterion_checks=(
                "date fields parse",
                "as-of timing is usable",
                "duplicate gvkey-date rows removed",
                "2010+ scope preserved",
                "column selection matches the intended contract",
            ),
        )

    factor_tokens = ("fama-french", "fama french", "liquidity", "daily factors", "monthly factors")
    factor_required = ("time_d", "time_avail_m")
    if any(token in combined for token in factor_tokens) and all(token in combined for token in factor_required):
        return ValidatorRoutingDecision(
            task_shape="external_dependency_task",
            task_shape_tags=("external_dependency_task", "data_contract_repair_task"),
            selected_validation_strategy="stubbed_connector_plus_factor_output_contract_validation",
            preferred_validator_kind="factor_output",
            fallback_validator_kinds=("factor_output", "dataframe"),
            why_selected=(
                "The task depends on external factor data but still requires local validation of real output contracts.",
                "The acceptance criteria are date-field and output-shape driven.",
            ),
            rejected_strategies=(
                "Reject syntax-only validation.",
                "Reject helper-only validation because the real factor output path must be exercised.",
            ),
            required_outputs=("daily_factors", "monthly_factors", "liquidity_output"),
            required_source_or_artifact=required_source,
            external_dependency_handling="If WRDS or remote downloads are unavailable, stub the connector and validate captured daily/monthly outputs locally.",
            expected_criterion_checks=(
                "daily factors use time_d",
                "monthly factors use time_avail_m",
                "date fields parse cleanly",
                "2010+ coverage preserved",
                "naming conventions are consistent",
            ),
        )

    contract_tokens = (
        "duplicate",
        "columns",
        "column set",
        "datetime",
        "null ",
        "2010+",
        "2010-01-01",
        "as-of",
        "time_d",
        "time_avail_m",
        "field contract",
    )
    script_tokens = ("download", "raw_sql", "to_pickle", "to_csv", "daily crsp", "rolling")
    if any(token in combined for token in script_tokens):
        return ValidatorRoutingDecision(
            task_shape="script_output_task",
            task_shape_tags=("script_output_task", "external_dependency_task"),
            selected_validation_strategy="script_output_sink_capture_validation",
            preferred_validator_kind="script_output",
            fallback_validator_kinds=("script_output", "dataframe"),
            why_selected=(
                "The task contract describes script-style output production rather than a pure library API call.",
                "Script-output validation can validate output files or captured sinks without assuming module-level DataFrames.",
            ),
            rejected_strategies=(
                "Reject module-level DataFrame exposure as the default validator surface.",
                "Reject syntax-only validation for a behavioral output task.",
            ),
            required_outputs=("script_output",),
            required_source_or_artifact=required_source,
            external_dependency_handling="If live dependencies are unavailable, stub them and capture the real output sink or returned frame.",
            expected_criterion_checks=(
                "required output columns exist",
                "date filtering matches the requested window",
                "null key rows removed",
            ),
        )

    if any(token in combined for token in contract_tokens):
        return ValidatorRoutingDecision(
            task_shape="data_contract_repair_task",
            task_shape_tags=("data_contract_repair_task",),
            selected_validation_strategy="concrete_data_contract_diagnostics",
            preferred_validator_kind="dataframe",
            fallback_validator_kinds=("dataframe",),
            why_selected=(
                "The task contract is dominated by column, date, duplicate, and shape requirements.",
            ),
            rejected_strategies=(
                "Reject syntax-only validation for a behavioral data contract task.",
            ),
            required_outputs=("validated_dataframe",),
            required_source_or_artifact=required_source,
            external_dependency_handling="Prefer local synthetic fixtures and direct contract diagnostics when live data is unavailable.",
            expected_criterion_checks=(
                "required columns present",
                "date fields parse",
                "duplicate keys removed",
                "scope window preserved",
            ),
        )

    return ValidatorRoutingDecision(
        task_shape="direct_function_task",
        task_shape_tags=("direct_function_task",),
        selected_validation_strategy="direct_callable_validation",
        preferred_validator_kind="dataframe",
        fallback_validator_kinds=("dataframe",),
        why_selected=("No stronger script/pipeline/notebook signals were detected.",),
        rejected_strategies=("Reject syntax-only validation for behavioral tasks.",),
        required_outputs=("direct_callable_result",),
        required_source_or_artifact=required_source,
        external_dependency_handling="Call the real helper directly when a callable entry point exists.",
        expected_criterion_checks=("direct callable behavior matches the requested contract",),
    )


def apply_validator_routing_overrides(
    decision: ValidatorRoutingDecision,
    overrides: dict[str, object],
) -> ValidatorRoutingDecision:
    if not overrides:
        return decision
    data = decision.to_record()
    data.update({key: value for key, value in overrides.items() if value is not None})
    return ValidatorRoutingDecision(**data)


def classify_validation_failure(
    raw_text: str,
    decision: ValidatorRoutingDecision,
) -> FailureSwitchRecommendation | None:
    lowered = " ".join(raw_text.lower().split())
    signature = normalize_failure_signature(
        raw_text,
        task_shape=decision.task_shape,
        strategy=decision.selected_validation_strategy,
    )
    if not lowered:
        return None
    if "did not expose a dataframe output to validate" in lowered:
        new_kind = decision.preferred_validator_kind
        if decision.task_shape == "script_output_task":
            new_strategy = "script_output_sink_capture_validation"
        elif decision.task_shape == "multi_output_pipeline_task":
            new_strategy = "captured_multi_output_contract_validation"
            new_kind = "pipeline_output"
        elif decision.task_shape == "annual_to_monthly_pipeline_task":
            new_strategy = "synthetic_annual_fixture_and_monthly_output_capture"
            new_kind = "annual_pipeline"
        elif decision.task_shape == "external_dependency_task":
            new_strategy = "stubbed_connector_plus_factor_output_contract_validation"
            new_kind = "factor_output"
        else:
            new_strategy = "output_capture_validation"
        return FailureSwitchRecommendation(
            failure_signature=signature,
            root_cause_category="direct_dataframe_exposure_missing",
            previous_strategy=decision.selected_validation_strategy,
            new_strategy=new_strategy,
            preferred_validator_kind=new_kind,
            reason="The validator assumed module-level DataFrame exposure for a task shape that should use output capture or pipeline validation.",
        )
    if "did not expose a callable download or writable output path to validate" in lowered:
        if decision.task_shape == "annual_to_monthly_pipeline_task":
            new_kind = "annual_pipeline"
            new_strategy = "synthetic_annual_fixture_and_monthly_output_capture"
        elif decision.task_shape == "script_output_task":
            new_kind = "script_output"
            new_strategy = "script_output_sink_capture_validation"
        else:
            new_kind = "pipeline_output"
            new_strategy = "captured_pipeline_output_validation"
        return FailureSwitchRecommendation(
            failure_signature=signature,
            root_cause_category="callable_download_missing",
            previous_strategy=decision.selected_validation_strategy,
            new_strategy=new_strategy,
            preferred_validator_kind=new_kind,
            reason="The validator chose the wrong entry surface; pipeline or sink-capture validation should replace callable-only assumptions.",
        )
    if "did not expose a writable output frame to validate" in lowered:
        new_kind = "annual_pipeline" if decision.task_shape == "annual_to_monthly_pipeline_task" else decision.preferred_validator_kind
        new_strategy = (
            "monkeypatched_sink_capture_validation"
            if new_kind != "annual_pipeline"
            else "synthetic_annual_fixture_and_monthly_output_capture"
        )
        return FailureSwitchRecommendation(
            failure_signature=signature,
            root_cause_category="writable_output_path_missing",
            previous_strategy=decision.selected_validation_strategy,
            new_strategy=new_strategy,
            preferred_validator_kind=new_kind,
            reason="The validator needs sink capture or a fixture-driven pipeline surface instead of assuming a ready-made writable output path.",
        )
    if "has no attribute" in lowered and "download_" in lowered:
        return FailureSwitchRecommendation(
            failure_signature=signature,
            root_cause_category="missing_function_chain",
            previous_strategy=decision.selected_validation_strategy,
            new_strategy="reopen_owner_detection_and_interface_repair",
            preferred_validator_kind=decision.preferred_validator_kind,
            reason="A production-path wrapper calls a missing download function chain, so the loop must reopen owner detection and interface repair instead of repeating the same validator surface.",
        )
    if "missing 1 required positional argument" in lowered or "takes " in lowered and " positional argument" in lowered:
        return FailureSwitchRecommendation(
            failure_signature=signature,
            root_cause_category="wrong_callable_signature",
            previous_strategy=decision.selected_validation_strategy,
            new_strategy="signature_aware_callable_capture",
            preferred_validator_kind=decision.preferred_validator_kind,
            reason="The validator called a real helper with the wrong signature and should switch to signature-aware invocation.",
        )
    if "no module named 'wrds'" in lowered or "modulenotfounderror: no module named 'wrds'" in lowered:
        return FailureSwitchRecommendation(
            failure_signature=signature,
            root_cause_category="missing_external_dependency",
            previous_strategy=decision.selected_validation_strategy,
            new_strategy="stubbed_external_dependency_validation",
            preferred_validator_kind=decision.preferred_validator_kind,
            reason="Live WRDS is unavailable; the validator should switch to local stubs or monkeypatched connectors.",
        )
    if "keyerror" in lowered or "get_loc" in lowered or "missing columns" in lowered:
        return FailureSwitchRecommendation(
            failure_signature=signature,
            root_cause_category="missing_column_contract",
            previous_strategy=decision.selected_validation_strategy,
            new_strategy="concrete_data_contract_diagnostics",
            preferred_validator_kind="dataframe" if decision.preferred_validator_kind == "dataframe" else decision.preferred_validator_kind,
            reason="The failure is a contract mismatch and should switch to concrete column/date diagnostics.",
        )
    if "requested notebook path escapes the active workspace" in lowered:
        return FailureSwitchRecommendation(
            failure_signature=signature,
            root_cause_category="notebook_path_escape",
            previous_strategy=decision.selected_validation_strategy,
            new_strategy="nbformat_parse_plus_nbclient_execution",
            preferred_validator_kind="notebook",
            reason="Notebook path validation must stay on the Route 1 notebook IO path.",
        )
    if "could not parse notebook" in lowered or "\ufeff{" in lowered:
        return FailureSwitchRecommendation(
            failure_signature=signature,
            root_cause_category="notebook_bom_parse",
            previous_strategy=decision.selected_validation_strategy,
            new_strategy="nbformat_parse_plus_nbclient_execution",
            preferred_validator_kind="notebook",
            reason="Notebook BOM or parse failures should switch to the Route 1 BOM-safe notebook reader.",
        )
    if "owning source" in lowered or "primary artifact" in lowered:
        return FailureSwitchRecommendation(
            failure_signature=signature,
            root_cause_category="source_owner_not_changed",
            previous_strategy=decision.selected_validation_strategy,
            new_strategy="reopen_owner_detection_and_primary_source_enforcement",
            preferred_validator_kind=decision.preferred_validator_kind,
            reason="The task passed checks without landing on the owning source or primary artifact.",
        )
    return None


def normalize_failure_signature(
    raw_text: str,
    *,
    task_shape: str,
    strategy: str,
    check_name: str = "python_validate",
    root_cause_category: str = "",
) -> str:
    normalized = " ".join(raw_text.strip().lower().split())
    normalized = normalized[:240]
    category = root_cause_category or _guess_root_cause_category(normalized)
    return f"{check_name}|{task_shape}|{strategy}|{category}|{normalized}"


def _guess_root_cause_category(lowered: str) -> str:
    if "dataframe output to validate" in lowered:
        return "direct_dataframe_exposure_missing"
    if "callable download or writable output path" in lowered:
        return "callable_download_missing"
    if "writable output frame to validate" in lowered:
        return "writable_output_path_missing"
    if "has no attribute" in lowered and "download_" in lowered:
        return "missing_function_chain"
    if "no module named 'wrds'" in lowered:
        return "missing_external_dependency"
    if "keyerror" in lowered or "get_loc" in lowered:
        return "missing_column_contract"
    if "requested notebook path escapes the active workspace" in lowered:
        return "notebook_path_escape"
    if "could not parse notebook" in lowered:
        return "notebook_bom_parse"
    if "owning source" in lowered or "primary artifact" in lowered:
        return "source_owner_not_changed"
    if "criterion fail" in lowered:
        return "criterion_fail_current_run"
    if "validation-plan error" in lowered:
        return "check_references_missing_file"
    return "generic_validation_failure"

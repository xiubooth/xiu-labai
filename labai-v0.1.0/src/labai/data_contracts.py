from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


DEFAULT_ACCEPTED_DATE_FIELDS = (
    "time_avail_m",
    "time_d",
    "date",
    "datadate",
    "datadateijal",
    "datacknowledgment",
    "as_of_date",
)


@dataclass(frozen=True)
class DataContractReport:
    label: str
    columns: tuple[str, ...]
    date_field: str
    missing_columns: tuple[str, ...]
    missing_alternative_groups: tuple[tuple[str, ...], ...]
    parse_failures: int
    duplicate_key_count: int
    null_date_rows: int
    min_date: str
    has_gvkey: bool
    issues: tuple[str, ...]


def expected_contract_for_label(label: str) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    lowered = label.lower()
    required_columns = ("gvkey",)
    if "rating" in lowered:
        return required_columns, (("credrat", "credrat_dwn", "splticrm", "rating"),)
    if "short" in lowered:
        return required_columns, (("shortint",),)
    if "pension" in lowered:
        return required_columns, (("paddml", "pbnaa", "pbnvv", "pbpro", "pbpru", "pcupsu", "pplao", "pplau", "pension"),)
    if "segment" in lowered:
        return required_columns, (("num_bus_seg", "segments"),)
    return required_columns, ()


def inspect_dataframe_contract(
    label: str,
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str] = (),
    alternative_column_groups: Iterable[Iterable[str]] = (),
    accepted_date_fields: Iterable[str] = DEFAULT_ACCEPTED_DATE_FIELDS,
    key_columns: Iterable[str] = ("gvkey",),
    min_date: str = "2010-01-01",
) -> DataContractReport:
    columns = tuple(str(column) for column in frame.columns)
    missing_columns = tuple(column for column in required_columns if column not in frame.columns)
    missing_alternative_groups = tuple(
        tuple(group)
        for group in alternative_column_groups
        if not any(option in frame.columns for option in group)
    )
    date_field = next((field for field in accepted_date_fields if field in frame.columns), "")
    issues: list[str] = []
    if missing_columns:
        issues.append(f"{label}:missing columns={list(missing_columns)}")
    if missing_alternative_groups:
        rendered = [list(group) for group in missing_alternative_groups]
        issues.append(f"{label}:missing alternative column groups={rendered}")
    if not date_field:
        issues.append(
            f"{label}:missing date-like column; accepted date fields={list(accepted_date_fields)} columns={list(columns)}"
        )
        return DataContractReport(
            label=label,
            columns=columns,
            date_field="",
            missing_columns=missing_columns,
            missing_alternative_groups=missing_alternative_groups,
            parse_failures=-1,
            duplicate_key_count=-1,
            null_date_rows=-1,
            min_date="",
            has_gvkey="gvkey" in frame.columns,
            issues=tuple(issues),
        )

    parsed = pd.to_datetime(frame[date_field], errors="coerce", format="mixed")
    parse_failures = int(parsed.isna().sum())
    null_date_rows = int(frame[date_field].isna().sum())
    if parse_failures > 0:
        issues.append(f"{label}:{date_field} parse_failures={parse_failures}")
    if null_date_rows > 0:
        issues.append(f"{label}:{date_field} null_date_rows={null_date_rows}")
    min_date_value = ""
    valid_dates = parsed.dropna()
    if not valid_dates.empty:
        min_date_value = str(valid_dates.min().date())
        if min_date_value < min_date:
            issues.append(f"{label}:min_date={min_date_value} before {min_date}")
    else:
        issues.append(f"{label}:{date_field} has no usable parsed dates")

    duplicate_key_count = -1
    normalized_key_columns = tuple(column for column in key_columns if column in frame.columns)
    if normalized_key_columns and date_field:
        duplicate_key_count = int(frame.duplicated(subset=[*normalized_key_columns, date_field]).sum())
        if duplicate_key_count > 0:
            issues.append(
                f"{label}:duplicate {','.join([*normalized_key_columns, date_field])} rows={duplicate_key_count}"
            )

    return DataContractReport(
        label=label,
        columns=columns,
        date_field=date_field,
        missing_columns=missing_columns,
        missing_alternative_groups=missing_alternative_groups,
        parse_failures=parse_failures,
        duplicate_key_count=duplicate_key_count,
        null_date_rows=null_date_rows,
        min_date=min_date_value,
        has_gvkey="gvkey" in frame.columns,
        issues=tuple(issues),
    )

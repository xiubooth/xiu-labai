from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / ".planning" / "phases" / "18-route1-hardening-ask-notebook-validator"
EVIDENCE_PATH = ARTIFACT_ROOT / "18-FALSE-POSITIVE-GUARD.jsonl"
DOC_PATH = ARTIFACT_ROOT / "18-FALSE-POSITIVE-GUARD.md"


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def main() -> int:
    errors: list[str] = []

    _require(DOC_PATH.exists(), "missing false-positive guard summary markdown", errors)
    _require(EVIDENCE_PATH.exists(), "missing false-positive guard evidence jsonl", errors)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    rows = _read_jsonl(EVIDENCE_PATH)
    by_name = {str(row.get("name")): row for row in rows}
    required_rows = (
        "criterion_fail_forces_check_failure",
        "failed_criterion_not_mapped_to_pass",
        "mixed_final_run_forces_failure",
        "fallback_cannot_hide_criterion_fail",
        "shallow_source_change_rejected",
        "task9_regression",
    )
    for name in required_rows:
        _require(name in by_name, f"missing evidence row `{name}`", errors)
        if name in by_name:
            _require(str(by_name[name].get("status")) == "passed", f"evidence row `{name}` is not passed", errors)

    criterion_row = by_name.get("criterion_fail_forces_check_failure", {})
    _require(bool(criterion_row.get("forced_failure")), "`criterion_fail_forces_check_failure` must prove forced failure", errors)
    _require(bool(criterion_row.get("exit_zero_override")), "`criterion_fail_forces_check_failure` must prove exit-code override", errors)

    mapping_row = by_name.get("failed_criterion_not_mapped_to_pass", {})
    _require(bool(mapping_row.get("failed_criterion_preserved")), "failed criterion must remain failed/open in acceptance mapping", errors)
    _require(bool(mapping_row.get("pass_promotion_blocked")), "failed criterion must not be promoted to PASS", errors)

    mixed_row = by_name.get("mixed_final_run_forces_failure", {})
    _require(bool(mixed_row.get("fail_wins")), "mixed PASS/FAIL final run must fail", errors)

    fallback_row = by_name.get("fallback_cannot_hide_criterion_fail", {})
    _require(bool(fallback_row.get("fallback_fail_propagates")), "dependency fallback must not hide criterion failure", errors)
    _require(bool(fallback_row.get("fallback_evidence_preserved")), "dependency fallback evidence must remain visible", errors)

    shallow_row = by_name.get("shallow_source_change_rejected", {})
    for field in ("validator_only_rejected", "helper_only_rejected", "syntax_only_rejected"):
        _require(bool(shallow_row.get(field)), f"shallow source-change guard missing `{field}=true`", errors)

    task9_row = by_name.get("task9_regression", {})
    outcome = str(task9_row.get("outcome", ""))
    _require(outcome in {"true_success", "honest_failure"}, "Task 9 regression outcome must be true_success or honest_failure", errors)
    if outcome == "honest_failure":
        _require(bool(task9_row.get("final_status_not_ok")), "honest failure must record final_status_not_ok=true", errors)
        _require(bool(task9_row.get("check_status_failed")), "honest failure must record check_status_failed=true", errors)
        _require(not bool(task9_row.get("false_success")), "honest failure must not be marked as false success", errors)
    if bool(task9_row.get("raw_current_criterion_fail_detected")):
        _require(task9_row.get("final_status") != "ok", "raw criterion fail must not end with status=ok", errors)
        _require(task9_row.get("check_status") != "passed", "raw criterion fail must not end with check_status=passed", errors)

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print("PASS: Phase 18 Route 1.1 false-positive guard verifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

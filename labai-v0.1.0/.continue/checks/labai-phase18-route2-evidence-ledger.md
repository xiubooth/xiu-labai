# Route 2 Evidence Ledger Check

Purpose: verify that final success is bound to recorded current-run evidence.

Pass conditions:
- task manifests, required reads, owner detection, edit ops, runtime results, typed validation, and final classification are written to `.labai/evidence/<task_run_id>.jsonl`
- final summaries prefer landed edits over intended edits
- missing evidence blocks success

Primary test coverage:
- `tests/test_route2_mature_loop.py::test_evidence_ledger_writes_required_sections`
- `tests/test_route2_mature_loop.py::test_build_landed_edit_evidence_uses_landed_files`
- `tests/test_route2_mature_loop.py::test_route2_context_enriches_task_contract_and_owner_failure`

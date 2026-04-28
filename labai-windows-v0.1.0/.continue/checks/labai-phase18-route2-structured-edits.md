# Route 2 Structured Edit Ops Check

Purpose: verify that LabAI plans and records typed edit operations instead of fuzzy prose-only edits.

Pass conditions:
- source-edit tasks include owning-source or primary-artifact edit ops
- notebook tasks include a notebook edit op
- validator-only or helper-only plans are rejected as success modes
- landed edit evidence records before/after hashes
- unified diffs are parsed through `unidiff.PatchSet`

Primary test coverage:
- `tests/test_route2_mature_loop.py::test_apply_unified_diff_text_uses_unidiff_parser`
- `tests/test_route2_mature_loop.py::test_apply_unified_diff_text_rejects_malformed_patch`
- `tests/test_route2_mature_loop.py::test_apply_unified_diff_text_rejects_parent_escape`
- `tests/test_route2_mature_loop.py::test_structured_edit_ops_require_source_and_notebook_targets`
- `tests/test_route2_mature_loop.py::test_landed_edit_evidence_records_hashes`

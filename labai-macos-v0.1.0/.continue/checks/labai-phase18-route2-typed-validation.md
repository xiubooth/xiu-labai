# Route 2 Typed Validation Check

Purpose: verify that LabAI records typed validation and criterion-level evidence.

Pass conditions:
- behavioral tasks cannot pass on syntax-only validation
- explicit `CRITERION FAIL` still forces failure
- dependency fallback cannot hide criterion failures
- notebook tasks record notebook execution validation
- data-contract tasks record concrete contract evidence

Primary test coverage:
- `tests/test_route2_mature_loop.py::test_typed_validation_rejects_syntax_only_behavioral_success`
- `tests/test_route2_mature_loop.py::test_typed_validation_marks_explicit_failed_criteria`
- `tests/test_editing.py::test_run_workspace_checks_fails_exit_zero_when_output_contains_criterion_fail`
- `tests/test_editing.py::test_run_workspace_checks_dependency_fallback_cannot_hide_criterion_fail`

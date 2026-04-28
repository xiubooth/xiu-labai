# Route 2 Task Manifest Check

Purpose: verify that coding tasks create a typed manifest before editing.

Pass conditions:
- prompt-named files are captured in `task_manifest.prompt_named_files`
- wildcard families expand into `task_manifest.wildcard_matches` when present
- primary artifacts are recorded in `task_manifest.primary_target_artifacts`
- owner candidates are recorded in `task_manifest.candidate_owner_files`

Primary test coverage:
- `tests/test_route2_mature_loop.py::test_task9_manifest_captures_primary_files`
- `tests/test_route2_mature_loop.py::test_task1_manifest_expands_wildcards`
- `tests/test_route2_mature_loop.py::test_task8_manifest_detects_reference_files`
- `tests/test_route2_mature_loop.py::test_notebook_manifest_marks_primary_artifact`

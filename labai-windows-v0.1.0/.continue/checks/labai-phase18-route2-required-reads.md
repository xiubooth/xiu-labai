# Route 2 Required Reads Check

Purpose: verify that LabAI records real inspection evidence before source-edit success.

Pass conditions:
- prompt-named files enter the required read set
- wildcard and reference files are inspected when present
- path listings alone do not count as inspection evidence
- coding tasks do not end with `read_strategy=none`

Primary test coverage:
- `tests/test_route2_mature_loop.py::test_required_read_set_and_inspection_include_prompt_named_files`
- `tests/test_route2_mature_loop.py::test_required_read_evidence_includes_wildcard_and_reference_files`
- `tests/test_route2_mature_loop.py::test_missing_path_listing_does_not_count_as_inspection`

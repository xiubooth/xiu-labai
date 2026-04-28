# Phase 18 Source Evidence Check

Scope:

- prompt-named source files must be inspected when present
- wildcard source families must be inspected when present
- source-required tasks cannot pass on validator-only or helper-only changes
- notebook-primary tasks cannot pass without durable notebook change

Machine mirror:

- `tests/test_editing.py`
- `tests/test_cli.py`
- `src/labai/phase16_source_and_evidence.py`
- `scripts/verify_phase18_route1_hardening.py`

# Phase 18 Validator Output Check

Scope:

- subprocess validator output is captured with UTF-8 plus replacement
- `PYTHONIOENCODING=utf-8` is set for local process execution
- repeated identical validator failures force a strategy switch
- Task 8 contract failures surface concrete field/date diagnostics

Machine mirror:

- `tests/test_providers.py`
- `tests/test_data_contracts.py`
- `tests/test_cli.py`
- `scripts/verify_phase18_route1_hardening.py`

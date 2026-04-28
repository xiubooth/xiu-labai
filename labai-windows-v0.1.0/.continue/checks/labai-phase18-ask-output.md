# Phase 18 Ask Output Check

Scope:

- `labai ask -- "Say exactly HELLO and nothing else."`

Pass conditions:

- stdout contains the assistant answer text
- stdout is not exactly `True`
- stdout is not exactly `False`
- Unicode output remains printable on Windows

Machine mirror:

- `tests/test_cli.py`
- `scripts/verify_phase18_route1_hardening.py`

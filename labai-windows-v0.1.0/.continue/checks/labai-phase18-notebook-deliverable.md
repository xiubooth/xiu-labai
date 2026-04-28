# Phase 18 Notebook Deliverable Check

Scope:

- notebook primary artifact stays inside the active workspace
- notebook is parsed with `nbformat`
- notebook executes with `nbclient`
- outputs are embedded back into the `.ipynb`
- helper-only change does not count as notebook success

Machine mirror:

- `tests/test_notebook_io.py`
- `tests/test_cli.py`
- `scripts/verify_phase18_route1_hardening.py`

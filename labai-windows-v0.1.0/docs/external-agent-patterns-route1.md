# Phase 18 Route 1 External Component Pattern

Route 1 hardens the existing LabAI shell instead of replacing it.

Runtime components used directly in product code:

- `nbformat` is used in [src/labai/notebook_io.py](/C:/Users/ASUS/Desktop/AI/src/labai/notebook_io.py) for notebook read/write and notebook creation.
- `nbclient.NotebookClient` is used in [src/labai/notebook_io.py](/C:/Users/ASUS/Desktop/AI/src/labai/notebook_io.py) for notebook execution inside the active workspace.
- `click.echo` is used in [src/labai/cli.py](/C:/Users/ASUS/Desktop/AI/src/labai/cli.py) for user-visible ask output and safe terminal emission.

Reference-only external design:

- Continue is a reference for source-controlled check specs and repeatable headless task validation.
- In this phase, Continue is not the workflow engine.
- LabAI remains the shell and Claw remains the runtime.
- Provider and model routing stay unchanged.

Out of scope for Route 1:

- OpenHands migration
- LangGraph migration
- Aider migration
- SWE-agent migration
- Continue runtime replacement
- Claw replacement
- provider/model strategy changes

If the Continue submodule cannot be added in the current workspace, Route 1 still proceeds with:

- repo-owned `.continue/checks/*.md` specs
- Python verifiers
- fresh regression evidence in `.planning/phases/18-route1-hardening-ask-notebook-validator/`

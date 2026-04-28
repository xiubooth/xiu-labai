# Phase 18 Route 2 External Patterns

Route 2 keeps LabAI as the shell and upgrades the internal edit loop with mature components and reference-system ideas.

## Runtime components

- `unidiff` is used in `src/labai/structured_edits.py` for unified diff parsing through `PatchSet`.
- `grep-ast` is wrapped by `src/labai/external/grep_ast_adapter.py`.
- `grep-ast` is optional in this workspace. When unavailable, `src/labai/repo_map.py` falls back to Python AST while still using the adapter boundary.
- Route 1 components remain active:
  - `nbformat` for notebook read and write
  - `nbclient` for notebook execution
  - `click` for CLI-safe output

## Borrowed ideas

- Aider:
  - repo map
  - explicit file ranking before edits
  - owner-aware context selection
- SWE-agent:
  - bounded ACI-style view, search, and structured edit workflow
  - explicit inspection evidence instead of assuming reads
- SWE-ReX:
  - structured runtime execution records with cwd, env, stdout, stderr, exit code, timeout, and duration
- Continue:
  - source-controlled checks that live in the repo
  - headless comparison mindset, without making Continue the runtime
- OpenHands:
  - sandbox/provider separation as a design boundary
- LangGraph:
  - future durable orchestration reference only

## Out of scope

- no platform migration
- no Aider runtime replacement
- no SWE-agent runtime replacement
- no SWE-ReX runtime replacement
- no OpenHands runtime replacement
- no Continue runtime replacement
- no LangGraph runtime migration
- no Docker requirement
- no provider or model strategy change

LabAI remains the CLI shell and the execution owner for `labai doctor`, `labai tools`, `labai ask`, and `labai workflow`.

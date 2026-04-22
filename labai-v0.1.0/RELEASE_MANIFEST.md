# Release Manifest

- Product: `labai`
- Version: `0.1.0`
- Release archive: `dist/labai-v0.1.0.zip`
- Release style: clean Windows-first RA/professor-facing package
- Default installed state after one-click bootstrap:
  - `active_profile = local`
  - `active_generation_provider = local`
  - `selected_runtime = claw`

## Included Core Install Surface

- `Launch-LabAI-Setup.cmd`
- `README.md`
- `.env.example`
- `pyproject.toml`
- `RELEASE_MANIFEST.md`
- `RELEASE_CHECKLIST.md`

## Included Windows Bootstrap Surface

- `scripts/windows/bootstrap-windows.ps1`
- `scripts/windows/install-labai.ps1`
- `scripts/windows/setup-local-ollama.ps1`
- `scripts/windows/setup-api-provider.ps1`
- `scripts/windows/switch-profile.ps1`
- `scripts/windows/verify-install.ps1`

## Included Runtime Assets

- `runtime-assets/claw/windows-x64/claw.exe`
- `runtime-assets/claw/windows-x64/README.md`

## Included Profile Templates

- `templates/profiles/local.toml`
- `templates/profiles/api-deepseek.toml`
- `templates/profiles/fallback.toml`

## Included User-Facing Docs

- `docs/INSTALL_WINDOWS.md`
- `docs/FIRST_RUN.md`
- `docs/PROFILES.md`
- `docs/API_PROVIDERS.md`
- `docs/TROUBLESHOOTING_INSTALL.md`
- `docs/external-agent-patterns-route1.md`
- `docs/external-agent-patterns-route2.md`

## Included Phase 18 Workflow And Runtime Modules

Route 1 and Route 1.1 runtime/support:

- `src/labai/notebook_io.py`
- `src/labai/data_contracts.py`

Route 2 and Route 2.1 framework surface:

- `src/labai/aci.py`
- `src/labai/task_manifest.py`
- `src/labai/repo_map.py`
- `src/labai/owner_detection.py`
- `src/labai/validator_routing.py`
- `src/labai/structured_edits.py`
- `src/labai/runtime_exec.py`
- `src/labai/typed_validation.py`
- `src/labai/evidence_ledger.py`
- `src/labai/external/__init__.py`
- `src/labai/external/grep_ast_adapter.py`

Existing core modules retained in the release:

- the current `src/labai/**/*.py` source tree, including CLI, workspace, workflow, providers, execution, runtime, research, papers, and tools packages

## Included Source-Controlled Checks

If present in the source repo, the release includes the clean `.continue/checks/` specs used by the current workflow framework:

- `.continue/checks/labai-phase18-ask-output.md`
- `.continue/checks/labai-phase18-notebook-deliverable.md`
- `.continue/checks/labai-phase18-source-evidence.md`
- `.continue/checks/labai-phase18-validator-output.md`
- `.continue/checks/labai-phase18-route2-task-manifest.md`
- `.continue/checks/labai-phase18-route2-required-reads.md`
- `.continue/checks/labai-phase18-route2-structured-edits.md`
- `.continue/checks/labai-phase18-route2-runtime-exec.md`
- `.continue/checks/labai-phase18-route2-typed-validation.md`
- `.continue/checks/labai-phase18-route2-evidence-ledger.md`

## Runtime Dependency Surface

Default install dependencies required by the shipped product:

- `click`
- `typer`
- `nbformat`
- `nbclient`
- `ipykernel`
- `pymupdf`
- `pypdf`
- `unidiff`
- `pandas`
- `numpy`

Optional dependency:

- `grep-ast`
  - optional because `src/labai/external/grep_ast_adapter.py` has a Python-AST fallback when `grep-ast` is unavailable

## Explicitly Excluded From The RA-Facing Zip

- `.planning/`
- `.labai/`
- `.claw/`
- `.codex/`
- `.pytest-workspaces/`
- `.pytest_cache/`
- `.pytest-route2/`
- `.pytest-tmp/`
- `.pytest-work/`
- `.release-staging/`
- `dist/`
- `tests/`
- `examples/`
- `third_party/`
- `__pycache__/`
- `*.pyc`
- `*.zip`
- `*.pdf`
- local `.env*` secrets
- local session, audit, output, and other dev-only traces
- maintainer-only packaging helpers such as `scripts/package_release.py`, `scripts/verify_release_archive.py`, and `scripts/windows/package-release.ps1`

## Verification Expectations For This Release

The intended maintainer verification flow is:

1. run the repo gates and Phase 18 verifiers
2. build the release zip with `scripts/windows/package-release.ps1`
3. verify archive contents with `scripts/verify_release_archive.py`
4. extract the zip to a fresh folder
5. run `Launch-LabAI-Setup.cmd`
6. rerun `scripts/windows/bootstrap-windows.ps1`
7. run `scripts/windows/verify-install.ps1`
8. confirm:
   - `labai doctor`
   - `labai tools`
   - `labai ask "hello"`
   - `labai workflow verify-workspace --preview`
   - local profile + managed Claw runtime + no developer `claw-code` path

## Third-Party / Reference Content

- `third_party/` is not bundled in this release.
- No reference submodules are required for normal RA usage.

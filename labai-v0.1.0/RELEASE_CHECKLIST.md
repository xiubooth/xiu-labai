# Release Checklist

## Package Scope

- [ ] Confirm the package is built from staging/copy/filtering, not by zipping the repo root.
- [ ] Confirm the release keeps the current public CLI surface:
  - `labai doctor`
  - `labai tools`
  - `labai ask`
  - `labai workflow ...`
- [ ] Confirm `labai ask` stays lightweight and does not auto-run heavy workflows.
- [ ] Confirm interactive progress goes to `stderr` and does not pollute exact-answer stdout output.
- [ ] Confirm the release keeps the default local-first target:
  - `active_profile = local`
  - `active_generation_provider = local`
  - `selected_runtime = claw`

## Required Product Files

- [ ] Confirm the release includes:
  - `Launch-LabAI-Setup.cmd`
  - `README.md`
  - `.env.example`
  - `pyproject.toml`
  - `RELEASE_MANIFEST.md`
  - `RELEASE_CHECKLIST.md`
- [ ] Confirm the release includes the shipped Windows scripts:
  - `bootstrap-windows.ps1`
  - `install-labai.ps1`
  - `setup-local-ollama.ps1`
  - `setup-api-provider.ps1`
  - `switch-profile.ps1`
  - `verify-install.ps1`
- [ ] Confirm the release includes the bundled Claw runtime asset.
- [ ] Confirm the release includes the three shipped profile templates.
- [ ] Confirm the release includes the core install docs and the two Phase 18 external-pattern docs.
- [ ] Confirm the release includes the full current `src/labai/**/*.py` source tree.
- [ ] Confirm the release includes `.continue/checks/*.md` when they exist in the source repo.

## Dependency Surface

- [ ] Confirm default install dependencies include:
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
- [ ] Confirm `grep-ast` is documented honestly as optional if the adapter fallback remains active.

## Forbidden Package Content

- [ ] Confirm the release zip does not include:
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
  - `.env`
  - `.env.local`
  - personal/dev-only traces
  - previous release zips

## Repo Verification Before Packaging

- [ ] Run:
  - `python -m pytest -q`
  - `labai doctor`
  - `labai tools`
  - `labai ask -- "Say exactly HELLO and nothing else."`
- [ ] Run:
  - `python scripts\verify_phase18_route1_hardening.py`
  - `python scripts\verify_phase18_route1_false_positive_guard.py`
  - `python scripts\verify_phase18_route2_mature_loop.py`
  - `python scripts\verify_phase18_route2_workflow_closure.py`
- [ ] If present, run:
  - `python scripts\verify_phase16_task_bank.py --min-total 60 --min-per-family 5 --min-public-repos 20 --min-public-tasks 20`
  - `python scripts\verify_phase16_isolation.py`
  - `python scripts\verify_phase16_validator_quality.py`
  - `python scripts\verify_phase16_dependency_fallback.py`
  - `python scripts\verify_phase16_source_and_evidence.py`

## Package Build And Archive Verification

- [ ] Run:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\windows\package-release.ps1`
- [ ] Confirm the archive path is:
  - `dist/labai-v0.1.0.zip`
- [ ] Run the direct archive verifier:
  - `python scripts\verify_release_archive.py --archive dist\labai-v0.1.0.zip`
- [ ] Confirm the archive contains the required Phase 18 modules and shipped `.continue/checks`.

## Fresh Extracted Install Verification

- [ ] Extract the zip to a fresh folder outside the dev repo.
- [ ] Run:
  - `Launch-LabAI-Setup.cmd -LauncherDir <fresh-local-bin> -SkipUserPathUpdate`
- [ ] Rerun:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1 -LauncherDir <fresh-local-bin> -SkipUserPathUpdate`
- [ ] Run:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\windows\verify-install.ps1 -LauncherDir <fresh-local-bin>`
- [ ] From the extracted install, run:
  - `labai doctor`
  - `labai tools`
  - `labai ask "hello"`
  - `labai workflow verify-workspace --preview`
- [ ] Run dependency and module import checks:
  - `python -c "import click, nbformat, nbclient, unidiff, pandas, numpy; print('deps_ok')"`
  - `python -c "import labai; import labai.notebook_io; import labai.validator_routing; import labai.task_manifest; import labai.structured_edits; import labai.runtime_exec; import labai.typed_validation; import labai.evidence_ledger; print('phase18_modules_ok')"`
- [ ] Confirm:
  - local profile is active
  - generation provider is local
  - selected runtime is claw
  - `runtime_fallback: none` for local ask
  - the managed Claw path is used
  - config does not reference developer `claw-code`
- [ ] Record the local performance smoke classification from `verify-install.ps1`:
  - `local_ready`
  - `local_works_but_slow`
  - `local_not_recommended`
  - `local_failed`

## Ship Readiness

- [ ] Confirm the package is clean, RA-facing, and free of development/personal traces.
- [ ] Confirm the package includes the current Phase 18 framework/runtime surface.
- [ ] Confirm the one-click setup flow provisions the intended local-first install successfully.

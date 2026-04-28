# macOS Release Manifest

- Product: `labai`
- Version: `0.1.0`
- Archive: `dist/labai-macos-v0.1.0.zip`
- Release style: clean macOS-focused package for real Mac validation
- Production readiness: contains real macOS Claw assets, but still requires real macOS validation before production claims

## Package Status

This package is named `labai-macos-v0.1.0.zip` because this repository now contains supplied GitHub Actions-built macOS Claw binaries at:

- `runtime-assets/claw/macos-arm64/claw`
- `runtime-assets/claw/macos-x64/claw`

The package includes the matching `claw.sha256` sidecars and README metadata for both architectures, and still supports `LABAI_CLAW_BINARY` as an override.

The macOS bootstrap now checks for Python 3.11 or newer before creating `.venv`, switches to a compliant Homebrew Python when available, attempts the official Homebrew install path if Homebrew is missing, configures Homebrew shellenv, attempts `brew install python@3.12` when no compliant interpreter is found, detects the Mac architecture, auto-configures a bundled macOS Claw binary when a real asset is present, installs Ollama automatically from the official macOS archive, starts the Ollama app or CLI service, waits for the local API, pulls only missing Qwen models, and records a direct Ollama versus LabAI smoke classification. If direct Ollama works but LabAI is blocked by a missing macOS Claw runtime, verification reports `blocked_by_claw`. This improves the scaffold but does not remove the need for a real macOS Claw binary.

Maintainer-only Claw build support is present in `scripts/mac/build-claw-macos.sh` and `.github/workflows/build-claw-macos.yml`. The local source checkout observed during this packaging pass was `C:\Users\ASUS\src\claw-code` at commit `11e2353585fac22568e2cd53d0cbffcd9d1b7e1b`. The supplied assets are Mach-O binaries; this Windows packaging host verified file presence, SHA256 sidecars, and Mach-O magic, while live execution remains part of the next Mac retest.

## Included macOS Surface

- `Launch-LabAI-Setup.command`
- `scripts/mac/build-claw-macos.sh`
- `scripts/mac/bootstrap-mac.sh`
- `scripts/mac/install-labai.sh`
- `scripts/mac/setup-local-ollama.sh`
- `scripts/mac/setup-api-provider.sh`
- `scripts/mac/verify-install.sh`
- `templates/profiles/local-mac.toml`
- `templates/profiles/api-deepseek-mac.toml`
- `templates/profiles/fallback-mac.toml`
- `runtime-assets/claw/macos-arm64/README.md`
- `runtime-assets/claw/macos-arm64/claw`
- `runtime-assets/claw/macos-arm64/claw.sha256`
- `runtime-assets/claw/macos-x64/README.md`
- `runtime-assets/claw/macos-x64/claw`
- `runtime-assets/claw/macos-x64/claw.sha256`

## Included Product Source

- `src/labai/**/*.py`
- Phase 18 notebook, validation, workflow, evidence, progress, and structured-runtime modules
- Phase 19 platform helper: `src/labai/runtime/platform.py`

## Included Docs

- `README.md`
- `docs/INSTALL_MAC.md`
- `docs/TROUBLESHOOTING_MAC.md`
- `docs/MAC_SMOKE_TEST.md`
- `docs/FIRST_RUN.md`
- `docs/PROFILES.md`
- `docs/API_PROVIDERS.md`

## Included Source-Controlled Checks

`.continue/` was inspected. The package includes only sanitized markdown check specs under `.continue/checks/` and excludes all other `.continue` content.

## Dependencies

Default runtime dependencies are declared in `pyproject.toml`, including:

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

`grep-ast` remains optional through the `route2` extra because the adapter has a Python-AST fallback.

## Explicitly Excluded

- Windows launcher and scripts
- Windows Claw runtime asset
- `.exe` and `.ps1` files
- `.planning/`
- `.labai/`
- `.claw/`
- `.codex/`
- `tests/`
- `examples/`
- `.pytest*`
- `.release-staging/`
- `dist/`
- `__pycache__/`
- `*.pyc`
- personal PDFs, prompts, logs, sessions, audits, outputs, and local runtime state

## Required Real Mac Validation

Before this can become a full macOS package, a Mac tester must validate:

- `chmod +x Launch-LabAI-Setup.command scripts/mac/*.sh`
- `./Launch-LabAI-Setup.command`
- `./scripts/mac/verify-install.sh`
- `labai doctor`
- `labai tools`
- `labai ask "what is 1+1"`
- `labai workflow verify-workspace --preview`
- macOS Claw binary path or `LABAI_CLAW_BINARY`
- Python auto-selection or Homebrew install behavior when the default `python3` is missing or below 3.11
- Ollama automatic install/start behavior
- Qwen model readiness and local performance classification

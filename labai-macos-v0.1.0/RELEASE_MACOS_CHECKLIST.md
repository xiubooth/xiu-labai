# macOS Release Checklist

## Package Identity

- [ ] Confirm the archive name is macOS-specific.
- [ ] If both real macOS Claw binaries are bundled, confirm the archive name is `labai-macos-v0.1.0.zip`.
- [ ] If a real macOS Claw binary is missing, confirm the archive name includes `scaffold`.
- [ ] Confirm the Windows package is not overwritten.
- [ ] Confirm the package is built by `python scripts/package_macos_release.py`.

## Required macOS Files

- [ ] `Launch-LabAI-Setup.command`
- [ ] `scripts/mac/build-claw-macos.sh`
- [ ] `scripts/mac/bootstrap-mac.sh`
- [ ] `scripts/mac/install-labai.sh`
- [ ] `scripts/mac/setup-local-ollama.sh`
- [ ] `scripts/mac/setup-api-provider.sh`
- [ ] `scripts/mac/verify-install.sh`
- [ ] `templates/profiles/local-mac.toml`
- [ ] `templates/profiles/api-deepseek-mac.toml`
- [ ] `templates/profiles/fallback-mac.toml`
- [ ] `runtime-assets/claw/macos-arm64/README.md`
- [ ] `runtime-assets/claw/macos-arm64/claw`
- [ ] `runtime-assets/claw/macos-arm64/claw.sha256`
- [ ] `runtime-assets/claw/macos-x64/README.md`
- [ ] `runtime-assets/claw/macos-x64/claw`
- [ ] `runtime-assets/claw/macos-x64/claw.sha256`
- [ ] `src/labai/**/*.py`
- [ ] `pyproject.toml`
- [ ] macOS install, troubleshooting, profile, API, first-run, and smoke-test docs
- [ ] Python version compliance and Homebrew install/switching path is present in `scripts/mac/install-labai.sh`
- [ ] Missing Homebrew triggers the automatic official Homebrew install attempt before manual fallback.
- [ ] maintainer-only macOS Claw build helper is present and does not run in normal RA setup.
- [ ] Ollama auto-install/start/readiness path is present in `scripts/mac/setup-local-ollama.sh`
- [ ] direct Ollama and LabAI ask performance smoke are present in `scripts/mac/verify-install.sh`
- [ ] missing Claw is reported as `blocked_by_claw` when direct Ollama works but LabAI ask cannot run.

## Forbidden Content

- [ ] `Launch-LabAI-Setup.cmd`
- [ ] `scripts/windows/`
- [ ] `runtime-assets/claw/windows-x64/`
- [ ] `*.exe`
- [ ] `*.ps1`
- [ ] `.planning/`
- [ ] `.labai/`
- [ ] `.claw/`
- [ ] `.codex/`
- [ ] `tests/`
- [ ] `examples/`
- [ ] `.pytest*`
- [ ] `.release-staging/`
- [ ] `dist/`
- [ ] `__pycache__/`
- [ ] `*.pyc`
- [ ] personal PDFs, prompts, logs, sessions, audits, outputs, and runtime state

## Static Validation

- [ ] Run `python scripts/verify_macos_release_archive.py --archive dist/labai-macos-v0.1.0.zip`.
- [ ] Confirm macOS scripts use LF line endings.
- [ ] Confirm macOS scripts use bash shebangs and `set -euo pipefail`.
- [ ] Confirm macOS scripts do not contain `%LOCALAPPDATA%`, `.exe`, `.ps1`, `winget`, `cmd.exe`, or PowerShell syntax.
- [ ] Confirm `install-labai.sh` enforces Python >= 3.11 before creating `.venv`.
- [ ] Confirm `install-labai.sh` tries Homebrew Python, automatic Homebrew install, shellenv setup, and `brew install python@3.12` before reporting a Python install blocker.
- [ ] Confirm `setup-local-ollama.sh` uses the official macOS Ollama archive and keeps manual install as fallback only.
- [ ] Confirm `bootstrap-mac.sh` prints separate Python, Python version, Python path, venv path, LabAI install, Ollama install, Ollama API, Qwen, Claw, verification, and local performance summary rows.
- [ ] If a macOS Claw binary is present, confirm it is non-empty, executable, not a Windows PE file, and has README smoke metadata.
- [ ] Run `bash -n scripts/mac/*.sh` if bash is available.

## Windows Regression

- [ ] Run `python -m pytest -q`.
- [ ] Run `labai doctor`.
- [ ] Run `labai ask "what is 1+1"`.
- [ ] Run `labai workflow verify-workspace --preview`.

## Real Mac Validation Required

- [ ] Run `chmod +x Launch-LabAI-Setup.command scripts/mac/*.sh`.
- [ ] Run `./Launch-LabAI-Setup.command`.
- [ ] Run `./scripts/mac/verify-install.sh`.
- [ ] Run `labai doctor`.
- [ ] Run `labai tools`.
- [ ] Run `labai ask "what is 1+1"`.
- [ ] Run `labai workflow verify-workspace --preview`.
- [ ] Confirm Claw works through a bundled macOS binary or `LABAI_CLAW_BINARY`.
- [ ] Confirm Python auto-selection or Homebrew Python installation works when the default `python3` is too old.
- [ ] Confirm Ollama/Qwen models are ready.
- [ ] Record local performance classification.

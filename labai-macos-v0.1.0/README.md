# labai

`labai` is a local-first research assistant shell. The current stable release path is Windows-first, with a macOS package now carrying GitHub Actions-built Claw runtime assets for real Mac retesting.

- one-click bootstrap
- local Ollama by default
- local Qwen generation and embedding models
- managed Claw runtime
- optional DeepSeek API later if you need it
- notebook read/write/execute support
- safer ask output and Windows-safe runtime handling
- evidence-oriented workflow verification and source-controlled checks

The public CLI stays small:

- `labai doctor`
- `labai tools`
- `labai ask <prompt>`
- `labai workflow <command> ...`

## Default Local Flow

The intended first release flow on Windows is:

1. unzip the release
2. double-click `Launch-LabAI-Setup.cmd`
3. wait for setup to finish
4. run:

```powershell
labai doctor
labai ask "hello"
labai workflow verify-workspace --preview
```

The default installed state is:

- `active_profile = local`
- `active_generation_provider = local`
- `selected_runtime = claw`
- local Ollama running
- managed `claw.exe` installed under `%LOCALAPPDATA%\LabAI\runtime\claw\claw.exe`
- required local models present:
  - `qwen2.5:7b`
  - `qwen2.5-coder:7b`
  - `qwen3-embedding:0.6b`

## What The Main Commands Do

### `labai doctor`

`labai doctor` checks whether the current install is ready. It reports the active profile, generation provider, runtime selection, local or API readiness, and the main missing prerequisites if something is not configured yet.

Example:

```powershell
labai doctor
```

### `labai ask`

`labai ask` is the lightweight direct-answer surface. It does not automatically scan the repo, read PDFs, edit files, or run workflows.

Examples:

```powershell
labai ask "hello"
labai ask "Summarize this repository"
```

If you actually want file, repo, PDF, or edit execution, use an explicit workflow command instead of `ask`.

### `labai workflow <command>`

`labai workflow` is the heavy execution surface for explicit file, repo, PDF, edit, and verification work. The current release keeps the command surface stable and adds stronger workspace verification and source-evidence handling behind it.

Example:

```powershell
labai workflow verify-workspace --preview
```

## Windows Quick Start

### One-click bootstrap

From the unzipped release folder:

- double-click `Launch-LabAI-Setup.cmd`

Equivalent PowerShell command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1
```

The bootstrap:

1. detects or installs Python
2. detects or installs Ollama
3. provisions the bundled managed Claw runtime
4. installs `labai`
5. applies the `local` profile
6. pulls missing local models
7. runs verification

## macOS Package

The macOS package is separate from the Windows package and is named honestly based on runtime contents. With real bundled macOS Claw binaries present for both supported architectures, the archive is `labai-macos-v0.1.0.zip`. It still needs real Mac retesting before claiming full production readiness.

After unzipping the macOS package on a Mac, start from Terminal:

```sh
chmod +x Launch-LabAI-Setup.command scripts/mac/*.sh
xattr -dr com.apple.quarantine .
./Launch-LabAI-Setup.command
```

Equivalent shell command:

```sh
scripts/mac/bootstrap-mac.sh
```

The macOS package:

1. finds or installs Python 3.11+
2. attempts to install Homebrew when the Mac needs it for Python setup
3. creates `.venv` with the compliant interpreter
4. installs `labai`
5. applies a macOS-safe local profile
6. detects or installs Ollama from the official macOS archive
7. starts Ollama and waits for `http://127.0.0.1:11434`
8. pulls only missing required Qwen models
9. looks for a managed or configured macOS Claw binary
10. runs install verification and reports local performance classification

macOS Claw is prepared through these paths:

- managed ARM64 binary when present: `runtime-assets/claw/macos-arm64/claw`
- managed Intel binary if added later: `runtime-assets/claw/macos-x64/claw`
- user-provided binary: `export LABAI_CLAW_BINARY="/path/to/claw"`
- source build only as an explicit developer fallback

Maintainers can rebuild the macOS Claw asset on a real Mac with:

```sh
scripts/mac/build-claw-macos.sh --source "$HOME/src/claw-code" --profile release
```

That script writes the real binary into the matching `runtime-assets/claw/macos-*` directory, records smoke metadata in the README, and is not part of normal RA setup. The current macOS package includes the supplied GitHub Actions-built `macos-arm64` and `macos-x64` Claw binaries; the next real Mac retest must run the version and end-to-end smoke checks.

Mac local Qwen performance depends on hardware. If 7B local models are too slow, use API mode or a smaller local model after validation.

If Ollama/Qwen succeeds but the selected architecture's macOS Claw binary is missing or fails smoke, the setup reports the exact Claw blocker and must not be treated as production-ready.

## Terminal Progress

Interactive runs now emit short progress messages to `stderr` while work is happening. This is not token streaming. It is step-by-step status such as:

- runtime/provider/model selection
- model call started
- waiting for model response
- reading workspace files
- running validation

The final answer body still stays on `stdout`, so exact-output asks remain clean.

Progress control:

```powershell
$env:LABAI_PROGRESS="auto"   # default
$env:LABAI_PROGRESS="on"
$env:LABAI_PROGRESS="off"
```

### Verify the install

```powershell
labai doctor
labai tools
labai ask "hello"
labai workflow verify-workspace --preview
```

On macOS the one-click setup creates `~/Library/Application Support/LabAI/bin/labai` and adds that directory to `~/.zprofile`. A setup script cannot mutate the parent Terminal after it exits, so if the same Terminal still says `labai: command not found`, run:

```sh
source ~/.zprofile
rehash
labai doctor
```

You can also run the launcher directly:

```sh
"$HOME/Library/Application Support/LabAI/bin/labai" doctor
```

## Generic Prompt Examples

These are intentionally generic and safe to reuse.

```powershell
labai doctor
labai tools
labai ask "hello"
labai ask "What is asset pricing?"
labai workflow verify-workspace --preview
```

## Local Performance Note

The install verification now runs a small local model smoke and reports one of:

- `local_ready`
- `local_works_but_slow`
- `local_not_recommended`
- `blocked_by_claw`
- `claw_model_syntax_failed`
- `local_failed`
- `not_measured`

If local Qwen works but is too slow on a weaker machine, the install can still succeed. In that case, API mode may be a better day-to-day option later.

`blocked_by_claw` means direct Ollama/Qwen can work, but LabAI cannot complete the local runtime path until a real macOS Claw binary is bundled or `LABAI_CLAW_BINARY` points to one. `claw_model_syntax_failed` means Claw is present but rejected the local model argument before completing the local Ollama path. `local_ready` is only valid when LabAI reports `runtime_used: claw`, `runtime_fallback: none`, a non-mock provider, and a non-mock selected model.

## Papers And Local Data

Local runtime data lives under `.labai/` inside your repo copy:

- `.labai/config.toml`
- `.labai/sessions/`
- `.labai/audit/`
- `.labai/outputs/`
- `.labai/library/`

If you want to work with PDFs, use a neutral folder such as `papers/` in your project or point to an allowed absolute path. The release package does not ship real sample papers.

## Profiles

The Windows release ships three user-facing profiles:

- `local`
- `api-deepseek`
- `fallback`

Switch profiles with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\switch-profile.ps1 -Profile local
powershell -ExecutionPolicy Bypass -File .\scripts\windows\switch-profile.ps1 -Profile api-deepseek
powershell -ExecutionPolicy Bypass -File .\scripts\windows\switch-profile.ps1 -Profile fallback
```

Local remains the default. DeepSeek is optional and can be configured later without changing the default local-first install.

macOS validation uses `templates/profiles/local-mac.toml` and `templates/profiles/api-deepseek-mac.toml` so Mac configs do not contain `%LOCALAPPDATA%`, `.exe`, or Windows launcher paths.

## Documentation

- [Windows install](docs/INSTALL_WINDOWS.md)
- [macOS install guide](docs/INSTALL_MAC.md)
- [macOS smoke test](docs/MAC_SMOKE_TEST.md)
- [First run](docs/FIRST_RUN.md)
- [Profiles](docs/PROFILES.md)
- [API providers](docs/API_PROVIDERS.md)
- [Install troubleshooting](docs/TROUBLESHOOTING_INSTALL.md)
- [macOS troubleshooting](docs/TROUBLESHOOTING_MAC.md)

## Development Install

If you are working from the source repo rather than the release zip:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

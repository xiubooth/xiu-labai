# labai

`labai` is a local-first research assistant shell for Windows. The release package is built for a practical first-run path:

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

The intended first release flow is:

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

`labai ask` runs a single prompt against the active local or API profile.

Examples:

```powershell
labai ask "hello"
labai ask "Summarize this repository"
```

### `labai workflow <command>`

`labai workflow` exposes higher-level helper commands on top of the same underlying runtime. The current release keeps the command surface stable and adds stronger workspace verification and source-evidence handling behind it.

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

### Verify the install

```powershell
labai doctor
labai tools
labai ask "hello"
labai workflow verify-workspace --preview
```

## Generic Prompt Examples

These are intentionally generic and safe to reuse.

```powershell
labai doctor
labai tools
labai ask "hello"
labai ask "Summarize this repository"
labai workflow verify-workspace --preview
```

## Papers And Local Data

Local runtime data lives under `.labai/` inside your repo copy:

- `.labai/config.toml`
- `.labai/sessions/`
- `.labai/audit/`
- `.labai/outputs/`
- `.labai/library/`

If you want to work with PDFs, use a neutral folder such as `papers/` in your project or point to an allowed absolute path. The release package does not ship real sample papers.

## Profiles

The release ships three user-facing profiles:

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

## Documentation

- [Windows install](docs/INSTALL_WINDOWS.md)
- [First run](docs/FIRST_RUN.md)
- [Profiles](docs/PROFILES.md)
- [API providers](docs/API_PROVIDERS.md)
- [Install troubleshooting](docs/TROUBLESHOOTING_INSTALL.md)

## Development Install

If you are working from the source repo rather than the release zip:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

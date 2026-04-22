# Install LabAI On Windows

## Who This Is For

Use this flow if you are a new RA or professor who wants a working local copy of `labai` without manually rebuilding the development setup.

## What You Need

- Windows PowerShell
- the release zip or a clone of this repo

Optional:

- a DeepSeek API key if you want API mode

The one-click bootstrap installs missing Python and Ollama automatically when `winget` is available. It also provisions the managed LabAI Claw runtime from the bundled release asset, so ordinary RAs do not need Rust, cargo, or a local `claw-code` repo.

## Recommended One-Click Install

From the release folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1
```

Or just double-click:

- `Launch-LabAI-Setup.cmd`

What the bootstrap does:

1. detects or installs Python 3
2. detects or installs Ollama
3. waits for the local Ollama API
4. provisions the bundled managed `claw.exe` to `%LOCALAPPDATA%\LabAI\runtime\claw\claw.exe`
5. creates or reuses `.venv`
   - during venv and pip bootstrap, the installer uses a repo-local temp directory under `.labai\temp\windows-bootstrap` to avoid Windows temp-permission failures
6. installs `labai`
7. applies the `local` profile
8. pulls the default local models if they are missing:
   - `qwen2.5:7b`
   - `qwen2.5-coder:7b`
   - `qwen3-embedding:0.6b`
9. runs:
   - `labai doctor`
   - `labai ask "hello"`
   - `scripts\windows\verify-install.ps1`

Default end state:

- `active_profile = local`
- `active_generation_provider = local`
- `selected_runtime = claw`

## Advanced Manual Install

If you only want the package/venv install without the full local runtime bootstrap:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\install-labai.ps1
```

That manual path still defaults to the lighter `fallback` profile.

## Choose A Different Starting Profile

Examples:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\install-labai.ps1 -Profile local
powershell -ExecutionPolicy Bypass -File .\scripts\windows\install-labai.ps1 -Profile api-deepseek
```

## Verify The Install

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\verify-install.ps1
labai doctor
labai tools
labai ask "hello"
```

## If `labai` Is Not Found Immediately

The installer creates a launcher directory and can add it to your user PATH. If the current shell does not see it yet:

1. open a new PowerShell window
2. go back to the release folder
3. run:

```powershell
labai doctor
```

## Next Setup Steps

- For local Ollama mode:
  - [FIRST_RUN.md](FIRST_RUN.md)
  - [PROFILES.md](PROFILES.md)
- For DeepSeek API mode:
  - [API_PROVIDERS.md](API_PROVIDERS.md)
- If something is missing:
  - [TROUBLESHOOTING_INSTALL.md](TROUBLESHOOTING_INSTALL.md)

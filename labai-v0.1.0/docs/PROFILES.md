# Profiles

## Release-Facing Profiles

`labai` ships three first-release profiles for non-expert setup:

1. `fallback`
2. `local`
3. `api-deepseek`

## `fallback`

Purpose:

- first install verification
- machines that are not fully runtime-ready yet

Behavior:

- runtime uses `native`
- generation stays lightweight
- `labai doctor`, `labai tools`, and `labai ask "hello"` should still work

Switch to it:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\switch-profile.ps1 -Profile fallback
```

## `local`

Purpose:

- managed Claw + local Ollama generation
- local embeddings and PDF retrieval

Conservative default model:

- `qwen2.5:7b`

Switch to it:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\switch-profile.ps1 -Profile local
```

Preferred helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1
```

Managed runtime details:

- bundled `claw.exe` is installed to `%LOCALAPPDATA%\LabAI\runtime\claw\claw.exe`
- local Ollama models are:
  - `qwen2.5:7b`
  - `qwen2.5-coder:7b`
  - `qwen3-embedding:0.6b`

## `api-deepseek`

Purpose:

- Claw + DeepSeek API through the OpenAI-compatible path

Requirements:

- `DEEPSEEK_API_KEY` must be set in the environment

Switch to it:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\switch-profile.ps1 -Profile api-deepseek
```

Preferred helper:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-api-provider.ps1 -Apply
```

## Backup Behavior

Every profile switch backs up the previous `.labai/config.toml` to:

- `.labai/config.backups/`

Profile switching does not delete user data.

## How `labai doctor` Reports Profiles

After switching, `labai doctor` reports:

- `active_profile`
- `active_generation_provider`
- `selected_runtime`
- the selected local or API provider status

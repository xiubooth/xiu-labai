# Troubleshooting Install

## One-Click Bootstrap Fails Before Install

Run the bootstrap from PowerShell so you can read the full message:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1
```

If the root launcher was double-clicked and failed, reopen PowerShell in the release folder and rerun the command above.

## `python` Or `py` Is Missing

The bootstrap tries to install Python automatically with `winget`.

If that is blocked in the current environment, install Python 3.11+ manually and make sure one of these works:

```powershell
python --version
py -3 --version
```

Then rerun:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1
```

## Virtual Environment Or `ensurepip` Fails

The installer uses a repo-local temp directory under:

- `.labai\temp\windows-bootstrap`

This avoids the common Windows `%LOCALAPPDATA%\Temp` permission failure during `python -m venv` and `ensurepip`.

If a previous attempt failed partway through:

1. remove the partially created `.venv` folder from the extracted release directory
2. rerun:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1
```

## `labai` Is Not Found After Setup

The installer creates a launcher directory. If the current shell does not see it yet:

1. open a new PowerShell window
2. go back to the release folder
3. run:

```powershell
labai doctor
```

If needed, rerun install and note the printed launcher directory.

## Ollama Is Missing

Check whether Ollama exists:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version
```

If that fails, install Ollama first, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1
```

## Required Ollama Models Are Missing

Preview:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-local-ollama.ps1 -Plan
```

Apply and allow model pulls:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-local-ollama.ps1 -Apply -Yes
```

## Claw Is Missing Or Not Ready

Run:

```powershell
labai doctor
```

Check:

- `selected_runtime`
- `claw_health`
- `runtime_check_claw_binary`

For a normal release install, the first fix is to rerun the bootstrap so the bundled managed `claw.exe` is copied to:

- `%LOCALAPPDATA%\LabAI\runtime\claw\claw.exe`

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1
```

If you still want to work without the local runtime, switch back to fallback mode:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\switch-profile.ps1 -Profile fallback
```

## DeepSeek Key Missing

Set the key for the current session:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

Then rerun:

```powershell
labai doctor
```

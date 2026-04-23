# First Run

## Minimum Successful First Run

After install, a new user should be able to run:

```powershell
labai doctor
labai tools
labai ask "hello"
labai workflow verify-workspace --preview
```

Interactive terminal runs may now print short progress messages to `stderr` while work is happening. The final answer body still stays on `stdout`.

## Recommended First Run

Use the one-click bootstrap:

1. unzip the release
2. double-click `Launch-LabAI-Setup.cmd`

Expected default end state:

- `active_profile = local`
- local Ollama running
- managed Claw runtime installed under `%LOCALAPPDATA%\LabAI\runtime\claw\claw.exe`
- required local models present:
  - `qwen2.5:7b`
  - `qwen2.5-coder:7b`
  - `qwen3-embedding:0.6b`
- `labai ask "hello"` uses the local Claw path with `runtime_fallback: none`

## Switch To Local Mode

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-local-ollama.ps1 -Plan
powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-local-ollama.ps1 -Apply
labai doctor
```

## Switch To API Mode

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-api-provider.ps1 -Apply
labai doctor
```

## Where Local Data Lives

Runtime data is stored inside the repo copy:

- `.labai/config.toml`
- `.labai/sessions/`
- `.labai/audit/`
- `.labai/outputs/`
- `.labai/library/`

These are local runtime files. They are excluded from the release package.

The managed bundled Claw runtime is stored outside the repo at:

- `%LOCALAPPDATA%\LabAI\runtime\claw\claw.exe`

## Working With PDFs

The release does not ship real sample papers. If you want to use PDF features, either:

- create a neutral folder such as `papers/` in your project and place your PDFs there
- or point `labai` at an allowed absolute path

Use the lightweight/heavy split intentionally:

- `labai ask ...` for direct prompt answers only
- `labai workflow ...` for actual file, repo, PDF, edit, and verification work

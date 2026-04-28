# macOS Smoke Test Checklist

Use this on a real Mac after Phase 19. Do not mark macOS support production-ready until this checklist passes on actual macOS hardware.

## Environment

```sh
sw_vers
uname -m
python3 --version
command -v brew
brew --version
ollama --version
curl -fsS http://127.0.0.1:11434/api/version
```

Expected:

- the bootstrap can find or install Python 3.11+
- if Homebrew is missing, the bootstrap attempts the official Homebrew install path before falling back to manual commands
- Ollama is installed and its local API is reachable
- architecture is recorded as `arm64` or `x86_64`

## Claw

Check one of these:

```sh
test -x runtime-assets/claw/macos-arm64/claw && echo managed_arm64_claw_ok
test -x runtime-assets/claw/macos-x64/claw && echo managed_x64_claw_ok
test -n "$LABAI_CLAW_BINARY" && test -x "$LABAI_CLAW_BINARY" && echo env_claw_ok
command -v claw
```

Expected:

- one real macOS Claw binary is available
- Windows `claw.exe` is not used
- if a bundled binary is present, bootstrap auto-configures it without requiring `LABAI_CLAW_BINARY`

Maintainer-only build command when the asset is missing:

```sh
scripts/mac/build-claw-macos.sh --source "$HOME/src/claw-code" --profile release
```

Expected:

- the script runs only on macOS
- output is written to the matching `runtime-assets/claw/macos-*/claw`
- `claw --version` smoke output is captured in the asset README
- Rust/Cargo remains a maintainer build requirement, not an RA setup requirement

## Bootstrap

```sh
chmod +x Launch-LabAI-Setup.command scripts/mac/*.sh
xattr -dr com.apple.quarantine .
./Launch-LabAI-Setup.command
```

Expected:

- `.venv` is created
- the final summary records the Python version and interpreter path used
- the final summary records the venv path used
- `.labai/config.toml` is created from the macOS local profile
- the selected bundled Claw is copied into `~/Library/Application Support/LabAI/runtime/claw/claw`
- `[claw].binary` points at the managed macOS Claw path
- launcher is created under `~/Library/Application Support/LabAI/bin`
- current setup process PATH includes the launcher directory
- `~/.zprofile` contains `export PATH="$HOME/Library/Application Support/LabAI/bin:$PATH"` exactly once
- missing Ollama is installed or a clear Ollama install blocker is reported
- Ollama is started and `http://127.0.0.1:11434` becomes reachable
- missing Qwen models are pulled one by one
- setup does not contain `%LOCALAPPDATA%` or `.exe` paths in the macOS config
- final summary separates Python, LabAI install, Ollama install, Ollama API, Qwen models, Claw, verification, and local performance
- if only Claw is missing, verification is reported as `blocked_by_claw` or partial instead of hiding the Python/Ollama/Qwen success state

## LabAI Commands

```sh
labai doctor
labai tools
labai ask "what is 1+1"
labai ask "只输出 3+5 的答案，不要解释"
labai workflow verify-workspace --preview
```

If `labai` is not found in the same Terminal after setup, open a new Terminal or run:

```sh
source ~/.zprofile
rehash
labai doctor
```

Or use the launcher directly:

```sh
"$HOME/Library/Application Support/LabAI/bin/labai" doctor
```

Expected:

- `active_profile: local`
- `active_generation_provider: local`
- `selected_runtime: claw`
- `runtime_fallback: none` when Claw and local Qwen are fully working
- ask outputs include `2` and `8`
- workflow preview resolves without mutation

## Direct Ollama Smoke

```sh
ollama run qwen2.5:7b "Say exactly 2 and nothing else."
```

Expected:

- output is `2`
- latency is acceptable for the tester's hardware

If this passes but LabAI smoke fails, record it as a LabAI/Claw runtime issue rather than an Ollama/Qwen issue.

## Install Verification

```sh
scripts/mac/verify-install.sh
```

Expected:

- dependency import check prints `deps_ok`
- Phase 18 module check prints `phase18_modules_ok`
- direct Ollama smoke is recorded
- LabAI ask smoke is recorded
- local performance classification is recorded as one of:
  - `local_ready`
  - `local_works_but_slow`
  - `local_not_recommended`
  - `blocked_by_claw`
  - `claw_model_syntax_failed`
  - `local_failed`
  - `not_measured`

`blocked_by_claw` means the direct Ollama smoke can run, but `labai ask` or workflow verification is blocked by the missing macOS Claw runtime.

`claw_model_syntax_failed` means Claw is present but rejected the local model argument. `local_ready` is only valid when LabAI ask reports the real local path: `runtime_used: claw`, `runtime_fallback: none`, a non-mock provider, and a non-mock selected model.

## Optional API Smoke

Only run this if a key is available:

```sh
export DEEPSEEK_API_KEY="your_key_here"
scripts/mac/setup-api-provider.sh --apply
labai doctor
labai ask "Say exactly 2 and nothing else."
```

Expected:

- API key is read from the environment
- no secret is written to config

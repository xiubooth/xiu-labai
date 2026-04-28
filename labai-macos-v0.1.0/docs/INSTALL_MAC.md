# macOS Install

The macOS package is separate from the Windows release path and now includes supplied GitHub Actions-built Claw runtime assets for arm64 and x64 Macs. Do not treat macOS as fully production-ready until a real Mac has completed the smoke checklist.

When both real bundled macOS Claw binaries are present, the archive is named:

```text
labai-macos-v0.1.0.zip
```

## Prerequisites

- macOS Terminal with `bash` or `zsh`
- Network access for Python/Ollama/Qwen setup when components are missing or too old
- Homebrew is optional at the start. If Python is missing or below 3.11 and Homebrew is missing, the bootstrap now attempts the official Homebrew install path automatically.
- Network access for Ollama/Qwen setup
- Admin permission if macOS needs it to install `Ollama.app` into `/Applications` or create `/usr/local/bin/ollama`
- A real macOS Claw binary from one of these paths:
  - bundled ARM64: `runtime-assets/claw/macos-arm64/claw`
  - bundled Intel: `runtime-assets/claw/macos-x64/claw`
  - user-provided override: `export LABAI_CLAW_BINARY="/path/to/claw"`
  - explicit developer source build fallback

The package must never use the Windows `claw.exe` on macOS.

Maintainers can produce the real macOS Claw asset on a Mac with:

```sh
scripts/mac/build-claw-macos.sh --source "$HOME/src/claw-code" --profile release
```

That maintainer script detects `arm64` versus `x86_64`, builds the real Rust binary, writes it into `runtime-assets/claw/macos-arm64/claw` or `runtime-assets/claw/macos-x64/claw`, sets executable permissions, runs `claw --version`, and records smoke metadata. Normal RA setup must not require Rust or Cargo.

## Bootstrap

From the unzipped macOS package root on a Mac:

```sh
chmod +x Launch-LabAI-Setup.command scripts/mac/*.sh
xattr -dr com.apple.quarantine .
./Launch-LabAI-Setup.command
```

Equivalent command:

```sh
scripts/mac/bootstrap-mac.sh
```

The bootstrap installs LabAI into `.venv`, creates `.labai/config.toml`, configures a user-local launcher under:

```text
~/Library/Application Support/LabAI/bin
```

The setup adds that launcher directory to the current setup process PATH and appends this line to `~/.zprofile` if it is not already present:

```sh
export PATH="$HOME/Library/Application Support/LabAI/bin:$PATH"
```

This is what makes `labai doctor`, `labai ask`, and `labai workflow ...` available in future zsh Terminal sessions. A setup script cannot update the parent Terminal PATH after it exits. If your current Terminal still says `labai: command not found` immediately after setup, open a new Terminal or run:

```sh
source ~/.zprofile
rehash
labai doctor
```

You can also bypass shell PATH activation with:

```sh
"$HOME/Library/Application Support/LabAI/bin/labai" doctor
```

During local setup the bootstrap now attempts to:

1. find a compliant Python interpreter, version 3.11 or newer
2. switch to Homebrew Python if the default `python3` is too old
3. install Homebrew when no compliant Python exists and `brew` is missing
4. configure Homebrew shellenv for the current setup run
5. add the Homebrew shellenv line to `~/.zprofile` for zsh if it is not already present
6. add the LabAI launcher PATH line to `~/.zprofile` without duplicating it
7. install `python@3.12` through Homebrew when no compliant Python is found
8. detect `/Applications/Ollama.app` or an existing `ollama` command
9. download the official macOS Ollama archive from Ollama if missing
10. install `Ollama.app` into `/Applications`
11. create or verify `/usr/local/bin/ollama`
12. start Ollama
13. wait for `http://127.0.0.1:11434`
14. pull only missing Qwen models

If automatic Homebrew setup asks for confirmation or a password, that is the normal Homebrew installer flow. If it finishes but the shell cannot see `brew`, open a new Terminal or run the shellenv line printed by the installer, then rerun:

```sh
./Launch-LabAI-Setup.command
```

If automatic Homebrew setup fails, use the fallback:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)" || eval "$(/usr/local/bin/brew shellenv)"
brew install python@3.12
./Launch-LabAI-Setup.command
```

If macOS asks for an admin password during Ollama setup, it is for installing the app under `/Applications` or creating the command link. If automatic Ollama install fails, install Ollama from `https://ollama.com/download/mac`, open it once, then rerun `./Launch-LabAI-Setup.command`.

## Local Profile

The macOS local profile is:

```text
templates/profiles/local-mac.toml
```

It uses:

- Claw: `$HOME/Library/Application Support/LabAI/runtime/claw/claw`
- Ollama command: `ollama`
- model: `qwen2.5:7b`
- required models: `qwen2.5:7b`, `qwen2.5-coder:7b`, `qwen3-embedding:0.6b`

## Ollama And Qwen

Check without pulling models:

```sh
scripts/mac/setup-local-ollama.sh --plan
```

Install/start Ollama if needed and pull missing models:

```sh
scripts/mac/setup-local-ollama.sh --apply --yes
```

Some MacBooks may be too slow for 7B local models. A slow local smoke is not the same as a broken install, but day-to-day use may be better with API mode or a smaller local model.

## API Mode

DeepSeek/API mode is optional and not the default.

For the current shell:

```sh
export DEEPSEEK_API_KEY="your_key_here"
```

To switch the generated config to the macOS API profile:

```sh
scripts/mac/setup-api-provider.sh --apply
```

Do not write real API keys into config files.

## Verify

```sh
scripts/mac/verify-install.sh
labai doctor
labai tools
labai ask "what is 1+1"
labai workflow verify-workspace --preview
```

Expected fully ready local state requires a working macOS Claw binary, reachable Ollama, and required Qwen models. When a bundled Claw binary is present for the Mac architecture, bootstrap copies it into `~/Library/Application Support/LabAI/runtime/claw/claw` and updates `[claw].binary` in `.labai/config.toml`.

If direct Ollama smoke passes but LabAI smoke fails, verification reports `blocked_by_claw`, `claw_model_syntax_failed`, or `local_failed` depending on the cause. Treat those as LabAI/Claw runtime issues, not as Ollama/Qwen install issues. `local_ready` requires `runtime_used: claw`, `runtime_fallback: none`, a non-mock provider, and a non-mock selected model.

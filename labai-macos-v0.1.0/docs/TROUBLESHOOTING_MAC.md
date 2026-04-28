# macOS Troubleshooting

The macOS package now includes supplied GitHub Actions-built Claw assets, while the Windows release path remains separate. Do not treat macOS as fully production-ready until a real Mac completes the smoke checklist.

## Script Will Not Open

If macOS blocks the root launcher:

```sh
chmod +x Launch-LabAI-Setup.command scripts/mac/*.sh
xattr -dr com.apple.quarantine .
./Launch-LabAI-Setup.command
```

You can always run the shell script directly:

```sh
scripts/mac/bootstrap-mac.sh
```

## `labai: command not found`

The macOS setup creates the launcher here:

```text
~/Library/Application Support/LabAI/bin/labai
```

The setup also updates the current setup process PATH and adds this exact line to `~/.zprofile` if it is not already present:

```sh
export PATH="$HOME/Library/Application Support/LabAI/bin:$PATH"
```

If your current Terminal still cannot find `labai` after setup, open a new Terminal or run:

```sh
source ~/.zprofile
rehash
labai doctor
```

You can also run the launcher directly:

```sh
"$HOME/Library/Application Support/LabAI/bin/labai" doctor
```

If writing `~/.zprofile` failed, the installer prints this manual command:

```sh
export PATH="$HOME/Library/Application Support/LabAI/bin:$PATH"
```

Re-running `./Launch-LabAI-Setup.command` will not duplicate the PATH line.

## Python Missing Or Too Old

LabAI requires Python 3.11 or newer. The bootstrap now checks the actual version before creating `.venv`.

If `python3` exists but is too old, setup tries these paths before failing:

1. a compliant `python3` already on PATH
2. Homebrew Python under `/opt/homebrew/bin` or `/usr/local/bin`
3. Python discovered from `brew --prefix`
4. automatic Homebrew install when `brew` is missing
5. `brew install python@3.12`

If Homebrew is missing, setup attempts the official Homebrew installer, configures Homebrew shellenv for the current setup run, and adds the zsh `~/.zprofile` shellenv line when safe and not already present.

If the Homebrew installer asks you to open a new Terminal, do that and rerun:

```sh
./Launch-LabAI-Setup.command
```

If automatic Homebrew install fails, use the manual fallback:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)" || eval "$(/usr/local/bin/brew shellenv)"
brew install python@3.12
./Launch-LabAI-Setup.command
```

Diagnostics:

```sh
command -v python3
python3 --version
command -v brew
brew --version
brew --prefix python@3.12
```

## Claw Missing

LabAI expects Claw for the local product path. The Windows `claw.exe` cannot run on macOS.

Supported macOS paths are:

- bundled ARM64 binary: `runtime-assets/claw/macos-arm64/claw`
- bundled Intel binary: `runtime-assets/claw/macos-x64/claw`
- user-provided binary:

```sh
export LABAI_CLAW_BINARY="/path/to/claw"
```

Source builds are a developer fallback, not the intended long-term RA install path.

Maintainers should build the bundled macOS asset on a real Mac:

```sh
scripts/mac/build-claw-macos.sh --source "$HOME/src/claw-code" --profile release
```

The script writes to `runtime-assets/claw/macos-arm64/claw` on Apple Silicon or `runtime-assets/claw/macos-x64/claw` on Intel, sets the executable bit, and runs a version smoke. If the current Mac architecture does not have a bundled asset, setup reports the exact expected path and remains blocked by Claw.

## Ollama Missing Or Stopped

The bootstrap now tries to install and start Ollama automatically. It uses the official macOS Ollama archive, installs `Ollama.app` into `/Applications`, and creates `/usr/local/bin/ollama` when needed.

If setup asks for an admin password, that is for `/Applications` or `/usr/local/bin`. If automatic installation fails, use the manual fallback:

1. Download Ollama from `https://ollama.com/download/mac`
2. Install and open Ollama
3. Rerun `./Launch-LabAI-Setup.command`

Diagnostics:

```sh
command -v ollama
ollama --version
curl -fsS http://127.0.0.1:11434/api/version
```

Then rerun:

```sh
scripts/mac/setup-local-ollama.sh --apply --yes
```

## Qwen Models Missing

Pull the default models:

```sh
scripts/mac/setup-local-ollama.sh --apply --yes
```

The defaults are:

- `qwen2.5:7b`
- `qwen2.5-coder:7b`
- `qwen3-embedding:0.6b`

## Local Model Is Too Slow

`scripts/mac/verify-install.sh` reports a local performance classification when the local profile can run:

- `local_ready`
- `local_works_but_slow`
- `local_not_recommended`
- `blocked_by_claw`
- `claw_model_syntax_failed`
- `local_failed`
- `not_measured`

If a Mac is slow but not broken, use API mode or a smaller local model after validation.

If direct Ollama smoke passes but `labai ask` fails, check the Claw section. That usually means the model stack is reachable but the LabAI Claw runtime is still missing, misconfigured, or rejecting the local model argument.

`blocked_by_claw` is not an Ollama/Qwen failure. It means direct local model execution was reachable, but the LabAI local runtime still needs a real macOS Claw binary or `LABAI_CLAW_BINARY`.

`claw_model_syntax_failed` means the macOS Claw binary is present, but the model call did not complete because Claw rejected the local model syntax. `local_ready` must not be recorded when LabAI falls back to native/mock, reports `selected_model: mock-static`, reports `provider_used: mock`, or returns a mock response.

## API Key

Set API keys through environment variables:

```sh
export DEEPSEEK_API_KEY="your_key_here"
```

Do not store secrets in `.labai/config.toml`.

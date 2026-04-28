# LabAI for macOS

Use this folder if you are on macOS.

## What This Setup Will Do

The one-click setup will:

- check or select Python 3.11 or newer
- install Homebrew if needed
- install LabAI
- install or start Ollama if needed
- pull the required Qwen models
- configure bundled macOS Claw if present
- configure the local profile
- run verification

## Step 0 - Get The Repository Onto Your Computer

### Option A: GitHub Browser Download

1. Open the GitHub repository page.
2. Click the green `Code` button.
3. Choose `Download ZIP`.
4. Open the downloaded repository folder.
5. Open `labai-macos-v0.1.0`.

### Option B: Command Line Clone

Open Terminal and run:

```sh
cd ~/Desktop
git clone https://github.com/xiubooth/xiu-labai.git
cd xiu-labai/labai-macos-v0.1.0
```

## Step 1 - Prepare The Launcher

Run these in Terminal from this folder:

```sh
chmod +x Launch-LabAI-Setup.command scripts/mac/*.sh
xattr -dr com.apple.quarantine .
```

`chmod` makes the setup files runnable. `xattr` removes download quarantine warnings.

## Step 2 - Run One-Click Setup

Run:

```sh
./Launch-LabAI-Setup.command
```

Alternative direct command:

```sh
./scripts/mac/bootstrap-mac.sh
```

## Step 3 - Refresh Terminal Path If Needed

The setup creates a launcher here:

```text
~/Library/Application Support/LabAI/bin/labai
```

If the same Terminal does not recognize `labai`, run:

```sh
source ~/.zprofile
rehash
```

Or run directly:

```sh
"$HOME/Library/Application Support/LabAI/bin/labai" doctor
```

Opening a new Terminal window should load `~/.zprofile` automatically.

## Step 4 - Verify Installation

Run these in Terminal:

```sh
labai doctor
labai tools
labai ask "hello"
labai workflow verify-workspace --preview
```

## Step 5 - Try Simple Student Examples

Run these in Terminal:

```sh
labai ask "What is 1+1?"
labai ask "Explain what a research assistant can use LabAI for in two sentences."
labai ask "Summarize this repository at a high level."
labai workflow verify-workspace --preview
```

## What The Main Commands Mean

- `labai doctor` checks whether setup is ready.
- `labai tools` lists available tools.
- `labai ask` asks a direct question.
- `labai workflow` runs structured workflows such as workspace verification.

## Troubleshooting

- If the launcher is blocked, rerun the `chmod` and `xattr` commands above.
- If Homebrew asks for a password, follow the prompt.
- If Python setup takes time, wait.
- If Ollama or model pulling is slow, wait. The first setup can take time.
- If local Qwen is slow, the Mac may be underpowered for local 7B models.
- If `labai` is not found, run `source ~/.zprofile`, then `rehash`.
- If Claw is missing, the bundled assets are expected at `runtime-assets/claw/macos-arm64/claw` and `runtime-assets/claw/macos-x64/claw`. You can override with `LABAI_CLAW_BINARY` for diagnostics.
- Optional API mode can be configured later if local performance is not enough.

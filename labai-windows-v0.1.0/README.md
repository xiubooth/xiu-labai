# LabAI for Windows

Use this folder if you are on Windows.

## What This Setup Will Do

The one-click setup will:

- install or select Python if needed
- install LabAI
- configure the managed Windows Claw runtime
- install or start Ollama if needed
- pull the required Qwen models
- configure the local profile
- run verification

## Step 0 - Get The Repository Onto Your Computer

### Option A: GitHub Browser Download

1. Open the GitHub repository page.
2. Click the green `Code` button.
3. Choose `Download ZIP`.
4. Open the downloaded repository folder.
5. Open `labai-windows-v0.1.0`.

### Option B: Command Line Clone

Open PowerShell and run:

```powershell
cd $HOME\Desktop
git clone https://github.com/xiubooth/xiu-labai.git
cd xiu-labai\labai-windows-v0.1.0
```

## Step 1 - Run One-Click Setup

Double-click:

```text
Launch-LabAI-Setup.cmd
```

Or run this in PowerShell from this folder:

```powershell
cd $HOME\Desktop\xiu-labai\labai-windows-v0.1.0
.\Launch-LabAI-Setup.cmd
```

Alternative direct PowerShell command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1
```

## Step 2 - Open A New Terminal If Needed

After setup, open a new PowerShell window if the `labai` command is not found. This lets Windows load any PATH changes made by setup.

The managed Windows Claw runtime is bundled at:

```text
runtime-assets/claw/windows-x64/claw.exe
```

Setup may install the managed runtime under:

```text
%LOCALAPPDATA%\LabAI\runtime\claw
```

## Step 3 - Verify Installation

Run these in PowerShell:

```powershell
labai doctor
labai tools
labai ask "hello"
labai workflow verify-workspace --preview
```

## Step 4 - Try Simple Student Examples

Run these in PowerShell:

```powershell
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

- If `labai` is not found, open a new PowerShell window or rerun setup.
- If PowerShell blocks scripts, use the `powershell -ExecutionPolicy Bypass -File .\scripts\windows\bootstrap-windows.ps1` command above.
- If Ollama or model pulling is slow, wait. The first setup can take time.
- If local Qwen is slow, the computer may be underpowered for local 7B models.
- Optional API profiles can be configured later if local performance is not enough.

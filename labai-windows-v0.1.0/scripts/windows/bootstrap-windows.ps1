[CmdletBinding()]
param(
    [string]$LauncherDir = "",
    [switch]$SkipUserPathUpdate,
    [string]$PythonWingetId = "Python.Python.3.12",
    [string]$OllamaWingetId = "Ollama.Ollama"
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$installScript = Join-Path $scriptRoot "install-labai.ps1"
$localSetupScript = Join-Path $scriptRoot "setup-local-ollama.ps1"
$verifyScript = Join-Path $scriptRoot "verify-install.ps1"
$bundledClawPath = Join-Path $repoRoot "runtime-assets\claw\windows-x64\claw.exe"
$defaultLauncherDir = Join-Path $env:LOCALAPPDATA "LabAI\bin"
$managedClawDir = Join-Path $env:LOCALAPPDATA "LabAI\runtime\claw"
$managedClawPath = Join-Path $managedClawDir "claw.exe"
$configPath = Join-Path $repoRoot ".labai\config.toml"
$venvLabai = Join-Path $repoRoot ".venv\Scripts\labai.exe"
$ollamaDefaultPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
$requiredModels = @("qwen2.5:7b", "qwen2.5-coder:7b", "qwen3-embedding:0.6b")

if ([string]::IsNullOrWhiteSpace($LauncherDir)) {
    $LauncherDir = $defaultLauncherDir
}
$resolvedLauncherDir = [System.IO.Path]::GetFullPath($LauncherDir)

function Write-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    Write-Host ""
    Write-Host "[labai-setup] $Message"
}

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @($machinePath, $userPath) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    if ($parts.Count -gt 0) {
        $env:PATH = ($parts -join ";")
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "Running: $Label"
    $global:LASTEXITCODE = 0
    & $Command
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode."
    }
}

function Invoke-Captured {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "Running: $Label"
    $global:LASTEXITCODE = 0
    $output = & $Command 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
    Write-Host $output.TrimEnd()
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode."
    }
    return $output
}

function Invoke-WingetInstall {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PackageId,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "$Description is missing and winget is not available. Install it manually, then rerun bootstrap-windows.ps1."
    }

    Invoke-Checked -Label "winget install $PackageId" -Command {
        & winget install --id $PackageId -e --accept-package-agreements --accept-source-agreements --silent
    }
    Refresh-ProcessPath
}

function Get-PythonLauncher {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    return $null
}

function Ensure-Python {
    $launcher = Get-PythonLauncher
    if ($launcher) {
        return $launcher
    }

    Invoke-WingetInstall -PackageId $PythonWingetId -Description "Python 3"

    $launcher = Get-PythonLauncher
    if ($launcher) {
        return $launcher
    }

    throw "Python installation finished, but Python is still not discoverable in the current shell. Open a new PowerShell window and rerun Launch-LabAI-Setup.cmd."
}

function Get-OllamaPath {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        return (Get-Command ollama).Source
    }
    if (Test-Path -LiteralPath $ollamaDefaultPath) {
        return $ollamaDefaultPath
    }
    return $null
}

function Test-OllamaApiReady {
    try {
        $null = Invoke-RestMethod "http://127.0.0.1:11434/api/version" -TimeoutSec 3
        return $true
    }
    catch {
        return $false
    }
}

function Wait-OllamaApi {
    param(
        [int]$TimeoutSeconds = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-OllamaApiReady) {
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Ollama did not become reachable at http://127.0.0.1:11434 within $TimeoutSeconds seconds."
}

function Ensure-Ollama {
    $ollamaPath = Get-OllamaPath
    if (-not $ollamaPath) {
        Invoke-WingetInstall -PackageId $OllamaWingetId -Description "Ollama"
        $ollamaPath = Get-OllamaPath
    }

    if (-not $ollamaPath) {
        throw "Ollama install completed, but ollama.exe was still not found. Reopen PowerShell or install Ollama manually, then rerun Launch-LabAI-Setup.cmd."
    }

    if (-not (Test-OllamaApiReady)) {
        Write-Host ""
        Write-Host "Starting Ollama..."
        Start-Process -FilePath $ollamaPath | Out-Null
        Wait-OllamaApi
    }

    return $ollamaPath
}

function Ensure-ManagedClaw {
    if (-not (Test-Path -LiteralPath $bundledClawPath)) {
        throw "Bundled Claw runtime asset is missing: $bundledClawPath"
    }

    New-Item -ItemType Directory -Force -Path $managedClawDir | Out-Null
    $copyNeeded = $true
    if (Test-Path -LiteralPath $managedClawPath) {
        $sourceInfo = Get-Item -LiteralPath $bundledClawPath
        $targetInfo = Get-Item -LiteralPath $managedClawPath
        $copyNeeded = ($sourceInfo.Length -ne $targetInfo.Length) -or ($sourceInfo.LastWriteTimeUtc -gt $targetInfo.LastWriteTimeUtc)
    }

    if ($copyNeeded) {
        Copy-Item -LiteralPath $bundledClawPath -Destination $managedClawPath -Force
    }

    return $managedClawPath
}

Write-Host "LabAI one-click Windows bootstrap"
Write-Host "Repo root: $repoRoot"
Write-Host "Launcher dir: $resolvedLauncherDir"
Write-Host "Managed Claw path: $managedClawPath"
Write-Host "Bundled Claw asset: $bundledClawPath"
Write-Host ""

Refresh-ProcessPath
Write-Step "Checking Python"
$pythonLauncher = Ensure-Python
Write-Host "Python launcher: $pythonLauncher"

Write-Step "Provisioning managed Claw runtime"
$managedClaw = Ensure-ManagedClaw
Write-Host "Managed Claw ready: $managedClaw"

Write-Step "Checking Ollama and local model service"
$ollamaPath = Ensure-Ollama
Write-Host "Ollama binary: $ollamaPath"
Write-Host "Ollama API: ready"

Write-Step "Installing LabAI into the local virtual environment"
Invoke-Checked -Label "install-labai.ps1" -Command {
    & $installScript -Profile local -LauncherDir $resolvedLauncherDir -SkipUserPathUpdate:$SkipUserPathUpdate
}

Write-Step "Applying the local profile and ensuring required local models"
Invoke-Checked -Label "setup-local-ollama.ps1" -Command {
    & $localSetupScript -Apply -Yes -GenerationModel "qwen2.5:7b" -EmbeddingModel "qwen3-embedding:0.6b"
}

if (-not (Test-Path -LiteralPath $venvLabai)) {
    throw "LabAI install did not produce the expected console entrypoint: $venvLabai"
}

$env:LABAI_CONFIG_PATH = $configPath
$env:PATH = "$resolvedLauncherDir;$($repoRoot)\.venv\Scripts;$env:PATH"

if (-not (Test-Path -LiteralPath $managedClawPath)) {
    throw "Managed Claw binary is missing after bootstrap: $managedClawPath"
}

Write-Step "Running doctor and lightweight ask smoke"
Invoke-Captured -Label "labai doctor" -Command { & $venvLabai doctor } | Out-Null
Invoke-Captured -Label 'labai ask "hello"' -Command { & $venvLabai ask "hello" } | Out-Null

Write-Step "Running shipped install verification"
Invoke-Checked -Label "verify-install.ps1" -Command {
    & $verifyScript -LauncherDir $resolvedLauncherDir
}

if (-not [string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
    Write-Host ""
    Write-Host "DEEPSEEK_API_KEY is present. API mode is available later through:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-api-provider.ps1 -Apply"
    Write-Host "The active profile remains local by default."
}

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Active profile target: local"
Write-Host "Managed Claw binary: $managedClawPath"
Write-Host "Required models: $($requiredModels -join ', ')"
Write-Host ""
Write-Host "Next commands:"
Write-Host "  labai doctor"
Write-Host '  labai ask "hello"'

[CmdletBinding()]
param(
    [ValidateSet("fallback", "local", "api-deepseek")]
    [string]$Profile = "fallback",
    [switch]$DevExtras,
    [switch]$ReplaceConfig,
    [string]$LauncherDir = "",
    [switch]$SkipUserPathUpdate
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$labaiDir = Join-Path $repoRoot ".labai"
$configPath = Join-Path $labaiDir "config.toml"
$venvDir = Join-Path $repoRoot ".venv"
$venvScripts = Join-Path $venvDir "Scripts"
$venvPython = Join-Path $venvScripts "python.exe"
$venvLabai = Join-Path $venvScripts "labai.exe"
$bootstrapTempRoot = Join-Path $labaiDir "temp\windows-bootstrap"
$bootstrapTempDir = Join-Path $bootstrapTempRoot ([System.Guid]::NewGuid().ToString("N"))
$defaultLauncherDir = Join-Path $env:LOCALAPPDATA "LabAI\bin"
if ([string]::IsNullOrWhiteSpace($LauncherDir)) {
    $LauncherDir = $defaultLauncherDir
}
$resolvedLauncherDir = [System.IO.Path]::GetFullPath($LauncherDir)
$templateMap = @{
    "local"        = "templates/profiles/local.toml"
    "api-deepseek" = "templates/profiles/api-deepseek.toml"
    "fallback"     = "templates/profiles/fallback.toml"
}
$templatePath = Join-Path $repoRoot $templateMap[$Profile]

function Resolve-PythonLauncher {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return @{
            Executable = $pythonCommand.Source
            Prefix = @()
        }
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        return @{
            Executable = $pyCommand.Source
            Prefix = @("-3")
        }
    }

    throw "Python was not found on PATH. Install Python 3.11+ first."
}

function Invoke-RootPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $invocationArgs = @($script:PythonLauncher.Prefix + $Arguments)
    & $script:PythonLauncher.Executable @invocationArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Invoke-VenvPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment Python command failed with exit code $LASTEXITCODE."
    }
}

function Ensure-LauncherPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BinDir
    )

    $env:PATH = "$BinDir;$venvScripts;$env:PATH"
    if ($SkipUserPathUpdate) {
        return
    }

    $currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $segments = @()
    if (-not [string]::IsNullOrWhiteSpace($currentUserPath)) {
        $segments = $currentUserPath.Split(";") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }
    if ($segments -contains $BinDir) {
        return
    }

    $updatedPath = if ($segments.Count -gt 0) {
        ($segments + $BinDir) -join ";"
    }
    else {
        $BinDir
    }
    [Environment]::SetEnvironmentVariable("Path", $updatedPath, "User")
}

function Restore-ProcessTemp {
    param(
        [string]$PreviousTemp,
        [string]$PreviousTmp
    )

    if ([string]::IsNullOrWhiteSpace($PreviousTemp)) {
        Remove-Item Env:TEMP -ErrorAction SilentlyContinue
    }
    else {
        $env:TEMP = $PreviousTemp
    }

    if ([string]::IsNullOrWhiteSpace($PreviousTmp)) {
        Remove-Item Env:TMP -ErrorAction SilentlyContinue
    }
    else {
        $env:TMP = $PreviousTmp
    }
}

$script:PythonLauncher = Resolve-PythonLauncher

Write-Host "LabAI installer"
Write-Host "Repo root: $repoRoot"
Write-Host "Selected profile: $Profile"
Write-Host "Launcher dir: $resolvedLauncherDir"
Write-Host "Dev extras: $($DevExtras.IsPresent)"
Write-Host "Bootstrap temp dir: $bootstrapTempDir"
Write-Host ""

if (-not (Test-Path -LiteralPath $templatePath)) {
    throw "Profile template not found: $templatePath"
}

New-Item -ItemType Directory -Force -Path $bootstrapTempDir | Out-Null
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$env:TEMP = $bootstrapTempDir
$env:TMP = $bootstrapTempDir

try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Host "Creating virtual environment..."
        Push-Location $repoRoot
        try {
            Invoke-RootPython -Arguments @("-m", "venv", $venvDir)
        }
        finally {
            Pop-Location
        }
    }

    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "Virtual environment Python was not created: $venvPython"
    }

    Push-Location $repoRoot
    try {
        Write-Host "Upgrading pip..."
        Invoke-VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")

        $installArgs = @("-m", "pip", "install", "-e")
        if ($DevExtras) {
            $installArgs += ".[dev]"
        }
        else {
            $installArgs += "."
        }
        Write-Host "Installing LabAI..."
        Invoke-VenvPython -Arguments $installArgs
    }
    finally {
        Pop-Location
    }
}
finally {
    Restore-ProcessTemp -PreviousTemp $previousTemp -PreviousTmp $previousTmp
}

if (-not (Test-Path -LiteralPath $venvLabai)) {
    throw "The LabAI console script was not created: $venvLabai"
}

New-Item -ItemType Directory -Force -Path $labaiDir | Out-Null
$configStatus = "preserved"
$backupPath = $null
if (-not (Test-Path -LiteralPath $configPath)) {
    Copy-Item -LiteralPath $templatePath -Destination $configPath -Force
    $configStatus = "created"
}
elseif ($ReplaceConfig) {
    $backupDir = Join-Path $labaiDir "config.backups"
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupPath = Join-Path $backupDir "config-$timestamp.toml"
    Copy-Item -LiteralPath $configPath -Destination $backupPath -Force
    Copy-Item -LiteralPath $templatePath -Destination $configPath -Force
    $configStatus = "replaced"
}

New-Item -ItemType Directory -Force -Path $resolvedLauncherDir | Out-Null
$launcherCmdPath = Join-Path $resolvedLauncherDir "labai.cmd"
$launcherPs1Path = Join-Path $resolvedLauncherDir "labai.ps1"

$launcherCmd = @"
@echo off
set "LABAI_CONFIG_PATH=$configPath"
"$venvLabai" %*
"@
$launcherPs1 = @"
`$env:LABAI_CONFIG_PATH = '$configPath'
& '$venvLabai' @args
"@

Set-Content -LiteralPath $launcherCmdPath -Value $launcherCmd -Encoding ascii
Set-Content -LiteralPath $launcherPs1Path -Value $launcherPs1 -Encoding utf8
Ensure-LauncherPath -BinDir $resolvedLauncherDir

Write-Host "Install complete."
Write-Host "Config status: $configStatus"
Write-Host "Config path: $configPath"
Write-Host "Launcher cmd: $launcherCmdPath"
Write-Host "Launcher ps1: $launcherPs1Path"
if ($backupPath) {
    Write-Host "Config backup: $backupPath"
}
Write-Host ""
Write-Host "Next commands:"
if (-not $SkipUserPathUpdate) {
    Write-Host "  Open a new PowerShell window if labai is not available in the current shell yet."
}
Write-Host "  For the full one-click local runtime bootstrap, run Launch-LabAI-Setup.cmd or scripts\\windows\\bootstrap-windows.ps1."
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\windows\verify-install.ps1 -LauncherDir `"$resolvedLauncherDir`""
Write-Host "  labai doctor"

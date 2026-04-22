[CmdletBinding()]
param(
    [ValidateSet("deepseek")]
    [string]$Provider = "deepseek",
    [switch]$Plan,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$switchScript = Join-Path $scriptRoot "switch-profile.ps1"
$configPath = Join-Path $repoRoot ".labai\config.toml"
$venvLabai = Join-Path $repoRoot ".venv\Scripts\labai.exe"
$keyPresent = -not [string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)

Write-Host "API provider setup"
Write-Host "Repo root: $repoRoot"
Write-Host "Provider: $Provider"
Write-Host "DEEPSEEK_API_KEY present: $keyPresent"
Write-Host ""
Write-Host "PowerShell example:"
Write-Host '  $env:DEEPSEEK_API_KEY="your_api_key_here"'
Write-Host ""
Write-Host "Usage:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-api-provider.ps1 -Plan"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-api-provider.ps1 -Apply"

if ($Plan -or -not $Apply) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -Apply to switch to the API profile."
    return
}

& $switchScript -Profile api-deepseek

if (Test-Path -LiteralPath $venvLabai) {
    Write-Host ""
    Write-Host "Running doctor with the API profile..."
    $env:LABAI_CONFIG_PATH = $configPath
    & $venvLabai doctor
}
else {
    Write-Host "LabAI is not installed yet. Run install-labai.ps1 first."
}

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("local", "api-deepseek", "fallback")]
    [string]$Profile
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$labaiDir = Join-Path $repoRoot ".labai"
$configPath = Join-Path $labaiDir "config.toml"
$backupDir = Join-Path $labaiDir "config.backups"

$templateMap = @{
    "local"        = "templates/profiles/local.toml"
    "api-deepseek" = "templates/profiles/api-deepseek.toml"
    "fallback"     = "templates/profiles/fallback.toml"
}

$templatePath = Join-Path $repoRoot $templateMap[$Profile]
if (-not (Test-Path -LiteralPath $templatePath)) {
    throw "Profile template not found: $templatePath"
}

New-Item -ItemType Directory -Force -Path $labaiDir | Out-Null

$backupPath = $null
if (Test-Path -LiteralPath $configPath) {
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupPath = Join-Path $backupDir "config-$timestamp.toml"
    Copy-Item -LiteralPath $configPath -Destination $backupPath -Force
}

Copy-Item -LiteralPath $templatePath -Destination $configPath -Force

Write-Host "Profile switched."
Write-Host "Repo root: $repoRoot"
Write-Host "Selected profile: $Profile"
Write-Host "Template: $templatePath"
Write-Host "Config: $configPath"
if ($backupPath) {
    Write-Host "Backup: $backupPath"
}
else {
    Write-Host "Backup: (none - config was created)"
}
Write-Host ""
Write-Host "Next step:"
Write-Host "  labai doctor"

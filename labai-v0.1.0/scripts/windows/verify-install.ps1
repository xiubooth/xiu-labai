[CmdletBinding()]
param(
    [string]$LauncherDir = ""
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$configPath = Join-Path $repoRoot ".labai\config.toml"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$venvLabai = Join-Path $repoRoot ".venv\Scripts\labai.exe"
if ([string]::IsNullOrWhiteSpace($LauncherDir)) {
    $LauncherDir = Join-Path $env:LOCALAPPDATA "LabAI\bin"
}
$resolvedLauncherDir = [System.IO.Path]::GetFullPath($LauncherDir)

function Invoke-CapturedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "Running: $Label"
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

function Assert-OutputContains {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Output,
        [Parameter(Mandatory = $true)]
        [string[]]$Needles,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    foreach ($needle in $Needles) {
        if ($Output -notmatch [regex]::Escape($needle)) {
            throw "$Label is missing expected text: $needle"
        }
    }
}

function Assert-OutputDoesNotContain {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Output,
        [Parameter(Mandatory = $true)]
        [string[]]$Needles,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    foreach ($needle in $Needles) {
        if ($Output -match [regex]::Escape($needle)) {
            throw "$Label contains forbidden text: $needle"
        }
    }
}

function Get-ActiveProfile {
    if (-not (Test-Path -LiteralPath $configPath)) {
        return ""
    }
    $content = Get-Content -LiteralPath $configPath -Raw -Encoding utf8
    $match = [regex]::Match($content, '(?ms)^\[app\].*?^active_profile\s*=\s*"([^"]+)"')
    if ($match.Success) {
        return $match.Groups[1].Value.Trim().ToLowerInvariant()
    }
    return ""
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment is missing: $venvPython"
}
if (-not (Test-Path -LiteralPath $venvLabai)) {
    throw "LabAI console entrypoint is missing: $venvLabai"
}

$env:LABAI_CONFIG_PATH = $configPath
$env:PATH = "$resolvedLauncherDir;$($repoRoot)\.venv\Scripts;$env:PATH"
$activeProfile = Get-ActiveProfile

Push-Location $repoRoot
try {
    Invoke-CapturedCommand -Label "python --version" -Command { & $venvPython --version } | Out-Null
    $dependencyCheck = Invoke-CapturedCommand -Label 'python -c "dependency imports"' -Command {
        & $venvPython -c "import click, nbclient, nbformat, numpy, pandas, typer, unidiff; import fitz; import pypdf; print('deps_ok')"
    }
    Assert-OutputContains -Output $dependencyCheck -Needles @(
        "deps_ok"
    ) -Label "dependency import check"

    $moduleCheck = Invoke-CapturedCommand -Label 'python -c "Phase 18 module imports"' -Command {
        & $venvPython -c "import labai; import labai.aci; import labai.data_contracts; import labai.notebook_io; import labai.owner_detection; import labai.repo_map; import labai.runtime_exec; import labai.structured_edits; import labai.task_manifest; import labai.typed_validation; import labai.validator_routing; import labai.evidence_ledger; import labai.external.grep_ast_adapter; print('phase18_modules_ok')"
    }
    Assert-OutputContains -Output $moduleCheck -Needles @(
        "phase18_modules_ok"
    ) -Label "Phase 18 module import check"

    $doctor = Invoke-CapturedCommand -Label "labai doctor" -Command { & $venvLabai doctor }
    Assert-OutputContains -Output $doctor -Needles @(
        "labai doctor",
        "active_profile:",
        "active_generation_provider:",
        "selected_runtime:"
    ) -Label "labai doctor"
    if ($activeProfile -eq "local") {
        $configText = Get-Content -LiteralPath $configPath -Raw -Encoding utf8
        $managedClawPath = Join-Path $env:LOCALAPPDATA "LabAI\runtime\claw\claw.exe"
        Assert-OutputContains -Output $doctor -Needles @(
            "active_profile: local",
            "active_generation_provider: local",
            "selected_runtime: claw"
        ) -Label "local profile doctor output"
        Assert-OutputDoesNotContain -Output $doctor -Needles @(
            "claw-code"
        ) -Label "local profile doctor output"
        if ($configText -notmatch [regex]::Escape("%LOCALAPPDATA%/LabAI/runtime/claw/claw.exe")) {
            throw "Local profile config does not point at the managed Claw runtime path."
        }
        if (-not (Test-Path -LiteralPath $managedClawPath)) {
            throw "Managed Claw binary is missing: $managedClawPath"
        }
    }

    $tools = Invoke-CapturedCommand -Label "labai tools" -Command { & $venvLabai tools }
    Assert-OutputContains -Output $tools -Needles @(
        "registered_tools:"
    ) -Label "labai tools"

    if ($activeProfile -eq "api-deepseek" -and [string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
        Assert-OutputContains -Output $doctor -Needles @(
            'missing `DEEPSEEK_API_KEY`'
        ) -Label "api profile doctor output"
        Write-Host ""
        Write-Host "API profile selected without DEEPSEEK_API_KEY. Skipping ask/workflow smoke."
    }
    else {
        $ask = Invoke-CapturedCommand -Label 'labai ask "hello"' -Command { & $venvLabai ask "hello" }
        Assert-OutputContains -Output $ask -Needles @(
            "status:",
            "runtime_used:",
            "answer:"
        ) -Label "labai ask"
        if ($activeProfile -eq "local") {
            Assert-OutputContains -Output $ask -Needles @(
                "runtime_used: claw",
                "runtime_fallback: none"
            ) -Label "local profile ask output"
        }

        $workflow = Invoke-CapturedCommand -Label "labai workflow verify-workspace --preview" -Command {
            & $venvLabai workflow verify-workspace --preview
        }
        Assert-OutputContains -Output $workflow -Needles @(
            "labai workflow verify-workspace --preview",
            "target_workspace_root:"
        ) -Label "workflow preview"
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Install verification passed."

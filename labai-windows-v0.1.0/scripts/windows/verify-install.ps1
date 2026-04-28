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
$ollamaDefaultPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
if ([string]::IsNullOrWhiteSpace($LauncherDir)) {
    $LauncherDir = Join-Path $env:LOCALAPPDATA "LabAI\bin"
}
$resolvedLauncherDir = [System.IO.Path]::GetFullPath($LauncherDir)

function Write-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    Write-Host ""
    Write-Host "[labai-install] $Message"
}

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

function Get-ConfigSectionValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Section,
        [Parameter(Mandatory = $true)]
        [string]$Key
    )

    if (-not (Test-Path -LiteralPath $configPath)) {
        return ""
    }
    $content = Get-Content -LiteralPath $configPath -Raw -Encoding utf8
    $sectionPattern = "(?ms)^\[$([regex]::Escape($Section))\]\s*(.*?)(?=^\[|\z)"
    $sectionMatch = [regex]::Match($content, $sectionPattern)
    if (-not $sectionMatch.Success) {
        return ""
    }
    $keyPattern = "(?m)^\s*$([regex]::Escape($Key))\s*=\s*`"([^`"]+)`"\s*$"
    $keyMatch = [regex]::Match($sectionMatch.Groups[1].Value, $keyPattern)
    if ($keyMatch.Success) {
        return $keyMatch.Groups[1].Value.Trim()
    }
    return ""
}

function Convert-ToProcessArgumentString {
    param(
        [string[]]$Arguments = @()
    )

    if (-not $Arguments -or $Arguments.Count -eq 0) {
        return ""
    }

    $quoted = foreach ($argument in $Arguments) {
        if ($null -eq $argument) {
            '""'
        }
        else {
            $text = [string]$argument
            if ($text -match '[\s"]') {
                '"' + ($text -replace '"', '\"') + '"'
            }
            else {
                $text
            }
        }
    }
    return ($quoted -join " ")
}

function Invoke-TimedProcessCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [int]$TimeoutSeconds = 120,
        [string]$WorkingDirectory = "",
        [hashtable]$EnvironmentOverrides = @{}
    )

    Write-Host ""
    Write-Host "Running: $Label"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = Convert-ToProcessArgumentString -Arguments $Arguments
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
        $psi.WorkingDirectory = $WorkingDirectory
    }
    if ($null -eq $EnvironmentOverrides) {
        $EnvironmentOverrides = @{}
    }
    foreach ($entry in $EnvironmentOverrides.GetEnumerator()) {
        $psi.Environment[$entry.Key] = [string]$entry.Value
    }

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $startTime = Get-Date
    $null = $process.Start()
    $timedOut = -not $process.WaitForExit($TimeoutSeconds * 1000)
    if ($timedOut) {
        try {
            $process.Kill($true)
        }
        catch {
        }
    }
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $durationMs = [int][Math]::Round(((Get-Date) - $startTime).TotalMilliseconds)
    $combined = @($stdout.TrimEnd(), $stderr.TrimEnd()) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    if ($combined.Count -gt 0) {
        Write-Host ($combined -join [Environment]::NewLine)
    }
    $exitCode = if ($timedOut) { -1 } else { $process.ExitCode }
    return [pscustomobject]@{
        Label = $Label
        ExitCode = $exitCode
        TimedOut = $timedOut
        DurationMs = $durationMs
        Stdout = $stdout
        Stderr = $stderr
        Output = ($combined -join [Environment]::NewLine)
    }
}

function Get-OllamaBinary {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    if (Test-Path -LiteralPath $ollamaDefaultPath) {
        return $ollamaDefaultPath
    }
    return ""
}

function Get-LocalPerformanceClassification {
    param(
        [Parameter(Mandatory = $true)]
        [int]$ElapsedMs,
        [switch]$Failed
    )

    if ($Failed) {
        return "local_failed"
    }
    if ($ElapsedMs -le 15000) {
        return "local_ready"
    }
    if ($ElapsedMs -le 45000) {
        return "local_works_but_slow"
    }
    return "local_not_recommended"
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
    Write-Step "Checking Python and shipped dependencies"
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

    Write-Step "Checking LabAI doctor and tool surface"
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
        Write-Step "Running lightweight ask and workflow smoke checks"
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

        if ($activeProfile -eq "local") {
            Write-Step "Running local performance smoke"
            $generationModel = Get-ConfigSectionValue -Section "models" -Key "general_model"
            if ([string]::IsNullOrWhiteSpace($generationModel)) {
                $generationModel = "qwen2.5:7b"
            }
            $ollamaBinary = Get-OllamaBinary
            $directOllama = $null
            if (-not [string]::IsNullOrWhiteSpace($ollamaBinary)) {
                $directOllama = Invoke-TimedProcessCapture `
                    -Label "ollama run $generationModel `"Say exactly 2 and nothing else.`"" `
                    -FilePath $ollamaBinary `
                    -Arguments @("run", $generationModel, "Say exactly 2 and nothing else.") `
                    -TimeoutSeconds 120 `
                    -WorkingDirectory $repoRoot
            }
            else {
                Write-Host "Skipping direct Ollama smoke because ollama.exe was not found."
            }

            $labaiAskSmoke = Invoke-TimedProcessCapture `
                -Label 'labai ask -- "Say exactly 2 and nothing else."' `
                -FilePath $venvLabai `
                -Arguments @("ask", "--", "Say exactly 2 and nothing else.") `
                -TimeoutSeconds 180 `
                -WorkingDirectory $repoRoot `
                -EnvironmentOverrides @{
                    LABAI_CONFIG_PATH = $configPath
                    LABAI_PROGRESS = "on"
                }

            if ($labaiAskSmoke.TimedOut) {
                throw "Local LabAI smoke timed out while waiting for the local model path."
            }
            if ($labaiAskSmoke.ExitCode -ne 0) {
                throw "Local LabAI smoke failed: $($labaiAskSmoke.Output)"
            }
            if (
                ($labaiAskSmoke.Stdout -notmatch "(?m)^2\s*$") -and
                ($labaiAskSmoke.Output -notmatch "(?m)^2\s*$")
            ) {
                throw "Local LabAI smoke did not produce the expected exact answer `2`."
            }

            $classification = Get-LocalPerformanceClassification -ElapsedMs $labaiAskSmoke.DurationMs
            $comparisonNote = "Only the LabAI path was measured."
            if ($directOllama -ne $null) {
                if ($directOllama.TimedOut) {
                    $comparisonNote = "Direct Ollama smoke timed out; the local model path is not healthy enough to recommend."
                }
                elseif ($directOllama.ExitCode -ne 0) {
                    $comparisonNote = "Direct Ollama smoke failed; check the Ollama model runtime separately."
                }
                elseif ($labaiAskSmoke.DurationMs -gt (($directOllama.DurationMs * 2) + 2000)) {
                    $comparisonNote = "The LabAI/Claw path is materially slower than direct Ollama on this machine."
                }
                elseif ($directOllama.DurationMs -gt ($labaiAskSmoke.DurationMs + 2000)) {
                    $comparisonNote = "Local model latency dominates more than the LabAI wrapper path."
                }
                else {
                    $comparisonNote = "Direct Ollama and the LabAI path were in the same general latency range."
                }
            }

            Write-Host "local_performance_classification: $classification"
            Write-Host "local_performance_labai_ask_ms: $($labaiAskSmoke.DurationMs)"
            if ($directOllama -ne $null) {
                Write-Host "local_performance_direct_ollama_ms: $($directOllama.DurationMs)"
            }
            Write-Host "local_performance_note: $comparisonNote"
            if ($classification -eq "local_not_recommended") {
                Write-Host "local_performance_next_step: Consider API mode if this machine feels too slow for day-to-day local use."
            }
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Install verification passed."

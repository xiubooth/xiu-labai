[CmdletBinding()]
param(
    [string]$GenerationModel = "qwen2.5:7b",
    [string]$CodeModel = "",
    [string]$EmbeddingModel = "qwen3-embedding:0.6b",
    [switch]$Plan,
    [switch]$Apply,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$switchScript = Join-Path $scriptRoot "switch-profile.ps1"
$configPath = Join-Path $repoRoot ".labai\config.toml"
$venvLabai = Join-Path $repoRoot ".venv\Scripts\labai.exe"
$ollamaDefaultPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"

if ([string]::IsNullOrWhiteSpace($CodeModel)) {
    if ($GenerationModel -like "gpt-oss:*") {
        $CodeModel = $GenerationModel
    }
    else {
        $CodeModel = "qwen2.5-coder:7b"
    }
}

$requiredModels = @($GenerationModel, $CodeModel, $EmbeddingModel | Select-Object -Unique)
$ollamaPath = if (Get-Command ollama -ErrorAction SilentlyContinue) {
    (Get-Command ollama).Source
}
elseif (Test-Path -LiteralPath $ollamaDefaultPath) {
    $ollamaDefaultPath
}
else {
    $null
}

function Set-SectionValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Content,
        [Parameter(Mandatory = $true)]
        [string]$Section,
        [Parameter(Mandatory = $true)]
        [string]$Key,
        [Parameter(Mandatory = $true)]
        [string]$ValueLiteral
    )

    $sectionPattern = "(?ms)(^\[$([regex]::Escape($Section))\]\r?\n)(.*?)(?=^\[|\z)"
    $sectionMatch = [regex]::Match($Content, $sectionPattern)
    if (-not $sectionMatch.Success) {
        throw "Could not find [$Section] in $configPath"
    }

    $sectionBody = $sectionMatch.Groups[2].Value
    $keyPattern = "(?m)(^$([regex]::Escape($Key))\s*=\s*).*$"
    if ($sectionBody -notmatch $keyPattern) {
        throw "Could not find [$Section] $Key in $configPath"
    }

    $updatedSectionBody = [regex]::Replace($sectionBody, $keyPattern, ('$1' + $ValueLiteral), 1)
    $prefix = $Content.Substring(0, $sectionMatch.Groups[2].Index)
    $suffixStart = $sectionMatch.Groups[2].Index + $sectionMatch.Groups[2].Length
    $suffix = $Content.Substring($suffixStart)
    return $prefix + $updatedSectionBody + $suffix
}

function Invoke-LabaiDoctor {
    if (-not (Test-Path -LiteralPath $venvLabai)) {
        Write-Host "Skipping doctor check because LabAI is not installed yet."
        return
    }
    $env:LABAI_CONFIG_PATH = $configPath
    & $venvLabai doctor
}

$serviceReachable = $false
try {
    $null = Invoke-RestMethod "http://127.0.0.1:11434/api/version" -TimeoutSec 3
    $serviceReachable = $true
}
catch {
    $serviceReachable = $false
}

$installedModels = @()
if ($ollamaPath -and $serviceReachable) {
    try {
        $modelList = & $ollamaPath list 2>&1 | Out-String
        foreach ($model in $requiredModels) {
            if ($modelList -match [regex]::Escape($model)) {
                $installedModels += $model
            }
        }
    }
    catch {
    }
}
$missingModels = @($requiredModels | Where-Object { $installedModels -notcontains $_ })

Write-Host "Local Ollama setup"
Write-Host "Repo root: $repoRoot"
if ($ollamaPath) {
    Write-Host "Ollama binary: $ollamaPath"
}
else {
    Write-Host "Ollama binary: (not found)"
}
Write-Host "Service reachable: $serviceReachable"
Write-Host "Generation model: $GenerationModel"
Write-Host "Code model: $CodeModel"
Write-Host "Embedding model: $EmbeddingModel"
Write-Host "Missing models: $($(if ($missingModels) { $missingModels -join ', ' } else { '(none)' }))"
Write-Host ""
Write-Host "Usage:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-local-ollama.ps1 -Plan"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup-local-ollama.ps1 -Apply"

if ($Plan -or -not $Apply) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -Apply to switch to the local profile and optionally pull models."
    return
}

& $switchScript -Profile local

$content = Get-Content -LiteralPath $configPath -Raw -Encoding utf8
$requiredLiteral = "[" + (($requiredModels | ForEach-Object { "`"$_`"" }) -join ", ") + "]"
$content = Set-SectionValue -Content $content -Section "models" -Key "general_model" -ValueLiteral "`"$GenerationModel`""
$content = Set-SectionValue -Content $content -Section "models" -Key "code_model" -ValueLiteral "`"$CodeModel`""
$content = Set-SectionValue -Content $content -Section "ollama" -Key "model" -ValueLiteral "`"$GenerationModel`""
$content = Set-SectionValue -Content $content -Section "ollama" -Key "required_models" -ValueLiteral $requiredLiteral
$content = Set-SectionValue -Content $content -Section "papers" -Key "embedding_model" -ValueLiteral "`"$EmbeddingModel`""
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($configPath, $content, $utf8NoBom)

if ($missingModels.Count -gt 0) {
    if (-not $ollamaPath) {
        Write-Host "Ollama is not installed, so missing models could not be pulled."
    }
    elseif (-not $Yes) {
        Write-Host "Missing models were detected but not pulled."
        Write-Host "Re-run with -Yes to pull:"
        Write-Host "  $($missingModels -join ', ')"
    }
    else {
        foreach ($model in $missingModels) {
            Write-Host "Pulling model: $model"
            & $ollamaPath pull $model
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to pull model: $model"
            }
        }
    }
}

Write-Host ""
Write-Host "Running doctor with the local profile..."
Invoke-LabaiDoctor

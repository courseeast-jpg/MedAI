# MEDAI-LOCAL-SELF-HEALING-VALIDATION-07 — Windows PowerShell runner.
#
# One operator command. Self-heals when the pipeline path produces zero
# facts by falling back to a deterministic adapter / MKB diagnostic via
# the Python aggregator.
#
# Pipeline:
#   1. cd to repo root.
#   2. verify branch is clinical-knowledge-architecture.
#   3. report (do not abort on) working-tree noise; print HEAD.
#   4. attempt a fast-forward pull only if clean; otherwise note it.
#   5. run scripts/local_runtime_prepare_minimum_05.ps1.
#   6. ensure test_input/ exists.
#   7. Windows file picker (PDF / TXT only) OR synthetic non-PHI fallback.
#   8. run run_medai_local_one_doc_operator_validation_05.py.
#   9. run run_medai_streamlit_launch_smoke_05.py.
#  10. invoke the Python 07 aggregator.
#  11. print exact next command: streamlit run app/main.py
#
# Hard rules:
#   * Never scan operator's whole computer; never auto-pick from
#     Downloads/Desktop/Documents.
#   * Never print raw filename, raw path, raw OCR text, raw document
#     text, private path, or PHI to any public report.
#   * Never enable auto-accept. Never call any external API.
#   * Never stage or commit selected local files / runtime DBs / OCR
#     dumps / private logs / .env.

$ErrorActionPreference = "Stop"

function Write-Step([string]$message) {
    Write-Host ("[self-heal-07] {0}" -f $message) -ForegroundColor Cyan
}

function Resolve-RepoRoot() {
    $scriptPath = $PSScriptRoot
    if (-not $scriptPath) {
        $scriptPath = (Split-Path -Parent $MyInvocation.MyCommand.Definition)
    }
    $candidate = Split-Path -Parent $scriptPath
    if (Test-Path (Join-Path $candidate ".git")) {
        return $candidate
    }
    if (Test-Path (Join-Path $scriptPath ".git")) {
        return $scriptPath
    }
    return (Get-Location).Path
}

$repoRoot = Resolve-RepoRoot
Set-Location $repoRoot
Write-Step ("repo root: {0}" -f $repoRoot)

# 2. Verify branch.
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "clinical-knowledge-architecture") {
    Write-Host ("[self-heal-07] branch is '{0}', expected 'clinical-knowledge-architecture'. Aborting." -f $branch) -ForegroundColor Red
    exit 2
}
$headShort = (& git rev-parse --short HEAD).Trim()
Write-Step ("branch={0} head={1}" -f $branch, $headShort)

# 3. Null-safe working-tree probe. Self-heal: do NOT abort on noise.
$gitStatusOutput = @(& git status --porcelain)
$gitStatus = ($gitStatusOutput -join "`n").Trim()
$workingTreeClean = $true
if ($gitStatus) {
    $workingTreeClean = $false
    Write-Step "working tree has local changes; continuing without aborting"
}

# 4. Optional fast-forward pull only when clean. Never destructive.
if ($workingTreeClean) {
    try {
        & git pull --ff-only origin clinical-knowledge-architecture 2>$null | Out-Null
        Write-Step "fast-forward pull attempted"
    } catch {
        Write-Step "pull skipped or failed; continuing"
    }
} else {
    Write-Step "pull skipped (tree not clean)"
}

# 5. Minimal runtime preparer.
Write-Step "running local_runtime_prepare_minimum_05.ps1"
$prepareScript = Join-Path $repoRoot "scripts\local_runtime_prepare_minimum_05.ps1"
$dependencyResult = "skipped"
try {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $prepareScript
    $dependencyResult = if ($LASTEXITCODE -eq 0) { "ok" } else { ("exit_{0}" -f $LASTEXITCODE) }
} catch {
    $dependencyResult = "exception"
}

# 6. Ensure test_input/ exists.
$testInputDir = Join-Path $repoRoot "test_input"
if (-not (Test-Path $testInputDir)) {
    New-Item -ItemType Directory -Path $testInputDir | Out-Null
}

# 6a. Clear ONLY this block's known neutral inputs.
foreach ($name in @("one_click_input.pdf", "one_click_input.txt", "one_click_synthetic_lab.txt", "self_healing_input.pdf", "self_healing_input.txt", "self_healing_synthetic_lab.txt")) {
    $target = Join-Path $testInputDir $name
    if (Test-Path $target) {
        Remove-Item $target -Force
    }
}

# 7. Operator-driven file picker, limited to PDF/TXT.
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Filter = "Documents (*.pdf;*.txt)|*.pdf;*.txt"
$dialog.Multiselect = $false
$dialog.Title = "MedAI self-healing validation: select ONE PDF or TXT (cancel for synthetic)"
$dialog.CheckFileExists = $true
$dialog.CheckPathExists = $true
$dialogResult = $dialog.ShowDialog()

$inputMode = "synthetic_fallback"
$inputSuffix = ".txt"
$inputSize = 0
$inputDest = $null

if ($dialogResult -eq [System.Windows.Forms.DialogResult]::OK -and $dialog.FileName) {
    $picked = $dialog.FileName
    $inputSuffix = [System.IO.Path]::GetExtension($picked).ToLower()
    if ($inputSuffix -ne ".pdf" -and $inputSuffix -ne ".txt") {
        Write-Step "picked file has an unsupported suffix; falling back to synthetic"
        $inputMode = "synthetic_fallback"
        $inputSuffix = ".txt"
    } else {
        $inputMode = "selected_file"
        $neutralName = if ($inputSuffix -eq ".pdf") { "self_healing_input.pdf" } else { "self_healing_input.txt" }
        $inputDest = Join-Path $testInputDir $neutralName
        Copy-Item -LiteralPath $picked -Destination $inputDest -Force
        $inputSize = (Get-Item $inputDest).Length
    }
}

if ($inputMode -eq "synthetic_fallback") {
    $inputSuffix = ".txt"
    $inputDest = Join-Path $testInputDir "self_healing_synthetic_lab.txt"
    $syntheticLines = @(
        "Lab result report",
        "Specimen: serum",
        "Reference range listed below.",
        "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]",
        "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)",
        "WBC: 7.2 x10E9/L (ref 4.0-11.0)",
        "Cholesterol: 4.1 mmol/L",
        "",
        "Lab result report (Russian section)",
        "Material: serum",
        "Glucose translated: Glyukoza 5.4 mmol/L",
        "Hemoglobin translated: Gemoglobin 135 g/L",
        "WBC translated: Leikotsity 7.2 x10E9/L",
        "",
        "End of synthetic fixture."
    )
    Set-Content -LiteralPath $inputDest -Value ($syntheticLines -join "`n") -Encoding UTF8
    $inputSize = (Get-Item $inputDest).Length
}

Write-Step ("input mode: {0}  suffix: {1}" -f $inputMode, $inputSuffix)

# Compute the safe handle (07 prefix) via the Python aggregator helper.
$inputSafeHash = & python -c "from scripts.run_medai_local_self_healing_validation_07 import safe_input_hash; print(safe_input_hash($inputSize, '$inputSuffix'))"
$inputSafeHash = $inputSafeHash.Trim()
Write-Step ("input safe hash: {0}" -f $inputSafeHash)

# 8. One-doc validation (05).
Write-Step "running run_medai_local_one_doc_operator_validation_05.py"
& python (Join-Path $repoRoot "scripts\run_medai_local_one_doc_operator_validation_05.py")
$oneDocExit = $LASTEXITCODE

# 9. UI smoke (05).
Write-Step "running run_medai_streamlit_launch_smoke_05.py"
& python (Join-Path $repoRoot "scripts\run_medai_streamlit_launch_smoke_05.py")
$smokeExit = $LASTEXITCODE
$streamlitLaunchAttempted = "false"
if ($env:MEDAI_STREAMLIT_LAUNCH_SMOKE) {
    $streamlitLaunchAttempted = "true"
}

# 10. Python self-healing aggregator.
Write-Step "running run_medai_local_self_healing_validation_07.py"
& python (Join-Path $repoRoot "scripts\run_medai_local_self_healing_validation_07.py") `
    --branch $branch `
    --head $headShort `
    --dependency-prepare-result $dependencyResult `
    --input-mode $inputMode `
    --input-suffix $inputSuffix.TrimStart('.') `
    --input-safe-hash $inputSafeHash `
    --streamlit-launch-attempted $streamlitLaunchAttempted | Out-Null
$aggregatorExit = $LASTEXITCODE

# 11. Print the next operator command.
Write-Step "done"
Write-Host ""
Write-Host "Next step: open the UI in your browser via:" -ForegroundColor Cyan
Write-Host "    streamlit run app/main.py" -ForegroundColor Green
Write-Host ""

exit $aggregatorExit

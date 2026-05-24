# MEDAI-LOCAL-ONE-CLICK-OPERATOR-VALIDATION-06
# Windows PowerShell orchestrator. Replaces the previous five-step
# manual flow with one local command.
#
# What this script does, in order:
#
#   1. cd to the repo root (handles being launched from scripts/).
#   2. Verify git branch is clinical-knowledge-architecture.
#   3. Print HEAD short SHA.
#   4. Refuse to proceed if git status carries unrelated uncommitted
#      changes outside of ignored / runtime folders.
#   5. Call scripts/local_runtime_prepare_minimum_05.ps1 to install only
#      the minimal local runtime.
#   6. Ensure test_input/ exists.
#   7. Open a Windows file picker limited to *.pdf and *.txt.
#   8. If the operator selects a file: copy it into test_input/ under a
#      neutral local name (one_click_input.pdf or one_click_input.txt).
#      Never print the original filename or path.
#   9. If the operator cancels: write a synthetic non-PHI TXT fixture
#      to test_input/one_click_synthetic_lab.txt with simple English
#      and Russian lab-style rows. No patient identity.
#  10. Run python scripts/run_medai_local_one_doc_operator_validation_05.py.
#  11. Run python scripts/run_medai_streamlit_launch_smoke_05.py.
#  12. Aggregate the two outputs into the 06 public-safe report via
#      scripts/run_medai_local_one_click_operator_validation_06_report.py.
#  13. Print the next operator command.
#
# Hard rules:
#   * Never scan the operator's whole computer.
#   * Never auto-select from Downloads / Desktop / Documents.
#   * Never write raw filename, raw path, raw OCR text, raw document
#     text, or PHI to any public report.
#   * Never enable auto-accept. Never call an external API.
#   * Never stage or commit selected local files, runtime DBs, OCR
#     dumps, private logs, .env, screenshots, or corpus artifacts.

$ErrorActionPreference = "Stop"

function Write-Step([string]$message) {
    Write-Host ("[one-click] {0}" -f $message) -ForegroundColor Cyan
}

function Resolve-RepoRoot() {
    $scriptPath = $PSScriptRoot
    if (-not $scriptPath) {
        $scriptPath = (Split-Path -Parent $MyInvocation.MyCommand.Definition)
    }
    # If we are inside scripts/, the parent is the repo root.
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
    Write-Host ("[one-click] branch is '{0}', expected 'clinical-knowledge-architecture'. Aborting." -f $branch) -ForegroundColor Red
    exit 2
}

# 3. Print HEAD.
$headShort = (& git rev-parse --short HEAD).Trim()
Write-Step ("branch={0} head={1}" -f $branch, $headShort)

# 4. Refuse if working tree has unrelated uncommitted changes.
$gitStatus = (& git status --porcelain).Trim()
if ($gitStatus) {
    # Allowed: report files inside the 06 report directory and queued
    # test_input/ entries that this script itself writes. Everything
    # else means the operator has work in flight; do not run.
    $unexpected = @()
    foreach ($line in ($gitStatus -split "`n")) {
        $path = $line.Substring(3).Replace("\\", "/").Trim()
        if ($path.StartsWith("reports/medai_local_one_click_operator_validation_06/")) { continue }
        if ($path.StartsWith("test_input/")) { continue }
        $unexpected += $path
    }
    if ($unexpected.Count -gt 0) {
        Write-Host "[one-click] unrelated uncommitted files detected; aborting:" -ForegroundColor Red
        foreach ($p in $unexpected) {
            Write-Host ("  - {0}" -f $p) -ForegroundColor Red
        }
        exit 3
    }
}

# 5. Run the minimal local runtime preparer.
Write-Step "running local_runtime_prepare_minimum_05.ps1"
$prepareScript = Join-Path $repoRoot "scripts\local_runtime_prepare_minimum_05.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $prepareScript
$dependencyExit = $LASTEXITCODE

# 6. Ensure test_input/ exists.
$testInputDir = Join-Path $repoRoot "test_input"
if (-not (Test-Path $testInputDir)) {
    New-Item -ItemType Directory -Path $testInputDir | Out-Null
}

# 7. Clear ONLY the script-managed neutral inputs from prior runs.
foreach ($name in @("one_click_input.pdf", "one_click_input.txt", "one_click_synthetic_lab.txt")) {
    $target = Join-Path $testInputDir $name
    if (Test-Path $target) {
        Remove-Item $target -Force
    }
}

# 8. Open the operator-driven file picker, limited to PDF and TXT.
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Filter = "Documents (*.pdf;*.txt)|*.pdf;*.txt"
$dialog.Multiselect = $false
$dialog.Title = "MedAI one-click validation: select ONE PDF or TXT (cancel for synthetic)"
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
        Write-Host "[one-click] picked file has an unsupported suffix; falling back to synthetic" -ForegroundColor Yellow
        $inputMode = "synthetic_fallback"
        $inputSuffix = ".txt"
    } else {
        $inputMode = "selected_file"
        $neutralName = if ($inputSuffix -eq ".pdf") { "one_click_input.pdf" } else { "one_click_input.txt" }
        $inputDest = Join-Path $testInputDir $neutralName
        Copy-Item -LiteralPath $picked -Destination $inputDest -Force
        $inputSize = (Get-Item $inputDest).Length
    }
}

if ($inputMode -eq "synthetic_fallback") {
    # 9. Synthetic non-PHI fallback. No patient identifiers, no real
    # dates, no facility names. Simple English + Russian lab-style rows.
    $inputSuffix = ".txt"
    $inputDest = Join-Path $testInputDir "one_click_synthetic_lab.txt"
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

# Compute a safe handle for the input — never the original filename.
$inputSafeHash = & python -c "from scripts.run_medai_local_one_click_operator_validation_06_report import safe_input_hash; print(safe_input_hash($inputSize, '$inputSuffix'))"
$inputSafeHash = $inputSafeHash.Trim()
Write-Step ("input safe hash: {0}" -f $inputSafeHash)

# 10. Run the one-doc validation.
Write-Step "running run_medai_local_one_doc_operator_validation_05.py"
& python (Join-Path $repoRoot "scripts\run_medai_local_one_doc_operator_validation_05.py")
$oneDocExit = $LASTEXITCODE

# 11. Run the Streamlit launch smoke (helper layer always; brief launch only when MEDAI_STREAMLIT_LAUNCH_SMOKE is set).
Write-Step "running run_medai_streamlit_launch_smoke_05.py"
& python (Join-Path $repoRoot "scripts\run_medai_streamlit_launch_smoke_05.py")
$smokeExit = $LASTEXITCODE

# 12. Aggregate into the 06 public-safe report.
Write-Step "aggregating one-click report"
& python (Join-Path $repoRoot "scripts\run_medai_local_one_click_operator_validation_06_report.py") `
    --branch $branch `
    --head $headShort `
    --dependency-prepare-exit-code $dependencyExit `
    --selected-input-mode $inputMode `
    --input-suffix $inputSuffix.TrimStart('.') `
    --input-safe-hash $inputSafeHash `
    --one-doc-validation-exit-code $oneDocExit `
    --streamlit-smoke-exit-code $smokeExit | Out-Null
$aggregatorExit = $LASTEXITCODE

# 13. Print the next operator command.
Write-Step "done"
Write-Host ""
Write-Host "Next step: open the UI in your browser via:" -ForegroundColor Cyan
Write-Host "    streamlit run app/main.py" -ForegroundColor Green
Write-Host ""

exit $aggregatorExit

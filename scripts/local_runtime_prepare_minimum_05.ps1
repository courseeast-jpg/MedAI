# MEDAI-LOCAL-RUNTIME-DEPENDENCY-AND-ONE-DOC-VALIDATION-05
# Minimal local runtime preparer for one-doc operator validation.
#
# This script installs ONLY the modules needed to exercise the practical
# MedAI extraction-to-MKB loop on the operator's machine:
#
#   * streamlit       — Run & Review UI
#   * spacy           — local NER fast-path used by the extraction layer
#   * en_core_web_sm  — minimal spaCy English model
#   * chromadb        — local vector store (required by the MKB quality
#                       gate import path; remains review-bound)
#   * PyPDF2          — minimal PDF text-layer extraction
#   * pytesseract     — local OCR fallback when PDFs lack a text layer
#
# This script intentionally does NOT install:
#
#   * any cloud SDK
#   * V2 / broader optional dependencies
#   * sqlcipher3 (operator may add encryption separately)
#
# The script never modifies .env. The current Python environment is
# used as-is (no venv creation, no system Python override). Run it
# from an Administrator-or-user PowerShell prompt at the repo root.

$ErrorActionPreference = "Stop"

function Write-Step([string]$message) {
    Write-Host ("[medai-runtime-prep] {0}" -f $message) -ForegroundColor Cyan
}

function Write-Verify([string]$module, [string]$status) {
    $color = if ($status -eq "present") { "Green" } else { "Yellow" }
    Write-Host ("  {0,-22} {1}" -f $module, $status) -ForegroundColor $color
}

# 1. Pick the current Python interpreter, do not change it.
$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) {
    Write-Host "[medai-runtime-prep] python not found on PATH. Aborting." -ForegroundColor Red
    exit 1
}
$pythonPath = $python.Source
Write-Step ("using python: {0}" -f $pythonPath)

# 2. Refuse to touch .env.
if (Test-Path ".env") {
    Write-Step ".env detected; this script will NOT modify it."
}

# 3. Install the minimal package set. Each is best-effort and idempotent.
$packages = @(
    "streamlit",
    "spacy",
    "chromadb",
    "PyPDF2",
    "pytesseract"
)
foreach ($pkg in $packages) {
    Write-Step ("pip install {0}" -f $pkg)
    & $pythonPath -m pip install --upgrade $pkg
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("[medai-runtime-prep] pip install {0} failed with exit {1}" -f $pkg, $LASTEXITCODE) -ForegroundColor Red
    }
}

# 4. Download the minimal spaCy English model only if spaCy is now
#    present. The download is a no-op when the model is already cached.
$spacyCheck = & $pythonPath -c "import importlib.util as iu; print('yes' if iu.find_spec('spacy') else 'no')"
if ($spacyCheck -eq "yes") {
    Write-Step "downloading minimal spaCy model en_core_web_sm"
    & $pythonPath -m spacy download en_core_web_sm
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[medai-runtime-prep] spaCy model download failed" -ForegroundColor Yellow
    }
} else {
    Write-Host "[medai-runtime-prep] spaCy not present; skipping model download" -ForegroundColor Yellow
}

# 5. Verification table.
Write-Step "verification:"
$verifyTargets = @(
    "streamlit",
    "spacy",
    "chromadb",
    "PyPDF2",
    "pytesseract",
    "pydantic",
    "sqlite3"
)
foreach ($mod in $verifyTargets) {
    $status = & $pythonPath -c "import importlib.util as iu; print('present' if iu.find_spec('$mod') else 'missing')"
    Write-Verify $mod $status
}

# Optional: spaCy model presence (model is loaded by name, not import).
$modelStatus = & $pythonPath -c "import spacy, sys; sys.exit(0 if spacy.util.is_package('en_core_web_sm') else 1)" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Verify "en_core_web_sm" "present"
} else {
    Write-Verify "en_core_web_sm" "missing"
}

Write-Step "done"
Write-Host "[medai-runtime-prep] Next step: put one PDF or TXT into test_input/ and run:"
Write-Host "[medai-runtime-prep]   python scripts/run_medai_local_one_doc_operator_validation_05.py"

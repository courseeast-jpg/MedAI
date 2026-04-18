# MedAI v1.1 - Automated Tesseract Installation
# Per Automation Operating Doctrine: Zero manual steps

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "MedAI v1.1 - Automated Tesseract OCR Installation" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# Alternative download sources (primary blocked by 403)
$sources = @(
    @{
        name = "GitHub Releases (Direct)"
        url = "https://github.com/tesseract-ocr/tesseract/releases/download/5.3.3/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
    },
    @{
        name = "Chocolatey Repository Mirror"
        url = "https://community.chocolatey.org/api/v2/package/tesseract/5.3.3"
    }
)

$installerPath = "$PSScriptRoot\tesseract_installer.exe"
$downloadSuccess = $false

foreach ($source in $sources) {
    Write-Host "[*] Attempting download from: $($source.name)" -ForegroundColor Yellow
    
    try {
        # Use WebClient with updated TLS settings
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        $webClient.DownloadFile($source.url, $installerPath)
        
        if (Test-Path $installerPath) {
            $fileSize = (Get-Item $installerPath).Length / 1MB
            if ($fileSize -gt 5) {  # Valid installer should be > 5MB
                Write-Host "[+] Download successful ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
                $downloadSuccess = $true
                break
            } else {
                Write-Host "[-] Downloaded file too small, trying next source..." -ForegroundColor Red
                Remove-Item $installerPath -Force
            }
        }
    }
    catch {
        Write-Host "[-] Failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

if (-not $downloadSuccess) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Red
    Write-Host "AUTO-DOWNLOAD FAILED - FALLBACK TO CHOCOLATEY" -ForegroundColor Red
    Write-Host "==================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Installing via Chocolatey package manager (automated)..." -ForegroundColor Yellow
    Write-Host ""
    
    # Check if Chocolatey is installed
    $chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue
    
    if (-not $chocoInstalled) {
        Write-Host "[*] Installing Chocolatey package manager..." -ForegroundColor Yellow
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        
        # Refresh environment
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    }
    
    Write-Host "[*] Installing Tesseract via Chocolatey..." -ForegroundColor Yellow
    choco install tesseract -y
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[+] Tesseract installed successfully via Chocolatey" -ForegroundColor Green
        $downloadSuccess = $true
    }
}

if ($downloadSuccess -and (Test-Path $installerPath)) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host "RUNNING INSTALLER (Silent Mode)" -ForegroundColor Green
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host ""
    
    # Run silent installation
    $installArgs = "/S /DIR=`"C:\Program Files\Tesseract-OCR`""
    Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -NoNewWindow
    
    # Add to PATH if not already there
    $tesseractPath = "C:\Program Files\Tesseract-OCR"
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    
    if ($currentPath -notlike "*$tesseractPath*") {
        Write-Host "[*] Adding Tesseract to system PATH..." -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$tesseractPath", "Machine")
        $env:Path += ";$tesseractPath"
        Write-Host "[+] PATH updated" -ForegroundColor Green
    }
    
    # Cleanup
    Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
    
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host "INSTALLATION COMPLETE" -ForegroundColor Green
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host ""
}

# Verify installation
Write-Host "[*] Verifying Tesseract installation..." -ForegroundColor Yellow
$tesseractTest = tesseract --version 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "[+] Tesseract is working correctly" -ForegroundColor Green
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host "NEXT STEP - Run validation again:" -ForegroundColor Cyan
    Write-Host "  python validate_system.py" -ForegroundColor Yellow
    Write-Host "==================================================================" -ForegroundColor Cyan
} else {
    Write-Host "[-] Tesseract not found in PATH. Restart terminal and try again." -ForegroundColor Red
    Write-Host "[!] If problem persists, manually add C:\Program Files\Tesseract-OCR to PATH" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to continue"

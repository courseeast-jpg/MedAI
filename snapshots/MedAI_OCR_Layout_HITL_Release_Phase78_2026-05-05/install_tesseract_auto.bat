@echo off
REM MedAI v1.1 - Automated Tesseract Installation (CMD Native)
REM Per Automation Operating Doctrine: Zero manual steps, all versions compatible

setlocal enabledelayedexpansion

echo ==================================================================
echo MedAI v1.1 - Automated Tesseract OCR Installation
echo ==================================================================
echo.

set "INSTALLER_PATH=%~dp0tesseract_installer.exe"
set "INSTALL_DIR=C:\Program Files\Tesseract-OCR"

REM Check if already installed
tesseract --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [+] Tesseract is already installed
    tesseract --version
    goto :validation
)

echo [*] Installing Chocolatey package manager...
echo.

REM Install Chocolatey
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" 

if %errorlevel% neq 0 (
    echo [-] Chocolatey installation failed
    goto :error
)

REM Refresh PATH
call refreshenv.cmd 2>nul
set "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"

echo.
echo [*] Installing Tesseract via Chocolatey...
echo.

choco install tesseract -y --no-progress

if %errorlevel% neq 0 (
    echo [-] Tesseract installation failed
    goto :error
)

REM Refresh PATH again
set "PATH=%PATH%;%INSTALL_DIR%"

:validation
echo.
echo ==================================================================
echo Verifying installation...
echo ==================================================================
echo.

tesseract --version
if %errorlevel% equ 0 (
    echo.
    echo ==================================================================
    echo [+] TESSERACT INSTALLED SUCCESSFULLY
    echo ==================================================================
    echo.
    echo Next step - Run validation:
    echo   python validate_system.py
    echo.
    goto :end
) else (
    echo.
    echo ==================================================================
    echo [!] Installation completed but PATH not updated in current session
    echo ==================================================================
    echo.
    echo CLOSE THIS WINDOW and open a NEW Command Prompt, then run:
    echo   F:
    echo   cd "My Drive\MED AI\medai_v1.1_validated\medai"
    echo   python validate_system.py
    echo.
    goto :end
)

:error
echo.
echo ==================================================================
echo [!] AUTOMATED INSTALLATION FAILED
echo ==================================================================
echo.
echo Fallback: Manual installation required
echo 1. Visit: https://github.com/UB-Mannheim/tesseract/wiki
echo 2. Download: tesseract-ocr-w64-setup-5.3.3.20231005.exe
echo 3. Run installer with default settings
echo 4. Restart Command Prompt
echo 5. Run: python validate_system.py
echo.

:end
pause

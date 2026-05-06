@echo off
setlocal

REM CKA-SEC-05 — encrypted-runtime launcher for MedAI.
REM
REM This wrapper launches the Python launcher
REM (scripts\start_cka_encrypted_runtime_ui.py), which prompts for the
REM SQLCipher encryption key via getpass TWICE and never echoes,
REM stores, or logs it. The key is passed to the Streamlit child
REM process only and is never written to disk.
REM
REM This .bat file does NOT contain any encryption key.
REM
REM Encrypted runtime remains opt-in. The default launcher is still
REM Start_MedAI_UI.bat (unencrypted local-only mode); this wrapper does
REM not replace it.
REM
REM To create the empty encrypted future store the FIRST time, run:
REM   python scripts\start_cka_encrypted_runtime_ui.py ^
REM         --store-path data\secure\cka_encrypted_future_store.db ^
REM         --create-if-missing

cd /d "%~dp0"

echo Starting MedAI in encrypted runtime mode...
echo Encryption key will be prompted by Python getpass (twice).
echo The key is never displayed, never written to disk, never logged.
echo.

start "" powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:8501'"

python scripts\start_cka_encrypted_runtime_ui.py ^
    --store-path data\secure\cka_encrypted_future_store.db ^
    --port 8501

echo.
echo MedAI encrypted runtime stopped or failed to start.
echo To rollback to the standard unencrypted local-only launcher, run:
echo   Start_MedAI_UI.bat
pause
endlocal

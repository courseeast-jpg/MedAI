@echo off
REM MedAI v1.1 - Tesseract Auto-Installer Launcher
REM Executes PowerShell script with elevated permissions

echo Starting automated Tesseract installation...
echo.

PowerShell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_tesseract.ps1"

pause

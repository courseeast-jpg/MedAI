@echo off
REM MedAI v1.1 - Windows Validation Launcher
REM Handles drive switching and path navigation automatically

echo Switching to F: drive...
F:

echo Navigating to medai directory...
cd "My Drive\MED AI\medai_v1.1_validated\medai"

echo Running validation script...
python validate_system.py

pause

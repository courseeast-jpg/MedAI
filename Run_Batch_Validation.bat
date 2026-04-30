@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo MedAI Batch Validation
echo ========================================
echo Current folder:
cd
echo.

if not exist "scripts\run_batch_validation.py" (
    echo ERROR: scripts\run_batch_validation.py not found.
    echo You are probably running this BAT from the wrong project folder.
    echo.
    pause
    exit /b 1
)

echo Running batch validation...
echo.

python scripts\run_batch_validation.py

set EXITCODE=%ERRORLEVEL%
echo.
echo ========================================
echo Batch validation finished.
echo Exit code: %EXITCODE%
echo ========================================
echo.

pause
exit /b %EXITCODE%

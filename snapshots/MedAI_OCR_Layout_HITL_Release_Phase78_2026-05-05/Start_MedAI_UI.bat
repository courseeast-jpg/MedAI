@echo off
setlocal

cd /d "%~dp0"

echo Starting MedAI...
echo Local-only mode ON
echo External APIs disabled by privacy gate
echo Browser opening at localhost:8501

set MEDAI_LOCAL_ONLY=1
set MEDAI_ALLOW_EXTERNAL_API=0
set MEDAI_REQUIRE_PII_SCRUB=1
set MEDAI_PRIVACY_AUDIT=1

start "" powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 3; Start-Process 'http://localhost:8501'"

python -m streamlit run app\main.py --server.port 8501

echo.
echo MedAI stopped or failed to start. Review any messages above.
pause

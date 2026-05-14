@echo off
setlocal
cd /d "%~dp0"
echo Starting MedAI test UI...
echo Local-only mode ON
echo External APIs disabled by privacy gate
set MEDAI_LOCAL_ONLY=1
set MEDAI_ALLOW_EXTERNAL_API=0
set MEDAI_REQUIRE_PII_SCRUB=1
set MEDAI_PRIVACY_AUDIT=1
python -m streamlit run app/main.py

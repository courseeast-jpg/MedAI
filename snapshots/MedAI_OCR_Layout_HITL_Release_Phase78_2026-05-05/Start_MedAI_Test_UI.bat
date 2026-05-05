@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run app/main.py

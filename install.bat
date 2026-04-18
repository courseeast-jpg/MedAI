@echo off
:: MedAI v1.1 - Installation Script (Windows)
:: Installs all dependencies and initializes databases

echo ======================================================================
echo MedAI v1.1 - Installation
echo ======================================================================
echo.

echo [1/5] Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [ERROR] pip upgrade failed
    pause
    exit /b 1
)

echo.
echo [2/5] Installing Python dependencies (~5 minutes)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependencies installation failed
    pause
    exit /b 1
)

echo.
echo [3/5] Downloading spaCy language model (~2 minutes)...
python -m spacy download en_core_web_trf
if %errorlevel% neq 0 (
    echo [WARNING] spaCy model download failed, trying smaller model...
    python -m spacy download en_core_web_sm
)

echo.
echo [4/5] Initializing SQLite database...
python scripts\init_db.py
if %errorlevel% neq 0 (
    echo [ERROR] Database initialization failed
    pause
    exit /b 1
)

echo.
echo [5/5] Initializing ChromaDB vector store...
python scripts\init_chroma.py
if %errorlevel% neq 0 (
    echo [ERROR] ChromaDB initialization failed
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Installation Complete!
echo ======================================================================
echo.
echo Next step: Launch MedAI
echo   streamlit run app\main.py
echo.
pause

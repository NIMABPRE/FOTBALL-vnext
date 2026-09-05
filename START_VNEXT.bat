@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title FOOTBALL vNext

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Python environment not found. Creating .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create .venv.
        echo Make sure Python 3.11+ is installed and available as ^"python^".
        pause
        exit /b 1
    )
)

if not exist ".venv\.vnext_deps_ok" (
    echo [2/4] Installing/checking project dependencies...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Dependency installation failed.
        echo If this is a network/PyPI problem, fix the connection and run this file again.
        pause
        exit /b 1
    )
    type nul > ".venv\.vnext_deps_ok"
) else (
    echo [2/4] Dependencies already installed.
)

set "PYTHONPATH=%CD%\src"

echo [3/4] Checking configuration...
.venv\Scripts\python.exe scripts\healthcheck.py
if errorlevel 1 (
    echo.
    echo ERROR: Health check failed.
    pause
    exit /b 1
)

echo [4/4] Starting FOOTBALL vNext Dashboard...
echo.
echo Dashboard: http://localhost:8501

echo Opening browser...
start "" http://localhost:8501

.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501

endlocal

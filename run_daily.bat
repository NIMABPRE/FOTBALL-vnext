@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv not found. Create it with: python -m venv .venv
  pause
  exit /b 1
)
set PYTHONPATH=%CD%\src
.venv\Scripts\python.exe scripts\daily_job.py --league "Premier League" --timezone "Europe/Berlin"
pause

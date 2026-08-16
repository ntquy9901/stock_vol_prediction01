@echo off
cd /d "%~dp0"
echo ============================================================
echo   ONE-TIME SETUP  -  create .venv and install requirements
echo ============================================================
echo.
python -m venv .venv
if errorlevel 1 (
  echo Failed to create the virtual environment. Is Python 3.11+ installed and on PATH?
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Setup complete. You can now double-click START_HERE.bat
echo.
pause

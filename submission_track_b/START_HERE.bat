@echo off
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo ============================================================
echo   TRACK-B VOLATILITY MODEL - REPRODUCIBILITY BUNDLE
echo   Interactive menu. Start here.
echo   (If this is the first run, double-click setup.bat once.)
echo ============================================================
echo.
"%PY%" reproduce.py
echo.
pause

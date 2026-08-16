@echo off
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo ============================================================
echo   VIEW ALL RESULTS  (no training, no data needed)
echo   Prints the P0-^>G1 table and writes output\results_table.md
echo   plus output\summary.png
echo ============================================================
echo.
"%PY%" reproduce.py view
echo.
pause

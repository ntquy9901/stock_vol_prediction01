@echo off
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo ============================================================
echo   TRAIN  -  FINAL model G1 (P3 backbone + graph layer)
echo   Trains and saves checkpoints\g1_final.pt, then scores the
echo   held-out test split. Requires the project dataset under
echo   data\ (price CSVs + news panel parquet + provenance).
echo ============================================================
echo.
"%PY%" reproduce.py train
echo.
pause

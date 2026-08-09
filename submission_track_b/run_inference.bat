@echo off
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo ============================================================
echo   RUN INFERENCE  -  FINAL model G1 on the test split
echo   Requires the project dataset under data\ and a G1
echo   checkpoint (run train_model.bat once if none exists).
echo ============================================================
echo.
"%PY%" reproduce.py infer
echo.
pause

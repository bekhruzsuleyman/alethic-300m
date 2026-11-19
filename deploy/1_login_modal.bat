@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo Log in to the Modal account that holds the $30 credit and the 'huggingface' secret.
.venv\Scripts\python.exe -m modal setup
echo.
echo Done. Close this window.
pause

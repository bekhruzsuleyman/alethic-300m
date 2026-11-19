@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -m modal app list
echo.
pause

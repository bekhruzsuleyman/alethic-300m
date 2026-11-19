@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=src
rem Shows the classic screen with no model: logo animation, spinner, tool blocks, colored code, file viewer.
.venv\Scripts\python.exe -m alethic.agent.preview
pause

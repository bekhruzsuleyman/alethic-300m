@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo This starts the PAID phase-2 run (cost cap: $26.5 of Modal credit).
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run --detach deploy/modal_phase2.py::start
echo.
echo Started. You can close this window; the run continues on Modal.
pause

@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MODAL_PROFILE=second
echo This starts the phase-3 run in the second workspace (free credit, cost cap: $20).
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run --detach deploy/modal_phase3.py::start
echo.
echo Started. You can close this window; the run continues on Modal.
pause

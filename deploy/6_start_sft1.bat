@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo This starts SFT-1 on Modal (one H100, about 10 minutes, about $1 to $3 at most).
echo It reads the base model from HF and writes runs/sft1 back to HF.
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run --detach deploy/modal_sft1.py::start
echo.
echo Started. You can close this window; the run continues on Modal.
pause

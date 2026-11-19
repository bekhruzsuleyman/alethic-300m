@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MODAL_PROFILE=second
echo This re-reads the saved CODE and THINKING replies with the fixed reader. It is FREE: no DeepSeek call.
echo Results go to the folder sft_code_full_rechecked and sft_traces_pilot_rechecked/ of your Hugging Face repo.
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run deploy/modal_expand.py::redo_b_d
if errorlevel 1 (
  echo.
  echo FAILED. It did NOT start. Copy the error above and send it to Claude.
  pause
  exit /b 1
)
echo.
echo Started. You can close this window; it keeps running on Modal.
pause

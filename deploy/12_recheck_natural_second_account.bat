@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MODAL_PROFILE=second
echo This re-checks the saved natural-run replies with the looser rules. It is FREE: no DeepSeek call.
echo Results go to the folder sft_expand_natural_rechecked/ of your Hugging Face repo.
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run deploy/modal_expand.py::redo_natural
if errorlevel 1 (
  echo.
  echo FAILED. It did NOT start. Copy the error above and send it to Claude.
  pause
  exit /b 1
)
echo.
echo Started. You can close this window; it keeps running on Modal.
pause

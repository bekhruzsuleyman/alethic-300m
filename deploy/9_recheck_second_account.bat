@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MODAL_PROFILE=second
echo This re-checks the saved DeepSeek replies with the fixed checker on your second's Modal account.
echo It is FREE for DeepSeek (no API call) and takes about 1 minute.
echo Results are pushed to the folder sft_expand_rechecked/ of your Hugging Face repo.
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run deploy/modal_expand.py::redo
if errorlevel 1 (
  echo.
  echo FAILED. Copy the error above and send it to Claude.
  pause
  exit /b 1
)
echo.
echo Done. Tell Claude.
pause

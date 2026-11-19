@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo This starts the overnight dataset expansion on Modal (CPU only, no GPU).
echo It calls the DeepSeek API with your key and spends up to 6 US dollars of your DeepSeek balance.
echo Passing examples are saved to the Modal volume and pushed to the folder sft_expand/ of your Hugging Face repo.
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run --detach deploy/modal_expand.py::start
if errorlevel 1 (
  echo.
  echo FAILED. It did NOT start. Copy the error above and send it to Claude.
  pause
  exit /b 1
)
echo.
echo Started. You can close this window; it keeps running on Modal.
pause

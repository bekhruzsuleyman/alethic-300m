@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MODAL_PROFILE=second
echo This tests THINKING TRACES: DeepSeek Pro solves 900 generated math problems step by step (the full D set) on your second's Modal account.
echo It calls the DeepSeek API and spends up to 8 US dollars (expected about 3 to 5 dollars on Pro) from the card.
echo Results go to the folder sft_traces_full/ of your Hugging Face repo.
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run --detach deploy/modal_expand.py::start_traces_full
if errorlevel 1 (
  echo.
  echo FAILED. It did NOT start. Copy the error above and send it to Claude.
  pause
  exit /b 1
)
echo.
echo Started. You can close this window; it keeps running on Modal.
pause

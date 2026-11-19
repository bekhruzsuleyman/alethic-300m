@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MODAL_PROFILE=second
echo This mixes the checked data into one 5,000-example set and trains SFT-2 on your second's Modal account.
echo Order: tests (cents) - mix the set - one H100 for about 10 to 20 minutes (cap about 3 dollars).
echo It uses the data already on Modal and Hugging Face. It does not call DeepSeek.
echo The result goes to runs/sft2_5k and sft2_5k_data in your Hugging Face repo.
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run --detach deploy/modal_sft2.py::start
if errorlevel 1 (
  echo.
  echo FAILED. It did NOT start. Copy the error above and send it to Claude.
  pause
  exit /b 1
)
echo.
echo Started. You can close this window; it keeps running on Modal.
pause

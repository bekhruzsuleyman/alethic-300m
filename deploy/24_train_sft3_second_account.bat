@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MODAL_PROFILE=second
echo SFT-3 on your second Modal account.
echo   1. uses the chat file that is already on the Modal Volume
echo   2. tests (cents), then mixes the stage B data
echo   3. stage A: 100,000 chat examples, one H100, about 70 to 130 minutes
echo   4. stage B: GOLD x3 + error traces + identity, about 20 to 30 minutes
echo Cost: about 8 to 15 dollars. The account had 10.75 dollars. If the credits run out the run STOPS and stage A is lost, so add credits first if you can.
echo Results go to runs/sft3a and runs/sft3b in your Hugging Face repo.
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run --detach deploy/modal_sft3.py::start
if errorlevel 1 goto fail
echo.
echo Started. You can close this window and go; it keeps running on Modal.
pause
exit /b 0
:fail
echo.
echo FAILED. Training did NOT start. Copy the error above and send it to Claude.
pause
exit /b 1

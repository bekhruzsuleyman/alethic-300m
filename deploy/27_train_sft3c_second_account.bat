@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MODAL_PROFILE=second
echo Stage B again on your second Modal account, from the finished stage A (runs/sft3a).
echo   1. tests (cents), then mixes the data: GOLD x2, 1,720 program traces, 1,168 file/folder command traces, 2,460 math-through-python traces, 2,350 edit/map/search/command traces, 594 long Gemini sessions (x2), error traces, identity, chat replay
echo   2. one H100, 2 epochs, keeps the best one, about 20 to 30 minutes
echo Cost: about 2 dollars. Check that the account has at least 3 dollars first.
echo Result goes to runs/sft3c in your Hugging Face repo. runs/sft3b stays as it is.
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run --detach deploy/modal_sft3.py::start_b
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

@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MODAL_PROFILE=second
echo ONE FULL RUN on your second's Modal account. It does, in this order:
echo   1. starts the 900 thinking traces on DeepSeek Pro (about 3 to 5 dollars, cap 8)
echo   2. starts 1,800 thinking examples (word problems, logic, units) on Pro (about 3 to 5 dollars, cap 8)
echo   3. starts 800 real projects with tool use (write, compile, run, fix) on Pro (about 4 to 6 dollars, cap 8)
echo   4. re-reads the saved code and thinking replies with the fixed reader (free)
echo New data types will be added to this same file as they are built.
echo Press any key to start, or close this window to cancel.
pause >nul
.venv\Scripts\python.exe -m modal run --detach deploy/modal_expand.py::start_traces_full
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m modal run --detach deploy/modal_expand.py::start_think_full
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m modal run --detach deploy/modal_expand.py::start_projects_full
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m modal run deploy/modal_expand.py::redo_b_d
if errorlevel 1 goto failed
echo.
echo All started. The thinking and project runs keep running on Modal; you can close this window.
pause
exit /b 0
:failed
echo.
echo FAILED. Copy the error above and send it to Claude.
pause
exit /b 1

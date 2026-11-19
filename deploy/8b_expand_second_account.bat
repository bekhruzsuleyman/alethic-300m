@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo This runs the overnight dataset expansion on your second's Modal account (profile "second").
echo Your own Modal account is not touched. It spends up to 6 US dollars of the DeepSeek balance.
echo.
findstr /c:"[second]" "%USERPROFILE%\.modal.toml" >nul 2>nul
if errorlevel 1 (
  echo First time: a browser will open. Log in to Modal as your second, then come back here.
  pause
  .venv\Scripts\python.exe -m modal token new --profile=second
  findstr /c:"[second]" "%USERPROFILE%\.modal.toml" >nul 2>nul
  if errorlevel 1 (
    echo Login did not finish. Close this window and double-click this file again.
    pause
    exit /b 1
  )
)
set MODAL_PROFILE=second
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

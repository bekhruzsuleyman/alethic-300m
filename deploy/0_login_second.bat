@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
rem Creates a SEPARATE Modal profile named "second" (not activated), so the main profile is never touched.
set MODAL_PROFILE=second
echo Creating the "second" Modal profile. A link will be printed below; it must be authorized in a browser
echo that is signed in to the second workspace. Leave this window open until it says the token is set.
.venv\Scripts\python.exe -m modal token new --no-activate
echo.
.venv\Scripts\python.exe -m modal profile list
pause

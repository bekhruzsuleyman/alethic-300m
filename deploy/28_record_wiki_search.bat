@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONPATH=.;src
echo Asking the real Wikipedia search API about 230 questions (about 2 minutes). Nothing is uploaded: it only reads.
.venv\Scripts\python.exe -m tools.skilltraces.record_wiki
if errorlevel 1 goto fail
echo.
echo Done. Tell Claude: wiki done.
pause
exit /b 0
:fail
echo Something failed. Run this file again, it continues where it stopped.
pause
exit /b 1

@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONPATH=src;.
where git >nul 2>nul
if errorlevel 1 goto nogit
echo Building the clean agent copy in ..\alethic-agent-public ...
.venv\Scripts\python.exe -m tools.release.export_agent . ..\alethic-agent-public
if errorlevel 1 goto scanfail
echo.
echo This publishes the folder alethic-agent-public to
echo   https://github.com/bekhruzsuleyman/alethic-agent
echo as ONE clean commit (it replaces anything already in that repo).
choice /c YN /m "Publish now? Y = publish, N = cancel"
if errorlevel 2 goto cancel
cd /d "%~dp0..\..\alethic-agent-public"
if errorlevel 1 goto nofolder
if not exist .alethic_export_ok goto nofolder
if exist .git rmdir /s /q .git
git init -b main
git config user.name "Bekhruz Suleyman"
git config user.email "bekhruzsuleyman@users.noreply.github.com"
git add -A
git commit -m "Alethic Agent: terminal harness for Alethic-300M"
git remote add origin https://github.com/bekhruzsuleyman/alethic-agent.git
git push --force -u origin main
if errorlevel 1 goto pushfail
echo.
echo Published. Open https://github.com/bekhruzsuleyman/alethic-agent to look.
pause
exit /b 0
:nofolder
echo Stopped: the folder alethic-agent-public is missing or was not made by the export. Nothing was touched.
pause
exit /b 1
:nogit
echo Git is not installed or not on PATH.
pause
exit /b 1
:scanfail
echo.
echo The copy failed or the scan found private text. Nothing was published. Send this window to Claude.
pause
exit /b 1
:pushfail
echo.
echo The push failed. If a login window opened, sign in to GitHub and double-click this file again.
pause
exit /b 1
:cancel
echo Cancelled. Nothing was published.
pause
exit /b 0

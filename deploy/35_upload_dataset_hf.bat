@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONPATH=src;.
.venv\Scripts\python.exe -m pip install -q huggingface_hub
echo Re-packing the shards so they carry the newest ids ...
.venv\Scripts\python.exe -m tools.release.pack_newborn docs\sft\record\built dataset
if errorlevel 1 goto bad
echo.
echo This uploads the shards and the README to
echo   https://huggingface.co/datasets/bekhruzsuleyman/alethic-newborn
echo and turns ON the access gate (people must accept the terms).
choice /c YN /m "Upload now? Y = upload, N = cancel"
if errorlevel 2 goto cancel
.venv\Scripts\python.exe -m tools.release.upload_dataset dataset
if errorlevel 1 goto fail
echo.
echo Done. Open the link above and look at the dataset page.
pause
exit /b 0
:bad
echo Packing failed. Send this window to Claude.
pause
exit /b 1
:fail
echo Upload failed. If it says you are not logged in, double-click 36_hf_login.bat first, then this file again.
pause
exit /b 1
:cancel
echo Cancelled. Nothing was uploaded.
pause
exit /b 0

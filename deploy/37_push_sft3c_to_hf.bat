@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONPATH=src;.
.venv\Scripts\python.exe -m pip install -q huggingface_hub
echo This copies the SFT-3c weights (runs/sft3c/best) inside
echo   https://huggingface.co/bekhruzsuleyman/alethic-300m
echo into the new folder sft/sft3c. The base model files are NOT touched.
echo The copy happens on the Hugging Face server, so it is fast.
echo Then it updates the model card README (the old one is saved first).
choice /c YN /m "Copy now? Y = copy, N = cancel"
if errorlevel 2 goto cancel
.venv\Scripts\python.exe -m tools.release.push_weights
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m tools.release.update_card
if errorlevel 1 goto fail
pause
exit /b 0
:fail
echo Failed. If it says you are not logged in, double-click 36_hf_login.bat first, then this file again.
pause
exit /b 1
:cancel
echo Cancelled. Nothing was changed.
pause
exit /b 0

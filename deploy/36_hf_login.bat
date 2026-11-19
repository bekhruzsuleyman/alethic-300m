@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONPATH=src;.
echo This logs you in to Hugging Face. Paste your own token when asked; Claude never sees it.
.venv\Scripts\python.exe -m pip install -q huggingface_hub
.venv\Scripts\python.exe -c "from huggingface_hub import login; login()"
pause

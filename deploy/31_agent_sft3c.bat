@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=src
if exist runs_sft3c\model.pt goto run
echo Downloading SFT-3c weights from Hugging Face, 1.2 GB, once...
.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; import shutil, os; p=d('bekhruzsuleyman/alethic-300m','runs/sft3c/best/model.pt'); os.makedirs('runs_sft3c',exist_ok=True); shutil.copyfile(p,'runs_sft3c/model.pt')"
if errorlevel 1 goto fail
:run
.venv\Scripts\python.exe -m alethic.agent --weights runs_sft3c\model.pt --tokenizer data\tokenizer\alethic-tokenizer-v1\alethic-tokenizer-v1.model --label SFT-3c
pause
exit /b 0
:fail
echo Could not download the weights. Check the internet and double-click this file again.
pause
exit /b 1

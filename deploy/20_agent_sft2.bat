@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=src
if not exist runs_sft2\model.pt (
  echo Downloading SFT-2 weights from Hugging Face, 1.2 GB, once...
  .venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; import shutil, os; p=d('bekhruzsuleyman/alethic-300m','runs/sft2_5k/best/model.pt'); os.makedirs('runs_sft2',exist_ok=True); shutil.copyfile(p,'runs_sft2/model.pt')"
)
.venv\Scripts\python.exe -m alethic.agent --weights runs_sft2\model.pt --tokenizer data\tokenizer\alethic-tokenizer-v1\alethic-tokenizer-v1.model --folder work --label SFT-2
pause

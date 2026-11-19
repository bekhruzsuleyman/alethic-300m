@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=src
if not exist runs_sft1\model.pt (
  echo Downloading SFT-1 weights from Hugging Face, 1.2 GB, once...
  .venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; import shutil; p=d('bekhruzsuleyman/alethic-300m','runs/sft1/best/model.pt'); import os; os.makedirs('runs_sft1',exist_ok=True); shutil.copyfile(p,'runs_sft1/model.pt')"
)
.venv\Scripts\python.exe -m alethic.sft.chat --weights runs_sft1\model.pt --tokenizer data\tokenizer\alethic-tokenizer-v1\alethic-tokenizer-v1.model
pause

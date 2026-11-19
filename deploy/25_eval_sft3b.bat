@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=.;src
set TOK=data\tokenizer\alethic-tokenizer-v1\alethic-tokenizer-v1.model
if exist runs_sft3b\model.pt goto have
echo Downloading SFT-3b weights from Hugging Face, 1.2 GB, once...
.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; import shutil, os; p=d('bekhruzsuleyman/alethic-300m','runs/sft3b/best/model.pt'); os.makedirs('runs_sft3b',exist_ok=True); shutil.copyfile(p,'runs_sft3b/model.pt')"
if errorlevel 1 goto fail
:have
set MODELS=--model sft3b=runs_sft3b\model.pt
if exist runs_sft2\model.pt set MODELS=--model sft3b=runs_sft3b\model.pt --model sft2=runs_sft2\model.pt
echo.
echo Running the 50 eval prompts, one model after the other. About 10 to 30 minutes per model.
echo You can close this window any time: the next run continues where this one stopped.
echo.
.venv\Scripts\python.exe -m tools.evalset.run %MODELS% --tokenizer %TOK%
if errorlevel 1 goto fail
echo.
echo Done. Paste the table above into the chat, and send reports\eval_answers.txt
pause
exit /b 0
:fail
echo Something failed. Take a screenshot of this window and send it.
pause
exit /b 1

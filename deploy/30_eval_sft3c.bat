@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=.;src
set TOK=data\tokenizer\alethic-tokenizer-v1\alethic-tokenizer-v1.model
if exist runs_sft3c\model.pt goto have
echo Downloading SFT-3c weights from Hugging Face, 1.2 GB, once...
.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; import shutil, os; p=d('bekhruzsuleyman/alethic-300m','runs/sft3c/best/model.pt'); os.makedirs('runs_sft3c',exist_ok=True); shutil.copyfile(p,'runs_sft3c/model.pt')"
if errorlevel 1 goto fail
:have
rem Order matters: the first lines are the ones that tell most. Close the window any time, the next run continues where it stopped.
set MODELS=--model sft3c@0.3=runs_sft3c\model.pt --model sft3c@0=runs_sft3c\model.pt
if exist runs_sft3b\model.pt set MODELS=%MODELS% --model sft3b@0=runs_sft3b\model.pt
set MODELS=%MODELS% --model sft3c@0.2=runs_sft3c\model.pt --model sft3c@0.5=runs_sft3c\model.pt
echo.
echo Running the 50 eval prompts for SFT-3c at temperature 0.3, 0, then 0.2 and 0.5 (and SFT-3b at 0 for comparison).
echo About 10 to 30 minutes for each line. The table is printed at the end.
echo.
.venv\Scripts\python.exe -m tools.evalset.run %MODELS% --tokenizer %TOK%
if errorlevel 1 goto fail
echo.
echo Done. Paste the table above into the chat, and send reports\eval_answers.txt
pause
exit /b 0
:fail
echo Something failed. Run this file again, it continues where it stopped.
pause
exit /b 1

@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONPATH=.;src
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo Building the stage A chat data from Smol-SmolTalk. Downloads about 1 GB, takes 10 to 30 minutes. You can leave it running.
.venv\Scripts\python.exe -m pip install --quiet huggingface_hub pyarrow sentencepiece
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m tools.chatconv.convert 100000
if errorlevel 1 goto fail
echo.
echo Done. File: reports\stage_a_chat.jsonl
pause
exit /b 0
:fail
echo Something failed. Take a screenshot of this window and send it.
pause
exit /b 1

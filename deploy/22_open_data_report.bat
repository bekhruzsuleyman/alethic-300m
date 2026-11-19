@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONPATH=.;src
echo Checking open datasets (license, size, how many fit 4096 tokens). Needs internet, about 5 to 15 minutes.
.venv\Scripts\python.exe -m pip install --quiet huggingface_hub pyarrow sentencepiece
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m tools.opendata.report
if errorlevel 1 goto fail
echo.
echo Done. Paste the text above into the chat, or send reports\open_data_report.txt
pause
exit /b 0
:fail
echo Something failed. Take a screenshot of this window and send it.
pause
exit /b 1

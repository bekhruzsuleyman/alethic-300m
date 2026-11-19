@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONPATH=.;src
echo Checking the search results against the real Wikipedia, about 1 minute...
.venv\Scripts\python.exe -m tools.skilltraces.replay docs\sft\record\built\gemini_sessions.jsonl --search-cache docs\sft\record\wiki_search_cache.json --search-only
echo.
echo Tell Claude the line that starts with "checked".
pause

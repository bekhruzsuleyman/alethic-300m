@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONPATH=src
echo Building alethic-agent.exe. This takes a few minutes and needs about 2 GB of disk.
.venv\Scripts\python.exe -m pip install --quiet pyinstaller
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --name alethic-agent --paths src ^
  --add-data "data\tokenizer\alethic-tokenizer-v1;data\tokenizer\alethic-tokenizer-v1" ^
  --add-data "configs\model_gqa.yaml;configs" ^
  --collect-submodules alethic --exclude-module pytest --exclude-module matplotlib ^
  deploy\agent_entry.py
if errorlevel 1 goto fail
rem PyInstaller drops a DLL that two packages share by name (sentencepiece needs its own copy), so copy every *.libs folder in
for /d %%D in (.venv\Lib\site-packages\*.libs) do (
  if not exist "dist\alethic-agent\_internal\%%~nxD" mkdir "dist\alethic-agent\_internal\%%~nxD"
  xcopy /y /q "%%D\*" "dist\alethic-agent\_internal\%%~nxD\" >nul
)
rem Other computers may not have the Microsoft Visual C++ runtime, so ship the DLLs next to torch (Microsoft allows this)
for %%F in (vcruntime140.dll vcruntime140_1.dll vcruntime140_threads.dll msvcp140.dll msvcp140_1.dll msvcp140_2.dll concrt140.dll) do (
  if exist "%SystemRoot%\System32\%%F" (
    copy /y "%SystemRoot%\System32\%%F" "dist\alethic-agent\_internal\" >nul
    copy /y "%SystemRoot%\System32\%%F" "dist\alethic-agent\_internal\torch\lib\" >nul 2>nul
  )
)
echo Checking that every library loads...
dist\alethic-agent\alethic-agent.exe --selftest
if errorlevel 1 goto fail
echo Making alethic-agent-windows.zip to carry to other computers (no weights inside, they download on first run)...
if exist alethic-agent-windows.zip del alethic-agent-windows.zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\alethic-agent' -DestinationPath 'alethic-agent-windows.zip' -CompressionLevel Fastest"
echo Building the installer dist\alethic-agent-setup.exe (the zip is embedded inside it)...
set CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC%" set CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe
"%CSC%" /nologo /target:winexe /optimize+ /out:dist\alethic-agent-setup.exe /resource:alethic-agent-windows.zip,bundle.zip /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.IO.Compression.dll /reference:System.IO.Compression.FileSystem.dll deploy\setup_installer.cs
if errorlevel 1 goto fail
if exist runs_sft3b\model.pt (
  echo Copying the SFT-3b weights next to the exe, so it does not download them again...
  copy /y runs_sft3b\model.pt dist\alethic-agent\model-sft3b.pt >nul
)
echo.
echo Done. Double-click:  dist\alethic-agent\alethic-agent.exe
echo For another computer: send dist\alethic-agent-setup.exe and run it (Windows 10 or 11).
pause
exit /b 0
:fail
echo.
echo The build failed. Send me the text above.
pause
exit /b 1

@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
set "RAIDERS_JUGGLING_PYTHONW=pythonw"
if exist "F:\Anaconda3\pythonw.exe" set "RAIDERS_JUGGLING_PYTHONW=F:\Anaconda3\pythonw.exe"
start "" "%RAIDERS_JUGGLING_PYTHONW%" -X utf8 tools\juggling_hot_reload.py %*
set "RAIDERS_JUGGLING_EXIT=%errorlevel%"
popd
exit /b %RAIDERS_JUGGLING_EXIT%

@echo off
setlocal
chcp 65001 >nul
title Raiders Task Terminal Launcher
pushd "%~dp0.."
if errorlevel 1 goto directory_error

set "PYTHONUTF8=1"
set "RAIDERS_LAUNCH_PYTHON=python"
if exist "F:\Anaconda3\python.exe" set "RAIDERS_LAUNCH_PYTHON=F:\Anaconda3\python.exe"
if defined RAIDERS_PYTHON set "RAIDERS_LAUNCH_PYTHON=%RAIDERS_PYTHON%"

set "RAIDERS_LAUNCH_PYTHONW=pythonw"
if exist "F:\Anaconda3\pythonw.exe" set "RAIDERS_LAUNCH_PYTHONW=F:\Anaconda3\pythonw.exe"
if defined RAIDERS_PYTHON for %%I in ("%RAIDERS_PYTHON%") do if exist "%%~dpIpythonw.exe" set "RAIDERS_LAUNCH_PYTHONW=%%~dpIpythonw.exe"

if "%~1"=="--help" goto show_help
start "" "%RAIDERS_LAUNCH_PYTHONW%" -X utf8 run.py task-terminal %*
set "RAIDERS_LAUNCH_EXIT=%errorlevel%"
popd
if not "%RAIDERS_LAUNCH_EXIT%"=="0" goto launch_error
exit /b %RAIDERS_LAUNCH_EXIT%

:show_help
"%RAIDERS_LAUNCH_PYTHON%" -X utf8 run.py task-terminal --help
set "RAIDERS_LAUNCH_EXIT=%errorlevel%"
popd
exit /b %RAIDERS_LAUNCH_EXIT%

:launch_error
echo [ERROR] Cannot open the graphical task terminal.
if not defined RAIDERS_NO_PAUSE pause
exit /b %RAIDERS_LAUNCH_EXIT%

:directory_error
echo [ERROR] Cannot open the project directory.
if not defined RAIDERS_NO_PAUSE pause
exit /b 1

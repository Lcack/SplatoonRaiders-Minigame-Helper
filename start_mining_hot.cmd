@echo off
setlocal
cd /d "%~dp0"
if exist "F:\Anaconda3\pythonw.exe" (
    start "" "F:\Anaconda3\pythonw.exe" "%~dp0tools\start_mining_hot.py"
    exit /b 0
)
where pythonw.exe >nul 2>nul
if not errorlevel 1 (
    start "" pythonw.exe "%~dp0tools\start_mining_hot.py"
    exit /b 0
)
echo Pythonw was not found. Install Python or use the Anaconda base environment.
pause
exit /b 1

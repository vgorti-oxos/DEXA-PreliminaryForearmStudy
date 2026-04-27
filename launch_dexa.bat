@echo off
setlocal
cd /d "%~dp0"

set "PYLAUNCH=py"
where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher "py" was not found. Please install Python 3.11 or newer.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local Python environment...
    %PYLAUNCH% -3 -m venv .venv
    if errorlevel 1 (
        echo Failed to create the Python environment.
        pause
        exit /b 1
    )
)

echo Installing/updating required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install required packages.
    pause
    exit /b 1
)

echo Launching DEXA analysis...
".venv\Scripts\python.exe" run_dexa.py
echo.
echo DEXA analysis finished.
pause

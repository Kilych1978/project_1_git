@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv .venv
    if errorlevel 1 exit /b 1
    goto install
)
where python >nul 2>nul
if errorlevel 1 (
    echo Install Python 3.12 from python.org, then run setup again.
    pause
    exit /b 1
)
python -m venv .venv
if errorlevel 1 exit /b 1
:install
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo Python 3.11 or newer is required. Install Python 3.12 and recreate the local .venv.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. Check your connection to PyPI.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" start_sync.py --check
set "setup_exit=%errorlevel%"
if not "%setup_exit%"=="0" echo Python packages installed. If FFmpeg is missing, follow README_RU.md.
pause
exit /b %setup_exit%

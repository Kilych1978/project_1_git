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
    echo Установите Python 3.12 с python.org, затем повторите запуск.
    pause
    exit /b 1
)
python -m venv .venv
if errorlevel 1 exit /b 1
:install
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo Требуется Python 3.11 или новее. Установите Python 3.12 и пересоздайте локальную .venv.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Не удалось установить зависимости. Проверьте доступ к PyPI.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" start_sync.py --check
set "setup_exit=%errorlevel%"
if not "%setup_exit%"=="0" echo Python-пакеты установлены. Если не найден FFmpeg, выполните инструкции в README_RU.md.
pause
exit /b %setup_exit%

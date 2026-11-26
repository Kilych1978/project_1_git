@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" start_sync.py %*
    goto finished
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 start_sync.py %*
    goto finished
)
where python >nul 2>nul
if not errorlevel 1 (
    python start_sync.py %*
    goto finished
)
echo Python не найден. Установите Python 3.12 с python.org и запустите SETUP_WINDOWS.cmd.
exit /b 1
:finished
set "sync_exit=%errorlevel%"
if not "%sync_exit%"=="0" echo Обработка завершилась с кодом %sync_exit%. Прочитайте сообщение и REPORT_RU.txt, если отчёт создан.
if "%~1"=="" pause
exit /b %sync_exit%

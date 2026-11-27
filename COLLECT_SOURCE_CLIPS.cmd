@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" source_clips.py %*
    goto finished
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 source_clips.py %*
    goto finished
)
where python >nul 2>nul
if not errorlevel 1 (
    python source_clips.py %*
    goto finished
)
echo Python не найден. Установите Python 3.12 с python.org.
pause
exit /b 2
:finished
set "clips_exit=%errorlevel%"
if not "%clips_exit%"=="0" echo Сбор фрагментов остановлен. Прочитайте сообщение и ERROR.txt в диагностической папке, если она создана.
pause
exit /b %clips_exit%

@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
call START_WINDOWS.cmd --micro-only
set "micro_exit=%errorlevel%"
pause
exit /b %micro_exit%

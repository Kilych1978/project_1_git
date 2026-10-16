@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
call START_WINDOWS.cmd --resume %*
set "resume_exit=%errorlevel%"
pause
exit /b %resume_exit%

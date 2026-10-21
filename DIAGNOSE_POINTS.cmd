@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
call START_WINDOWS.cmd --diagnose-points %*
set "points_exit=%errorlevel%"
pause
exit /b %points_exit%

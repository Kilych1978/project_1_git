@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0ASSEMBLE_VIDEO.ps1"
set "KATALA_ASSEMBLY_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %KATALA_ASSEMBLY_EXIT%

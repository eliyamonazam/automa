@echo off
rem Automa web launcher: double-click this file, or make a desktop shortcut to it.
rem It runs automa-web.ps1 next to it (arguments are passed on, e.g. -Port 8766).
rem -ExecutionPolicy Bypass applies only to this one PowerShell process.
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0automa-web.ps1" %*
set "AUTOMA_EXIT=%ERRORLEVEL%"
if not "%AUTOMA_EXIT%"=="0" pause
exit /b %AUTOMA_EXIT%

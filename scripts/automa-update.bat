@echo off
rem Automa updater: double-click this file, or make a desktop shortcut to it.
rem It runs automa-update.ps1 next to it (arguments are passed on, e.g. -RepoDir automa2).
rem -ExecutionPolicy Bypass applies only to this one PowerShell process.
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0automa-update.ps1" %*
exit /b %ERRORLEVEL%

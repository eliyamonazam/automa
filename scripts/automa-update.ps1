# Automa updater for Windows: double-click automa-update.bat (runs inside WSL, D170).
#
# What it does:
#   1. runs `automa update` inside WSL, from ~/automa with that folder's .venv:
#      git fast-forward to the latest main (never force/reset), and pip install -e .
#      only when pyproject.toml changed;
#   2. shows the result in colour (green = done, yellow = nothing changed, see why);
#   3. asks "Open Automa now?" and, on yes, starts automa-web.ps1 next to this file.
#
# This file is ASCII only on purpose: Windows PowerShell 5.1 reads files without a
# BOM in the legacy code page. Usage: see the "update" part of README.md (Windows).

[CmdletBinding()]
param(
    # Folder of the Automa clone, relative to the WSL home directory.
    [ValidatePattern('^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$')]
    [string]$RepoDir = 'automa',

    # WSL distribution name (empty = an Ubuntu distro, else the first usable one).
    [ValidatePattern('^[A-Za-z0-9._-]*$')]
    [string]$Distro = '',

    # Do not ask to open Automa afterwards (used by automa-web.ps1 -Update).
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'

if ($RepoDir -match '(^|/)\.\.?(/|$)') {
    Write-Host "RepoDir must not contain '.' or '..' parts: $RepoDir" -ForegroundColor Red
    exit 2
}

$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wsl) {
    Write-Host 'WSL was not found. Install it first (README.md, section Windows): wsl --install' -ForegroundColor Red
    exit 2
}

# Same distro choice as automa-web.ps1: prefer Ubuntu, never Docker Desktop's distro.
if (-not $Distro) {
    $prevEncoding = [Console]::OutputEncoding
    try {
        [Console]::OutputEncoding = [System.Text.Encoding]::Unicode  # wsl -l prints UTF-16
        $listed = & $wsl.Source -l -q 2>$null
    } catch {
        $listed = @()
    } finally {
        [Console]::OutputEncoding = $prevEncoding
    }
    $names = @($listed | ForEach-Object { ("$_" -replace "`0", '').Trim() } | Where-Object { $_ })
    $pick = $names | Where-Object { $_ -match '^Ubuntu' } | Select-Object -First 1
    if (-not $pick) { $pick = $names | Where-Object { $_ -notmatch '^docker-desktop' } | Select-Object -First 1 }
    if ($pick -and $pick -match '^[A-Za-z0-9._-]+$') { $Distro = $pick }
}

# Fixed command for bash inside WSL. No quotes are needed: RepoDir passed the
# pattern above. Exit code 3 = missing clone or venv.
$inner = "cd ~/$RepoDir 2>/dev/null && test -x .venv/bin/automa || exit 3; exec .venv/bin/automa update"

$wslArgs = @()
if ($Distro) { $wslArgs += @('-d', $Distro) }
$wslArgs += @('-e', 'bash', '-lc', $inner)

Write-Host ''
Write-Host '=== Updating Automa ===' -ForegroundColor Cyan
Write-Host "Folder in WSL: ~/$RepoDir"
Write-Host ''
& $wsl.Source @wslArgs
$code = $LASTEXITCODE
Write-Host ''
if ($code -eq 0) {
    Write-Host 'Done: Automa is up to date.' -ForegroundColor Green
} elseif ($code -eq 3) {
    Write-Host "No Automa install found at ~/$RepoDir/.venv in WSL. Follow README.md (section Windows) first." -ForegroundColor Red
} else {
    Write-Host 'Not updated (see the lines above). Your current version still works as before.' -ForegroundColor Yellow
}

if ($NoOpen) { exit $code }

if ($code -ne 3) {
    Write-Host ''
    $answer = Read-Host 'Open Automa now? [Y/n]'
    if ($answer -eq '' -or $answer -match '^[Yy]') {
        $webArgs = @{ RepoDir = $RepoDir }
        if ($Distro) { $webArgs['Distro'] = $Distro }
        & (Join-Path $PSScriptRoot 'automa-web.ps1') @webArgs
        exit $LASTEXITCODE
    }
}
Read-Host 'Press Enter to close this window' | Out-Null
exit $code

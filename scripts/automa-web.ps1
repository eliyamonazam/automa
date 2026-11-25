# Automa web launcher for Windows (runs Automa inside WSL; Automa itself is POSIX only, D90).
#
# What it does:
#   1. starts `automa web --allow-actions` (or read-only with -ReadOnly) inside WSL,
#      from ~/automa with that folder's .venv, in this console window;
#   2. waits until the page answers on http://127.0.0.1:<Port>/ and opens it in the
#      default browser;
#   3. with -PublicHost automa.example.com, also starts the Cloudflare Tunnel of
#      docs/REMOTE-ACCESS.md (scripts/automa-tunnel.sh, `automa web --public`), so the
#      family can open https://automa.example.com/ from outside; Automa still listens
#      only on 127.0.0.1 and the tunnel stops together with it;
#   4. keeps running until you press Ctrl+C or close this window (that also stops it,
#      and a background run started from the page is paused cleanly).
#
# This file is ASCII only on purpose: Windows PowerShell 5.1 reads files without a
# BOM in the legacy code page. Usage: see the Windows section of README.md.

[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,

    # Folder of the Automa clone, relative to the WSL home directory.
    [ValidatePattern('^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$')]
    [string]$RepoDir = 'automa',

    # WSL distribution name (empty = the default distribution).
    [ValidatePattern('^[A-Za-z0-9._-]*$')]
    [string]$Distro = '',

    # Serve the read-only page (no approve/reject, new idea or start/stop buttons).
    [switch]$ReadOnly,

    # Do not open the browser.
    [switch]$NoBrowser,

    # Update Automa first (automa-update.ps1 next to this file), then open it.
    [switch]$Update,

    # Public name of the Cloudflare Tunnel (docs/REMOTE-ACCESS.md), e.g. automa.example.com.
    [ValidatePattern('^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$')]
    [string]$PublicHost = '',

    # Name given to `cloudflared tunnel create` (docs/REMOTE-ACCESS.md).
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$TunnelName = 'automa'
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

# The default WSL distro may be one without bash (e.g. Docker Desktop's
# "docker-desktop"), which fails with "execvpe(bash) failed". Without -Distro, prefer
# an Ubuntu distro, else any distro that is not Docker Desktop's.
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
    if ($pick -and $pick -match '^[A-Za-z0-9._-]+$') {
        $Distro = $pick
        Write-Host "Using WSL distro: $Distro"
    }
}

if ($Update) {
    $updateArgs = @{ RepoDir = $RepoDir; NoOpen = $true }
    if ($Distro) { $updateArgs['Distro'] = $Distro }
    & (Join-Path $PSScriptRoot 'automa-update.ps1') @updateArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Opening the current version without the update.' -ForegroundColor Yellow
    }
}

$Url = "http://127.0.0.1:$Port/"

function Test-Port([int]$P) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync('127.0.0.1', $P)
        return ($task.Wait(500) -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Open-Page {
    if (-not $NoBrowser) { Start-Process $Url }
}

if (Test-Port $Port) {
    Write-Host "Something already listens on port $Port (probably Automa from another window)."
    Write-Host "Opening $Url ; to start a new server use another port: -Port 8766"
    if ($Update) {
        Write-Host 'Close that other Automa window and open Automa again to use the updated version.' -ForegroundColor Yellow
    }
    Open-Page
    exit 0
}

$actions = if ($ReadOnly) { '' } else { ' --allow-actions' }
# Both values passed the patterns above; the host name is sent in lower case.
$launch = '.venv/bin/automa web'
if ($PublicHost) {
    $PublicHost = $PublicHost.ToLowerInvariant()
    $launch = "bash scripts/automa-tunnel.sh $TunnelName $PublicHost"
}
# Fixed command for bash inside WSL. No quotes are needed: Port is an integer and
# RepoDir, TunnelName and PublicHost passed the patterns above. Exit code 3 = missing
# clone or venv, 4 = cloudflared missing, 127 = clone too old for -PublicHost.
$inner = "cd ~/$RepoDir 2>/dev/null && test -x .venv/bin/automa || exit 3; exec $launch --port $Port$actions"

# One argument string: Start-Process does not quote array items that contain spaces.
# $inner has no double quotes, so wrapping it in them is safe.
$wslArgs = ''
if ($Distro) { $wslArgs = "-d $Distro " }
$wslArgs += "-e bash -lc `"$inner`""

Write-Host "Starting Automa in WSL: ~/$RepoDir ($Url). Press Ctrl+C here to stop."
if ($PublicHost) {
    Write-Host "From outside the house: https://$PublicHost/ (only while this window is open)."
}
$proc = Start-Process -FilePath $wsl.Source -ArgumentList $wslArgs -NoNewWindow -PassThru
$null = $proc.Handle  # keep the handle so ExitCode is available later

$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    if ($proc.HasExited) { break }
    if (Test-Port $Port) { $ready = $true; break }
    Start-Sleep -Milliseconds 500
}
if ($ready) {
    Open-Page
} elseif (-not $proc.HasExited) {
    Write-Host "The page did not answer within 30 seconds. Try $Url by hand; see README.md (Windows) if it never opens." -ForegroundColor Yellow
}

$proc.WaitForExit()
$code = $proc.ExitCode
if ($code -eq 3) {
    Write-Host "No Automa install found at ~/$RepoDir/.venv in WSL. Follow README.md (section Windows) first." -ForegroundColor Red
} elseif ($code -eq 4) {
    Write-Host 'cloudflared is not installed in WSL. Follow docs/REMOTE-ACCESS.md (step 2), or open Automa without -PublicHost.' -ForegroundColor Red
} elseif ($code -eq 127 -and $PublicHost) {
    Write-Host 'This Automa copy has no scripts/automa-tunnel.sh yet. Update Automa first (automa-update.bat).' -ForegroundColor Red
} elseif ($code -ne 0 -and $null -ne $code) {
    Write-Host "Automa web stopped with exit code $code (see the messages above)." -ForegroundColor Yellow
}
exit $code

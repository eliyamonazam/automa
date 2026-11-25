"""Static and behavioural checks of the Windows update scripts (D170).

Like ``test_windows_launcher.py``: PowerShell is not available in CI, so the script
text is checked and the exact bash command handed to WSL is run against a fake
``~/automa/.venv/bin/automa``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PS1 = ROOT / "scripts" / "automa-update.ps1"
BAT = ROOT / "scripts" / "automa-update.bat"
WEB_PS1 = ROOT / "scripts" / "automa-web.ps1"
README = ROOT / "README.md"


def _text(path: Path) -> str:
    return path.read_bytes().decode("ascii")


def _inner_template() -> str:
    match = re.search(r'^\$inner = "(?P<cmd>[^"\r\n]*)"\r?$', _text(PS1), re.MULTILINE)
    assert match, 'the bash command must be a single $inner = "..." line'
    return match.group("cmd")


@pytest.mark.parametrize("path", [PS1, BAT])
def test_scripts_exist_ascii_and_crlf(path: Path) -> None:
    data = path.read_bytes()
    data.decode("ascii")
    assert b"\r\n" in data
    assert b"\n" not in data.replace(b"\r\n", b""), "bare LF line ending"
    assert b"\r" not in data.replace(b"\r\n", b""), "bare CR"


def test_bat_only_runs_the_update_ps1_next_to_it() -> None:
    lines = [
        line for line in _text(BAT).split("\r\n") if line and not line.lower().startswith("rem ")
    ]
    assert lines[0] == "@echo off"
    assert [line for line in lines if "powershell" in line.lower()] == [
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0automa-update.ps1" %*'
    ]


def test_ps1_validates_parameters_like_the_web_launcher() -> None:
    text = _text(PS1)
    repo_pattern = "[ValidatePattern('^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$')]"
    assert repo_pattern + "\r\n    [string]$RepoDir" in text
    assert "[ValidatePattern('^[A-Za-z0-9._-]*$')]\r\n    [string]$Distro" in text
    assert "if ($RepoDir -match '(^|/)\\.\\.?(/|$)')" in text
    assert "'^Ubuntu'" in text and "docker-desktop" in text


def test_ps1_command_is_fixed_and_runs_only_automa_update() -> None:
    template = _inner_template()
    assert "'" not in template and '"' not in template and "`" not in template
    assert set(re.findall(r"\$\w+", template)) == {"$RepoDir"}
    assert template.endswith("exec .venv/bin/automa update")
    # Passed as separate arguments with the call operator, not through a shell string.
    assert "$wslArgs += @('-e', 'bash', '-lc', $inner)" in _text(PS1)
    assert "& $wsl.Source @wslArgs" in _text(PS1)


@pytest.mark.parametrize(
    "forbidden",
    [
        "Invoke-Expression",
        "iex ",
        "Set-ExecutionPolicy",
        "-EncodedCommand",
        "DownloadString",
        "Invoke-WebRequest",
        "--force",
        "reset --hard",
        "Start-Process",
    ],
)
def test_ps1_avoids_dangerous_constructs(forbidden: str) -> None:
    assert forbidden not in _text(PS1)


def test_ps1_reports_result_in_colour_and_asks_before_opening() -> None:
    text = _text(PS1)
    assert "-ForegroundColor Green" in text
    assert "-ForegroundColor Yellow" in text
    assert "if ($code -eq 3) {" in text and "No Automa install found" in text
    assert "Read-Host 'Open Automa now? [Y/n]'" in text
    assert "& (Join-Path $PSScriptRoot 'automa-web.ps1') @webArgs" in text
    assert "if ($NoOpen) { exit $code }" in text
    assert "Read-Host 'Press Enter to close this window'" in text


def test_web_launcher_has_update_switch() -> None:
    text = _text(WEB_PS1)
    assert "[switch]$Update,\r\n" in text
    assert "$updateArgs = @{ RepoDir = $RepoDir; NoOpen = $true }" in text
    assert "& (Join-Path $PSScriptRoot 'automa-update.ps1') @updateArgs" in text
    # The update happens before the port check (the server is started after it).
    assert text.index("automa-update.ps1") < text.index("if (Test-Port $Port) {")
    assert "Opening the current version without the update." in text


def test_readme_explains_the_update_path() -> None:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Windows", 1)[1].split("\n## ", 1)[0]
    for needle in (
        "automa-update.bat",
        "automa update",
        "-Update",
        "~/automa/scripts/automa-*",
    ):
        assert needle in section


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _run(home: Path, repo_dir: str = "automa") -> subprocess.CompletedProcess[str]:
    command = _inner_template().replace("$RepoDir", repo_dir)
    bash = shutil.which("bash")
    assert bash is not None
    return subprocess.run(  # noqa: S603 - fixed script text from the repository
        [bash, "--noprofile", "--norc", "-c", command],
        env={"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _fake_install(home: Path, repo_dir: str = "automa") -> None:
    bin_dir = home / repo_dir / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / "automa"
    fake.write_text('#!/bin/sh\necho "cwd=$(pwd)"\necho "args=$*"\nexit 5\n', encoding="utf-8")
    fake.chmod(0o755)


@needs_bash
def test_command_runs_automa_update_from_the_clone(tmp_path: Path) -> None:
    _fake_install(tmp_path, "work/automa2")
    result = _run(tmp_path, "work/automa2")
    assert result.returncode == 5
    assert f"cwd={tmp_path / 'work' / 'automa2'}" in result.stdout
    assert "args=update" in result.stdout.splitlines()


@needs_bash
def test_command_exits_3_without_install(tmp_path: Path) -> None:
    assert _run(tmp_path).returncode == 3
    (tmp_path / "automa").mkdir()
    assert _run(tmp_path).returncode == 3

"""Static and behavioural checks of the Windows launcher scripts (D130).

PowerShell and cmd.exe are not available in CI, so these tests check the text of
``scripts/automa-web.ps1``/``.bat`` and run the exact bash command the PowerShell
script hands to WSL against a fake ``~/automa/.venv/bin/automa``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from automa.webui import server as webui_server

ROOT = Path(__file__).resolve().parent.parent
PS1 = ROOT / "scripts" / "automa-web.ps1"
BAT = ROOT / "scripts" / "automa-web.bat"
README = ROOT / "README.md"
GITATTRIBUTES = ROOT / ".gitattributes"


def _text(path: Path) -> str:
    return path.read_bytes().decode("ascii")


def _inner_template() -> str:
    match = re.search(r'^\$inner = "(?P<cmd>[^"\r\n]*)"\r?$', _text(PS1), re.MULTILINE)
    assert match, 'the bash command must be a single $inner = "..." line'
    return match.group("cmd")


def _inner(port: int, repo_dir: str = "automa", *, read_only: bool = False) -> str:
    actions = "" if read_only else " --allow-actions"
    return (
        _inner_template()
        .replace("$RepoDir", repo_dir)
        .replace("$Port", str(port), 1)
        .replace("$actions", actions)
    )


@pytest.mark.parametrize("path", [PS1, BAT])
def test_scripts_exist_ascii_and_crlf(path: Path) -> None:
    data = path.read_bytes()
    data.decode("ascii")  # PowerShell 5.1 reads BOM-less files in the ANSI code page
    assert b"\r\n" in data
    assert b"\n" not in data.replace(b"\r\n", b""), "bare LF line ending"
    assert b"\r" not in data.replace(b"\r\n", b""), "bare CR"


def test_gitattributes_keeps_crlf_for_windows_scripts() -> None:
    lines = GITATTRIBUTES.read_text(encoding="utf-8").splitlines()
    assert "*.bat text eol=crlf" in lines
    assert "*.ps1 text eol=crlf" in lines


def test_ps1_default_port_matches_cli_default() -> None:
    match = re.search(r"\[int\]\$Port = (\d+)", _text(PS1))
    assert match
    assert int(match.group(1)) == webui_server.DEFAULT_PORT


def test_ps1_validates_every_parameter_that_reaches_the_command() -> None:
    text = _text(PS1)
    assert "[ValidateRange(1024, 65535)]\r\n    [int]$Port" in text
    repo_pattern = "[ValidatePattern('^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$')]"
    assert repo_pattern + "\r\n    [string]$RepoDir" in text
    assert "[ValidatePattern('^[A-Za-z0-9._-]*$')]\r\n    [string]$Distro" in text
    assert "if ($RepoDir -match '(^|/)\\.\\.?(/|$)')" in text


def test_ps1_command_has_no_quote_or_injection_points() -> None:
    template = _inner_template()
    assert "'" not in template and '"' not in template and "`" not in template
    # Only these variables are interpolated, all validated or fixed.
    assert set(re.findall(r"\$[A-Za-z]+", template)) == {"$RepoDir", "$Port", "$actions"}
    text = _text(PS1)
    assert "$actions = if ($ReadOnly) { '' } else { ' --allow-actions' }" in text
    assert '$wslArgs += "-e bash -lc `"$inner`""' in text


@pytest.mark.parametrize(
    "forbidden",
    [
        "Invoke-Expression",
        "iex ",
        "0.0.0.0",  # noqa: S104 - asserted absent
        "--host",
        "Set-ExecutionPolicy",
        "-EncodedCommand",
        "powershell -Command",
        "DownloadString",
    ],
)
def test_ps1_avoids_dangerous_constructs(forbidden: str) -> None:
    assert forbidden not in _text(PS1)


def test_ps1_opens_only_the_loopback_page() -> None:
    text = _text(PS1)
    assert '$Url = "http://127.0.0.1:$Port/"' in text
    assert re.findall(r"Start-Process (?:\$|-)[^\r\n]*", text) == [
        "Start-Process $Url }",
        "Start-Process -FilePath $wsl.Source -ArgumentList $wslArgs -NoNewWindow -PassThru",
    ]
    assert "ConnectAsync('127.0.0.1', $P)" in text


def test_bat_only_runs_the_ps1_next_to_it() -> None:
    text = _text(BAT)
    lines = [line for line in text.split("\r\n") if line and not line.lower().startswith("rem ")]
    assert lines[0] == "@echo off"
    commands = [line for line in lines if "powershell" in line.lower()]
    assert commands == [
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0automa-web.ps1" %*'
    ]
    assert "exit /b %AUTOMA_EXIT%" in lines


def test_readme_explains_the_launcher_and_shortcut() -> None:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Windows", 1)[1].split("\n## ", 1)[0]
    for needle in (
        "automa-web.bat",
        "automa-web.ps1",
        "Desktop (create shortcut)",
        "-ReadOnly",
        "-Port 8766",
        "networkingMode=mirrored",
        "~/automa/scripts/automa-*.bat",
    ):
        assert needle in section


# --- the bash command itself, run against a fake install -------------------------

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _fake_install(home: Path, repo_dir: str = "automa") -> Path:
    bin_dir = home / repo_dir / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / "automa"
    fake.write_text('#!/bin/sh\necho "cwd=$(pwd)"\necho "args=$*"\nexit 7\n', encoding="utf-8")
    fake.chmod(0o755)
    return fake


def _run(command: str, home: Path) -> subprocess.CompletedProcess[str]:
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    bash = shutil.which("bash")
    assert bash is not None
    return subprocess.run(  # noqa: S603 - fixed script text from the repository
        [bash, "--noprofile", "--norc", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@needs_bash
def test_command_starts_automa_web_with_actions_from_the_clone(tmp_path: Path) -> None:
    _fake_install(tmp_path)
    result = _run(_inner(8765), tmp_path)
    assert result.returncode == 7  # exec: the exit code is automa's own
    assert f"cwd={tmp_path / 'automa'}" in result.stdout
    assert "args=web --port 8765 --allow-actions" in result.stdout


@needs_bash
def test_command_read_only_and_custom_port_and_dir(tmp_path: Path) -> None:
    _fake_install(tmp_path, "work/automa2")
    result = _run(_inner(9001, "work/automa2", read_only=True), tmp_path)
    assert result.returncode == 7
    assert f"cwd={tmp_path / 'work' / 'automa2'}" in result.stdout
    assert "args=web --port 9001" in result.stdout
    assert "--allow-actions" not in result.stdout


@needs_bash
def test_command_exits_3_without_clone(tmp_path: Path) -> None:
    result = _run(_inner(8765), tmp_path)
    assert result.returncode == 3
    assert result.stdout == ""


@needs_bash
def test_command_exits_3_without_venv(tmp_path: Path) -> None:
    (tmp_path / "automa").mkdir()
    result = _run(_inner(8765), tmp_path)
    assert result.returncode == 3


@needs_bash
def test_command_exits_3_when_automa_not_executable(tmp_path: Path) -> None:
    _fake_install(tmp_path).chmod(0o644)
    result = _run(_inner(8765), tmp_path)
    assert result.returncode == 3


def test_ps1_reports_missing_install_for_exit_code_3() -> None:
    text = _text(PS1)
    assert "if ($code -eq 3) {" in text
    assert "No Automa install found" in text
    assert "exit $code" in text


def test_ps1_prefers_an_ubuntu_distro_when_none_is_given() -> None:
    # Seen live: the default WSL distro had no bash ("execvpe(bash) failed"), as with
    # Docker Desktop's docker-desktop distro.
    text = (ROOT / "scripts" / "automa-web.ps1").read_text(encoding="ascii")
    assert "-l -q" in text and "'^Ubuntu'" in text and "docker-desktop" in text
    assert "[System.Text.Encoding]::Unicode" in text
    assert "$pick -match '^[A-Za-z0-9._-]+$'" in text

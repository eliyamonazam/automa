"""Remote access part 2 (D188): ``scripts/automa-tunnel.sh``, the ``-PublicHost``
option of the Windows launcher and the owner's guide ``docs/REMOTE-ACCESS.md``.

No network and no real cloudflared: the script runs against a fake ``cloudflared``
on PATH and a fake ``.venv/bin/automa`` in a temporary clone.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from typer.main import get_command

from automa import cli
from automa.webui import server as webui_server

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "automa-tunnel.sh"
PS1 = ROOT / "scripts" / "automa-web.ps1"
GUIDE = ROOT / "docs" / "REMOTE-ACCESS.md"
README = ROOT / "README.md"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _ps1() -> str:
    return PS1.read_bytes().decode("ascii")


# --- the shell script, statically ---------------------------------------------------


def test_script_is_lf_executable_and_parses() -> None:
    data = SCRIPT.read_bytes()
    data.decode("ascii")
    assert b"\r" not in data
    assert data.startswith(b"#!/usr/bin/env bash\n")
    assert os.access(SCRIPT, os.X_OK)
    assert "*.sh text eol=lf" in (ROOT / ".gitattributes").read_text(encoding="utf-8")
    bash = shutil.which("bash")
    if bash:
        subprocess.run([bash, "-n", str(SCRIPT)], check=True, timeout=10)  # noqa: S603


def test_script_never_binds_beyond_loopback() -> None:
    text = SCRIPT.read_text(encoding="ascii")
    assert "0.0.0.0" not in text  # noqa: S104 - asserted absent
    assert "--host " not in text and "--host=" not in text
    assert '--url "http://127.0.0.1:$port"' in text
    assert "eval" not in text and "sudo" not in text


# --- the shell script, run against fakes --------------------------------------------


def _clone(tmp_path: Path, automa_body: str | None = None) -> Path:
    if automa_body is None:
        # Wait (at most 5 s) until the fake tunnel has written its arguments, so the
        # cleanup on exit cannot kill it before that.
        pid = tmp_path / "cloudflared.pid"
        automa_body = (
            f'i=0; while [ ! -s "{pid}" ] && [ "$i" -lt 50 ]; do sleep 0.1; i=$((i+1)); done\n'
            'echo "args=$*"\nexit 7\n'
        )
    clone = tmp_path / "automa"
    bin_dir = clone / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / "automa"
    fake.write_text("#!/bin/sh\n" + automa_body, encoding="utf-8")
    fake.chmod(0o755)
    return clone


def _fake_cloudflared(tmp_path: Path, body: str | None = None) -> Path:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "cloudflared"
    state = tmp_path / "cloudflared"
    fake.write_text(
        "#!/bin/sh\n"
        + (
            body
            if body is not None
            else f'echo "$*" > "{state}.args"\necho $$ > "{state}.pid"\nexec sleep 60\n'
        ),
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return bin_dir


def _run(
    tmp_path: Path, args: list[str], *, cloudflared: bool = True, timeout: float = 30
) -> subprocess.CompletedProcess[str]:
    clone = tmp_path / "automa"
    clone.mkdir(exist_ok=True)
    bins = [str(_fake_cloudflared(tmp_path))] if cloudflared else []
    system = "/usr/bin:/bin"
    env = {"HOME": str(tmp_path / "home"), "PATH": ":".join([*bins, system])}
    bash = shutil.which("bash")
    assert bash is not None
    return subprocess.run(  # noqa: S603 - repository script with test arguments
        [bash, str(SCRIPT), *args],
        cwd=clone,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@needs_bash
def test_runs_tunnel_and_public_automa_then_stops_the_tunnel(tmp_path: Path) -> None:
    _clone(tmp_path)
    started = time.monotonic()
    result = _run(tmp_path, ["automa", "automa.example.com", "--port", "8765", "--allow-actions"])
    assert time.monotonic() - started < 6, "must not wait for the tunnel watcher"
    assert result.returncode == 7  # Automa's own exit code
    assert (
        "args=web --port 8765 --allow-actions --public --host-name automa.example.com"
        in result.stdout
    )
    assert "https://automa.example.com/" in result.stdout
    args = (tmp_path / "cloudflared.args").read_text(encoding="utf-8").split()
    assert args == ["tunnel", "run", "--url", "http://127.0.0.1:8765", "automa"]
    pid = int((tmp_path / "cloudflared.pid").read_text(encoding="utf-8"))
    for _ in range(50):
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    assert not _pid_alive(pid), "the tunnel must stop together with Automa"
    log = tmp_path / "home" / ".cloudflared" / "automa-tunnel-automa.log"
    assert log.is_file()


@needs_bash
def test_read_only_and_custom_port_and_tunnel_name(tmp_path: Path) -> None:
    _clone(tmp_path)
    result = _run(tmp_path, ["family-1", "a.b.example.org", "--port", "9001"])
    assert result.returncode == 7
    assert "args=web --port 9001 --public --host-name a.b.example.org" in result.stdout
    assert "--allow-actions" not in result.stdout
    args = (tmp_path / "cloudflared.args").read_text(encoding="utf-8").split()
    assert args[-2:] == ["http://127.0.0.1:9001", "family-1"]


@needs_bash
def test_exit_4_without_cloudflared_and_no_automa_started(tmp_path: Path) -> None:
    _clone(tmp_path)
    result = _run(tmp_path, ["automa", "automa.example.com", "--port", "8765"], cloudflared=False)
    assert result.returncode == 4
    assert "REMOTE-ACCESS.md" in result.stderr
    assert "args=" not in result.stdout


@needs_bash
def test_exit_3_without_install(tmp_path: Path) -> None:
    result = _run(tmp_path, ["automa", "automa.example.com", "--port", "8765"])
    assert result.returncode == 3
    assert not (tmp_path / "cloudflared.args").exists()


@needs_bash
@pytest.mark.parametrize(
    "args",
    [
        [],
        ["automa", "automa.example.com"],
        ["automa", "automa.example.com", "--port"],
        ["automa", "automa.example.com", "--prt", "8765"],
        ["automa", "automa.example.com", "--port", "80"],
        ["automa", "automa.example.com", "--port", "99999"],
        ["automa", "automa.example.com", "--port", "87a5"],
        ["automa", "automa.example.com", "--port", "8765", "--host", "0.0.0.0"],  # noqa: S104
        ["automa", "automa.example.com", "--port", "8765", "--allow-actions", "x"],
        ["automa", "https://automa.example.com", "--port", "8765"],
        ["automa", "automa.example.com:443", "--port", "8765"],
        ["automa", "localhost", "--port", "8765"],
        ["automa", "Automa.Example.com", "--port", "8765"],
        ["automa", "a..example.com", "--port", "8765"],
        ["automa", "-bad.example.com", "--port", "8765"],
        ["au tomа", "automa.example.com", "--port", "8765"],
        ["x;id", "automa.example.com", "--port", "8765"],
        ["$(id)", "automa.example.com", "--port", "8765"],
    ],
)
def test_bad_arguments_exit_2_before_anything_starts(tmp_path: Path, args: list[str]) -> None:
    _clone(tmp_path)
    result = _run(tmp_path, args)
    assert result.returncode == 2
    assert "args=" not in result.stdout
    assert not (tmp_path / "cloudflared.args").exists()


@needs_bash
def test_warns_when_the_tunnel_dies_at_once(tmp_path: Path) -> None:
    # Automa keeps running for a while; the fake tunnel fails immediately.
    _clone(tmp_path, "sleep 10\nexit 0\n")
    _fake_cloudflared(tmp_path, 'echo "error: no such tunnel" >&2\nexit 1\n')
    clone = tmp_path / "automa"
    bash = shutil.which("bash")
    assert bash is not None
    env = {"HOME": str(tmp_path / "home"), "PATH": f"{tmp_path / 'fakebin'}:/usr/bin:/bin"}
    result = subprocess.run(  # noqa: S603
        [bash, str(SCRIPT), "automa", "automa.example.com", "--port", "8765"],
        cwd=clone,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "WARNING: the tunnel stopped" in result.stderr
    log = tmp_path / "home" / ".cloudflared" / "automa-tunnel-automa.log"
    assert "no such tunnel" in log.read_text(encoding="utf-8")


def test_script_host_pattern_agrees_with_the_server() -> None:
    """Names the script accepts are names ``automa web --host-name`` accepts."""
    text = SCRIPT.read_text(encoding="ascii")
    label = re.search(r"^label='([^']+)'$", text, re.MULTILINE)
    assert label
    pattern = re.compile(rf"^{label.group(1)}(\.{label.group(1)})+$")
    for name in ("automa.example.com", "a-b.c.example.co", "x1.example.ir"):
        assert pattern.fullmatch(name)
        assert webui_server.validate_public_host_name(name) == name
    for name in ("localhost", "example", "a..b", "-a.example.com", "A.example.com"):
        assert not pattern.fullmatch(name)


# --- the Windows launcher option ----------------------------------------------------


def test_ps1_public_host_and_tunnel_name_are_validated() -> None:
    text = _ps1()
    assert "[string]$PublicHost = ''" in text
    assert "[ValidatePattern('^[A-Za-z0-9._-]+$')]\r\n    [string]$TunnelName = 'automa'" in text
    match = re.search(r"\[ValidatePattern\('([^']+)'\)\]\r\n    \[string\]\$PublicHost", text)
    assert match
    pattern = re.compile(match.group(1), re.IGNORECASE)  # PowerShell -match ignores case
    for bad in ("https://a.example.com", "a.example.com:443", "a b.example.com", "a;b.com", "x"):
        assert not pattern.fullmatch(bad)
    assert pattern.fullmatch("Automa.Example.com")
    assert "$PublicHost = $PublicHost.ToLowerInvariant()" in text


def test_ps1_public_launch_runs_only_the_tunnel_script() -> None:
    text = _ps1()
    assert "$launch = '.venv/bin/automa web'" in text
    assert '$launch = "bash scripts/automa-tunnel.sh $TunnelName $PublicHost"' in text
    assert "exec $launch --port $Port$actions" in text
    assert "if ($code -eq 4) {" in text and "docs/REMOTE-ACCESS.md" in text
    assert "if ($code -eq 127 -and $PublicHost)" in text.replace("} elseif", "if")
    # The page opened on this computer stays the loopback one.
    assert '$Url = "http://127.0.0.1:$Port/"' in text


@needs_bash
def test_ps1_public_command_reaches_the_script_end_to_end(tmp_path: Path) -> None:
    """The exact WSL command with -PublicHost, run against a fake clone."""
    match = re.search(r'^\$inner = "(?P<cmd>[^"\r\n]*)"\r?$', _ps1(), re.MULTILINE)
    assert match
    home = tmp_path / "home"
    clone = home / "automa"
    (clone / ".venv" / "bin").mkdir(parents=True)
    (clone / "scripts").mkdir()
    shutil.copy(SCRIPT, clone / "scripts" / "automa-tunnel.sh")
    fake = clone / ".venv" / "bin" / "automa"
    fake.write_text('#!/bin/sh\necho "args=$*"\nexit 5\n', encoding="utf-8")
    fake.chmod(0o755)
    bins = _fake_cloudflared(tmp_path)
    command = (
        match.group("cmd")
        .replace("$RepoDir", "automa")
        .replace("$launch", "bash scripts/automa-tunnel.sh automa automa.example.com")
        .replace("$Port", "8765", 1)
        .replace("$actions", " --allow-actions")
    )
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(  # noqa: S603
        [bash, "--noprofile", "--norc", "-c", command],
        env={"HOME": str(home), "PATH": f"{bins}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 5
    assert (
        "args=web --port 8765 --allow-actions --public --host-name automa.example.com"
        in result.stdout
    )


# --- the guide ----------------------------------------------------------------------


def test_guide_covers_every_step_the_roadmap_asks_for() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    for needle in (
        "automa users add",
        "--role owner",
        "pkg.cloudflare.com/cloudflared any main",
        "sudo apt-get install cloudflared",
        "cloudflared tunnel login",
        "cloudflared tunnel create automa",
        "cloudflared tunnel route dns automa automa.example.com",
        "-PublicHost automa.example.com",
        "-TunnelName",
        "shell:startup",
        "powercfg /change standby-timeout-ac 0",
        "Active hours",
        "automa users disable",
        "~/.cloudflared/automa-tunnel-automa.log",
    ):
        assert needle in text, needle


def test_guide_quotes_the_real_messages() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="ascii")
    assert "cloudflared is not installed in WSL" in script
    assert "WARNING: the tunnel stopped" in script
    assert "From outside the house: https://" in _ps1()
    assert "This Automa copy has no scripts/automa-tunnel.sh yet" in _ps1()
    server = (ROOT / "src" / "automa" / "webui" / "server.py").read_text(encoding="utf-8")
    assert "--public needs at least one active account first" in server
    for quoted in (
        "cloudflared is not installed in WSL",
        "--public needs at least one active account first",
        "This Automa copy has no scripts/automa-tunnel.sh yet",
        "WARNING: the tunnel stopped",
        "From outside the house: https://automa.example.com/",
    ):
        assert quoted in text, quoted


def test_guide_commands_use_real_cli_options() -> None:
    """Every ``automa users …`` sub-command and ``web`` option the guide names exists."""
    group = get_command(cli.app)
    users = group.commands["users"]  # type: ignore[attr-defined]
    text = GUIDE.read_text(encoding="utf-8")
    for sub in re.findall(r"automa users ([a-z]+)", text):
        assert sub in users.commands, sub  # type: ignore[attr-defined]
    web_opts = {opt for param in group.commands["web"].params for opt in param.opts}  # type: ignore[attr-defined]
    for opt in re.findall(r"automa web ((?:--[a-z-]+ ?(?:[a-z0-9.]+ )?)+)", text):
        for flag in re.findall(r"--[a-z-]+", opt):
            assert flag in web_opts, flag
    assert str(webui_server.DEFAULT_PORT) in text


def test_guide_never_tells_to_open_a_port_or_store_secrets() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    assert "0.0.0.0" not in text  # noqa: S104 - asserted absent
    assert "port forwarding" not in text.lower()
    assert "sudo cloudflared" not in text


def test_readme_and_architecture_point_to_the_guide() -> None:
    assert "docs/REMOTE-ACCESS.md" in README.read_text(encoding="utf-8")
    arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "automa-tunnel.sh" in arch and "REMOTE-ACCESS.md" in arch

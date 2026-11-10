"""``run_command``: no shell, confined cwd, clean environment, timeout, output cap.

Only small local programs (``sh``, ``echo``, ``sleep``) are run; no model, no network.
"""

from __future__ import annotations

import ast
import os
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from automa.events import MemoryEventSink
from automa.tools import (
    PermissionPolicy,
    RiskLevel,
    RunCommand,
    RunCommandArgs,
    ToolError,
    ToolExecutor,
    Workspace,
    clean_env,
    default_tools,
)
from automa.tools.base import MAX_TOOL_OUTPUT_CHARS
from automa.tools.command import MAX_ARGV_ITEMS, MAX_STREAM_BYTES, MAX_TIMEOUT_S, _format

SRC = Path(__file__).resolve().parents[1] / "src" / "automa"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "files"
    r.mkdir()
    return r


@pytest.fixture
def ws(root: Path) -> Workspace:
    return Workspace(root)


def run(ws: Workspace, argv: list[str], **kw: object) -> str:
    return RunCommand().run(RunCommandArgs(argv=argv, **kw), ws)  # type: ignore[arg-type]


def alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    except OSError:  # pragma: no cover - no procfs
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def wait_dead(pid: int, seconds: float = 3.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not alive(pid):
            return True
        time.sleep(0.02)
    return False


def executor(
    root: Path, *, max_risk: RiskLevel, allowed: set[str] | None = None
) -> tuple[ToolExecutor, MemoryEventSink]:
    sink = MemoryEventSink()
    policy = PermissionPolicy(
        allowed_tools={"run_command"} if allowed is None else allowed, max_risk=max_risk
    )
    return ToolExecutor(default_tools(), policy, Workspace(root), sink, "proj"), sink


# --------------------------------------------------------------------------- basics


class TestBasics:
    def test_stdout_stderr_and_exit_code(self, ws: Workspace) -> None:
        out = run(ws, ["sh", "-c", "echo out; echo err >&2; exit 3"])
        assert out.startswith("exit_code: 3\n")
        assert "--- stdout ---\nout\n" in out
        assert "--- stderr ---\nerr\n" in out

    def test_success(self, ws: Workspace) -> None:
        assert run(ws, ["echo", "hello"]).startswith("exit_code: 0\n")

    def test_risk_is_execute(self) -> None:
        assert RunCommand.risk is RiskLevel.EXECUTE

    @pytest.mark.parametrize(
        "arg",
        ["a; touch pwned", "$(touch pwned)", "`touch pwned`", "*", "a | tee pwned", "> pwned"],
    )
    def test_no_shell_interpretation(self, ws: Workspace, root: Path, arg: str) -> None:
        (root / "only.txt").write_text("x")
        out = run(ws, ["echo", arg])
        assert f"\n{arg}\n" in out
        assert sorted(p.name for p in root.iterdir()) == ["only.txt"]

    def test_stdin_is_closed(self, ws: Workspace) -> None:
        start = time.monotonic()
        out = run(ws, ["cat"], timeout_s=5)
        assert out.startswith("exit_code: 0\n")
        assert time.monotonic() - start < 3

    def test_runs_in_workspace_root(self, ws: Workspace, root: Path) -> None:
        run(ws, ["sh", "-c", "pwd > where.txt"])
        assert (root / "where.txt").read_text().strip() == str(root)


# --------------------------------------------------------------------------- argv


class TestArgv:
    @pytest.mark.parametrize(
        "argv",
        [
            [],
            [""],
            [" ls"],
            ["ls "],
            ["/bin/sh"],
            ["../bin/sh"],
            ["a/../../sh"],
            ["C:\\x"],
            ["ls", "a\x00b"],
            ["ls", "x" * 4097],
            ["echo"] * (MAX_ARGV_ITEMS + 1),
        ],
    )
    def test_invalid_argv(self, argv: list[str]) -> None:
        with pytest.raises(ValidationError):
            RunCommandArgs(argv=argv)

    @pytest.mark.parametrize("timeout", [0, -1, MAX_TIMEOUT_S + 1, float("inf"), float("nan")])
    def test_invalid_timeout(self, timeout: float) -> None:
        with pytest.raises(ValidationError):
            RunCommandArgs(argv=["ls"], timeout_s=timeout)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunCommandArgs.model_validate({"argv": ["ls"], "shell": True})

    def test_argv_must_be_strings(self) -> None:
        with pytest.raises(ValidationError):
            RunCommandArgs.model_validate({"argv": "ls -la"})

    def test_relative_program_is_normalized(self) -> None:
        assert RunCommandArgs(argv=["sub/./run.sh"]).argv == ["./sub/run.sh"]
        assert RunCommandArgs(argv=["./run.sh", "a"]).argv == ["./run.sh", "a"]

    def test_workspace_script(self, ws: Workspace, root: Path) -> None:
        (root / "bin").mkdir()
        script = root / "bin" / "hi.sh"
        script.write_text('#!/bin/sh\necho from-script "$1"\n')
        script.chmod(0o755)
        assert "from-script X" in run(ws, ["bin/hi.sh", "X"])

    def test_program_not_found(self, ws: Workspace) -> None:
        with pytest.raises(ToolError, match="not found"):
            run(ws, ["definitely-not-a-program-xyz"])

    def test_not_executable(self, ws: Workspace, root: Path) -> None:
        (root / "plain.sh").write_text("echo hi\n")
        with pytest.raises(ToolError, match="not executable|cannot start"):
            run(ws, ["./plain.sh"])


class TestAllowedPrograms:
    def test_allow_list(self, ws: Workspace) -> None:
        tool = RunCommand(allowed_programs={"echo"})
        assert tool.run(RunCommandArgs(argv=["echo", "ok"]), ws).startswith("exit_code: 0")
        with pytest.raises(ToolError, match="not allowed"):
            tool.run(RunCommandArgs(argv=["sh", "-c", "echo no"]), ws)

    def test_empty_allow_list_denies_everything(self, ws: Workspace) -> None:
        with pytest.raises(ToolError, match="not allowed"):
            RunCommand(allowed_programs=[]).run(RunCommandArgs(argv=["echo"]), ws)

    def test_string_rejected(self) -> None:
        with pytest.raises(TypeError):
            RunCommand(allowed_programs="echo")

    def test_allow_list_is_frozen(self) -> None:
        names = ["echo"]
        tool = RunCommand(allowed_programs=names)
        names.append("sh")
        assert tool.allowed_programs == frozenset({"echo"})


# --------------------------------------------------------------------------- cwd


class TestCwd:
    def test_subdirectory(self, ws: Workspace, root: Path) -> None:
        (root / "sub" / "deep").mkdir(parents=True)
        run(ws, ["sh", "-c", "pwd > here.txt"], cwd="sub/deep")
        assert (root / "sub" / "deep" / "here.txt").read_text().strip() == str(root / "sub/deep")

    @pytest.mark.parametrize("cwd", ["..", "../x", "/etc", "a/../..", "a\x00b"])
    def test_escape_refused(self, ws: Workspace, cwd: str) -> None:
        with pytest.raises(ToolError):
            run(ws, ["pwd"], cwd=cwd)

    def test_missing_dir(self, ws: Workspace) -> None:
        with pytest.raises(ToolError, match="not found"):
            run(ws, ["pwd"], cwd="nope")

    def test_file_is_not_a_dir(self, ws: Workspace, root: Path) -> None:
        (root / "f").write_text("x")
        with pytest.raises(ToolError):
            run(ws, ["pwd"], cwd="f")

    def test_symlink_refused(self, ws: Workspace, root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "link").symlink_to(outside)
        with pytest.raises(ToolError):
            run(ws, ["sh", "-c", "touch pwned"], cwd="link")
        assert not (outside / "pwned").exists()

    def test_symlink_in_middle_refused(self, ws: Workspace, root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        (outside / "inner").mkdir(parents=True)
        (root / "link").symlink_to(outside)
        with pytest.raises(ToolError):
            run(ws, ["pwd"], cwd="link/inner")

    def test_dir_path_root(self, ws: Workspace, root: Path) -> None:
        assert ws.dir_path("") == root
        assert ws.dir_path(".") == root


# --------------------------------------------------------------------------- environment


class TestEnvironment:
    SECRETS = {
        "OPENAI_API_KEY": "sk-secret-openai",
        "AUTOMA_API_KEY": "sk-secret-automa",
        "ANTHROPIC_API_KEY": "sk-secret-anthropic",
        "GITHUB_TOKEN": "ghp-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "LD_PRELOAD": "/opt/evil.so",
        "PYTHONPATH": "/opt/evil",
    }

    def test_secrets_not_passed(self, ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
        for k, v in self.SECRETS.items():
            monkeypatch.setenv(k, v)
        out = run(ws, ["env"])
        for k, v in self.SECRETS.items():
            assert k not in out
            assert v not in out

    def test_clean_env_allow_list(self, root: Path) -> None:
        parent = {**self.SECRETS, "PATH": "/x/bin", "LANG": "fa_IR.UTF-8", "TZ": "UTC"}
        env = clean_env(str(root), parent)
        assert env == {
            "PATH": "/x/bin",
            "LANG": "fa_IR.UTF-8",
            "TZ": "UTC",
            "HOME": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def test_clean_env_defaults(self, root: Path) -> None:
        env = clean_env(str(root), {})
        assert env["PATH"] and env["LANG"] == "C.UTF-8" and env["HOME"] == str(root)
        assert set(env) == {"PATH", "LANG", "HOME", "PYTHONDONTWRITEBYTECODE"}

    def test_home_is_workspace(self, ws: Workspace, root: Path) -> None:
        assert f"HOME={root}\n" in run(ws, ["env"])


# --------------------------------------------------------------------------- timeout and processes


class TestTimeout:
    def test_timeout_kills(self, ws: Workspace) -> None:
        start = time.monotonic()
        with pytest.raises(ToolError, match="timed out"):
            run(ws, ["sleep", "30"], timeout_s=0.3)
        assert time.monotonic() - start < 5

    def test_timeout_reports_partial_output(self, ws: Workspace) -> None:
        with pytest.raises(ToolError, match="before-hang"):
            run(ws, ["sh", "-c", "echo before-hang; sleep 30"], timeout_s=0.5)

    def test_timeout_kills_whole_group(self, ws: Workspace, root: Path) -> None:
        with pytest.raises(ToolError, match="timed out"):
            run(ws, ["sh", "-c", "sleep 30 & echo $! > bg.pid; wait"], timeout_s=0.5)
        pid = int((root / "bg.pid").read_text())
        assert wait_dead(pid)

    def test_background_child_of_finished_command_is_killed(
        self, ws: Workspace, root: Path
    ) -> None:
        start = time.monotonic()
        out = run(ws, ["sh", "-c", "sleep 30 & echo $! > bg.pid"], timeout_s=10)
        assert out.startswith("exit_code: 0")
        assert time.monotonic() - start < 5
        assert wait_dead(int((root / "bg.pid").read_text()))

    def test_detached_output_holder_does_not_hang(self, ws: Workspace) -> None:
        start = time.monotonic()
        out = run(ws, ["sh", "-c", "(sleep 30; echo late) & echo early"], timeout_s=10)
        assert "early" in out and "late" not in out
        assert time.monotonic() - start < 5


# --------------------------------------------------------------------------- output


class TestOutput:
    def test_large_output_is_capped(self, ws: Workspace) -> None:
        out = run(ws, ["sh", "-c", "yes 0123456789 | head -c 5000000"], timeout_s=30)
        assert len(out) <= MAX_TOOL_OUTPUT_CHARS
        assert "truncated" in out
        assert out.startswith("exit_code: 0")

    def test_stream_cap_reports_discarded(self, ws: Workspace) -> None:
        n = MAX_STREAM_BYTES + 1000
        out = run(ws, ["sh", "-c", f"yes | head -c {n} >&2"], timeout_s=30)
        assert "1000 more bytes discarded" in out

    def test_invalid_utf8_replaced(self, ws: Workspace) -> None:
        out = run(ws, ["printf", "\\377\\376ok"])
        assert "ok" in out and "\ufffd" in out

    def test_format_keeps_tails(self) -> None:
        big = "a" * 30_000 + "OUT-END"
        out = _format(1, big, "small-err")
        assert len(out) <= MAX_TOOL_OUTPUT_CHARS
        assert "OUT-END" in out and "small-err" in out
        both = _format(0, "o" * 30_000 + "O!", "e" * 30_000 + "E!")
        assert len(both) <= MAX_TOOL_OUTPUT_CHARS
        assert "O!" in both and "E!" in both
        err_big = _format(0, "tiny", "e" * 30_000 + "E!")
        assert "tiny" in err_big and "E!" in err_big and len(err_big) <= MAX_TOOL_OUTPUT_CHARS

    def test_format_no_truncation(self) -> None:
        assert _format(0, "a", "b") == "exit_code: 0\n--- stdout ---\na\n--- stderr ---\nb"


# --------------------------------------------------------------------------- executor + policy


class TestExecutorIntegration:
    @pytest.mark.parametrize("max_risk", [RiskLevel.READ, RiskLevel.WRITE])
    def test_denied_below_execute(self, root: Path, max_risk: RiskLevel) -> None:
        ex, sink = executor(root, max_risk=max_risk)
        res = ex.execute("run_command", {"argv": ["sh", "-c", "touch pwned"]})
        assert not res.ok and res.denied
        assert not (root / "pwned").exists()
        assert sink.events[-1].type == "tool.denied"
        assert "run_command" not in {s["function"]["name"] for s in ex.specs()}

    def test_denied_when_not_listed(self, root: Path) -> None:
        ex, _ = executor(root, max_risk=RiskLevel.EXECUTE, allowed={"read_file"})
        res = ex.execute("run_command", {"argv": ["sh", "-c", "touch pwned"]})
        assert res.denied and not (root / "pwned").exists()

    def test_allowed_with_execute(self, root: Path) -> None:
        ex, sink = executor(root, max_risk=RiskLevel.EXECUTE)
        res = ex.execute("run_command", '{"argv": ["echo", "hi"]}')
        assert res.ok and "hi" in res.content
        assert "run_command" in {s["function"]["name"] for s in ex.specs()}
        ev = sink.events[-1]
        assert ev.type == "tool.call"
        assert ev.data == {
            "tool": "run_command",
            "risk": "execute",
            "ok": True,
            "program": "echo",
            "cwd": "",
        }

    def test_arguments_not_logged(self, root: Path) -> None:
        ex, sink = executor(root, max_risk=RiskLevel.EXECUTE)
        ex.execute("run_command", {"argv": ["echo", "sk-very-secret-value"]})
        assert "sk-very-secret-value" not in str([e.data for e in sink.events])

    def test_failure_is_result_not_exception(self, root: Path) -> None:
        ex, sink = executor(root, max_risk=RiskLevel.EXECUTE)
        res = ex.execute("run_command", {"argv": ["sleep", "5"], "timeout_s": 0.2})
        assert not res.ok and "timed out" in (res.error or "")
        assert sink.events[-1].data["ok"] is False

    def test_invalid_args_via_executor(self, root: Path) -> None:
        ex, _ = executor(root, max_risk=RiskLevel.EXECUTE)
        for bad in ({"argv": ["/bin/sh"]}, {"argv": ["ls"], "cwd": ".."}, {"cmd": "ls"}):
            res = ex.execute("run_command", bad)
            assert not res.ok and not res.denied

    def test_spec_has_no_shell_option(self) -> None:
        props = RunCommand().spec()["function"]["parameters"]["properties"]
        assert set(props) == {"argv", "cwd", "timeout_s"}


# --------------------------------------------------------------------------- structure


def test_only_command_module_spawns_processes() -> None:
    """Process-spawning APIs appear only in tools/command.py."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                mods = {a.name for a in node.names} | {getattr(node, "module", None) or ""}
                if "subprocess" in mods or "pty" in mods:
                    offenders.append(f"{rel}:{node.lineno}")
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and (
                    node.attr in {"system", "popen", "fork", "forkpty", "posix_spawn"}
                    or node.attr.startswith(("exec", "spawn"))
                )
            ):
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders and all(o.startswith("tools/command.py:") for o in offenders)


def test_command_never_uses_shell() -> None:
    tree = ast.parse((SRC / "tools" / "command.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "shell":
            assert isinstance(node.value, ast.Constant) and node.value.value is False
        if isinstance(node, ast.keyword) and node.arg == "env":
            assert isinstance(node.value, ast.Name) and node.value.id == "env"

"""Tests for roles as data files (``automa.roles``). Local only, no model calls."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from automa.config import Tier
from automa.events import MemoryEventSink
from automa.roles import (
    BUILTIN_ROLES_DIR,
    MAX_ROLE_FILE_BYTES,
    Role,
    RoleError,
    RoleRegistry,
    parse_role,
)
from automa.tools import (
    ApprovalRequired,
    Decision,
    PermissionPolicy,
    RiskLevel,
    Tool,
    ToolArgs,
    ToolExecutor,
    Workspace,
    default_tools,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "automa"

GOOD = """
name = "{name}"
description = "A test role."
tier = "small"
tools = ["read_file", "list_dir"]
max_risk = "read"
max_steps = 5
instructions = "Do the task."
"""


def write_role(directory: Path, name: str, text: str | None = None) -> Path:
    path = directory / f"{name}.toml"
    path.write_text(text if text is not None else GOOD.format(name=name), encoding="utf-8")
    return path


def role(**kw: object) -> Role:
    data: dict[str, object] = {"name": "tester", "description": "d", "instructions": "i"}
    data.update(kw)
    return Role.model_validate(data)


class Net(Tool):
    name: ClassVar[str] = "http_get"
    description: ClassVar[str] = "Fetch a URL."
    risk: ClassVar[RiskLevel] = RiskLevel.EXTERNAL
    args_model: ClassVar[type[ToolArgs]] = ToolArgs

    def run(self, args: ToolArgs, workspace: Workspace) -> str:  # pragma: no cover
        raise AssertionError("external tools must never run without approval")


# -- built-in roles ---------------------------------------------------------------


def test_builtin_roles_exist_and_validate_against_real_tools() -> None:
    reg = RoleRegistry.builtin(tools=default_tools())
    assert reg.names() == {"researcher", "architect", "developer", "writer"}
    assert len(reg) == 4
    assert [r.name for r in reg] == sorted(reg.names())


def test_builtin_dir_is_inside_package() -> None:
    assert BUILTIN_ROLES_DIR.is_dir()
    assert BUILTIN_ROLES_DIR.is_relative_to(SRC)


@pytest.mark.parametrize("name", ["researcher", "architect", "writer"])
def test_non_developer_roles_cannot_execute(name: str) -> None:
    r = RoleRegistry.builtin().get(name)
    assert r.max_risk.rank < RiskLevel.EXECUTE.rank
    assert "run_command" not in r.tools
    assert r.policy().decide("run_command", RiskLevel.EXECUTE).decision is Decision.DENY


def test_developer_can_run_commands_but_not_external() -> None:
    p = RoleRegistry.builtin().get("developer").policy()
    assert p.decide("run_command", RiskLevel.EXECUTE).allowed
    assert p.decide("write_file", RiskLevel.WRITE).allowed
    assert p.decide("http_get", RiskLevel.EXTERNAL).decision is Decision.NEEDS_APPROVAL


def test_builtin_roles_are_sane() -> None:
    for r in RoleRegistry.builtin():
        assert r.max_risk is not RiskLevel.EXTERNAL
        assert r.instructions.strip()
        assert 1 <= r.max_steps <= 50
        assert isinstance(r.tier, Tier)
        # no secrets / keys in prompts
        low = r.instructions.lower()
        assert "sk-" not in low and "api_key=" not in low


def test_builtin_role_files_are_plain_files() -> None:
    for entry in BUILTIN_ROLES_DIR.iterdir():
        assert not entry.is_symlink()
        assert entry.suffix == ".toml"
        assert entry.stat().st_size < MAX_ROLE_FILE_BYTES


# -- Role model ---------------------------------------------------------------------


def test_defaults_are_least_privilege() -> None:
    r = role()
    assert r.tools == ()
    assert r.max_risk is RiskLevel.READ
    assert r.tier is Tier.MEDIUM
    p = r.policy()
    assert p.allowed_tools == frozenset()
    assert p.decide("read_file", RiskLevel.READ).decision is Decision.DENY


def test_policy_matches_role() -> None:
    r = role(tools=["read_file", "write_file"], max_risk="write")
    p = r.policy()
    assert isinstance(p, PermissionPolicy)
    assert p.allowed_tools == {"read_file", "write_file"}
    assert p.max_risk is RiskLevel.WRITE


def test_role_is_immutable() -> None:
    r = role()
    with pytest.raises(ValidationError):
        r.max_risk = RiskLevel.EXECUTE  # type: ignore[misc]


def test_policy_cannot_be_widened_through_role() -> None:
    r = role(tools=["read_file"])
    p = r.policy()
    with pytest.raises(ValidationError):
        p.max_risk = RiskLevel.EXECUTE  # type: ignore[misc]
    assert r.policy().decide("write_file", RiskLevel.WRITE).decision is Decision.DENY


@pytest.mark.parametrize(
    "kw",
    [
        {"max_risk": "external"},
        {"max_risk": "root"},
        {"name": "Developer"},
        {"name": "../x"},
        {"name": ""},
        {"name": "a" * 40},
        {"description": ""},
        {"description": "   "},
        {"instructions": ""},
        {"instructions": " \n "},
        {"instructions": "x" * 20_001},
        {"tools": "read_file"},
        {"tools": ["read_file", "read_file"]},
        {"tools": ["Read-File"]},
        {"tools": [1]},
        {"tier": "huge"},
        {"max_steps": 0},
        {"max_steps": 201},
        {"schema_version": 2},
        {"model": "gpt-x"},  # roles ask for a tier, never a model name
        {"api_key": "nope"},
    ],
)
def test_invalid_role_fields(kw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        role(**kw)


# -- parse_role -----------------------------------------------------------------


def test_parse_role_ok() -> None:
    r = parse_role(GOOD.format(name="x"), expected_name="x")
    assert r.name == "x" and r.tier is Tier.SMALL and r.max_steps == 5


def test_parse_role_name_must_match_file() -> None:
    with pytest.raises(RoleError, match="does not match"):
        parse_role(GOOD.format(name="x"), expected_name="y")


def test_parse_role_bad_toml() -> None:
    with pytest.raises(RoleError, match="invalid TOML"):
        parse_role("name = ", expected_name="x", source="x.toml")


def test_parse_role_reports_fields() -> None:
    with pytest.raises(RoleError) as e:
        parse_role('name = "x"\nmax_risk = "external"\n', expected_name="x", source="x.toml")
    msg = str(e.value)
    assert msg.startswith("x.toml: invalid role")
    assert "max_risk" in msg and "description" in msg and "instructions" in msg


# -- loading from a directory -------------------------------------------------------


def test_from_dir_loads_roles(tmp_path: Path) -> None:
    write_role(tmp_path, "alpha")
    write_role(tmp_path, "beta")
    (tmp_path / "README.md").write_text("ignored", encoding="utf-8")
    (tmp_path / ".hidden.toml").write_text("broken = ", encoding="utf-8")
    reg = RoleRegistry.from_dir(tmp_path, tools=default_tools())
    assert reg.names() == {"alpha", "beta"}
    assert "alpha" in reg and "gamma" not in reg
    assert reg.get("beta").tier is Tier.SMALL


def test_unknown_role_lookup(tmp_path: Path) -> None:
    write_role(tmp_path, "alpha")
    with pytest.raises(RoleError, match="unknown role"):
        RoleRegistry.from_dir(tmp_path).get("omega")


def test_empty_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RoleError, match="no roles"):
        RoleRegistry.from_dir(tmp_path)


def test_missing_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RoleError, match="not found"):
        RoleRegistry.from_dir(tmp_path / "nope")


def test_symlinked_dir_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    write_role(real, "alpha")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(RoleError):
        RoleRegistry.from_dir(tmp_path / "link")


def test_one_broken_file_fails_the_whole_registry(tmp_path: Path) -> None:
    write_role(tmp_path, "alpha")
    write_role(tmp_path, "beta", 'name = "beta"\n')
    with pytest.raises(RoleError, match="beta.toml"):
        RoleRegistry.from_dir(tmp_path)


def test_file_name_mismatch(tmp_path: Path) -> None:
    write_role(tmp_path, "alpha", GOOD.format(name="beta"))
    with pytest.raises(RoleError, match="does not match"):
        RoleRegistry.from_dir(tmp_path)


def test_invalid_file_stem(tmp_path: Path) -> None:
    write_role(tmp_path, "Bad Name", GOOD.format(name="Bad Name"))
    with pytest.raises(RoleError):
        RoleRegistry.from_dir(tmp_path)


def test_symlinked_role_file_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    target = write_role(outside, "alpha")
    roles = tmp_path / "roles"
    roles.mkdir()
    (roles / "alpha.toml").symlink_to(target)
    with pytest.raises(RoleError, match="cannot open"):
        RoleRegistry.from_dir(roles)


def test_directory_named_toml_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "alpha.toml").mkdir()
    with pytest.raises(RoleError):
        RoleRegistry.from_dir(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_fifo_role_file_is_rejected_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "alpha.toml")
    # O_RDONLY on a FIFO with no writer would block; we must not hang.
    import threading

    errors: list[BaseException] = []

    def load() -> None:
        try:
            RoleRegistry.from_dir(tmp_path)
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=load, daemon=True)
    t.start()
    t.join(5)
    if t.is_alive():  # unblock the reader so the test process can exit
        with open(tmp_path / "alpha.toml", "wb"):
            pass
        t.join(5)
        pytest.fail("loading a FIFO role file blocked")
    assert errors and isinstance(errors[0], RoleError)


def test_oversized_file_is_rejected(tmp_path: Path) -> None:
    text = GOOD.format(name="alpha") + "\n#" + "x" * MAX_ROLE_FILE_BYTES
    write_role(tmp_path, "alpha", text)
    with pytest.raises(RoleError, match="larger than"):
        RoleRegistry.from_dir(tmp_path)


def test_non_utf8_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "alpha.toml").write_bytes(b'name = "\xff"\n')
    with pytest.raises(RoleError, match="UTF-8"):
        RoleRegistry.from_dir(tmp_path)


# -- registry consistency with the tool set ---------------------------------------


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(RoleError, match="unknown tools"):
        RoleRegistry([role(tools=["delete_everything"])], tools=default_tools())


def test_tool_above_max_risk_is_rejected() -> None:
    with pytest.raises(RoleError, match="above max_risk"):
        RoleRegistry([role(tools=["write_file"], max_risk="read")], tools=default_tools())
    with pytest.raises(RoleError, match="run_command"):
        RoleRegistry(
            [role(tools=["read_file", "run_command"], max_risk="write")], tools=default_tools()
        )


def test_external_tool_may_be_listed_but_always_needs_approval(tmp_path: Path) -> None:
    r = role(tools=["http_get"], max_risk="read")
    reg = RoleRegistry([r], tools=[*default_tools(), Net()])
    ws_dir = tmp_path / "files"
    ws_dir.mkdir()
    ex = ToolExecutor(
        [Net()], reg.get("tester").policy(), Workspace(ws_dir), MemoryEventSink(), "p"
    )
    with pytest.raises(ApprovalRequired):
        ex.execute("http_get", "{}")


def test_without_tools_only_structure_is_checked() -> None:
    reg = RoleRegistry([role(tools=["anything"])])
    assert reg.get("tester").tools == ("anything",)


def test_duplicate_roles_rejected() -> None:
    with pytest.raises(RoleError, match="duplicate"):
        RoleRegistry([role(), role()])


def test_registry_accepts_only_roles() -> None:
    with pytest.raises(TypeError):
        RoleRegistry([{"name": "x"}])  # type: ignore[list-item]


def test_builtin_policies_through_executor(tmp_path: Path) -> None:
    ws_dir = tmp_path / "files"
    ws_dir.mkdir()
    ws = Workspace(ws_dir)
    writer = ToolExecutor(
        default_tools(),
        RoleRegistry.builtin().get("writer").policy(),
        ws,
        MemoryEventSink(),
        "p",
    )
    assert {s["function"]["name"] for s in writer.specs()} == {
        "read_file",
        "list_dir",
        "write_file",
    }
    res = writer.execute("run_command", '{"argv": ["true"]}')
    assert not res.ok and res.denied
    ok = writer.execute("write_file", '{"path": "a.md", "content": "hi"}')
    assert ok.ok, ok.error
    assert (ws_dir / "a.md").read_text(encoding="utf-8") == "hi"


# -- structure: roles are data, not code -------------------------------------------


def test_roles_package_does_not_touch_llm_or_run_tools() -> None:
    for py in (SRC / "roles").glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("automa.llm"), py.name
                assert not node.module.startswith("automa.budget"), py.name
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"run", "execute", "complete"}, f"{py.name}: {node.attr}"

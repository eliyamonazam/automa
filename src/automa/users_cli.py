"""``automa users``: the owner manages web UI accounts from the command line (D180).

Passwords are never taken as command-line arguments (they would end up in the
shell history and in ``ps``): they are asked twice with hidden input, or read
from one line of standard input with ``--password-stdin``. Nothing here prints
a hash or a salt.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from automa.config import ConfigError, load_config
from automa.users import UserRole, UsersError, UserStore

EXIT_USER = 1
EXIT_CONFIG = 2

users_app = typer.Typer(
    help="Web UI accounts (owner only): add, list, change password, disable or enable. "
    "With no accounts the web UI stays loopback-only as before.",
    no_args_is_help=True,
)

ConfigOpt = Annotated[Path | None, typer.Option("--config", help="Path to automa.toml.")]
StdinOpt = Annotated[
    bool,
    typer.Option(
        "--password-stdin",
        help="Read the password from one line of standard input instead of asking.",
    ),
]


def _fail(message: str, code: int = EXIT_USER) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def _store(config_path: Path | None) -> UserStore:
    try:
        config = load_config(config_path)
    except ConfigError as e:
        raise _fail(str(e), EXIT_CONFIG) from None
    return UserStore(config.workspace)


def _read_password(from_stdin: bool) -> str:
    if from_stdin:
        line = sys.stdin.readline()
        if not line:
            raise _fail("no password on standard input")
        return line.removesuffix("\n").removesuffix("\r")
    return str(typer.prompt("password", hide_input=True, confirmation_prompt="repeat password"))


def _status(locked_until: datetime | None, disabled: bool, now: datetime) -> str:
    if disabled:
        return "disabled"
    if locked_until is not None and now < locked_until:
        return f"locked until {locked_until.astimezone(UTC):%Y-%m-%d %H:%M} UTC"
    return "active"


@users_app.command("add")
def add(
    name: Annotated[str, typer.Argument(help="Login name (lowercase letters, digits, - or _).")],
    role: Annotated[
        UserRole, typer.Option("--role", help="owner (sees and approves everything) or member.")
    ] = UserRole.MEMBER,
    password_stdin: StdinOpt = False,
    config_path: ConfigOpt = None,
) -> None:
    """Create an account. The first account must be an owner."""
    store = _store(config_path)
    password = _read_password(password_stdin)
    try:
        user = store.add(name, password, role)
    except UsersError as e:
        raise _fail(str(e)) from None
    typer.echo(f"added {user.role.value} {user.name!r} (saved in {store.path})")


@users_app.command("list")
def list_users(config_path: ConfigOpt = None) -> None:
    """Show the accounts (never passwords or hashes)."""
    store = _store(config_path)
    try:
        users = store.list()
    except UsersError as e:
        raise _fail(str(e)) from None
    if not users:
        typer.echo("no accounts; the web UI is loopback-only (add one: automa users add <name>)")
        return
    now = datetime.now(UTC)
    for u in users:
        typer.echo(
            f"{u.name:<20} {u.role.value:<7} {_status(u.locked_until, u.disabled, now):<34} "
            f"since {u.created_at.astimezone(UTC):%Y-%m-%d}"
        )


@users_app.command("passwd")
def passwd(
    name: Annotated[str, typer.Argument(help="Account to change.")],
    password_stdin: StdinOpt = False,
    config_path: ConfigOpt = None,
) -> None:
    """Set a new password (also lifts a lock after failed logins)."""
    store = _store(config_path)
    password = _read_password(password_stdin)
    try:
        store.set_password(name, password)
    except UsersError as e:
        raise _fail(str(e)) from None
    typer.echo(f"password of {name!r} changed")


@users_app.command("disable")
def disable(
    name: Annotated[str, typer.Argument(help="Account to disable.")],
    config_path: ConfigOpt = None,
) -> None:
    """Stop an account from logging in (its projects stay). The last owner cannot be."""
    store = _store(config_path)
    try:
        store.disable(name)
    except UsersError as e:
        raise _fail(str(e)) from None
    typer.echo(f"{name!r} disabled")


@users_app.command("enable")
def enable(
    name: Annotated[str, typer.Argument(help="Account to enable again.")],
    config_path: ConfigOpt = None,
) -> None:
    """Allow a disabled account to log in again."""
    store = _store(config_path)
    try:
        store.enable(name)
    except UsersError as e:
        raise _fail(str(e)) from None
    typer.echo(f"{name!r} enabled")

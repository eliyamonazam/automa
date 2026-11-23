"""Tests for web UI accounts (D180): ``automa.users`` and ``automa users``.

No network and no model calls. scrypt uses the smallest allowed N (2**14) to keep
the suite fast; the stored-parameter checks are tested separately.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.users import (
    LOCKOUT,
    MAX_FAILED_ATTEMPTS,
    SCRYPT_N,
    LoginOutcome,
    PasswordHash,
    UserRole,
    UsersError,
    UserStore,
    check_password,
    hash_password,
    verify_password,
)

FAST_N = 2**14
PW = "correct horse 9"


class Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock: Clock) -> UserStore:
    return UserStore(tmp_path, now=clock, scrypt_n=FAST_N)


@pytest.fixture
def family(store: UserStore) -> UserStore:
    store.add("eli", PW, UserRole.OWNER)
    store.add("dad", "another pass 1", UserRole.MEMBER)
    return store


# -- hashing ------------------------------------------------------------------


def test_hash_roundtrip_and_random_salt() -> None:
    a = hash_password(PW, n=FAST_N)
    b = hash_password(PW, n=FAST_N)
    assert a.salt != b.salt and a.hash != b.hash
    assert verify_password(PW, a) and verify_password(PW, b)
    assert not verify_password(PW + "x", a)
    assert not verify_password("", a)


def test_default_parameters_are_strong() -> None:
    assert SCRYPT_N >= 2**15
    h = hash_password(PW)
    assert h.n == SCRYPT_N and h.r == 8 and h.p == 1


def test_unicode_password_is_normalised() -> None:
    composed = "رمزِ خوبِ من ۱۲۳"
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "café-pass-1")
    h = hash_password(decomposed, n=FAST_N)
    assert verify_password(unicodedata.normalize("NFC", decomposed), h)
    assert verify_password(composed, hash_password(composed, n=FAST_N))


@pytest.mark.parametrize(
    "bad",
    ["short", " leading-space-1", "trailing-space-1 ", "has\x00null-byte", "zw​space-1", "x" * 257],
)
def test_bad_passwords_are_refused(bad: str) -> None:
    with pytest.raises(UsersError):
        check_password(bad)


def test_overlong_password_never_verifies() -> None:
    assert not verify_password("x" * 10_000, hash_password(PW, n=FAST_N))


@pytest.mark.parametrize(
    "field,value",
    [
        ("n", 2**10),
        ("n", 3 * 2**14),
        ("n", 2**21),
        ("r", 1),
        ("p", 0),
        ("salt", "AAAA"),
        ("hash", "!!"),
    ],
)
def test_weak_or_malformed_stored_hash_refused(field: str, value: object) -> None:
    data = hash_password(PW, n=FAST_N).model_dump()
    data[field] = value
    with pytest.raises(ValueError):
        PasswordHash.model_validate(data)


# -- store --------------------------------------------------------------------


def test_no_file_means_no_users(store: UserStore) -> None:
    assert store.list() == [] and not store.has_users()
    assert not store.path.exists()


def test_add_writes_private_file_without_plaintext(family: UserStore) -> None:
    mode = stat.S_IMODE(os.stat(family.path).st_mode)
    assert mode == 0o600
    text = family.path.read_text("utf-8")
    assert PW not in text and "another pass 1" not in text
    data = json.loads(text)
    assert [u["name"] for u in data["users"]] == ["eli", "dad"]
    assert family.has_users()
    assert family.get("dad").role is UserRole.MEMBER  # type: ignore[union-attr]


def test_first_user_must_be_owner(store: UserStore) -> None:
    with pytest.raises(UsersError, match="owner first"):
        store.add("dad", PW)
    assert not store.path.exists()


@pytest.mark.parametrize("name", ["A", "a", "Eli", "1abc", "a b", "../x", "علی", "a" * 33, ""])
def test_bad_names_refused(store: UserStore, name: str) -> None:
    with pytest.raises(UsersError):
        store.add(name, PW, UserRole.OWNER)


def test_duplicate_name_refused(family: UserStore) -> None:
    with pytest.raises(UsersError, match="already exists"):
        family.add("dad", PW)


def test_last_owner_cannot_be_disabled(family: UserStore) -> None:
    with pytest.raises(UsersError, match="last active owner"):
        family.disable("eli")
    family.add("mom", PW, UserRole.OWNER)
    family.disable("eli")
    assert family.get("eli").disabled  # type: ignore[union-attr]
    with pytest.raises(UsersError, match="last active owner"):
        family.disable("mom")


def test_unknown_user_operations(family: UserStore) -> None:
    for op in (family.disable, family.enable):
        with pytest.raises(UsersError, match="no user"):
            op("ghost")
    with pytest.raises(UsersError, match="no user"):
        family.set_password("ghost", PW)


def test_symlinked_file_refused(tmp_path: Path, clock: Clock) -> None:
    real = tmp_path / "elsewhere.json"
    real.write_text('{"schema_version": 1, "users": []}')
    os.chmod(real, 0o600)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "users.json").symlink_to(real)
    with pytest.raises(UsersError, match="symlink"):
        UserStore(ws, now=clock).list()


def test_world_readable_file_refused(family: UserStore) -> None:
    os.chmod(family.path, 0o644)
    with pytest.raises(UsersError, match="chmod 600"):
        family.list()
    with pytest.raises(UsersError):
        family.authenticate("eli", PW)


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        b"\xff\xfe",
        b'{"schema_version": 2, "users": []}',
        b'{"schema_version": 1, "users": [], "extra": 1}',
    ],
)
def test_damaged_file_refused(store: UserStore, content: bytes) -> None:
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_bytes(content)
    os.chmod(store.path, 0o600)
    with pytest.raises(UsersError, match="damaged"):
        store.list()


def test_duplicate_names_in_file_refused(family: UserStore) -> None:
    data = json.loads(family.path.read_text())
    data["users"].append(data["users"][0])
    family.path.write_text(json.dumps(data))
    with pytest.raises(UsersError, match="damaged"):
        family.list()


def test_oversized_file_refused(store: UserStore) -> None:
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_bytes(b" " * (1024 * 1024 + 1))
    os.chmod(store.path, 0o600)
    with pytest.raises(UsersError, match="1 MiB"):
        store.list()


def test_directory_in_place_of_file_refused(store: UserStore) -> None:
    store.path.mkdir(parents=True)
    os.chmod(store.path, 0o700)
    with pytest.raises(UsersError):
        store.list()


def test_tampered_weak_hash_in_file_refused(family: UserStore) -> None:
    data = json.loads(family.path.read_text())
    data["users"][0]["password"]["n"] = 2
    family.path.write_text(json.dumps(data))
    with pytest.raises(UsersError, match="damaged"):
        family.authenticate("eli", PW)


# -- login and lockout --------------------------------------------------------


def test_login_ok(family: UserStore) -> None:
    r = family.authenticate("eli", PW)
    assert r.ok and r.user is not None and r.user.role is UserRole.OWNER


@pytest.mark.parametrize("name", ["ghost", "Eli", "../eli", "", "x" * 100])
def test_unknown_or_malformed_name_is_bad_credentials(family: UserStore, name: str) -> None:
    assert family.authenticate(name, PW).outcome is LoginOutcome.BAD_CREDENTIALS


def test_wrong_password(family: UserStore) -> None:
    r = family.authenticate("eli", "wrong password")
    assert r.outcome is LoginOutcome.BAD_CREDENTIALS and r.user is None
    assert family.get("eli").failed_attempts == 1  # type: ignore[union-attr]


def test_lockout_after_repeated_failures(family: UserStore, clock: Clock) -> None:
    for _ in range(MAX_FAILED_ATTEMPTS - 1):
        assert family.authenticate("eli", "nope nope").outcome is LoginOutcome.BAD_CREDENTIALS
    r = family.authenticate("eli", "nope nope")
    assert r.outcome is LoginOutcome.LOCKED and r.locked_until == clock.t + LOCKOUT
    # Even the right password is refused while locked (guessing cannot go on).
    assert family.authenticate("eli", PW).outcome is LoginOutcome.LOCKED
    clock.t += LOCKOUT - timedelta(seconds=1)
    assert family.authenticate("eli", PW).outcome is LoginOutcome.LOCKED
    clock.t += timedelta(seconds=2)
    assert family.authenticate("eli", PW).ok
    assert family.get("eli").failed_attempts == 0  # type: ignore[union-attr]
    assert family.get("eli").locked_until is None  # type: ignore[union-attr]


def test_after_expired_lock_count_restarts(family: UserStore, clock: Clock) -> None:
    for _ in range(MAX_FAILED_ATTEMPTS):
        family.authenticate("dad", "nope nope")
    clock.t += LOCKOUT + timedelta(seconds=1)
    r = family.authenticate("dad", "nope nope")
    assert r.outcome is LoginOutcome.BAD_CREDENTIALS
    assert family.get("dad").failed_attempts == 1  # type: ignore[union-attr]


def test_success_resets_count(family: UserStore) -> None:
    family.authenticate("eli", "nope nope")
    family.authenticate("eli", "nope nope")
    assert family.authenticate("eli", PW).ok
    for _ in range(MAX_FAILED_ATTEMPTS - 1):
        family.authenticate("eli", "nope nope")
    assert family.get("eli").locked_until is None  # type: ignore[union-attr]


def test_lock_of_one_user_does_not_affect_another(family: UserStore) -> None:
    for _ in range(MAX_FAILED_ATTEMPTS):
        family.authenticate("dad", "nope nope")
    assert family.authenticate("eli", PW).ok


def test_passwd_changes_password_and_lifts_lock(family: UserStore) -> None:
    for _ in range(MAX_FAILED_ATTEMPTS):
        family.authenticate("dad", "nope nope")
    family.set_password("dad", "brand new pass")
    assert family.authenticate("dad", "another pass 1").outcome is LoginOutcome.BAD_CREDENTIALS
    assert family.authenticate("dad", "brand new pass").ok


def test_passwd_refuses_weak_password_and_keeps_old(family: UserStore) -> None:
    with pytest.raises(UsersError):
        family.set_password("dad", "short")
    assert family.authenticate("dad", "another pass 1").ok


def test_disabled_user(family: UserStore) -> None:
    family.disable("dad")
    assert family.authenticate("dad", "another pass 1").outcome is LoginOutcome.DISABLED
    # A wrong password does not reveal that the account is disabled.
    assert family.authenticate("dad", "nope nope").outcome is LoginOutcome.BAD_CREDENTIALS
    family.enable("dad")
    assert family.authenticate("dad", "another pass 1").ok


# -- CLI ----------------------------------------------------------------------

runner = CliRunner()


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import automa.users as users_mod

    monkeypatch.setattr(users_mod, "SCRYPT_N", FAST_N)
    path = tmp_path / "automa.toml"
    path.write_text(f'workspace = "{tmp_path / "ws"}"\n')
    return path


def _invoke(cfg: Path, *args: str, stdin: str | None = None):  # type: ignore[no-untyped-def]
    return runner.invoke(cli.app, ["users", *args, "--config", str(cfg)], input=stdin)


def test_cli_full_cycle(cfg: Path, tmp_path: Path) -> None:
    r = _invoke(cfg, "list")
    assert r.exit_code == 0 and "loopback-only" in r.output

    r = _invoke(cfg, "add", "dad", "--password-stdin", stdin="another pass 1\n")
    assert r.exit_code == 1 and "owner first" in r.output

    r = _invoke(cfg, "add", "eli", "--role", "owner", "--password-stdin", stdin=PW + "\n")
    assert r.exit_code == 0, r.output
    r = _invoke(cfg, "add", "dad", "--password-stdin", stdin="another pass 1\n")
    assert r.exit_code == 0, r.output

    r = _invoke(cfg, "list")
    assert r.exit_code == 0
    assert "eli" in r.output and "owner" in r.output and "dad" in r.output
    assert "scrypt" not in r.output and PW not in r.output

    store = UserStore(tmp_path / "ws")
    assert store.authenticate("dad", "another pass 1").ok

    r = _invoke(cfg, "disable", "eli")
    assert r.exit_code == 1 and "last active owner" in r.output
    r = _invoke(cfg, "disable", "dad")
    assert r.exit_code == 0
    assert "disabled" in _invoke(cfg, "list").output
    r = _invoke(cfg, "enable", "dad")
    assert r.exit_code == 0

    r = _invoke(cfg, "passwd", "dad", "--password-stdin", stdin="brand new pass\n")
    assert r.exit_code == 0
    assert store.authenticate("dad", "brand new pass").ok


def test_cli_interactive_password_must_match(cfg: Path) -> None:
    r = _invoke(cfg, "add", "eli", "--role", "owner", stdin=f"{PW}\n{PW}x\n{PW}\n{PW}\n")
    assert r.exit_code == 0, r.output
    assert PW not in r.output  # hidden input is never echoed


def test_cli_password_is_not_an_argument(cfg: Path) -> None:
    r = _invoke(cfg, "add", "eli", PW)
    assert r.exit_code != 0


def test_cli_rejects_weak_password_and_empty_stdin(cfg: Path) -> None:
    r = _invoke(cfg, "add", "eli", "--role", "owner", "--password-stdin", stdin="short\n")
    assert r.exit_code == 1 and "at least" in r.output
    r = _invoke(cfg, "add", "eli", "--role", "owner", "--password-stdin", stdin="")
    assert r.exit_code == 1 and "no password" in r.output


def test_cli_unknown_user(cfg: Path) -> None:
    r = _invoke(cfg, "disable", "ghost")
    assert r.exit_code == 1 and "no user" in r.output


def test_concurrent_failures_are_all_counted(family: UserStore) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: family.authenticate("dad", "nope nope"), range(4)))
    assert family.get("dad").failed_attempts == 4  # type: ignore[union-attr]

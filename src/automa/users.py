"""User accounts for the web UI (M15, D180): password hashing, the users file and
the lockout after repeated failed logins.

Only the owner manages accounts, from the CLI (``automa users``); nothing here
talks to the network or a model. The file is ``<workspace>/users.json``:

* written atomically (temp file + ``os.replace``) with mode 0600, under an
  exclusive ``flock`` on ``users.json.lock`` for every read-modify-write;
* refused when it is a symlink, not a regular file, larger than 1 MiB, readable
  or writable by group/others, not UTF-8 or not the exact schema - never
  guessed or "repaired";
* passwords are stored only as ``hashlib.scrypt`` hashes with a random 16-byte
  salt per password; the parameters are stored with the hash and must not be
  weaker than the defaults.

With no users file (or an empty one) nothing changes for the web UI: it keeps
its loopback-only behaviour (D78); with an account every page needs a sign-in
(:mod:`automa.webui.auth`, D181).
"""

from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import tempfile
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

USERS_FILE = "users.json"
LOCK_FILE = "users.json.lock"
MAX_FILE_BYTES = 1024 * 1024
MAX_USERS = 50

NAME_PATTERN = r"^[a-z][a-z0-9_-]{1,31}$"
MIN_PASSWORD_CHARS = 8
MAX_PASSWORD_CHARS = 256

# scrypt: N=2**15, r=8, p=1 needs ~32 MiB and is well above OWASP's minimum
# (N=2**17 is the 2023 recommendation for r=8 only with p=1 and enough RAM; on a
# family laptop under WSL 2**15 keeps a login around 0.1 s).
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16
_MIN_N = 2**14
_MAX_N = 2**20
_MAXMEM = 256 * 1024 * 1024

MAX_FAILED_ATTEMPTS = 5
LOCKOUT = timedelta(minutes=15)


class UsersError(Exception):
    """The users file cannot be used, or an account operation was refused."""


class UserRole(StrEnum):
    OWNER = "owner"
    MEMBER = "member"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str, size: int) -> bytes:
    try:
        raw = base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError):
        raise ValueError("not base64") from None
    if len(raw) != size:
        raise ValueError(f"expected {size} bytes")
    return raw


class PasswordHash(BaseModel):
    """One scrypt hash with its parameters; weaker-than-default values are refused."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algo: Literal["scrypt"] = "scrypt"
    n: int
    r: Annotated[int, Field(ge=SCRYPT_R, le=16)]
    p: Annotated[int, Field(ge=1, le=4)]
    salt: str
    hash: str

    @field_validator("n")
    @classmethod
    def _n(cls, v: int) -> int:
        if v < _MIN_N or v > _MAX_N or v & (v - 1):
            raise ValueError("n must be a power of two between 2**14 and 2**20")
        return v

    @field_validator("salt")
    @classmethod
    def _salt(cls, v: str) -> str:
        _unb64(v, SALT_BYTES)
        return v

    @field_validator("hash")
    @classmethod
    def _hash(cls, v: str) -> str:
        _unb64(v, SCRYPT_DKLEN)
        return v


def check_password(password: str) -> str:
    """The password, normalised (NFC), or ``UsersError`` saying what is wrong."""
    if not isinstance(password, str):
        raise UsersError("the password must be text")
    pw = unicodedata.normalize("NFC", password)
    if len(pw) < MIN_PASSWORD_CHARS:
        raise UsersError(f"the password must be at least {MIN_PASSWORD_CHARS} characters")
    if len(pw) > MAX_PASSWORD_CHARS:
        raise UsersError(f"the password must be at most {MAX_PASSWORD_CHARS} characters")
    if any(unicodedata.category(c) in ("Cc", "Cf") for c in pw):
        raise UsersError("the password must not contain control or invisible characters")
    if pw.strip() != pw:
        raise UsersError("the password must not start or end with a space")
    return pw


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        unicodedata.normalize("NFC", password).encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=SCRYPT_DKLEN,
        maxmem=_MAXMEM,
    )


def hash_password(password: str, *, n: int | None = None) -> PasswordHash:
    """A fresh scrypt hash with a random salt (the password is checked first)."""
    pw = check_password(password)
    n = SCRYPT_N if n is None else n
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(pw, salt, n, SCRYPT_R, SCRYPT_P)
    return PasswordHash(n=n, r=SCRYPT_R, p=SCRYPT_P, salt=_b64(salt), hash=_b64(digest))


def verify_password(password: str, stored: PasswordHash) -> bool:
    """Constant-time comparison; any password that could never be set is False."""
    if not isinstance(password, str) or len(password) > MAX_PASSWORD_CHARS:
        return False
    digest = _derive(password, _unb64(stored.salt, SALT_BYTES), stored.n, stored.r, stored.p)
    return hmac.compare_digest(digest, _unb64(stored.hash, SCRYPT_DKLEN))


# Spent on unknown names so that "no such user" takes as long as "wrong password".
_DUMMY_HASH = PasswordHash(
    n=SCRYPT_N,
    r=SCRYPT_R,
    p=SCRYPT_P,
    salt=_b64(b"\0" * SALT_BYTES),
    hash=_b64(b"\0" * SCRYPT_DKLEN),
)


class User(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, Field(pattern=NAME_PATTERN)]
    role: UserRole
    password: PasswordHash
    created_at: datetime
    password_changed_at: datetime
    disabled: bool = False
    failed_attempts: Annotated[int, Field(ge=0, le=1000)] = 0
    locked_until: datetime | None = None
    # D191: when this account dismissed the three-step first-visit card of the web UI;
    # ``None`` (also every account made before D191): not yet shown or not dismissed.
    welcomed_at: datetime | None = None

    @field_validator("created_at", "password_changed_at", "locked_until", "welcomed_at")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("timestamps must carry a time zone")
        return v

    def locked(self, now: datetime) -> bool:
        return self.locked_until is not None and now < self.locked_until


class UsersFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    users: Annotated[list[User], Field(max_length=MAX_USERS)] = Field(default_factory=list)

    @field_validator("users")
    @classmethod
    def _unique(cls, v: list[User]) -> list[User]:
        names = [u.name for u in v]
        if len(set(names)) != len(names):
            raise ValueError("duplicate user names")
        return v


class LoginOutcome(StrEnum):
    OK = "ok"
    BAD_CREDENTIALS = "bad_credentials"  # unknown name or wrong password
    LOCKED = "locked"
    DISABLED = "disabled"


class LoginResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: LoginOutcome
    user: User | None = None
    locked_until: datetime | None = None

    @property
    def ok(self) -> bool:
        return self.outcome is LoginOutcome.OK


def check_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(NAME_PATTERN, name):
        raise UsersError(
            "a user name is 2-32 characters: lowercase English letters, digits, '-' or '_', "
            "starting with a letter"
        )
    return name


class UserStore:
    """``<workspace>/users.json``; every change happens under one file lock."""

    def __init__(
        self,
        workspace: Path,
        *,
        now: Callable[[], datetime] = _utc_now,
        scrypt_n: int | None = None,
    ) -> None:
        self._dir = Path(workspace)
        self._now = now
        self._n = scrypt_n

    @property
    def path(self) -> Path:
        return self._dir / USERS_FILE

    # -- file handling -------------------------------------------------------

    def _read(self) -> UsersFile:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(self.path, flags)
        except FileNotFoundError:
            return UsersFile()
        except OSError as e:
            raise UsersError(f"cannot open {USERS_FILE} (a symlink is refused): {e}") from None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise UsersError(f"{USERS_FILE} is not a regular file")
            if st.st_mode & 0o077:
                raise UsersError(
                    f"{USERS_FILE} is readable by other users of this computer; "
                    f"fix it with: chmod 600 {self.path}"
                )
            if st.st_size > MAX_FILE_BYTES:
                raise UsersError(f"{USERS_FILE} is larger than 1 MiB")
            chunks: list[bytes] = []
            total = 0
            while chunk := os.read(fd, 65536):
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise UsersError(f"{USERS_FILE} is larger than 1 MiB")
                chunks.append(chunk)
        finally:
            os.close(fd)
        try:
            data = json.loads(b"".join(chunks).decode("utf-8"))
            return UsersFile.model_validate(data)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as e:
            raise UsersError(f"{USERS_FILE} is damaged and was not used: {e}") from None

    def _write(self, data: UsersFile) -> None:
        text = data.model_dump_json(indent=2) + "\n"
        fd, tmp = tempfile.mkstemp(prefix=f".{USERS_FILE}.", suffix=".tmp", dir=self._dir)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._dir.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(self._dir / LOCK_FILE, flags, 0o600)
        except OSError as e:
            raise UsersError(f"cannot open {LOCK_FILE}: {e}") from None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    # -- queries -------------------------------------------------------------

    def list(self) -> list[User]:
        return list(self._read().users)

    def get(self, name: str) -> User | None:
        return next((u for u in self._read().users if u.name == name), None)

    def has_users(self) -> bool:
        """True when at least one account exists (the web UI then needs a login)."""
        return bool(self._read().users)

    # -- changes (owner, from the CLI) ---------------------------------------

    def add(self, name: str, password: str, role: UserRole = UserRole.MEMBER) -> User:
        check_name(name)
        role = UserRole(role)
        pw_hash = hash_password(password, n=self._n)
        with self._locked():
            data = self._read()
            if any(u.name == name for u in data.users):
                raise UsersError(f"user {name!r} already exists")
            if len(data.users) >= MAX_USERS:
                raise UsersError(f"at most {MAX_USERS} users")
            if role is UserRole.MEMBER and not _active_owners(data.users):
                raise UsersError("add an owner first: automa users add <name> --role owner")
            now = self._now()
            user = User(
                name=name, role=role, password=pw_hash, created_at=now, password_changed_at=now
            )
            data.users.append(user)
            self._write(data)
            return user

    def set_password(self, name: str, password: str) -> User:
        """New password; also clears the failed-attempt count and any lock."""
        pw_hash = hash_password(password, n=self._n)
        return self._update(
            name,
            password=pw_hash,
            password_changed_at=self._now(),
            failed_attempts=0,
            locked_until=None,
        )

    def disable(self, name: str) -> User:
        with self._locked():
            data = self._read()
            user = _find(data, name)
            if (
                user.role is UserRole.OWNER
                and not user.disabled
                and len(_active_owners(data.users)) == 1
            ):
                raise UsersError("the last active owner cannot be disabled")
            return self._replace(data, user, disabled=True)

    def enable(self, name: str) -> User:
        return self._update(name, disabled=False, failed_attempts=0, locked_until=None)

    # -- web UI (the account itself) -----------------------------------------

    def mark_welcomed(self, name: str) -> User:
        """Record that ``name`` dismissed the first-visit card (D191).

        Idempotent: the first time is kept. Touches nothing that ends a session
        (role, password, disabled), so the browser stays signed in."""
        with self._locked():
            data = self._read()
            user = _find(data, name)
            if user.welcomed_at is not None:
                return user
            return self._replace(data, user, welcomed_at=self._now())

    # -- login ---------------------------------------------------------------

    def authenticate(self, name: str, password: str) -> LoginResult:
        """Check a login and record the outcome.

        Unknown names cost one scrypt too and always answer ``BAD_CREDENTIALS``.
        ``MAX_FAILED_ATTEMPTS`` wrong passwords in a row lock the account for
        ``LOCKOUT``; while locked the password is not even checked (so guessing
        cannot continue), and a correct login resets the count.
        """
        if not isinstance(name, str) or not re.fullmatch(NAME_PATTERN, name):
            verify_password(password if isinstance(password, str) else "", _DUMMY_HASH)
            return LoginResult(outcome=LoginOutcome.BAD_CREDENTIALS)
        with self._locked():
            data = self._read()
            user = next((u for u in data.users if u.name == name), None)
            now = self._now()
            if user is None:
                verify_password(password, _DUMMY_HASH)
                return LoginResult(outcome=LoginOutcome.BAD_CREDENTIALS)
            if user.locked(now):
                return LoginResult(outcome=LoginOutcome.LOCKED, locked_until=user.locked_until)
            good = verify_password(password, user.password)
            if user.disabled:
                # Only someone who knows the password learns the account is disabled.
                outcome = LoginOutcome.DISABLED if good else LoginOutcome.BAD_CREDENTIALS
                return LoginResult(outcome=outcome)
            if good:
                if user.failed_attempts or user.locked_until is not None:
                    user = self._replace(data, user, failed_attempts=0, locked_until=None)
                return LoginResult(outcome=LoginOutcome.OK, user=user)
            # An expired lock starts a fresh count.
            failed = 1 if user.locked_until is not None else user.failed_attempts + 1
            if failed >= MAX_FAILED_ATTEMPTS:
                until = now + LOCKOUT
                self._replace(data, user, failed_attempts=failed, locked_until=until)
                return LoginResult(outcome=LoginOutcome.LOCKED, locked_until=until)
            self._replace(data, user, failed_attempts=failed, locked_until=None)
            return LoginResult(outcome=LoginOutcome.BAD_CREDENTIALS)

    # -- helpers -------------------------------------------------------------

    def _update(self, name: str, **changes: object) -> User:
        with self._locked():
            data = self._read()
            return self._replace(data, _find(data, name), **changes)

    def _replace(self, data: UsersFile, user: User, **changes: object) -> User:
        new = User.model_validate({**user.model_dump(), **changes})
        data.users = [new if u.name == user.name else u for u in data.users]
        self._write(data)
        return new


def _find(data: UsersFile, name: str) -> User:
    user = next((u for u in data.users if u.name == name), None)
    if user is None:
        raise UsersError(f"no user named {name!r}")
    return user


def _active_owners(users: list[User]) -> list[User]:
    return [u for u in users if u.role is UserRole.OWNER and not u.disabled]

"""Login for the web UI (M15, D181): sessions, the session cookie and the login limiter.

Only active when ``<workspace>/users.json`` has at least one account (D180); with no
account the web UI keeps its loopback-only behaviour unchanged.

* A session is a random 256-bit token (``secrets.token_urlsafe(32)``) kept only in
  this server process, keyed by its SHA-256 (the raw token never sits in a dict a
  timing side channel could probe). A server restart logs everyone out.
* The cookie is ``HttpOnly; Secure; SameSite=Strict; Path=/`` with ``Max-Age``.
  Browsers accept ``Secure`` cookies on ``http://127.0.0.1``/``localhost`` (a
  potentially trustworthy origin) and the remote access of M15 is HTTPS.
* A session ends after ``SESSION_IDLE`` without use, ``SESSION_MAX`` after login, on
  logout, or as soon as its account is disabled, removed, changes role or gets a new
  password (checked against ``users.json`` on every request).
* Each session has its own CSRF token for every form of the page; the login form has
  a per-server token. Both are compared in constant time.
* :class:`LoginLimiter` refuses a client after ``MAX_CLIENT_FAILURES`` failed logins
  within ``CLIENT_WINDOW`` and everyone after ``MAX_ATTEMPTS_PER_MINUTE`` attempts in a
  minute (each attempt costs one scrypt); the per-account lock of D180 still applies.

Pure apart from the clock: no files (the accounts are :class:`automa.users.UserStore`),
no network, no model.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

from automa.users import User, UserRole, UserStore

SESSION_COOKIE = "automa_session"
LOGIN_PATH = "/login"
LOGOUT_PATH = "/logout"
AUTH_PATHS = frozenset({LOGIN_PATH, LOGOUT_PATH})
LOGIN_FIELDS = frozenset({"csrf", "name", "password"})
LOGOUT_FIELDS = frozenset({"csrf"})

SESSION_IDLE = timedelta(days=7)
SESSION_MAX = timedelta(days=30)
MAX_SESSIONS = 200

MAX_CLIENT_FAILURES = 10
CLIENT_WINDOW = timedelta(minutes=15)
MAX_ATTEMPTS_PER_MINUTE = 60
MAX_TRACKED_CLIENTS = 1000

_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43}")
MAX_COOKIE_HEADER = 8192


def _utc_now() -> datetime:
    return datetime.now(UTC)


def new_token() -> str:
    """A fresh random token (43 URL-safe characters, 256 bits)."""
    return secrets.token_urlsafe(32)


def _key(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def tokens_equal(given: str, expected: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


@dataclass(frozen=True)
class Session:
    """One logged-in browser."""

    user: str
    role: UserRole
    csrf_token: str
    created_at: datetime
    last_seen: datetime
    # The account's password time at login: a new password ends the session.
    password_changed_at: datetime

    def expired(self, now: datetime) -> bool:
        return now >= self.created_at + SESSION_MAX or now >= self.last_seen + SESSION_IDLE

    def matches(self, user: User | None) -> bool:
        """Whether the account still allows this session."""
        return (
            user is not None
            and not user.disabled
            and user.role is self.role
            and user.password_changed_at == self.password_changed_at
        )


class SessionStore:
    """The sessions of this server process (thread-safe, at most ``MAX_SESSIONS``)."""

    def __init__(self, *, now: Callable[[], datetime] = _utc_now) -> None:
        self._now = now
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def create(self, user: User) -> tuple[str, Session]:
        """A new session (and its cookie token) for a just-authenticated ``user``."""
        now = self._now()
        token = new_token()
        session = Session(
            user=user.name,
            role=user.role,
            csrf_token=new_token(),
            created_at=now,
            last_seen=now,
            password_changed_at=user.password_changed_at,
        )
        with self._lock:
            self._purge(now)
            while len(self._sessions) >= MAX_SESSIONS:
                self._sessions.popitem(last=False)  # the least recently used
            self._sessions[_key(token)] = session
        return token, session

    def get(self, token: str | None) -> Session | None:
        """The live session of ``token`` (its idle time restarts), else ``None``."""
        if token is None or not _TOKEN_RE.fullmatch(token):
            return None
        now = self._now()
        key = _key(token)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return None
            if session.expired(now):
                del self._sessions[key]
                return None
            session = replace(session, last_seen=now)
            self._sessions[key] = session
            self._sessions.move_to_end(key)
            return session

    def peek(self, token: str | None) -> Session | None:
        """Like :meth:`get` but without counting as use (body-size checks)."""
        if token is None or not _TOKEN_RE.fullmatch(token):
            return None
        now = self._now()
        with self._lock:
            session = self._sessions.get(_key(token))
            return None if session is None or session.expired(now) else session

    def drop(self, token: str | None) -> None:
        if token is None or not _TOKEN_RE.fullmatch(token):
            return
        with self._lock:
            self._sessions.pop(_key(token), None)

    def _purge(self, now: datetime) -> None:
        for key in [k for k, s in self._sessions.items() if s.expired(now)]:
            del self._sessions[key]


def cookie_token(headers: Sequence[tuple[str, str]]) -> str | None:
    """The session token from the ``Cookie`` header(s).

    ``None`` unless exactly one well-formed ``automa_session`` value is present: two
    copies (e.g. one planted for a parent domain) are ambiguous and count as none."""
    found: list[str] = []
    total = 0
    for name, value in headers:
        if name.lower() != "cookie":
            continue
        total += len(value)
        if total > MAX_COOKIE_HEADER:
            return None
        for part in value.split(";"):
            key, sep, val = part.strip().partition("=")
            if sep and key.strip() == SESSION_COOKIE:
                found.append(val.strip())
    if len(found) != 1 or not _TOKEN_RE.fullmatch(found[0]):
        return None
    return found[0]


_COOKIE_FLAGS = "Path=/; HttpOnly; Secure; SameSite=Strict"


def session_cookie(token: str) -> str:
    """The ``Set-Cookie`` value of a new session."""
    if not _TOKEN_RE.fullmatch(token):
        raise ValueError("invalid session token")
    return f"{SESSION_COOKIE}={token}; Max-Age={int(SESSION_MAX.total_seconds())}; {_COOKIE_FLAGS}"


def clear_cookie() -> str:
    """The ``Set-Cookie`` value that removes the session cookie."""
    return f"{SESSION_COOKIE}=; Max-Age=0; {_COOKIE_FLAGS}"


@dataclass
class _Client:
    failures: deque[datetime] = field(default_factory=deque)


class LoginLimiter:
    """Failed-login limits per client and a global cap on login attempts."""

    def __init__(self, *, now: Callable[[], datetime] = _utc_now) -> None:
        self._now = now
        self._lock = threading.Lock()
        self._clients: OrderedDict[str, _Client] = OrderedDict()
        self._attempts: deque[datetime] = deque()

    def retry_after(self, client: str) -> int | None:
        """Seconds to wait before ``client`` may try again, or ``None`` (allowed).

        An allowed call counts as one attempt towards the global cap."""
        now = self._now()
        with self._lock:
            while self._attempts and now - self._attempts[0] >= timedelta(minutes=1):
                self._attempts.popleft()
            entry = self._clients.get(client)
            if entry is not None:
                while entry.failures and now - entry.failures[0] >= CLIENT_WINDOW:
                    entry.failures.popleft()
                if len(entry.failures) >= MAX_CLIENT_FAILURES:
                    wait = entry.failures[0] + CLIENT_WINDOW - now
                    return max(1, int(wait.total_seconds()) + 1)
            if len(self._attempts) >= MAX_ATTEMPTS_PER_MINUTE:
                wait = self._attempts[0] + timedelta(minutes=1) - now
                return max(1, int(wait.total_seconds()) + 1)
            self._attempts.append(now)
            return None

    def failed(self, client: str) -> None:
        now = self._now()
        with self._lock:
            entry = self._clients.pop(client, None) or _Client()
            entry.failures.append(now)
            self._clients[client] = entry  # most recent last
            while len(self._clients) > MAX_TRACKED_CLIENTS:
                self._clients.popitem(last=False)

    def succeeded(self, client: str) -> None:
        with self._lock:
            self._clients.pop(client, None)


@dataclass(frozen=True)
class AuthSettings:
    """What the login of one server needs."""

    users: UserStore
    origins: frozenset[str]
    login_token: str = field(default_factory=new_token)
    sessions: SessionStore = field(default_factory=SessionStore)
    limiter: LoginLimiter = field(default_factory=LoginLimiter)
    now: Callable[[], datetime] = _utc_now


def normalize_login_name(raw: str) -> str:
    """Phones capitalise the first letter: names are lower-case anyway (D180)."""
    return raw.strip().lower()


def minutes_left(until: datetime | None, now: datetime) -> int:
    """Whole minutes (at least 1) until ``until``."""
    if until is None:
        return 1
    return max(1, -(-int((until - now).total_seconds()) // 60))

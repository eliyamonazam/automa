"""``CostLedger``: durable SQLite record of every model call and its cost.

Money is stored as integer nano-USD, always rounded *up*, so summed spending
can never be lower than the real spending because of float rounding. Days are
UTC calendar days. The ledger never swallows a write error: if a call cannot be
recorded, the caller sees the exception (the budget layer must fail closed).
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from automa.config import Tier
from automa.llm.gateway import CallEstimate
from automa.llm.types import LLMResult

SCHEMA_VERSION = 2
# Version 2 (D183) added the ``user`` column (the web UI account a project belongs
# to); version-1 files are migrated in place with ``ALTER TABLE ... ADD COLUMN``.
NANO_PER_USD = 1_000_000_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT    NOT NULL,
    day                TEXT    NOT NULL,
    project_id         TEXT,
    task_id            TEXT,
    tier               TEXT    NOT NULL,
    model              TEXT    NOT NULL,
    response_model     TEXT    NOT NULL,
    input_tokens       INTEGER NOT NULL CHECK (input_tokens >= 0),
    output_tokens      INTEGER NOT NULL CHECK (output_tokens >= 0),
    cost_nusd          INTEGER NOT NULL CHECK (cost_nusd >= 0),
    estimate_nusd      INTEGER NOT NULL CHECK (estimate_nusd >= 0),
    reported_cost_nusd INTEGER CHECK (reported_cost_nusd IS NULL OR reported_cost_nusd >= 0),
    user               TEXT
);
CREATE INDEX IF NOT EXISTS calls_project ON calls (project_id);
CREATE INDEX IF NOT EXISTS calls_day ON calls (day);
CREATE INDEX IF NOT EXISTS calls_user ON calls (user);
"""

_MIGRATE_V1 = """
ALTER TABLE calls ADD COLUMN user TEXT;
CREATE INDEX IF NOT EXISTS calls_user ON calls (user);
"""

USER_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
MONTH_PATTERN = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")

Clock = Callable[[], datetime]


class LedgerError(Exception):
    """The ledger database is unusable (unknown schema, invalid input)."""


def usd_to_nusd(usd: float) -> int:
    """Convert USD to integer nano-USD, rounding up (never under-count)."""
    if not math.isfinite(usd) or usd < 0:
        raise LedgerError("cost must be a finite, non-negative number")
    return math.ceil(usd * NANO_PER_USD)


def nusd_to_usd(nusd: int) -> float:
    return nusd / NANO_PER_USD


def _utc_now() -> datetime:
    return datetime.now(UTC)


class LedgerEntry(BaseModel):
    """One recorded call."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    id: int
    ts: datetime
    project_id: str | None
    task_id: str | None
    tier: Tier
    model: str
    response_model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_nusd: int = Field(ge=0)
    estimate_nusd: int = Field(ge=0)
    reported_cost_nusd: int | None = Field(default=None, ge=0)
    user: str | None = Field(default=None, pattern=USER_PATTERN.pattern)

    @property
    def cost_usd(self) -> float:
        return nusd_to_usd(self.cost_nusd)


_COLUMNS = (
    "id, ts, project_id, task_id, tier, model, response_model, input_tokens, "
    "output_tokens, cost_nusd, estimate_nusd, reported_cost_nusd, user"
)


class CostLedger:
    """Append-only ledger of model calls backed by SQLite.

    ``path`` may be ``":memory:"`` (tests). ``record`` matches the ``record`` half
    of ``CallGuard`` so the budget guard can delegate to it.

    With ``read_only=True`` (for advisory readers such as ``automa estimate``, D63)
    the file must already exist and is opened with SQLite ``mode=ro``: nothing is
    created, migrated or written, and ``record`` raises. A read-only ledger is never
    given to the ``BudgetGuard``.
    """

    def __init__(self, path: Path | str, *, clock: Clock = _utc_now, read_only: bool = False):
        self._path = str(path)
        self._read_only = read_only
        self._empty = False  # read-only open of a file without the schema yet
        self._has_user = True  # False only for a read-only open of a version-1 file
        target, uri = self._path, False
        if self._path == ":memory:":
            if read_only:
                raise LedgerError("an in-memory ledger cannot be opened read-only")
        elif read_only:
            p = Path(self._path)
            if p.is_symlink():
                raise LedgerError(f"{p}: refusing to open a symlinked ledger")
            if not p.is_file():
                raise LedgerError(f"{p}: no ledger to read")
            target, uri = p.resolve().as_uri() + "?mode=ro", True
        else:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        # One connection shared across threads, serialized by ``_lock``.
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(target, check_same_thread=False, uri=uri)
        except sqlite3.Error as e:
            if not read_only:
                raise
            raise LedgerError(f"cannot open ledger {self._path}: {e}") from e
        try:
            self._init_schema()
        except sqlite3.Error as e:
            self._conn.close()
            if not read_only:
                raise
            raise LedgerError(f"{self._path}: not a usable ledger ({e})") from e
        except BaseException:
            self._conn.close()
            raise

    @property
    def read_only(self) -> bool:
        return self._read_only

    @classmethod
    def for_workspace(cls, workspace: Path, *, clock: Clock = _utc_now) -> CostLedger:
        """Open the ledger at ``<workspace>/ledger.sqlite3``."""
        return cls(workspace / "ledger.sqlite3", clock=clock)

    def _init_schema(self) -> None:
        (version,) = self._conn.execute("PRAGMA user_version").fetchone()
        if version not in (0, 1, SCHEMA_VERSION):
            raise LedgerError(f"unsupported ledger schema version {version}")
        if version == 0:
            tables = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if tables:
                raise LedgerError("database is not an Automa ledger (unexpected tables)")
        if self._read_only:
            self._empty = version == 0
            # A version-1 file read without migrating: there is no ``user`` column.
            self._has_user = version != 1
            return
        with self._conn:
            if version == 1:
                self._conn.executescript(_MIGRATE_V1)
            else:
                self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> CostLedger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise LedgerError("clock must return a timezone-aware datetime")
        return now.astimezone(UTC)

    def now(self) -> datetime:
        """Current UTC time according to the ledger clock."""
        return self._now()

    def today(self) -> date:
        """Current UTC day according to the ledger clock."""
        return self._now().date()

    def record(
        self, estimate: CallEstimate, result: LLMResult, *, user: str | None = None
    ) -> LedgerEntry:
        """Persist one completed call. Raises on any failure (fail closed).

        ``user`` (D183) is the account the project belongs to, or ``None``.
        """
        if self._read_only:
            raise LedgerError("ledger is open read-only")
        if user is not None and not USER_PATTERN.fullmatch(user):
            raise LedgerError("invalid user name")
        if result.tier is not estimate.tier:
            raise LedgerError("result tier does not match estimate tier")
        now = self._now()
        usage = result.response.usage
        reported = (
            usd_to_nusd(usage.reported_cost_usd) if usage.reported_cost_usd is not None else None
        )
        row = (
            now.isoformat(),
            now.date().isoformat(),
            estimate.context.project_id,
            estimate.context.task_id,
            estimate.tier.value,
            estimate.model,
            result.response.model,
            usage.input_tokens,
            usage.output_tokens,
            usd_to_nusd(result.cost_usd),
            usd_to_nusd(estimate.max_cost_usd),
            reported,
            user,
        )
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO calls (ts, day, project_id, task_id, tier, model, response_model,"
                " input_tokens, output_tokens, cost_nusd, estimate_nusd, reported_cost_nusd,"
                " user) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        entry_id = cur.lastrowid
        assert entry_id is not None  # noqa: S101 - sqlite always sets it after INSERT
        return self.get(entry_id)

    def get(self, entry_id: int) -> LedgerEntry:
        if self._empty:
            raise KeyError(entry_id)
        with self._lock:
            row = self._conn.execute(
                f"SELECT {self._columns()} FROM calls WHERE id = ?",  # noqa: S608 - constant
                (entry_id,),
            ).fetchone()
        if row is None:
            raise KeyError(entry_id)
        return _row_to_entry(row)

    def _columns(self) -> str:
        return _COLUMNS if self._has_user else _COLUMNS.replace(", user", ", NULL")

    def _user_filter_ok(self, user: str | None) -> bool:
        """False when a user filter can match nothing (version-1 file read as is)."""
        if user is not None and not USER_PATTERN.fullmatch(user):
            raise LedgerError("invalid user name")
        return user is None or self._has_user

    def entries(
        self,
        *,
        project_id: str | None = None,
        day: date | None = None,
        user: str | None = None,
        month: str | None = None,
    ) -> list[LedgerEntry]:
        """All entries in insertion order, optionally filtered by project, UTC day,
        user and/or UTC month (``YYYY-MM``)."""
        if self._empty or not self._user_filter_ok(user):
            return []
        where, params = _filters(project_id, day, None, user, month)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._columns()} FROM calls{where} ORDER BY id",  # noqa: S608
                params,
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def spent_nusd(
        self,
        *,
        project_id: str | None = None,
        day: date | None = None,
        task_id: str | None = None,
        user: str | None = None,
        month: str | None = None,
    ) -> int:
        """Exact total spending in nano-USD for the given filters.

        ``month`` is a UTC calendar month ``YYYY-MM`` (D183, the per-user cap).
        """
        if self._empty or not self._user_filter_ok(user):
            return 0
        where, params = _filters(project_id, day, task_id, user, month)
        with self._lock:
            (total,) = self._conn.execute(
                f"SELECT COALESCE(SUM(cost_nusd), 0) FROM calls{where}",  # noqa: S608
                params,
            ).fetchone()
        return int(total)

    def total_cost(self) -> float:
        return nusd_to_usd(self.spent_nusd())

    def project_cost(self, project_id: str) -> float:
        return nusd_to_usd(self.spent_nusd(project_id=project_id))

    def daily_cost(self, day: date | None = None) -> float:
        """Spending on UTC ``day`` (default: today according to the ledger clock)."""
        return nusd_to_usd(self.spent_nusd(day=day or self.today()))


def month_of(day: date) -> str:
    """The ``YYYY-MM`` key of the UTC calendar month containing ``day``."""
    return f"{day.year:04d}-{day.month:02d}"


def _filters(
    project_id: str | None,
    day: date | None,
    task_id: str | None = None,
    user: str | None = None,
    month: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    clauses: list[str] = []
    params: list[str] = []
    if month is not None:
        if not MONTH_PATTERN.fullmatch(month):
            raise LedgerError("month must be YYYY-MM")
        # ``day`` is always ``YYYY-MM-DD``: a fixed-width prefix comparison.
        clauses.append("substr(day, 1, 7) = ?")
        params.append(month)
    if user is not None:
        clauses.append("user = ?")
        params.append(user)
    if task_id is not None:
        clauses.append("task_id = ?")
        params.append(task_id)
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if day is not None:
        clauses.append("day = ?")
        params.append(day.isoformat())
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, tuple(params)


def _row_to_entry(row: tuple[object, ...]) -> LedgerEntry:
    keys = [c.strip() for c in _COLUMNS.split(",")]
    data = dict(zip(keys, row, strict=True))
    return LedgerEntry.model_validate(data)

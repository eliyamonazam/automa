"""``QualityLog``: verifier outcome next to the cost of every judged attempt (D56).

One row per judged attempt ``(project, task, attempt)``: the task type (its role),
the tier and model the worker used, the worker's result status, the verifier
outcome and the worker cost of that attempt. The data router (M8) reads the
aggregated ``stats`` to pick the cheapest model with a good enough pass rate.

The log is *derived* data: the source of truth stays the write-once
``TaskResult``/``Verdict`` files of the ``ProjectStore``. So

* ``record`` is idempotent: recording the same attempt again is a no-op, a
  *different* row for an attempt already recorded is a ``QualityError`` (verdicts
  are write-once, so that means a corrupt log or a bug);
* ``sync_project`` (re)imports every judged attempt of a project, so a row that
  could not be written during a run is never lost for good.

The log never authorizes or blocks anything: it is not part of the budget layer
or of the permission policy.

Storage: ``<workspace>/quality.sqlite3``, money as integer nano-USD (as in the
ledger), schema version in ``user_version``; unknown schema or foreign tables
fail closed.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from automa.config import Tier
from automa.plan.models import (
    ROLE_PATTERN,
    TASK_ID_PATTERN,
    ResultStatus,
    Task,
    TaskResult,
    UtcDatetime,
    Verdict,
    VerdictOutcome,
)
from automa.project.store import PROJECT_ID_PATTERN, ProjectStore

SCHEMA_VERSION = 1
QUALITY_FILE = "quality.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    project_id      TEXT    NOT NULL,
    task_id         TEXT    NOT NULL,
    attempt         INTEGER NOT NULL CHECK (attempt >= 1),
    role            TEXT    NOT NULL,
    tier            TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    result_status   TEXT    NOT NULL,
    outcome         TEXT    NOT NULL CHECK (outcome IN ('pass', 'fail')),
    model_evaluated INTEGER NOT NULL CHECK (model_evaluated IN (0, 1)),
    cost_nusd       INTEGER NOT NULL CHECK (cost_nusd >= 0),
    judged_at       TEXT    NOT NULL,
    PRIMARY KEY (project_id, task_id, attempt)
);
CREATE INDEX IF NOT EXISTS attempts_role_model ON attempts (role, model);
"""

_COLUMNS = (
    "project_id",
    "task_id",
    "attempt",
    "role",
    "tier",
    "model",
    "result_status",
    "outcome",
    "model_evaluated",
    "cost_nusd",
    "judged_at",
)


class QualityError(Exception):
    """The quality log is unusable, or a row contradicts one already recorded."""


class QualityRecord(BaseModel):
    """One judged attempt: what was tried, by which model, what it cost, how it went."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    project_id: str = Field(pattern=PROJECT_ID_PATTERN)
    task_id: str = Field(pattern=TASK_ID_PATTERN)
    attempt: int = Field(ge=1)
    role: str = Field(pattern=ROLE_PATTERN)  # the task type
    tier: Tier
    model: str = Field(min_length=1, max_length=200)
    result_status: ResultStatus
    outcome: VerdictOutcome
    model_evaluated: bool
    cost_nusd: int = Field(ge=0)  # the worker's cost of this attempt
    judged_at: UtcDatetime

    @property
    def passed(self) -> bool:
        return self.outcome is VerdictOutcome.PASS

    @classmethod
    def from_attempt(
        cls, project_id: str, task: Task, result: TaskResult, verdict: Verdict
    ) -> QualityRecord:
        """Build the row of one judged attempt; the three must belong together."""
        if not verdict.matches(task, result):
            raise QualityError(f"verdict does not match task {task.id!r} attempt {result.attempt}")
        return cls(
            project_id=project_id,
            task_id=task.id,
            attempt=result.attempt,
            role=task.role,
            tier=result.tier,
            model=result.model,
            result_status=result.status,
            outcome=verdict.outcome,
            model_evaluated=verdict.model_evaluated,
            cost_nusd=result.cost_nusd,
            judged_at=verdict.created_at,
        )


class QualityStats(BaseModel):
    """Aggregate of all recorded attempts of one task type (role) with one model."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    role: str
    model: str
    tiers: list[Tier]  # tiers this model was used under, sorted
    attempts: int = Field(ge=1)
    passed: int = Field(ge=0)
    cost_nusd: int = Field(ge=0)

    @property
    def failed(self) -> int:
        return self.attempts - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.attempts

    @property
    def avg_cost_nusd(self) -> int:
        """Mean worker cost per attempt, rounded up (never under-count)."""
        return math.ceil(self.cost_nusd / self.attempts)

    @property
    def cost_per_pass_nusd(self) -> int | None:
        """All spending of this pair divided by its passes (rounded up), or ``None``."""
        return math.ceil(self.cost_nusd / self.passed) if self.passed else None


_TIER_ORDER = {t: i for i, t in enumerate(Tier)}


class QualityLog:
    """Durable record of judged attempts backed by SQLite (thread-safe).

    ``path`` may be ``":memory:"`` (tests).
    """

    def __init__(self, path: Path | str):
        self._path = str(path)
        if self._path != ":memory:":
            p = Path(self._path)
            if p.is_symlink():
                raise QualityError(f"{p}: refusing to open a symlinked quality log")
            p.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
        except sqlite3.Error as e:
            raise QualityError(f"cannot open quality log {self._path}: {e}") from e
        try:
            self._init_schema()
        except sqlite3.Error as e:
            self._conn.close()
            raise QualityError(f"{self._path}: not a usable quality log ({e})") from e
        except BaseException:
            self._conn.close()
            raise

    @classmethod
    def for_workspace(cls, workspace: Path) -> QualityLog:
        """Open the log at ``<workspace>/quality.sqlite3``."""
        return cls(Path(workspace) / QUALITY_FILE)

    def _init_schema(self) -> None:
        (version,) = self._conn.execute("PRAGMA user_version").fetchone()
        if version not in (0, SCHEMA_VERSION):
            raise QualityError(f"unsupported quality log schema version {version}")
        if version == 0:
            tables = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if tables:
                raise QualityError("database is not an Automa quality log (unexpected tables)")
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> QualityLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writing -------------------------------------------------------------

    def record(self, rec: QualityRecord) -> bool:
        """Store ``rec``; ``True`` if new, ``False`` if the identical row exists.

        A different row for the same attempt raises ``QualityError`` and changes
        nothing.
        """
        if not isinstance(rec, QualityRecord):
            raise TypeError("rec must be a QualityRecord")
        rec = QualityRecord.model_validate(rec.model_dump())
        row = _to_row(rec)
        with self._lock:
            try:
                existing = self._get_row(rec.project_id, rec.task_id, rec.attempt)
                if existing is not None:
                    if _row_to_record(existing) != rec:
                        raise QualityError(
                            f"attempt {rec.attempt} of {rec.project_id}/{rec.task_id} is "
                            "already recorded with different data"
                        )
                    return False
                with self._conn:
                    self._conn.execute(
                        f"INSERT INTO attempts ({', '.join(_COLUMNS)})"  # noqa: S608
                        f" VALUES ({', '.join('?' * len(_COLUMNS))})",
                        row,
                    )
            except sqlite3.Error as e:
                raise QualityError(f"cannot write quality log: {e}") from e
        return True

    # -- reading -------------------------------------------------------------

    def _get_row(self, project_id: str, task_id: str, attempt: int) -> tuple[object, ...] | None:
        row: tuple[object, ...] | None = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM attempts"  # noqa: S608 - constant columns
            " WHERE project_id = ? AND task_id = ? AND attempt = ?",
            (project_id, task_id, attempt),
        ).fetchone()
        return row

    def records(
        self,
        *,
        role: str | None = None,
        model: str | None = None,
        project_id: str | None = None,
    ) -> list[QualityRecord]:
        """Recorded attempts (oldest judgement first), optionally filtered."""
        clauses: list[str] = []
        params: list[str] = []
        for column, value in (("role", role), ("model", model), ("project_id", project_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT {', '.join(_COLUMNS)} FROM attempts{where}"  # noqa: S608
                    " ORDER BY judged_at, project_id, task_id, attempt",
                    params,
                ).fetchall()
            except sqlite3.Error as e:
                raise QualityError(f"cannot read quality log: {e}") from e
        return [_row_to_record(r) for r in rows]

    def stats(self, *, role: str | None = None) -> list[QualityStats]:
        """Aggregates per (role, model), sorted by role then model.

        Computed in Python from validated rows, so a corrupt row fails closed
        instead of skewing the numbers.
        """
        groups: dict[tuple[str, str], list[QualityRecord]] = {}
        for rec in self.records(role=role):
            groups.setdefault((rec.role, rec.model), []).append(rec)
        return [
            QualityStats(
                role=r,
                model=m,
                tiers=sorted({x.tier for x in recs}, key=_TIER_ORDER.__getitem__),
                attempts=len(recs),
                passed=sum(1 for x in recs if x.passed),
                cost_nusd=sum(x.cost_nusd for x in recs),
            )
            for (r, m), recs in sorted(groups.items())
        ]


def sync_project(log: QualityLog, store: ProjectStore, project_id: str) -> int:
    """Record every judged attempt of ``project_id``; returns how many rows were new.

    A project without a plan has nothing to record (0). Attempts without a verdict
    are skipped (they may still be judged later). Store errors propagate.
    """
    if not store.has_plan(project_id):
        return 0
    plan = store.load_plan(project_id)
    added = 0
    for task in plan.tasks:
        for result, verdict in store.task_history(project_id, task.id):
            if verdict is None:
                continue
            if log.record(QualityRecord.from_attempt(project_id, task, result, verdict)):
                added += 1
    return added


def _to_row(rec: QualityRecord) -> tuple[object, ...]:
    return (
        rec.project_id,
        rec.task_id,
        rec.attempt,
        rec.role,
        rec.tier.value,
        rec.model,
        rec.result_status.value,
        rec.outcome.value,
        int(rec.model_evaluated),
        rec.cost_nusd,
        rec.judged_at.isoformat(),
    )


def _row_to_record(row: tuple[object, ...]) -> QualityRecord:
    data = dict(zip(_COLUMNS, row, strict=True))
    if data["model_evaluated"] not in (0, 1):
        raise QualityError("corrupt quality log row (model_evaluated)")
    data["model_evaluated"] = bool(data["model_evaluated"])
    try:
        return QualityRecord.model_validate(data)
    except ValueError as e:
        raise QualityError(f"corrupt quality log row: {e}") from e

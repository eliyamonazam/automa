"""Tests for ``CostLedger(read_only=True)`` (D63): advisory readers never write.

SQLite in tmp_path only; no provider, no network.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from automa.budget import BudgetGuard, CostLedger, LedgerError
from automa.config import BudgetLimits
from test_ledger import FixedClock, make_pair

CLOCK = FixedClock(datetime(2026, 11, 9, 12, 0, tzinfo=UTC))


def snapshot(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), path.stat().st_mtime_ns


def filled(tmp_path: Path) -> Path:
    path = tmp_path / "ledger.sqlite3"
    with CostLedger(path, clock=CLOCK) as led:
        led.record(*make_pair(0.01, project="p1"))
        led.record(*make_pair(0.02, project="p2"))
    return path


def test_reads_the_same_totals(tmp_path: Path) -> None:
    path = filled(tmp_path)
    with CostLedger(path, clock=CLOCK) as rw:
        expected = (rw.spent_nusd(), rw.spent_nusd(project_id="p1"), len(rw.entries()))
    with CostLedger(path, clock=CLOCK, read_only=True) as ro:
        assert ro.read_only
        got = (ro.spent_nusd(), ro.spent_nusd(project_id="p1"), len(ro.entries()))
        assert ro.spent_nusd(day=ro.today()) == expected[0]
        assert ro.get(1).project_id == "p1"
    assert got == expected


def test_opening_read_only_changes_nothing(tmp_path: Path) -> None:
    path = filled(tmp_path)
    before = snapshot(path)
    files = sorted(p.name for p in tmp_path.iterdir())
    with CostLedger(path, read_only=True) as ro:
        ro.spent_nusd(project_id="p1")
    assert snapshot(path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == files


def test_record_is_refused(tmp_path: Path) -> None:
    path = filled(tmp_path)
    before = snapshot(path)
    with CostLedger(path, read_only=True) as ro, pytest.raises(LedgerError, match="read-only"):
        ro.record(*make_pair(0.01))
    assert snapshot(path) == before


def test_missing_file_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "ledger.sqlite3"
    with pytest.raises(LedgerError, match="no ledger"):
        CostLedger(path, read_only=True)
    assert not path.exists()
    assert not path.parent.exists()


def test_memory_cannot_be_read_only() -> None:
    with pytest.raises(LedgerError):
        CostLedger(":memory:", read_only=True)


def test_symlink_is_refused(tmp_path: Path) -> None:
    real = filled(tmp_path)
    link = tmp_path / "link.sqlite3"
    link.symlink_to(real)
    with pytest.raises(LedgerError, match="symlink"):
        CostLedger(link, read_only=True)


def test_garbage_file_is_ledger_error(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    path.write_bytes(b"not a database" * 100)
    with pytest.raises(LedgerError):
        CostLedger(path, read_only=True)


def test_foreign_database_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE other (x)")
    conn.commit()
    conn.close()
    with pytest.raises(LedgerError, match="not an Automa ledger"):
        CostLedger(path, read_only=True)


def test_unsupported_version_is_refused(tmp_path: Path) -> None:
    path = filled(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(LedgerError, match="schema version"):
        CostLedger(path, read_only=True)


def test_empty_database_reads_as_empty_without_migrating(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    sqlite3.connect(path).close()  # zero-byte SQLite file, no schema
    before = snapshot(path)
    with CostLedger(path, read_only=True) as ro:
        assert ro.spent_nusd() == 0
        assert ro.entries() == []
        with pytest.raises(KeyError):
            ro.get(1)
    assert snapshot(path) == before


def test_writable_open_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "new" / "ledger.sqlite3"
    with CostLedger(path, clock=CLOCK) as led:
        assert not led.read_only
        led.record(*make_pair(0.01))
        assert led.spent_nusd() > 0


def test_guard_over_read_only_ledger_fails_closed(tmp_path: Path) -> None:
    # Never done in the code; if it were, spending could not go unrecorded.
    path = filled(tmp_path)
    before = snapshot(path)
    with CostLedger(path, clock=CLOCK, read_only=True) as ro:
        guard = BudgetGuard(BudgetLimits(), ro)
        est, result = make_pair(0.01, project="p1")
        guard.authorize(est)
        with pytest.raises(LedgerError):
            guard.record(est, result)
    assert snapshot(path) == before

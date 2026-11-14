"""Tests for ``automa quality`` and ``automa quality sync`` (D57), and the read-only
mode of ``QualityLog`` they use.

Local only: FakeProvider behind the real gateway/guard/ledger (only to produce real
judged attempts with ``automa run``), real ProjectStore. No network, no real model
calls; the quality commands themselves never build a provider.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import test_cli_run as cr
from automa import cli
from automa.config import Tier
from automa.plan.models import ResultStatus, VerdictOutcome
from automa.quality import QUALITY_FILE, QualityError, QualityLog, QualityRecord

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def rec(**kw: Any) -> QualityRecord:
    data: dict[str, Any] = {
        "project_id": "p1",
        "task_id": "a",
        "attempt": 1,
        "role": "writer",
        "tier": Tier.MEDIUM,
        "model": "m-medium",
        "result_status": ResultStatus.COMPLETED,
        "outcome": VerdictOutcome.PASS,
        "model_evaluated": False,
        "cost_nusd": 1000,
        "judged_at": T0,
    }
    data.update(kw)
    return QualityRecord(**data)


def snapshot(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), path.stat().st_mtime_ns


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


# --- QualityLog read-only mode ---------------------------------------------------


def test_read_only_missing_file_raises_and_creates_nothing(tmp_path: Path) -> None:
    path = tmp_path / "ws" / QUALITY_FILE
    with pytest.raises(QualityError, match="no quality log"):
        QualityLog(path, read_only=True)
    assert not (tmp_path / "ws").exists()


def test_read_only_memory_is_refused() -> None:
    with pytest.raises(QualityError, match="read-only"):
        QualityLog(":memory:", read_only=True)


def test_read_only_reads_rows_and_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / QUALITY_FILE
    with QualityLog(path) as log:
        log.record(rec())
        log.record(rec(task_id="b", outcome=VerdictOutcome.FAIL, cost_nusd=3000))
    before = snapshot(path)
    with QualityLog(path, read_only=True) as ro:
        assert ro.read_only
        assert len(ro.records()) == 2
        (s,) = ro.stats()
        assert (s.attempts, s.passed, s.cost_nusd) == (2, 1, 4000)
        with pytest.raises(QualityError, match="read-only"):
            ro.record(rec(task_id="c"))
    assert snapshot(path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [QUALITY_FILE]


def test_read_only_empty_file_is_empty_and_untouched(tmp_path: Path) -> None:
    path = tmp_path / QUALITY_FILE
    path.write_bytes(b"")
    with QualityLog(path, read_only=True) as ro:
        assert ro.records() == []
        assert ro.stats() == []
    assert path.read_bytes() == b""


@pytest.mark.parametrize("kind", ["garbage", "foreign", "version"])
def test_read_only_fails_closed(tmp_path: Path, kind: str) -> None:
    path = tmp_path / QUALITY_FILE
    if kind == "garbage":
        path.write_bytes(b"garbage" * 200)
    else:
        conn = sqlite3.connect(path)
        if kind == "foreign":
            conn.execute("CREATE TABLE other (x)")
        else:
            conn.execute("PRAGMA user_version = 99")
        conn.commit()
        conn.close()
    before = snapshot(path)
    with pytest.raises(QualityError):
        QualityLog(path, read_only=True)
    assert snapshot(path) == before


def test_read_only_refuses_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.sqlite3"
    QualityLog(real).close()
    link = tmp_path / QUALITY_FILE
    link.symlink_to(real)
    with pytest.raises(QualityError, match="symlink"):
        QualityLog(link, read_only=True)


def test_read_only_path_with_special_characters(tmp_path: Path) -> None:
    d = tmp_path / "a b?c#d%20"
    d.mkdir()
    with QualityLog.for_workspace(d) as log:
        log.record(rec())
    with QualityLog.for_workspace(d, read_only=True) as ro:
        assert ro.records() == [rec()]


class _Racing(QualityLog):
    """Pretends the row is absent at the first check (another process wins the race)."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.hidden = True

    def _get_row(self, *args: Any) -> Any:
        if self.hidden:
            self.hidden = False
            return None
        return super()._get_row(*args)


def test_record_race_with_identical_row_is_a_no_op(tmp_path: Path) -> None:
    path = tmp_path / QUALITY_FILE
    with QualityLog(path) as other:
        other.record(rec())
    with _Racing(path) as log:
        assert log.record(rec()) is False
        assert log.records() == [rec()]


def test_record_race_with_different_row_is_a_conflict(tmp_path: Path) -> None:
    path = tmp_path / QUALITY_FILE
    with QualityLog(path) as other:
        other.record(rec())
    with _Racing(path) as log:
        with pytest.raises(QualityError, match="different data"):
            log.record(rec(outcome=VerdictOutcome.FAIL))
        assert log.records() == [rec()]


# --- CLI helpers -------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> cr.Env:
    return cr.Env(tmp_path, monkeypatch)


def quality(env: cr.Env, *args: str) -> Any:
    return CliRunner().invoke(cli.app, ["quality", *args, "--config", str(env.config)])


def sync(env: cr.Env, *args: str) -> Any:
    return CliRunner().invoke(cli.app, ["quality", "sync", *args, "--config", str(env.config)])


def run_project(env: cr.Env) -> None:
    """A real run: attempt 1 fails verification, attempt 2 passes."""
    env.project(cr.task_dict("a"))
    env.script(*cr.write("a", "bad"), *cr.write("a"))
    r = env.run(cr.PID, "--max-attempts", "2")
    assert r.exit_code == cli.EXIT_OK, r.output


def seed(env: cr.Env, *records: QualityRecord) -> Path:
    env.ws.mkdir(parents=True, exist_ok=True)
    path = env.ws / QUALITY_FILE
    with QualityLog(path) as log:
        for r in records:
            log.record(r)
    return path


# --- automa quality ----------------------------------------------------------------


def test_quality_without_log_creates_nothing(env: cr.Env, tmp_path: Path) -> None:
    before = tree(tmp_path)
    r = quality(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "no quality data yet" in r.output
    assert "automa quality sync" in r.output
    assert tree(tmp_path) == before
    assert not env.ws.exists()
    assert env.factory_calls == 0


def test_quality_json_without_log(env: cr.Env) -> None:
    r = quality(env, "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert json.loads(r.output) == []
    assert not env.ws.exists()


def test_quality_after_run_shows_pass_rate_and_cost(env: cr.Env) -> None:
    run_project(env)
    calls = len(env.provider.requests)
    path = env.ws / QUALITY_FILE
    before = snapshot(path)
    r = quality(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    header, row = r.output.strip().splitlines()
    assert header.split()[:3] == ["role", "model", "tiers"]
    cols = row.split()
    assert cols[0] == "writer"
    assert cols[3:6] == ["2", "1", "50.0%"]
    assert snapshot(path) == before
    assert len(env.provider.requests) == calls
    assert env.factory_calls == 1  # only the run built a provider


def test_quality_json_values(env: cr.Env) -> None:
    seed(
        env,
        rec(cost_nusd=1000),
        rec(task_id="b", outcome=VerdictOutcome.FAIL, cost_nusd=2001, tier=Tier.SMALL),
        rec(task_id="c", role="developer", model="m-large", tier=Tier.LARGE, cost_nusd=5),
    )
    r = quality(env, "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    rows = json.loads(r.output)
    assert [(x["role"], x["model"]) for x in rows] == [
        ("developer", "m-large"),
        ("writer", "m-medium"),
    ]
    w = rows[1]
    assert w["tiers"] == ["small", "medium"]
    assert (w["attempts"], w["passed"], w["failed"]) == (2, 1, 1)
    assert w["pass_rate"] == 0.5
    assert w["cost_nusd"] == 3001
    assert w["avg_cost_nusd"] == 1501  # rounded up
    assert w["cost_per_pass_nusd"] == 3001
    assert w["cost_usd"] == pytest.approx(3001e-9)


def test_quality_never_passed_shows_dash_and_null(env: cr.Env) -> None:
    seed(env, rec(outcome=VerdictOutcome.FAIL))
    r = quality(env)
    assert r.output.strip().splitlines()[1].split()[-1] == "-"
    (row,) = json.loads(quality(env, "--json").output)
    assert row["cost_per_pass_nusd"] is None
    assert row["cost_per_pass_usd"] is None
    assert row["pass_rate"] == 0.0


def test_quality_role_filter(env: cr.Env) -> None:
    seed(env, rec(), rec(task_id="b", role="developer"))
    rows = json.loads(quality(env, "--role", "developer", "--json").output)
    assert [x["role"] for x in rows] == ["developer"]
    r = quality(env, "--role", "researcher")
    assert r.exit_code == cli.EXIT_OK
    assert "no quality data yet for role researcher" in r.output


@pytest.mark.parametrize("bad", ["Writer", "../x", "a b", "", "x" * 40])
def test_quality_invalid_role_is_user_error(env: cr.Env, bad: str) -> None:
    seed(env, rec())
    r = quality(env, "--role", bad)
    assert r.exit_code == cli.EXIT_USER, r.output


def test_quality_broken_log_is_config_error_and_untouched(env: cr.Env) -> None:
    env.ws.mkdir(parents=True)
    path = env.ws / QUALITY_FILE
    path.write_bytes(b"garbage" * 200)
    r = quality(env)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert path.read_bytes() == b"garbage" * 200


def test_quality_symlinked_log_is_config_error(env: cr.Env, tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.sqlite3"
    QualityLog(real).close()
    env.ws.mkdir(parents=True)
    (env.ws / QUALITY_FILE).symlink_to(real)
    assert quality(env).exit_code == cli.EXIT_CONFIG


def test_quality_dangling_symlink_is_config_error(env: cr.Env, tmp_path: Path) -> None:
    env.ws.mkdir(parents=True)
    (env.ws / QUALITY_FILE).symlink_to(tmp_path / "missing.sqlite3")
    assert quality(env).exit_code == cli.EXIT_CONFIG
    assert not (tmp_path / "missing.sqlite3").exists()


def test_quality_corrupt_row_is_config_error(env: cr.Env) -> None:
    path = seed(env, rec())
    conn = sqlite3.connect(path)
    conn.execute("UPDATE attempts SET role = 'Not A Role'")
    conn.commit()
    conn.close()
    assert quality(env).exit_code == cli.EXIT_CONFIG


def test_quality_escapes_model_name(env: cr.Env) -> None:
    seed(env, rec(model="evil\x1b[31m‮model\nx"))
    r = quality(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "\x1b" not in r.output
    assert "‮" not in r.output
    assert "\\x1b" in r.output
    assert len(r.output.strip().splitlines()) == 2  # the newline does not add a row


def test_quality_needs_no_api_key(env: cr.Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seed(env, rec())

    def boom(config: Any) -> Any:
        raise AssertionError("provider built")

    monkeypatch.setattr(cli, "make_provider", boom)
    assert quality(env).exit_code == cli.EXIT_OK
    assert sync(env).exit_code == cli.EXIT_OK


def test_quality_bad_config_is_config_error(env: cr.Env) -> None:
    env.config.write_text("not = [valid", encoding="utf-8")
    assert quality(env).exit_code == cli.EXIT_CONFIG
    assert sync(env).exit_code == cli.EXIT_CONFIG


def test_quality_help_lists_sync() -> None:
    r = CliRunner().invoke(cli.app, ["quality", "--help"])
    assert r.exit_code == 0
    assert "sync" in r.output
    assert "--role" in r.output


# --- automa quality sync -------------------------------------------------------------


def test_sync_imports_past_project_once(env: cr.Env) -> None:
    run_project(env)
    (env.ws / QUALITY_FILE).unlink()  # as if the project ran before the log existed
    calls = len(env.provider.requests)
    r = sync(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert f"{cr.PID}  2 new" in r.output
    assert "imported 2 new row(s) from 1 project(s)" in r.output
    r = sync(env, cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert f"{cr.PID}  0 new" in r.output
    assert len(env.provider.requests) == calls
    rows = json.loads(quality(env, "--json").output)
    assert [(x["attempts"], x["passed"]) for x in rows] == [(2, 1)]


def test_sync_after_run_adds_nothing(env: cr.Env) -> None:
    run_project(env)
    before = snapshot(env.ws / QUALITY_FILE)[0]
    r = sync(env, "--json")
    assert r.exit_code == cli.EXIT_OK, r.output
    assert json.loads(r.output) == {
        "projects": [{"id": cr.PID, "added": 0, "error": None}],
        "added": 0,
        "failed": 0,
    }
    with QualityLog.for_workspace(env.ws) as log:
        assert len(log.records()) == 2
    assert len(before) > 0


def test_sync_without_projects_creates_nothing(env: cr.Env, tmp_path: Path) -> None:
    before = tree(tmp_path)
    r = sync(env)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "no projects yet" in r.output
    assert tree(tmp_path) == before
    r = sync(env, "--json")
    assert json.loads(r.output) == {"projects": [], "added": 0, "failed": 0}
    assert tree(tmp_path) == before


def test_sync_project_without_plan_is_zero(env: cr.Env) -> None:
    env.project(plan=False)
    r = sync(env, cr.PID)
    assert r.exit_code == cli.EXIT_OK, r.output
    assert f"{cr.PID}  0 new" in r.output


@pytest.mark.parametrize("pid", ["missing", "../x", "Bad_Id"])
def test_sync_unknown_or_invalid_project_creates_no_log(env: cr.Env, pid: str) -> None:
    env.project(cr.task_dict("a"))
    r = sync(env, pid)
    assert r.exit_code == cli.EXIT_USER, r.output
    assert not (env.ws / QUALITY_FILE).exists()


def test_sync_reports_broken_project_and_imports_the_others(env: cr.Env) -> None:
    run_project(env)
    (env.ws / QUALITY_FILE).unlink()
    env.store.create("other goal", project_id="proj-2")
    env.store.plan_file("proj-2").write_text("{not json", encoding="utf-8")
    r = sync(env)
    assert r.exit_code == cli.EXIT_USER, r.output
    assert f"{cr.PID}  2 new" in r.output
    assert "proj-2  error:" in r.output
    assert "; 1 failed" in r.output
    data = json.loads(sync(env, "--json").output)
    assert data["failed"] == 1
    assert [p["id"] for p in data["projects"]] == [cr.PID, "proj-2"]
    assert data["projects"][1]["error"]
    with QualityLog.for_workspace(env.ws) as log:
        assert len(log.records(project_id=cr.PID)) == 2


def test_sync_conflicting_row_is_reported(env: cr.Env) -> None:
    run_project(env)
    path = env.ws / QUALITY_FILE
    conn = sqlite3.connect(path)
    conn.execute("UPDATE attempts SET cost_nusd = cost_nusd + 1 WHERE attempt = 1")
    conn.commit()
    conn.close()
    r = sync(env)
    assert r.exit_code == cli.EXIT_USER, r.output
    assert "different data" in r.output


def test_sync_broken_log_is_config_error(env: cr.Env) -> None:
    env.project(cr.task_dict("a"))
    env.ws.mkdir(parents=True, exist_ok=True)
    path = env.ws / QUALITY_FILE
    path.write_bytes(b"garbage" * 200)
    r = sync(env)
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert path.read_bytes() == b"garbage" * 200


def test_sync_skips_symlinked_project_dirs(env: cr.Env, tmp_path: Path) -> None:
    run_project(env)
    (env.ws / QUALITY_FILE).unlink()
    target = env.store.project_dir(cr.PID)
    os.symlink(target, env.store.root / "proj-link")
    data = json.loads(sync(env, "--json").output)
    assert [p["id"] for p in data["projects"]] == [cr.PID]

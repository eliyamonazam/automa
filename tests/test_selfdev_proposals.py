"""Tests of automa.selfdev_proposals (D84): no model, no network, no repository writes."""

from __future__ import annotations

import ast
import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from automa import selfdev_proposals as sp
from automa.plan.models import PlanError
from automa.selfdev import parse_roadmap
from automa.selfdev_proposals import (
    MAX_PROPOSALS,
    DecisionStatus,
    ItemProposal,
    ProposalNotFound,
    ProposalReply,
    ProposalStore,
    ProposalStoreError,
    Screened,
    parse_proposal_reply,
    roadmap_digest,
    roadmap_line,
    screen_proposals,
)

ROADMAP = """\
# نقشه راه

## M9 — رابط‌ها
- [x] اعلان (هسته): ماژول notify
- [!] اعلان (سرویس واقعی و اتصال)

## M10 — تکامل خودکار
- [x] خودتوسعه (هسته)
- [ ] خودتوسعه (پیشنهاد مورد تازه)
"""

M10 = "M10 — تکامل خودکار"
COMMIT = "a" * 40


def _p(text: str = "گزارش هفتگی هزینه در automa finance", milestone: str = M10, **kw: str):
    return ItemProposal(milestone=milestone, text=text, **kw)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 16, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def store(tmp_path: Path) -> ProposalStore:
    return ProposalStore.for_workspace(tmp_path / "ws", clock=Clock())


def _screened(*texts: str) -> Screened:
    reply = ProposalReply(proposals=[_p(t) for t in texts])
    return screen_proposals(reply, parse_roadmap(ROADMAP))


# -- ItemProposal --------------------------------------------------------------


def test_proposal_basic_and_line() -> None:
    p = _p("  گزارش   هفتگی هزینه\tدر finance  ", reason="کمک به صاحب مخزن")
    assert p.text == "گزارش هفتگی هزینه در finance"
    assert p.line == "- [ ] گزارش هفتگی هزینه در finance"
    assert roadmap_line(p.text) == p.line
    # The line round-trips through the roadmap parser as one open item.
    parsed = parse_roadmap(f"## {M10}\n{p.line}\n")
    assert [i.text for i in parsed.items] == [p.text]


def test_persian_joiners_kept() -> None:
    p = _p("ساخت‌خودکار مهارت‌ها با benchmark")
    assert "‌" in p.text


@pytest.mark.parametrize(
    "text",
    [
        "line one\nline two is here",
        "line one\rline two is here",
        "line one line two is here",
        "line one\x85line two is here",
        "line one\x0bline two is here",
    ],
)
def test_multiline_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="single line"):
        _p(text)


@pytest.mark.parametrize(
    "text",
    [
        "evil ‮ bidi override here",
        "zero width ​ space here",
        "bell \x07 control char here",
        "escape \x1b[31m red text here",
        "bom ﻿ inside the text",
    ],
)
def test_control_and_format_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="control or format"):
        _p(text)


@pytest.mark.parametrize(
    "text",
    [
        "[x] already done, mark it",
        "[!] waiting for the owner here",
        "[] empty marker at the start",
        "# heading instead of an item",
        "> quoted text instead of item",
        "- nested list item instead",
        "* nested list item instead",
        "1. numbered item instead here",
        "```python fence at the start",
        "<script>alert(1)</script> html",
    ],
)
def test_leading_structure_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="Markdown structure"):
        _p(text)


def test_fence_inside_rejected() -> None:
    with pytest.raises(ValueError, match="code fence"):
        _p("some text then ``` a fence")


@pytest.mark.parametrize("text", ["", "   ", "short"])
def test_too_short_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="at least"):
        _p(text)


def test_too_long_rejected() -> None:
    with pytest.raises(ValueError, match="longer than"):
        _p("x" * (sp.MAX_PROPOSAL_TEXT_CHARS + 1))
    assert len(_p("x" * sp.MAX_PROPOSAL_TEXT_CHARS).text) == sp.MAX_PROPOSAL_TEXT_CHARS


def test_reason_and_milestone_rules() -> None:
    with pytest.raises(ValueError):
        _p(reason="a\nb")
    with pytest.raises(ValueError):
        _p(reason="r" * (sp.MAX_REASON_CHARS + 1))
    with pytest.raises(ValueError):
        _p(milestone="")
    with pytest.raises(ValueError):
        _p(milestone="M10‮")
    with pytest.raises(ValueError):
        _p(milestone="m" * 201)


def test_proposal_is_frozen_and_strict() -> None:
    with pytest.raises(ValueError):
        ItemProposal.model_validate({"milestone": M10, "text": "valid text here", "status": "x"})
    p = _p()
    with pytest.raises(ValueError):
        p.text = "changed text here"  # type: ignore[misc]


# -- parse_proposal_reply ------------------------------------------------------


def test_parse_reply_ok_and_fenced() -> None:
    body = {"proposals": [{"milestone": M10, "text": "a valid new item", "reason": "why"}]}
    reply = parse_proposal_reply("```json\n" + json.dumps(body, ensure_ascii=False) + "\n```")
    assert reply.proposals[0].text == "a valid new item"
    assert parse_proposal_reply('{"proposals": []}').proposals == []
    assert parse_proposal_reply("{}").proposals == []


@pytest.mark.parametrize(
    "text, message",
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        ('{"proposals": [{"milestone": "M", "text": "[x] done item here"}]}', "validation"),
        ('{"proposals": [], "extra": 1}', "validation"),
        ('{"proposals": [{"milestone": "M", "text": "ok text here", "status": "!"}]}', "valid"),
    ],
)
def test_parse_reply_errors(text: str, message: str) -> None:
    with pytest.raises(PlanError, match=message) as info:
        parse_proposal_reply(text)
    assert info.value.details


def test_parse_reply_too_many() -> None:
    items = [{"milestone": M10, "text": f"proposal number {i}"} for i in range(MAX_PROPOSALS + 1)]
    with pytest.raises(PlanError):
        parse_proposal_reply(json.dumps({"proposals": items}))


def test_parse_reply_not_text() -> None:
    with pytest.raises(PlanError):
        parse_proposal_reply(None)  # type: ignore[arg-type]


# -- screen_proposals ----------------------------------------------------------


def test_screen_drops_duplicates_and_unknown_milestones() -> None:
    roadmap = parse_roadmap(ROADMAP)
    reply = ProposalReply(
        proposals=[
            _p("ساخت خودکار Skill با benchmark"),
            _p("خودتوسعه   (هسته)"),  # done item, other spacing
            _p("ساخت خودکار SKILL با benchmark"),  # duplicate of the first (case)
            _p("an item for a milestone that does not exist", milestone="M99"),
            _p("item with a milestone in other case", milestone="m10 — تکامل خودکار"),
        ]
    )
    s = screen_proposals(reply, roadmap)
    assert [p.text for p in s.proposals] == [
        "ساخت خودکار Skill با benchmark",
        "item with a milestone in other case",
    ]
    assert s.proposals[1].milestone == M10  # canonical milestone name
    assert s.dropped_duplicates == 2
    assert s.dropped_unknown_milestone == 1


def test_screen_drops_waiting_item_again() -> None:
    reply = ProposalReply(proposals=[_p("اعلان (سرویس واقعی و اتصال)", milestone="M9 — رابط‌ها")])
    s = screen_proposals(reply, parse_roadmap(ROADMAP))
    assert s.proposals == [] and s.dropped_duplicates == 1


def test_screen_empty() -> None:
    s = screen_proposals(ProposalReply(), parse_roadmap(ROADMAP))
    assert s.proposals == [] and s.dropped_duplicates == 0


# -- ProposalStore -------------------------------------------------------------


def test_save_get_and_layout(store: ProposalStore, tmp_path: Path) -> None:
    pset = store.save(
        _screened("first new item", "second new item"), roadmap_text=ROADMAP, commit=COMMIT
    )
    assert store.root == tmp_path / "ws" / "selfdev" / "proposals"
    assert sorted(os.listdir(store.root)) == [f"{pset.id}.json"]
    assert pset.commit == COMMIT
    assert pset.roadmap_sha256 == roadmap_digest(ROADMAP)
    assert store.get(pset.id) == pset
    assert store.all() == [pset]
    assert store.pending(pset.id) == [1, 2]
    assert store.decisions(pset.id) == {}


def test_empty_store(store: ProposalStore) -> None:
    assert store.all() == []
    assert not store.root.exists()


def test_save_empty_set(store: ProposalStore) -> None:
    pset = store.save(_screened(), roadmap_text=ROADMAP)
    assert pset.proposals == [] and store.pending(pset.id) == []


def test_accept_reject_once(store: ProposalStore) -> None:
    pset = store.save(_screened("first new item", "second new item"), roadmap_text=ROADMAP)
    d1 = store.accept(pset.id, 1)
    d2 = store.reject(pset.id, 2)
    assert d1.status is DecisionStatus.ACCEPTED and d2.status is DecisionStatus.REJECTED
    assert store.pending(pset.id) == []
    assert store.decisions(pset.id) == {1: d1, 2: d2}
    with pytest.raises(ProposalStoreError, match="already accepted"):
        store.reject(pset.id, 1)
    with pytest.raises(ProposalStoreError, match="already rejected"):
        store.accept(pset.id, 2)


def test_accept_never_writes_roadmap(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ROADMAP.md").write_text(ROADMAP, encoding="utf-8")
    before = {p: p.read_bytes() for p in repo.rglob("*")}
    store = ProposalStore.for_workspace(tmp_path / "ws", clock=Clock())
    pset = store.save(_screened("a brand new item"), roadmap_text=ROADMAP)
    store.accept(pset.id, 1)
    assert {p: p.read_bytes() for p in repo.rglob("*")} == before


@pytest.mark.parametrize("number", [0, 3, -1, True, "1", 1.0])
def test_bad_numbers(store: ProposalStore, number: object) -> None:
    pset = store.save(_screened("first new item", "second new item"), roadmap_text=ROADMAP)
    with pytest.raises(ProposalStoreError):
        store.accept(pset.id, number)  # type: ignore[arg-type]


@pytest.mark.parametrize("set_id", ["", "sdp-1", "../x", "sdp-20261116-120000-ABCDEF", 5])
def test_bad_set_ids(store: ProposalStore, set_id: object) -> None:
    with pytest.raises(ProposalStoreError):
        store.get(set_id)  # type: ignore[arg-type]


def test_missing_set(store: ProposalStore) -> None:
    with pytest.raises(ProposalNotFound):
        store.get("sdp-20261116-120000-abcdef")
    with pytest.raises(ProposalNotFound):
        store.accept("sdp-20261116-120000-abcdef", 1)


def test_save_rechecks_against_roadmap(store: ProposalStore) -> None:
    unknown = Screened(proposals=[_p(milestone="M77 — نامعلوم")])
    with pytest.raises(ProposalStoreError, match="milestone"):
        store.save(unknown, roadmap_text=ROADMAP)
    dup = Screened(proposals=[_p("خودتوسعه (هسته)")])
    with pytest.raises(ProposalStoreError, match="existing"):
        store.save(dup, roadmap_text=ROADMAP)
    with pytest.raises(ProposalStoreError, match="invalid roadmap"):
        store.save(_screened(), roadmap_text="- [?] broken\n")
    with pytest.raises(ProposalStoreError, match="same item twice"):
        store.save(Screened(proposals=[_p(), _p()]), roadmap_text=ROADMAP)
    assert store.all() == []


def test_save_type_checks(store: ProposalStore) -> None:
    with pytest.raises(TypeError):
        store.save(ProposalReply(), roadmap_text=ROADMAP)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        store.save(_screened(), roadmap_text=None)  # type: ignore[arg-type]


@pytest.mark.parametrize("commit", ["abc", "A" * 40, "g" * 40, "a" * 41])
def test_bad_commit(store: ProposalStore, commit: str) -> None:
    with pytest.raises(ProposalStoreError):
        store.save(_screened(), roadmap_text=ROADMAP, commit=commit)


def test_naive_clock(tmp_path: Path) -> None:
    store = ProposalStore.for_workspace(tmp_path, clock=lambda: datetime(2026, 1, 1))
    with pytest.raises(ProposalStoreError, match="naive"):
        store.save(_screened(), roadmap_text=ROADMAP)


def test_max_sets(store: ProposalStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp, "MAX_SETS", 2)
    store.save(_screened(), roadmap_text=ROADMAP)
    store.save(_screened(), roadmap_text=ROADMAP)
    with pytest.raises(ProposalStoreError, match="already 2"):
        store.save(_screened(), roadmap_text=ROADMAP)


def test_all_sorted(store: ProposalStore) -> None:
    a = store.save(_screened("first new item"), roadmap_text=ROADMAP)
    b = store.save(_screened("second new item"), roadmap_text=ROADMAP)
    assert [s.id for s in store.all()] == [a.id, b.id]


# -- fail closed on disk -------------------------------------------------------


def _saved(store: ProposalStore) -> sp.ProposalSet:
    return store.save(_screened("first new item", "second new item"), roadmap_text=ROADMAP)


def test_unknown_file_fails(store: ProposalStore) -> None:
    _saved(store)
    (store.root / "notes.txt").write_text("x")
    with pytest.raises(ProposalStoreError, match="unexpected"):
        store.all()


def test_temp_files_ignored(store: ProposalStore) -> None:
    pset = _saved(store)
    (store.root / ".tmp-file").write_text("x")
    assert store.all() == [pset]


def test_orphan_decision_fails(store: ProposalStore) -> None:
    _saved(store)
    (store.root / "sdp-20200101-000000-abcdef.d1.json").write_text("{}")
    with pytest.raises(ProposalStoreError, match="missing proposal set"):
        store.all()


def test_corrupt_set_fails(store: ProposalStore) -> None:
    pset = _saved(store)
    path = store.root / f"{pset.id}.json"
    path.chmod(0o644)
    path.write_text("{not json")
    with pytest.raises(ProposalStoreError, match="unreadable"):
        store.get(pset.id)


def test_tampered_set_fails(store: ProposalStore) -> None:
    pset = _saved(store)
    path = store.root / f"{pset.id}.json"
    data = json.loads(path.read_text())
    data["proposals"][0]["text"] = "[x] marked done by tampering"
    path.write_text(json.dumps(data))
    with pytest.raises(ProposalStoreError, match="invalid proposal set"):
        store.get(pset.id)


def test_set_under_other_name_fails(store: ProposalStore) -> None:
    pset = _saved(store)
    other = "sdp-20200101-000000-abcdef"
    os.rename(store.root / f"{pset.id}.json", store.root / f"{other}.json")
    with pytest.raises(ProposalStoreError, match="does not belong"):
        store.get(other)


def test_decision_tampering_fails(store: ProposalStore) -> None:
    pset = _saved(store)
    store.accept(pset.id, 1)
    os.rename(store.root / f"{pset.id}.d1.json", store.root / f"{pset.id}.d2.json")
    with pytest.raises(ProposalStoreError, match="does not belong"):
        store.decisions(pset.id)


def test_decision_beyond_set_fails(store: ProposalStore) -> None:
    pset = _saved(store)
    data = {
        "schema_version": 1,
        "set_id": pset.id,
        "number": 3,
        "status": "accepted",
        "decided_at": "2026-11-16T12:00:00Z",
    }
    (store.root / f"{pset.id}.d3.json").write_text(json.dumps(data))
    with pytest.raises(ProposalStoreError, match="does not have"):
        store.all()


def test_symlinked_set_fails(store: ProposalStore, tmp_path: Path) -> None:
    pset = _saved(store)
    path = store.root / f"{pset.id}.json"
    target = tmp_path / "elsewhere.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ProposalStoreError, match="cannot open"):
        store.get(pset.id)


def test_symlinked_dir_fails(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "ws" / "selfdev").mkdir(parents=True)
    (tmp_path / "ws" / "selfdev" / "proposals").symlink_to(real)
    store = ProposalStore.for_workspace(tmp_path / "ws", clock=Clock())
    with pytest.raises(ProposalStoreError, match="symlink"):
        store.all()
    with pytest.raises(ProposalStoreError, match="symlink"):
        store.save(_screened(), roadmap_text=ROADMAP)
    assert list(real.iterdir()) == []


def test_symlinked_parent_dir_fails(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "selfdev").symlink_to(real)
    store = ProposalStore.for_workspace(tmp_path / "ws", clock=Clock())
    with pytest.raises(ProposalStoreError, match="symlink"):
        store.save(_screened(), roadmap_text=ROADMAP)


def test_proposals_path_is_file(tmp_path: Path) -> None:
    (tmp_path / "selfdev").mkdir()
    (tmp_path / "selfdev" / "proposals").write_text("x")
    store = ProposalStore.for_workspace(tmp_path, clock=Clock())
    with pytest.raises(ProposalStoreError, match="not a directory"):
        store.all()


def test_oversized_set_fails(store: ProposalStore) -> None:
    pset = _saved(store)
    path = store.root / f"{pset.id}.json"
    path.write_text(" " * (sp.MAX_SET_FILE_BYTES + 1))
    with pytest.raises(ProposalStoreError, match="larger"):
        store.get(pset.id)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_fifo_set_fails(store: ProposalStore) -> None:
    pset = _saved(store)
    path = store.root / f"{pset.id}.json"
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ProposalStoreError, match="not a regular"):
        store.get(pset.id)


def test_concurrent_decisions_once(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    pset = ProposalStore.for_workspace(root, clock=Clock()).save(
        _screened("first new item"), roadmap_text=ROADMAP
    )
    # Separate store objects: no shared in-process lock, only the exclusive file.
    stores = [ProposalStore.for_workspace(root, clock=Clock()) for _ in range(8)]
    results: list[str] = []
    barrier = threading.Barrier(len(stores))

    def run(s: ProposalStore) -> None:
        barrier.wait()
        try:
            s.accept(pset.id, 1)
            results.append("ok")
        except ProposalStoreError:
            results.append("err")

    threads = [threading.Thread(target=run, args=(s,)) for s in stores]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1
    assert [p.name for p in sorted(root.rglob("*.d1.json"))] == [f"{pset.id}.d1.json"]


def test_files_are_pretty_json_and_contain_only_proposals(store: ProposalStore) -> None:
    pset = store.save(_screened("first new item"), roadmap_text=ROADMAP, commit=COMMIT)
    data = json.loads((store.root / f"{pset.id}.json").read_text(encoding="utf-8"))
    assert set(data) == {
        "schema_version",
        "id",
        "commit",
        "roadmap_sha256",
        "proposals",
        "dropped_duplicates",
        "dropped_unknown_milestone",
        "saved_at",
    }
    assert ROADMAP not in json.dumps(data, ensure_ascii=False)  # only its digest


# -- structure -----------------------------------------------------------------


def test_module_has_no_model_network_or_process_dependency() -> None:
    tree = ast.parse(Path(sp.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = (
        "httpx",
        "socket",
        "subprocess",
        "automa.llm",
        "automa.budget",
        "automa.tools",
        "automa.agents",
    )
    for name in imported:
        assert not name.startswith(forbidden), name


def test_module_writes_only_through_write_once_helper() -> None:
    tree = ast.parse(Path(sp.__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    attrs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                attrs.append(f.attr)
            elif isinstance(f, ast.Name):
                names.append(f.id)
            if isinstance(f, ast.Attribute) and f.attr == "fdopen":
                assert isinstance(node.args[1], ast.Constant) and node.args[1].value == "rb"
    assert "open" not in names
    banned = ("write_text", "write_bytes", "rename", "unlink", "rmtree", "remove", "rmdir")
    for name in banned + ("symlink_to", "chmod", "link", "truncate"):
        assert name not in attrs, name
    # The only os.open is the O_RDONLY reader; the only writer is the write-once helper.
    assert attrs.count("open") == 1
    assert "_write_json_exclusive" in names

"""Self-development proposals (core, D84): new roadmap items, proposed but never applied.

A later step lets the Executive read VISION/ROADMAP and *propose* new roadmap items.
This module is everything around that model call, and contains no model call itself:

* ``ItemProposal`` / ``ProposalReply`` — the reply format. A proposal is always a new
  ``[ ]`` item: one line, no Markdown structure of its own, no control/bidi characters,
  and it must round-trip through ``parse_roadmap`` (D80) unchanged, so accepting it can
  never produce an ambiguous ROADMAP;
* ``parse_proposal_reply`` — strict JSON parsing with model-readable errors (``PlanError``);
* ``screen_proposals`` — code-side filtering against the roadmap the model saw: unknown
  milestones and duplicates (of existing items or of each other) are dropped and counted;
* ``ProposalStore`` — stores proposal sets under ``<workspace>/selfdev/proposals/`` and
  records the owner's decision on each proposal (accepted / rejected), each file written
  once with ``os.link``.

Accepting a proposal only records the decision. Nothing here writes ``ROADMAP.md`` or
any repository file; ``roadmap_line`` gives the exact line the owner can add by hand.

Layout::

    <set_id>.json        the stored set (its proposals never change)
    <set_id>.d<n>.json   the decision on proposal ``n`` (1-based), at most one

Symlinks, non-regular or oversized files, invalid JSON, ids that do not match the file
name, orphaned decisions and unknown files raise ``ProposalStoreError`` (fail closed).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import threading
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from automa.memory.models import normalize_text
from automa.plan.models import PlanError, _format_errors, strip_code_fence
from automa.project.store import ProjectConflict, ProjectStoreError, _write_json_exclusive
from automa.selfdev import MAX_MILESTONE_CHARS, ItemStatus, Roadmap, RoadmapError, parse_roadmap

PROPOSAL_SCHEMA_VERSION = 1
PROPOSALS_DIR = Path("selfdev") / "proposals"
MAX_PROPOSALS = 5
MAX_PROPOSAL_TEXT_CHARS = 1_000
MIN_PROPOSAL_TEXT_CHARS = 10
MAX_REASON_CHARS = 500
MAX_SETS = 200
MAX_SET_FILE_BYTES = 64 * 1024
MAX_DECISION_FILE_BYTES = 4 * 1024

SET_ID_PATTERN = r"^sdp-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$"
_SET_ID_RE = re.compile(SET_ID_PATTERN)
_SET_FILE_RE = re.compile(r"^(sdp-[0-9]{8}-[0-9]{6}-[0-9a-f]{6})\.json$")
_DECISION_FILE_RE = re.compile(r"^(sdp-[0-9]{8}-[0-9]{6}-[0-9a-f]{6})\.d([1-9][0-9]*)\.json$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = r"^[0-9a-f]{64}$"
# Text that would look like Markdown structure at the start of a roadmap line.
_LEADING_STRUCTURE_RE = re.compile(r"^(\[.?\]|#|>|[-*+](\s|$)|[0-9]+[.)](\s|$)|```|~~~|<)")
_ALLOWED_FORMAT_CHARS = frozenset({"\u200c", "\u200d"})  # Persian joiners (ZWNJ, ZWJ)
# Every character str.splitlines() breaks on (the roadmap parser reads lines with it).
_LINE_BREAKS = frozenset("\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ProposalStoreError(Exception):
    """A proposal set cannot be stored, read or decided."""


class ProposalNotFound(ProposalStoreError):
    pass


class DecisionStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


def _single_line(v: str, *, what: str, max_chars: int, min_chars: int = 1) -> str:
    """NFC, whitespace collapsed to single spaces; no control or format characters."""
    if not isinstance(v, str):
        raise ValueError(f"{what} must be a string")
    v = unicodedata.normalize("NFC", v)
    for ch in v:
        if ch in _ALLOWED_FORMAT_CHARS:
            continue
        if ch in _LINE_BREAKS:
            raise ValueError(f"{what} must be a single line")
        if ch.isspace():
            continue
        if unicodedata.category(ch) in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            raise ValueError(f"{what} must not contain control or format characters")
    v = " ".join(v.split())
    if len(v) < min_chars:
        raise ValueError(f"{what} must have at least {min_chars} characters")
    if len(v) > max_chars:
        raise ValueError(f"{what} is longer than {max_chars} characters")
    return v


def roadmap_line(text: str) -> str:
    """The exact ROADMAP line for a proposal text (always a new ``[ ]`` item)."""
    return f"- [ ] {text}"


class ItemProposal(BaseModel):
    """One proposed roadmap item: always a new ``[ ]`` item under an existing milestone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    milestone: str
    text: str
    reason: str = ""

    @field_validator("milestone")
    @classmethod
    def _milestone(cls, v: str) -> str:
        return _single_line(v, what="milestone", max_chars=MAX_MILESTONE_CHARS)

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        v = _single_line(
            v,
            what="text",
            max_chars=MAX_PROPOSAL_TEXT_CHARS,
            min_chars=MIN_PROPOSAL_TEXT_CHARS,
        )
        if _LEADING_STRUCTURE_RE.match(v):
            raise ValueError("text must not start with Markdown structure (marker, heading, list)")
        if "```" in v or "~~~" in v:
            raise ValueError("text must not contain a code fence")
        # The line must parse back as exactly this one open item (D80 parser).
        try:
            parsed = parse_roadmap(f"## m\n{roadmap_line(v)}\n")
        except RoadmapError as e:
            raise ValueError(f"text is not a valid roadmap item: {e}") from None
        if (
            len(parsed.items) != 1
            or parsed.items[0].text != v
            or parsed.items[0].status is not ItemStatus.TODO
        ):
            raise ValueError("text is not a valid roadmap item")
        return v

    @field_validator("reason")
    @classmethod
    def _reason(cls, v: str) -> str:
        if v == "":
            return v
        return _single_line(v, what="reason", max_chars=MAX_REASON_CHARS)

    @property
    def line(self) -> str:
        return roadmap_line(self.text)


class ProposalReply(BaseModel):
    """The model's reply. An empty list means "nothing worth adding"."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposals: list[ItemProposal] = Field(default_factory=list, max_length=MAX_PROPOSALS)


def parse_proposal_reply(text: str) -> ProposalReply:
    """Parse the model's reply. Raises ``PlanError`` with model-readable details."""
    if not isinstance(text, str):
        raise PlanError("proposal reply must be text", "- <root>: expected text")
    raw = strip_code_fence(text)
    try:
        data: Any = json.loads(raw)
    except ValueError as e:
        raise PlanError("proposals are not valid JSON", f"- <root>: invalid JSON ({e})") from e
    if not isinstance(data, dict):
        raise PlanError("proposals must be a JSON object", "- <root>: expected a JSON object")
    try:
        return ProposalReply.model_validate(data)
    except ValidationError as e:
        raise PlanError("proposals failed validation", _format_errors(e)) from e


class Screened(BaseModel):
    """Result of ``screen_proposals``: what is kept and how much was dropped."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposals: list[ItemProposal] = Field(default_factory=list, max_length=MAX_PROPOSALS)
    dropped_duplicates: int = Field(default=0, ge=0)
    dropped_unknown_milestone: int = Field(default=0, ge=0)


def screen_proposals(reply: ProposalReply, roadmap: Roadmap) -> Screened:
    """Drop proposals for milestones the roadmap does not have and duplicates.

    Duplicates are compared with ``normalize_text`` against every roadmap item (any
    status, so a done or ``[!]`` item is never proposed again) and against earlier
    proposals of the same reply.
    """
    milestones = {normalize_text(m): m for m in roadmap.milestones}
    seen = {normalize_text(item.text) for item in roadmap.items}
    kept: list[ItemProposal] = []
    duplicates = 0
    unknown = 0
    for p in reply.proposals:
        milestone = milestones.get(normalize_text(p.milestone))
        if milestone is None:
            unknown += 1
            continue
        key = normalize_text(p.text)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if p.milestone != milestone:
            p = p.model_copy(update={"milestone": milestone})
        kept.append(p)
    return Screened(
        proposals=kept, dropped_duplicates=duplicates, dropped_unknown_milestone=unknown
    )


def roadmap_digest(text: str) -> str:
    """SHA-256 of the roadmap text the proposals were made against."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_set_id(set_id: str) -> str:
    if not isinstance(set_id, str) or not _SET_ID_RE.fullmatch(set_id):
        raise ProposalStoreError(f"invalid proposal set id: {set_id!r}")
    return set_id


def _aware(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return v.astimezone(UTC)


class ProposalSet(BaseModel):
    """Contents of ``<set_id>.json``. Immutable; written once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = PROPOSAL_SCHEMA_VERSION
    id: str = Field(pattern=SET_ID_PATTERN)
    # The commit of the snapshot the model read (None: a working-tree roadmap).
    commit: str | None = None
    roadmap_sha256: str = Field(pattern=_SHA256_RE)
    proposals: list[ItemProposal] = Field(default_factory=list, max_length=MAX_PROPOSALS)
    dropped_duplicates: int = Field(default=0, ge=0)
    dropped_unknown_milestone: int = Field(default=0, ge=0)
    saved_at: datetime

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != PROPOSAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported proposal schema version {v}")
        return v

    @field_validator("commit")
    @classmethod
    def _commit(cls, v: str | None) -> str | None:
        if v is not None and not _COMMIT_RE.fullmatch(v):
            raise ValueError("commit must be a full 40-character lowercase hex id")
        return v

    @field_validator("saved_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _aware(v)

    @model_validator(mode="after")
    def _no_duplicates(self) -> ProposalSet:
        seen: set[str] = set()
        for p in self.proposals:
            key = normalize_text(p.text)
            if key in seen:
                raise ValueError("a proposal set cannot propose the same item twice")
            seen.add(key)
        return self

    def proposal(self, number: int) -> ItemProposal:
        """Proposal ``number`` (1-based, as shown to the owner)."""
        if isinstance(number, bool) or not isinstance(number, int):
            raise ProposalStoreError(f"invalid proposal number: {number!r}")
        if not 1 <= number <= len(self.proposals):
            raise ProposalStoreError(
                f"proposal set {self.id!r} has no proposal {number} (it has {len(self.proposals)})"
            )
        return self.proposals[number - 1]


class ProposalDecision(BaseModel):
    """Contents of ``<set_id>.d<n>.json``. Immutable; written once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = PROPOSAL_SCHEMA_VERSION
    set_id: str = Field(pattern=SET_ID_PATTERN)
    number: int = Field(ge=1, le=MAX_PROPOSALS)
    status: DecisionStatus
    decided_at: datetime

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != PROPOSAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported proposal schema version {v}")
        return v

    @field_validator("decided_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _aware(v)


def _read_json_file(path: Path, limit: int, what: str) -> Any:
    # O_NONBLOCK: opening a FIFO must not hang; it is then rejected as non-regular.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise ProposalNotFound(f"{path.name}: {what} file is missing") from None
    except OSError as e:
        raise ProposalStoreError(f"{path.name}: cannot open {what} file ({e.strerror})") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ProposalStoreError(f"{path.name}: {what} file is not a regular file")
        if st.st_size > limit:
            raise ProposalStoreError(f"{path.name}: {what} file larger than {limit} bytes")
        with os.fdopen(fd, "rb", closefd=True) as f:
            fd = -1
            data = f.read(limit + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > limit:
        raise ProposalStoreError(f"{path.name}: {what} file larger than {limit} bytes")
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ProposalStoreError(f"{path.name}: unreadable {what} file") from None


class ProposalStore:
    """Stores proposal sets and the owner's decisions. Never writes the repository."""

    def __init__(self, root: str | Path, *, clock: Clock = _utc_now):
        self._root = Path(root)
        self._clock = clock
        self._lock = threading.Lock()

    @classmethod
    def for_workspace(cls, workspace: str | Path, *, clock: Clock = _utc_now) -> ProposalStore:
        return cls(Path(workspace) / PROPOSALS_DIR, clock=clock)

    @property
    def root(self) -> Path:
        return self._root

    def _check_dir(self) -> bool:
        for d in (self._root.parent, self._root):
            if d.is_symlink():
                raise ProposalStoreError(f"{d.name}: proposals directory must not be a symlink")
            if d.exists() and not d.is_dir():
                raise ProposalStoreError(f"{d.name}: proposals path is not a directory")
        return self._root.is_dir()

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ProposalStoreError("clock returned a naive datetime")
        return now.astimezone(UTC)

    # -- reading --------------------------------------------------------------

    def _scan(self) -> tuple[list[str], dict[str, dict[int, Path]]]:
        if not self._check_dir():
            return [], {}
        sets: list[str] = []
        decisions: dict[str, dict[int, Path]] = {}
        for child in sorted(self._root.iterdir(), key=lambda p: p.name):
            name = child.name
            if name.startswith("."):
                continue  # temp file of an in-progress write
            if m := _SET_FILE_RE.fullmatch(name):
                sets.append(m.group(1))
            elif m := _DECISION_FILE_RE.fullmatch(name):
                decisions.setdefault(m.group(1), {})[int(m.group(2))] = child
            else:
                raise ProposalStoreError(f"{name!r}: unexpected entry in the proposals directory")
        for sid in decisions:
            if sid not in sets:
                raise ProposalStoreError(f"decision for missing proposal set {sid!r}")
        return sets, decisions

    def _load_set(self, set_id: str) -> ProposalSet:
        path = self._root / f"{set_id}.json"
        raw = _read_json_file(path, MAX_SET_FILE_BYTES, "proposal set")
        try:
            pset = ProposalSet.model_validate(raw)
        except ValidationError as e:
            raise ProposalStoreError(
                f"{path.name}: invalid proposal set file:\n{_format_errors(e)}"
            ) from None
        if pset.id != set_id:
            raise ProposalStoreError(f"{path.name}: proposal set file does not belong here")
        return pset

    def _load_decision(self, pset: ProposalSet, number: int, path: Path) -> ProposalDecision:
        raw = _read_json_file(path, MAX_DECISION_FILE_BYTES, "decision")
        try:
            decision = ProposalDecision.model_validate(raw)
        except ValidationError as e:
            raise ProposalStoreError(
                f"{path.name}: invalid decision file:\n{_format_errors(e)}"
            ) from None
        if decision.set_id != pset.id or decision.number != number:
            raise ProposalStoreError(f"{path.name}: decision file does not belong here")
        if number > len(pset.proposals):
            raise ProposalStoreError(f"{path.name}: decision for a proposal the set does not have")
        return decision

    def all(self) -> list[ProposalSet]:
        """Every proposal set, oldest first (sets and decisions fully validated)."""
        ids, decisions = self._scan()
        sets = [self._load_set(sid) for sid in ids]
        for pset in sets:
            for number, path in decisions.get(pset.id, {}).items():
                self._load_decision(pset, number, path)
        return sorted(sets, key=lambda s: (s.saved_at, s.id))

    def get(self, set_id: str) -> ProposalSet:
        validate_set_id(set_id)
        ids, _ = self._scan()
        if set_id not in ids:
            raise ProposalNotFound(f"no proposal set {set_id!r}")
        return self._load_set(set_id)

    def decisions(self, set_id: str) -> dict[int, ProposalDecision]:
        """Decisions of one set, keyed by proposal number."""
        validate_set_id(set_id)
        ids, decisions = self._scan()
        if set_id not in ids:
            raise ProposalNotFound(f"no proposal set {set_id!r}")
        pset = self._load_set(set_id)
        return {
            n: self._load_decision(pset, n, path)
            for n, path in sorted(decisions.get(set_id, {}).items())
        }

    def pending(self, set_id: str) -> list[int]:
        """Numbers of the proposals not decided yet."""
        pset = self.get(set_id)
        decided = self.decisions(set_id)
        return [n for n in range(1, len(pset.proposals) + 1) if n not in decided]

    # -- writing --------------------------------------------------------------

    def check_room(self) -> None:
        """Raise ``ProposalStoreError`` if ``save`` would be refused for lack of room or
        an unreadable store. Callers run this *before* paying for a model call."""
        with self._lock:
            ids, _ = self._scan()
            if len(ids) >= MAX_SETS:
                raise ProposalStoreError(f"there are already {MAX_SETS} proposal sets")

    def save(
        self, screened: Screened, *, roadmap_text: str, commit: str | None = None
    ) -> ProposalSet:
        """Store screened proposals (write-once). Writes nothing else."""
        if not isinstance(screened, Screened):
            raise TypeError("save() needs the result of screen_proposals")
        if not isinstance(roadmap_text, str):
            raise TypeError("roadmap_text must be the roadmap the proposals were made against")
        # Re-check that the proposals were screened against this very roadmap.
        try:
            roadmap = parse_roadmap(roadmap_text)
        except RoadmapError as e:
            raise ProposalStoreError(f"invalid roadmap: {e}") from None
        known = {normalize_text(m) for m in roadmap.milestones}
        existing = {normalize_text(i.text) for i in roadmap.items}
        for p in screened.proposals:
            if normalize_text(p.milestone) not in known:
                raise ProposalStoreError("a proposal names a milestone the roadmap does not have")
            if normalize_text(p.text) in existing:
                raise ProposalStoreError("a proposal repeats an existing roadmap item")
        with self._lock:
            ids, _ = self._scan()
            if len(ids) >= MAX_SETS:
                raise ProposalStoreError(f"there are already {MAX_SETS} proposal sets")
            now = self._now()
            try:
                stored = ProposalSet.model_validate(
                    {
                        "id": f"sdp-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}",
                        "commit": commit,
                        "roadmap_sha256": roadmap_digest(roadmap_text),
                        "proposals": [p.model_dump() for p in screened.proposals],
                        "dropped_duplicates": screened.dropped_duplicates,
                        "dropped_unknown_milestone": screened.dropped_unknown_milestone,
                        "saved_at": now,
                    }
                )
            except ValidationError as e:
                raise ProposalStoreError(f"invalid proposal set:\n{_format_errors(e)}") from None
            self._root.mkdir(parents=True, exist_ok=True)
            self._check_dir()
            try:
                _write_json_exclusive(
                    self._root / f"{stored.id}.json", stored.model_dump(mode="json")
                )
            except ProjectStoreError as e:
                raise ProposalStoreError(str(e)) from None
        return stored

    def accept(self, set_id: str, number: int) -> ProposalDecision:
        """Record that proposal ``number`` is accepted. ROADMAP.md is not written."""
        return self._decide(set_id, number, DecisionStatus.ACCEPTED)

    def reject(self, set_id: str, number: int) -> ProposalDecision:
        """Record that proposal ``number`` is rejected."""
        return self._decide(set_id, number, DecisionStatus.REJECTED)

    def _decide(self, set_id: str, number: int, status: DecisionStatus) -> ProposalDecision:
        with self._lock:
            pset = self.get(set_id)
            pset.proposal(number)
            existing = self.decisions(set_id).get(number)
            if existing is not None:
                raise ProposalStoreError(
                    f"proposal {number} of set {set_id!r} is already {existing.status.value}"
                )
            decision = ProposalDecision(
                set_id=set_id, number=number, status=status, decided_at=self._now()
            )
            path = self._root / f"{set_id}.d{number}.json"
            try:
                _write_json_exclusive(path, decision.model_dump(mode="json"))
            except ProjectConflict:
                raise ProposalStoreError(
                    f"proposal {number} of set {set_id!r} was decided concurrently"
                ) from None
            except ProjectStoreError as e:
                raise ProposalStoreError(str(e)) from None
        return decision

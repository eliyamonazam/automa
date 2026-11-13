"""Storage and approval of post-project review proposals (M7, D53).

``ReviewAgent`` (D52) only *proposes* memory entries. This module stores a review
in the project's folder and lets the user decide on each proposal:

* accepted → the proposal becomes one ``MemoryEntry`` with ``source = review``
  (``project`` proposals in that project's memory, ``lesson`` proposals in the
  lessons layer);
* rejected → nothing is written to memory.

Nothing else ever writes a review proposal to memory, and this module never calls
a model.

Layout (``<project>/reviews/``, every file written once with ``os.link``)::

    <review_id>.json        the stored review (its proposals never change)
    <review_id>.d<n>.json   the decision on proposal ``n`` (1-based), at most one

Guarantees (fail closed):

* A review can be stored only for an existing project that is ``done`` or
  ``failed`` and still has the status the review was made for.
* Each proposal is decided at most once, even across processes (exclusive file).
  If the decision cannot be recorded after the memory entry was written, that
  entry is removed again; if the process died in between, accepting again finds
  the identical entry and records it instead of writing a second one.
* Symlinks, non-regular or oversized files, invalid JSON, ids that do not match
  the file name, decisions without a review and unknown files raise
  ``ReviewStoreError``; nothing is repaired or skipped silently.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from automa.agents.reviewer import MAX_PROPOSALS, REVIEWABLE_STATUSES, Proposal, Review
from automa.events import Event
from automa.memory.models import MEMORY_ID_PATTERN, MemoryScope, MemorySource, normalize_text
from automa.memory.store import MemoryStore, MemoryStoreError
from automa.plan.models import _format_errors
from automa.project import ProjectStoreError, validate_project_id
from automa.project.store import ProjectConflict, ProjectStatus, ProjectStore, _write_json_exclusive

REVIEW_SCHEMA_VERSION = 1
REVIEWS_DIR = "reviews"
REVIEW_ID_PATTERN = r"^rev-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$"
MAX_REVIEWS_PER_PROJECT = 50
MAX_REVIEW_FILE_BYTES = 64 * 1024
MAX_DECISION_FILE_BYTES = 4 * 1024

_REVIEW_ID_RE = re.compile(REVIEW_ID_PATTERN)
_REVIEW_FILE_RE = re.compile(r"^(rev-[0-9]{8}-[0-9]{6}-[0-9a-f]{6})\.json$")
_DECISION_FILE_RE = re.compile(r"^(rev-[0-9]{8}-[0-9]{6}-[0-9a-f]{6})\.d([1-9][0-9]*)\.json$")

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ReviewStoreError(Exception):
    """A review cannot be stored, read or decided."""


class ReviewNotFound(ReviewStoreError):
    pass


class DecisionStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


def validate_review_id(review_id: str) -> str:
    if not isinstance(review_id, str) or not _REVIEW_ID_RE.fullmatch(review_id):
        raise ReviewStoreError(f"invalid review id: {review_id!r}")
    return review_id


def _aware(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return v.astimezone(UTC)


class StoredReview(BaseModel):
    """Contents of ``reviews/<id>.json``. Immutable; written once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = REVIEW_SCHEMA_VERSION
    id: str = Field(pattern=REVIEW_ID_PATTERN)
    project_id: str
    project_status: ProjectStatus
    proposals: list[Proposal] = Field(default_factory=list, max_length=MAX_PROPOSALS)
    dropped_duplicates: int = Field(default=0, ge=0)
    created_at: datetime
    saved_at: datetime

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != REVIEW_SCHEMA_VERSION:
            raise ValueError(f"unsupported review schema version {v}")
        return v

    @field_validator("project_id")
    @classmethod
    def _project_id(cls, v: str) -> str:
        try:
            return validate_project_id(v)
        except ProjectStoreError as e:
            raise ValueError(str(e)) from None

    @field_validator("project_status")
    @classmethod
    def _finished(cls, v: ProjectStatus) -> ProjectStatus:
        if v not in REVIEWABLE_STATUSES:
            raise ValueError("only done or failed projects have reviews")
        return v

    @field_validator("created_at", "saved_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _aware(v)

    @model_validator(mode="after")
    def _no_duplicates(self) -> StoredReview:
        seen: set[tuple[str, str]] = set()
        for p in self.proposals:
            key = (p.scope, normalize_text(p.text))
            if key in seen:
                raise ValueError("a review cannot propose the same note twice")
            seen.add(key)
        return self

    def proposal(self, number: int) -> Proposal:
        """Proposal ``number`` (1-based, as shown to the user)."""
        if isinstance(number, bool) or not isinstance(number, int):
            raise ReviewStoreError(f"invalid proposal number: {number!r}")
        if not 1 <= number <= len(self.proposals):
            raise ReviewStoreError(
                f"review {self.id!r} has no proposal {number} (it has {len(self.proposals)})"
            )
        return self.proposals[number - 1]


class ProposalDecision(BaseModel):
    """Contents of ``reviews/<review_id>.d<n>.json``. Immutable; written once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = REVIEW_SCHEMA_VERSION
    review_id: str = Field(pattern=REVIEW_ID_PATTERN)
    project_id: str
    number: int = Field(ge=1, le=MAX_PROPOSALS)
    status: DecisionStatus
    # The memory entry of an accepted proposal (None when rejected).
    memory_id: str | None = Field(default=None, pattern=MEMORY_ID_PATTERN)
    # True when an identical entry already existed and was recorded instead.
    memory_existed: bool = False
    decided_at: datetime

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != REVIEW_SCHEMA_VERSION:
            raise ValueError(f"unsupported review schema version {v}")
        return v

    @field_validator("project_id")
    @classmethod
    def _project_id(cls, v: str) -> str:
        try:
            return validate_project_id(v)
        except ProjectStoreError as e:
            raise ValueError(str(e)) from None

    @field_validator("decided_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _aware(v)

    @model_validator(mode="after")
    def _consistent(self) -> ProposalDecision:
        accepted = self.status is DecisionStatus.ACCEPTED
        if accepted != (self.memory_id is not None):
            raise ValueError("memory_id must be set exactly when the proposal is accepted")
        if self.memory_existed and not accepted:
            raise ValueError("memory_existed only applies to accepted proposals")
        return self


def _read_file(path: Path, limit: int, what: str) -> Any:
    # O_NONBLOCK: opening a FIFO must not hang; it is then rejected as non-regular.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise ReviewNotFound(f"{path}: {what} file is missing") from None
    except OSError as e:
        raise ReviewStoreError(f"{path}: cannot open {what} file ({e.strerror})") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ReviewStoreError(f"{path}: {what} file is not a regular file")
        if st.st_size > limit:
            raise ReviewStoreError(f"{path}: {what} file larger than {limit} bytes")
        with os.fdopen(fd, "rb", closefd=True) as f:
            fd = -1
            data = f.read(limit + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > limit:
        raise ReviewStoreError(f"{path}: {what} file larger than {limit} bytes")
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ReviewStoreError(f"{path}: unreadable {what} file") from None


class ReviewStore:
    """Stores reviews in ``<project>/reviews/`` and turns accepted proposals into memory.

    Never calls a model. Memory is written only by ``accept``, always with
    ``source = review``.
    """

    def __init__(self, projects: ProjectStore, memory: MemoryStore, *, clock: Clock = _utc_now):
        self._projects = projects
        self._memory = memory
        self._clock = clock
        self._lock = threading.Lock()

    @classmethod
    def for_workspace(cls, workspace: str | Path, *, clock: Clock = _utc_now) -> ReviewStore:
        return cls(
            ProjectStore.for_workspace(workspace, clock=clock),
            MemoryStore.for_workspace(workspace, clock=clock),
            clock=clock,
        )

    # -- paths ----------------------------------------------------------------

    def reviews_dir(self, project_id: str) -> Path:
        """``<project>/reviews`` of an existing project (not created here)."""
        self._projects.load(project_id)  # ProjectNotFound / corrupt project fail first
        return self._projects.project_dir(project_id) / REVIEWS_DIR

    @staticmethod
    def _check_dir(directory: Path) -> bool:
        if directory.is_symlink():
            raise ReviewStoreError(f"{directory}: reviews directory must not be a symlink")
        if directory.exists() and not directory.is_dir():
            raise ReviewStoreError(f"{directory}: reviews path is not a directory")
        return directory.is_dir()

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ReviewStoreError("clock returned a naive datetime")
        return now.astimezone(UTC)

    # -- reading --------------------------------------------------------------

    def _scan(self, project_id: str) -> tuple[list[str], dict[str, dict[int, Path]]]:
        """Review ids and decision files; any unknown or orphaned file fails closed."""
        directory = self.reviews_dir(project_id)
        if not self._check_dir(directory):
            return [], {}
        reviews: list[str] = []
        decisions: dict[str, dict[int, Path]] = {}
        for child in sorted(directory.iterdir(), key=lambda p: p.name):
            name = child.name
            if name.startswith("."):
                continue  # temp file of an in-progress write
            if m := _REVIEW_FILE_RE.fullmatch(name):
                reviews.append(m.group(1))
            elif m := _DECISION_FILE_RE.fullmatch(name):
                decisions.setdefault(m.group(1), {})[int(m.group(2))] = child
            else:
                raise ReviewStoreError(f"{child}: unexpected entry in the reviews directory")
        for rid in decisions:
            if rid not in reviews:
                raise ReviewStoreError(f"{directory}: decision for missing review {rid!r}")
        return reviews, decisions

    def _load_review(self, project_id: str, review_id: str) -> StoredReview:
        path = self.reviews_dir(project_id) / f"{review_id}.json"
        raw = _read_file(path, MAX_REVIEW_FILE_BYTES, "review")
        try:
            review = StoredReview.model_validate(raw)
        except ValidationError as e:
            raise ReviewStoreError(f"{path}: invalid review file:\n{_format_errors(e)}") from None
        if review.id != review_id or review.project_id != project_id:
            raise ReviewStoreError(f"{path}: review file does not belong here")
        return review

    def _load_decision(self, review: StoredReview, number: int, path: Path) -> ProposalDecision:
        raw = _read_file(path, MAX_DECISION_FILE_BYTES, "decision")
        try:
            decision = ProposalDecision.model_validate(raw)
        except ValidationError as e:
            raise ReviewStoreError(f"{path}: invalid decision file:\n{_format_errors(e)}") from None
        if (
            decision.review_id != review.id
            or decision.project_id != review.project_id
            or decision.number != number
        ):
            raise ReviewStoreError(f"{path}: decision file does not belong here")
        if number > len(review.proposals):
            raise ReviewStoreError(f"{path}: decision for a proposal the review does not have")
        return decision

    def all(self, project_id: str) -> list[StoredReview]:
        """All reviews of a project, oldest first (each fully validated)."""
        ids, decisions = self._scan(project_id)
        reviews = [self._load_review(project_id, rid) for rid in ids]
        for review in reviews:  # validate decision files too: fail closed everywhere
            for number, path in decisions.get(review.id, {}).items():
                self._load_decision(review, number, path)
        return sorted(reviews, key=lambda r: (r.saved_at, r.id))

    def latest(self, project_id: str) -> StoredReview | None:
        reviews = self.all(project_id)
        return reviews[-1] if reviews else None

    def get(self, project_id: str, review_id: str) -> StoredReview:
        validate_review_id(review_id)
        ids, _ = self._scan(project_id)
        if review_id not in ids:
            raise ReviewNotFound(f"project {project_id!r} has no review {review_id!r}")
        return self._load_review(project_id, review_id)

    def decisions(self, project_id: str, review_id: str) -> dict[int, ProposalDecision]:
        """Decisions of one review, keyed by proposal number."""
        validate_review_id(review_id)
        ids, decisions = self._scan(project_id)
        if review_id not in ids:
            raise ReviewNotFound(f"project {project_id!r} has no review {review_id!r}")
        review = self._load_review(project_id, review_id)
        return {
            n: self._load_decision(review, n, path)
            for n, path in sorted(decisions.get(review_id, {}).items())
        }

    def pending(self, project_id: str, review_id: str) -> list[int]:
        """Numbers of the proposals not decided yet."""
        review = self.get(project_id, review_id)
        decided = self.decisions(project_id, review_id)
        return [n for n in range(1, len(review.proposals) + 1) if n not in decided]

    # -- writing --------------------------------------------------------------

    def save(self, review: Review) -> StoredReview:
        """Store a review (write-once) and log ``review.saved``. Writes no memory."""
        if not isinstance(review, Review):
            raise TypeError("save() needs a Review from ReviewAgent")
        try:
            review = Review.model_validate(review.model_dump())
        except ValidationError as e:
            raise ReviewStoreError(f"invalid review:\n{_format_errors(e)}") from None
        project_id = review.project_id
        with self._lock:
            project = self._projects.load(project_id)
            if project.status not in REVIEWABLE_STATUSES:
                raise ReviewStoreError(
                    f"project {project_id!r} is {project.status.value}; "
                    "only reviews of done or failed projects can be stored"
                )
            if project.status is not review.project_status:
                raise ReviewStoreError(
                    f"review was made when project {project_id!r} was "
                    f"{review.project_status.value}, but it is now {project.status.value}"
                )
            ids, _ = self._scan(project_id)
            if len(ids) >= MAX_REVIEWS_PER_PROJECT:
                raise ReviewStoreError(
                    f"project {project_id!r} already has {MAX_REVIEWS_PER_PROJECT} reviews"
                )
            now = self._now()
            try:
                stored = StoredReview.model_validate(
                    {
                        "id": f"rev-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}",
                        "project_id": project_id,
                        "project_status": review.project_status,
                        "proposals": [p.model_dump() for p in review.proposals],
                        "dropped_duplicates": review.dropped_duplicates,
                        "created_at": review.created_at,
                        "saved_at": now,
                    }
                )
            except ValidationError as e:
                raise ReviewStoreError(f"invalid review:\n{_format_errors(e)}") from None
            directory = self.reviews_dir(project_id)
            directory.mkdir(exist_ok=True)
            self._check_dir(directory)
            try:
                _write_json_exclusive(
                    directory / f"{stored.id}.json", stored.model_dump(mode="json")
                )
            except ProjectStoreError as e:
                raise ReviewStoreError(str(e)) from None
        self._emit(
            "review.saved",
            project_id,
            {
                "review_id": stored.id,
                "proposals": len(stored.proposals),
                "lessons": sum(1 for p in stored.proposals if p.scope == "lesson"),
            },
        )
        return stored

    def accept(self, project_id: str, review_id: str, number: int) -> ProposalDecision:
        """Write proposal ``number`` to memory (``source = review``) and record the decision."""
        return self._decide(project_id, review_id, number, accept=True)

    def reject(self, project_id: str, review_id: str, number: int) -> ProposalDecision:
        """Record that proposal ``number`` is rejected. Writes no memory."""
        return self._decide(project_id, review_id, number, accept=False)

    def _decide(
        self, project_id: str, review_id: str, number: int, *, accept: bool
    ) -> ProposalDecision:
        with self._lock:
            review = self.get(project_id, review_id)
            proposal = review.proposal(number)
            existing = self.decisions(project_id, review_id).get(number)
            if existing is not None:
                raise ReviewStoreError(
                    f"proposal {number} of review {review_id!r} is already {existing.status.value}"
                )
            memory_id: str | None = None
            memory_existed = False
            created = False
            if accept:
                memory_id, memory_existed = self._write_memory(project_id, proposal)
                created = not memory_existed
            now = self._now()
            decision = ProposalDecision(
                review_id=review_id,
                project_id=project_id,
                number=number,
                status=DecisionStatus.ACCEPTED if accept else DecisionStatus.REJECTED,
                memory_id=memory_id,
                memory_existed=memory_existed,
                decided_at=now,
            )
            path = self.reviews_dir(project_id) / f"{review_id}.d{number}.json"
            try:
                _write_json_exclusive(path, decision.model_dump(mode="json"))
            except BaseException as e:
                # Never leave a memory entry behind without its recorded decision.
                if created and memory_id is not None:
                    self._memory.forget(memory_id)
                if isinstance(e, ProjectConflict):
                    raise ReviewStoreError(
                        f"proposal {number} of review {review_id!r} was decided concurrently"
                    ) from None
                if isinstance(e, ProjectStoreError):
                    raise ReviewStoreError(str(e)) from None
                raise
        data: dict[str, Any] = {
            "review_id": review_id,
            "number": number,
            "scope": proposal.scope,
        }
        if accept:
            data["memory_id"] = memory_id
            data["memory_existed"] = memory_existed
        self._emit(
            "review.proposal_accepted" if accept else "review.proposal_rejected", project_id, data
        )
        return decision

    def _write_memory(self, project_id: str, proposal: Proposal) -> tuple[str, bool]:
        """The id of the memory entry for ``proposal`` and whether it already existed."""
        scope = proposal.memory_scope
        pid = project_id if scope is MemoryScope.PROJECT else None
        key = normalize_text(proposal.text)
        try:
            for entry in self._memory.entries(scope, pid):
                if normalize_text(entry.text) == key:
                    return entry.id, True
            entry = self._memory.add(
                scope, proposal.text, project_id=pid, source=MemorySource.REVIEW
            )
        except MemoryStoreError as e:
            raise ReviewStoreError(f"cannot write the accepted proposal to memory: {e}") from e
        return entry.id, False

    def _emit(self, type_: str, project_id: str, data: dict[str, Any]) -> None:
        self._projects.events(project_id).emit(
            Event(ts=self._now(), type=type_, project_id=project_id, data=data)
        )

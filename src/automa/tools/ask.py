"""``ask_user``: the worker asks the user for something only they can give (D159).

Not a permission-governed action: it reaches nothing outside the project, it only
records a question (``UserRequestStore``) and pauses the project. So the executor
offers it to every task that has a request store, whatever the role's tools are,
and runs it without the ``PermissionPolicy`` (it can never widen what a role may
do: the answer is data, and files the user gives land in ``files/inputs/``).

* The same question already answered → the answer comes back at once (no pause).
* Already declined → a result that says so: continue without it, never invent it.
* Otherwise the question is recorded (or the identical pending one reused) and
  ``UserInputRequired`` is raised: the attempt stops, nothing is stored for it,
  and the orchestrator pauses the project (``StopReason.USER_INPUT``).
* A task may ask at most ``MAX_REQUESTS_PER_TASK`` different questions.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from automa.tools.base import RiskLevel, Tool, ToolArgs, ToolError
from automa.tools.workspace import Workspace
from automa.userinput import (
    MAX_QUESTION_CHARS,
    MAX_REQUESTS_PER_TASK,
    MAX_WHY_CHARS,
    RequestStatus,
    UserRequest,
    UserRequestError,
    UserRequestStore,
)

ASK_TOOL = "ask_user"


class UserInputRequired(Exception):  # noqa: N818 - a control-flow signal, not an error
    """The task needs an answer from the user; the project should pause."""

    def __init__(self, request_id: str, question: str) -> None:
        super().__init__(question)
        self.request_id = request_id
        self.question = question
        # Set by the worker (D166): the attempt so far, so it can continue later.
        self.suspended: object | None = None


class AskUserArgs(ToolArgs):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description=(
            "The exact thing you need, as a short clear request in the user's language "
            "(for example: 'Please upload the sales spreadsheet for 2025')."
        ),
    )
    why: str = Field(
        default="",
        max_length=MAX_WHY_CHARS,
        description="Why you need it and what you will do with it, in one sentence.",
    )
    kind: Literal["text", "file"] = Field(
        default="text", description="'file' when you need the user to upload a file."
    )


class AskUser(Tool):
    name: ClassVar[str] = ASK_TOOL
    description: ClassVar[str] = (
        "Ask the user for something only they can provide: a file, data you cannot "
        "reach (no access, login, paid source, live data), a key or account detail, or "
        "a decision that changes the result. The work pauses until they answer; their "
        "answer (and uploaded files, under inputs/) is given to you when the work "
        "continues. Use it instead of guessing, inventing data or delivering a fake "
        "or partial result. Do not use it for things you can do or find yourself."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.READ
    args_model: ClassVar[type[ToolArgs]] = AskUserArgs

    def __init__(
        self, store: UserRequestStore, task_id: str, *, max_requests: int = MAX_REQUESTS_PER_TASK
    ) -> None:
        if not isinstance(store, UserRequestStore):
            raise TypeError("store must be a UserRequestStore")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id is required")
        self._store = store
        self._task_id = task_id
        self._max = max_requests

    @property
    def task_id(self) -> str:
        return self._task_id

    def answered(self) -> list[UserRequest]:
        """Answered or declined requests of this task (oldest first)."""
        try:
            return [
                r
                for r in self._store.for_task(self._task_id)
                if r.status is not RequestStatus.PENDING
            ]
        except UserRequestError:
            return []

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        return self.ask(args)

    def ask(self, args: ToolArgs) -> str:
        """The answer text, or ``UserInputRequired`` (the executor's only entry point)."""
        assert isinstance(args, AskUserArgs)  # noqa: S101 - the executor validated it
        if not args.question.strip():
            raise ToolError("question must not be blank")
        try:
            found = self._store.find(self._task_id, args.question)
            if found is not None and found.status is not RequestStatus.PENDING:
                return found.as_text()
            if found is None and len(self._store.for_task(self._task_id)) >= self._max:
                raise ToolError(
                    f"question limit reached ({self._max} per task): do not ask again. "
                    "Finish with what you have and say clearly in your final summary "
                    "what is missing."
                )
            req = self._store.request(self._task_id, args.question, why=args.why, kind=args.kind)
        except UserRequestError as e:
            raise ToolError(f"could not record the question: {e}") from None
        raise UserInputRequired(req.id, req.question)

    def audit(self, args: ToolArgs) -> dict[str, str]:
        assert isinstance(args, AskUserArgs)  # noqa: S101
        return {"kind": args.kind, "question": args.question[:200]}

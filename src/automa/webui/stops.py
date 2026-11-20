"""Why a paused project stopped, in plain words, and what to do next (M13, D151).

Two pure steps:

* :func:`latest_stop` reads the project's recorded events (``orchestrator.*`` and the
  BudgetGuard's ``budget.stop``) and returns a structured :class:`StopInfo` for the
  latest run, or ``None`` when that run did not end in a pause. Recorded error text
  (``detail``) is only *matched* against fixed patterns to tell a gateway balance,
  key or network problem apart; it is never shown.
* :func:`describe_stop` turns a :class:`StopInfo` into one sentence in the page
  language and says whether continuing makes sense right now (for the daily cap:
  only after it resets at 00:00 UTC, shown in the local time zone).

Nothing here writes, calls a model, authorizes or starts anything: a "Continue"
button is the same start form (CSRF, same-origin, cost confirmation) as the run
panel's, and pressing it early only meets the same BudgetGuard again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict

from automa.events import Event
from automa.l10n import LANGUAGES, Language, absolute_time, localize_digits
from automa.webui.i18n import check_messages, template_fields

StopKind = Literal[
    "daily_cap",
    "project_cap",
    "per_call_cap",
    "budget",
    "balance",
    "key",
    "network",
    "provider",
    "approval",
    "user",
    "verification",
    "tool_layer",
    "config",
    "task_brake",
    "other",
]
StopTone = Literal["info", "bad"]

STARTED = "orchestrator.started"
PAUSED = "orchestrator.paused"
FINISHED = "orchestrator.finished"
BUDGET_STOP = "budget.stop"

_SCOPE_KIND: Mapping[str, StopKind] = MappingProxyType(
    {"daily": "daily_cap", "project": "project_cap", "per_call": "per_call_cap"}
)
_REASON_KIND: Mapping[str, StopKind] = MappingProxyType(
    {
        "approval": "approval",
        "user": "user",
        "verification": "verification",
        "tool_layer": "tool_layer",
        "config": "config",
        "task_brake": "task_brake",
    }
)
# Matched against the lower-cased recorded provider error, never shown.
_BALANCE_MARKS = (
    "http 402",
    "insufficient",
    "balance",
    "credit",
    "quota",
    "payment",
    "موجودی",
)
_KEY_MARKS = ("http 401", "http 403", "api key", "api_key", "unauthorized", "forbidden")
_NETWORK_MARKS = ("could not connect", "timed out", "network error")

# Continuing makes sense once the user has done the next step (or right away).
_RESUMABLE: frozenset[StopKind] = frozenset(
    {"balance", "network", "provider", "approval", "user", "verification", "tool_layer"}
)
_INFO_TONE: frozenset[StopKind] = frozenset({"approval", "user"})


class StopInfo(BaseModel):
    """The structured reason of the latest pause (no recorded text)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: StopKind
    at: datetime
    reset_at: datetime | None = None  # daily cap: when the next UTC day starts


def classify_provider(detail: str) -> StopKind:
    """``balance`` / ``key`` / ``network`` / ``provider`` from a provider error text."""
    low = detail.lower()
    if any(m in low for m in _BALANCE_MARKS):
        return "balance"
    if any(m in low for m in _KEY_MARKS):
        return "key"
    if any(m in low for m in _NETWORK_MARKS):
        return "network"
    return "provider"


def _day_reset(key: object, at: datetime) -> datetime:
    """00:00 UTC of the day after the capped day (its key, else the stop's day)."""
    try:
        day = date.fromisoformat(key) if isinstance(key, str) else at.astimezone(UTC).date()
    except ValueError:
        day = at.astimezone(UTC).date()
    return datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=UTC)


def latest_stop(events: Sequence[Event]) -> StopInfo | None:
    """Why the latest run paused, or ``None`` (still running, finished, never run)."""
    start = max((i for i, e in enumerate(events) if e.type == STARTED), default=-1)
    run = events[start + 1 :]
    if any(e.type == FINISHED for e in run):
        return None
    pause = next((e for e in reversed(run) if e.type == PAUSED), None)
    budget = next((e for e in reversed(run) if e.type == BUDGET_STOP), None)
    if pause is None:
        if budget is None:
            return None
        return _budget_stop(budget)
    reason = pause.data.get("reason")
    if reason == "budget":
        return _budget_stop(budget) if budget is not None else StopInfo(kind="budget", at=pause.ts)
    if reason == "provider":
        detail = pause.data.get("detail")
        kind = classify_provider(detail if isinstance(detail, str) else "")
        return StopInfo(kind=kind, at=pause.ts)
    return StopInfo(kind=_REASON_KIND.get(str(reason), "other"), at=pause.ts)


def _budget_stop(event: Event) -> StopInfo:
    kind = _SCOPE_KIND.get(str(event.data.get("scope")), "budget")
    reset = _day_reset(event.data.get("key"), event.ts) if kind == "daily_cap" else None
    return StopInfo(kind=kind, at=event.ts, reset_at=reset)


_RAW: dict[str, dict[Language, str]] = {
    "stop.daily_wait": {
        "en": "Paused: today's spending cap (daily_usd) is used up. It can continue from {when}.",
        "fa": "متوقف شد: سقف خرج روزانه (daily_usd) پر شده؛ از {when} می‌شه ادامه داد.",
    },
    "stop.daily_ready": {
        "en": "Paused by the daily spending cap, which has reset since. It can continue now.",
        "fa": "به‌خاطر سقف خرج روزانه متوقف شده بود؛ سقف دوباره باز شده و الان می‌شه ادامه داد.",
    },
    "stop.when.today": {"en": "today at {clock}", "fa": "امروز ساعت {clock}"},
    "stop.when.tomorrow": {"en": "tomorrow at {clock}", "fa": "فردا ساعت {clock}"},
    "stop.project_cap": {
        "en": "Paused: the project reached its spending cap. Raise project_usd in "
        "automa.toml, then press Start to finish.",
        "fa": "متوقف شد: خرج پروژه به سقفش رسید. در automa.toml مقدار project_usd رو "
        "بیشتر کن و بعد «شروع» رو بزن تا تموم بشه.",
    },
    "stop.per_call_cap": {
        "en": "Paused: one model call would cost more than the per-call cap. Raise "
        "per_call_usd in automa.toml, then continue.",
        "fa": "متوقف شد: یک درخواست به مدل از سقف هر درخواست بیشتر می‌شد. در automa.toml "
        "مقدار per_call_usd رو بیشتر کن و بعد ادامه بده.",
    },
    "stop.budget": {
        "en": "Paused: the spending guard did not allow going on (see the log).",
        "fa": "متوقف شد: نگهبان هزینه اجازهٔ ادامه نداد (گزارش رو ببین).",
    },
    "stop.balance": {
        "en": "Paused: the model gateway account is out of credit. Top it up, then continue.",
        "fa": "متوقف شد: موجودی حساب درگاه مدل تموم شده؛ حساب رو شارژ کن و بعد ادامه بده.",
    },
    "stop.key": {
        "en": "Paused: the model gateway did not accept the key. Check the key in .env.",
        "fa": "متوقف شد: درگاه مدل کلید رو قبول نکرد؛ کلید داخل فایل .env رو بررسی کن.",
    },
    "stop.network": {
        "en": "Paused: could not reach the model gateway (internet or the gateway "
        "itself). Continue when you are online.",
        "fa": "متوقف شد: به درگاه مدل وصل نشد (اینترنت یا خود درگاه)؛ وقتی وصل بودی ادامه بده.",
    },
    "stop.provider": {
        "en": "Paused: the model gateway returned an error. Try to continue a bit later.",
        "fa": "متوقف شد: درگاه مدل خطا داد؛ کمی بعد دوباره ادامه بده.",
    },
    "stop.approval": {
        "en": "Paused for your approval. Once you decided, it can continue.",
        "fa": "منتظر تأیید تو متوقف شد؛ وقتی تصمیم گرفتی، می‌شه ادامه داد.",
    },
    "stop.user": {
        "en": "Stopped as you asked. Continue whenever you like.",
        "fa": "به درخواست تو متوقف شد؛ هر وقت خواستی ادامه بده.",
    },
    "stop.verification": {
        "en": "Paused: checking a step gave no usable answer. It can try again.",
        "fa": "متوقف شد: بررسی یک مرحله جواب قابل‌استفاده نداد؛ می‌شه دوباره امتحان کرد.",
    },
    "stop.tool_layer": {
        "en": "Paused: the tool layer failed (see the log). It can try again.",
        "fa": "متوقف شد: لایهٔ ابزارها خطا داد (گزارش رو ببین)؛ می‌شه دوباره امتحان کرد.",
    },
    "stop.config": {
        "en": "Paused: the settings are incomplete (for example no model for this kind "
        "of task). Check automa.toml.",
        "fa": "متوقف شد: تنظیمات کامل نیست (مثلاً برای این نوع کار مدلی تعریف نشده)؛ "
        "automa.toml رو بررسی کن.",
    },
    "stop.task_brake": {
        "en": "Paused by the spending brake: one step cost far more than estimated. "
        "To allow one more step of the same size, run in the terminal: "
        "automa brake continue <project> <task>",
        "fa": "ترمز خرج متوقفش کرد: یک مرحله خیلی بیشتر از پیش‌بینی خرج کرد. اگه ادامه "
        "می‌ارزه، در ترمینال بزن: automa brake continue <project> <task>",
    },
    "stop.other": {
        "en": "Paused (see the log).",
        "fa": "متوقف شد (گزارش رو ببین).",
    },
    "stop.continue": {"en": "Continue", "fa": "ادامه"},
}

check_messages(_RAW)
MESSAGES: Mapping[str, Mapping[Language, str]] = MappingProxyType(
    {k: MappingProxyType(dict(v)) for k, v in _RAW.items()}
)


def st(key: str, lang: Language, **params: str) -> str:
    """Like :func:`automa.webui.i18n.t` for this module's table."""
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language: {lang!r}")
    template = MESSAGES[key][lang]
    if set(params) != template_fields(template):
        raise ValueError(f"{key}: parameters {sorted(params)}")
    return template.format(**params)


@dataclass(frozen=True)
class StopLine:
    text: str  # plain text (escaped by the caller)
    tone: StopTone
    resume: bool  # continuing now makes sense (a Continue button may be offered)


def _when(reset: datetime, now: datetime, tz: tzinfo, lang: Language) -> str:
    local = reset.astimezone(tz)
    days = (local.date() - now.astimezone(tz).date()).days
    clock = localize_digits(f"{local.hour}:{local.minute:02d}", lang)
    if days == 0:
        return st("stop.when.today", lang, clock=clock)
    if days == 1:
        return st("stop.when.tomorrow", lang, clock=clock)
    return absolute_time(reset, tz, lang, now=now)


def describe_stop(stop: StopInfo, *, lang: Language, tz: tzinfo, now: datetime) -> StopLine:
    """The sentence and next step of ``stop`` at ``now`` (local times in ``tz``)."""
    if stop.kind == "daily_cap" and stop.reset_at is not None:
        if now >= stop.reset_at:
            return StopLine(st("stop.daily_ready", lang), "info", True)
        when = _when(stop.reset_at, now, tz, lang)
        return StopLine(st("stop.daily_wait", lang, when=when), "bad", False)
    kind = "budget" if stop.kind == "daily_cap" else stop.kind
    tone: StopTone = "info" if kind in _INFO_TONE else "bad"
    return StopLine(st(f"stop.{kind}", lang), tone, kind in _RESUMABLE)

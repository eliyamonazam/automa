"""Home-page project cards (M16, D160): one delivery-centred state per project.

Pure: from what :func:`automa.webui.views.build_overview` already read (project
status, delivery state, open questions and approvals) to one of a few states the
user cares about — "waiting for your answer", "ready", "incomplete", "working",
"stopped", "ready to start" — with a fixed inline SVG icon, a short label and one
plain sentence. Waiting for the user always comes first: nothing moves until they
answer. The icons are fixed markup (no recorded text, no external resource), so they
are safe inside the page under the strict CSP.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from automa.l10n import LANGUAGES, Language, localize_digits
from automa.webui.i18n import check_messages, template_fields


class CardState(StrEnum):
    WAITING = "waiting"  # an open question, unanswered idea questions or an approval
    WITH_OWNER = "with_owner"  # a member's project: an approval only the owner decides (D186)
    READY = "ready"  # delivery ready: done with every main file present
    PARTIAL = "partial"  # delivery incomplete: failed, or done with a main file missing
    WORKING = "working"  # planning or running right now
    PAUSED = "paused"  # stopped mid-way; can be continued
    TO_START = "to_start"  # an idea or a plan that has not started
    UNKNOWN = "unknown"  # an unrecognised recorded status: shown as recorded, never guessed


KNOWN_STATUSES = frozenset({"new", "planning", "running", "paused", "done", "failed"})


def card_state(
    status: str,
    delivery: str | None,
    *,
    open_asks: int = 0,
    pending_approvals: int = 0,
    owner_approvals: int = 0,
    questions_waiting: bool = False,
    running: bool = False,
) -> CardState:
    """The card state of one project.

    ``status`` is the recorded project status, ``delivery`` the
    :class:`automa.delivery.DeliveryState` value (``None`` before a plan or when the
    report could not be built). Anything waiting for the user wins; then a finished
    delivery; then whether work is going on. A project marked ``done`` whose delivery
    could not be built is never shown as ready (it is "incomplete": the product could
    not be checked).

    ``owner_approvals`` (D186) are approvals of a member's project seen by that
    member: only the owner decides them, so they never say "waiting for you"; the
    member's own questions still come first."""
    if open_asks > 0 or pending_approvals > 0 or questions_waiting:
        return CardState.WAITING
    if owner_approvals > 0:
        return CardState.WITH_OWNER
    if status not in KNOWN_STATUSES:
        return CardState.UNKNOWN
    if delivery == "ready":
        return CardState.READY
    if delivery == "partial" or status in ("done", "failed"):
        return CardState.PARTIAL
    if running or status in ("running", "planning"):
        return CardState.WORKING
    if status == "paused" and delivery != "not_started":
        return CardState.PAUSED
    return CardState.TO_START


# Inline SVG icons (24x24, stroke = currentColor). Fixed strings, never recorded text.
_SVG = (
    '<svg class="icon" viewBox="0 0 24 24" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true" focusable="false">{}</svg>'
)
ICONS: Mapping[str, str] = MappingProxyType(
    {
        CardState.WAITING: _SVG.format(
            '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3'
            'c-.6.3-1 .9-1 1.6V14"/><path d="M12 17.5h.01"/>'
        ),
        CardState.WITH_OWNER: _SVG.format(
            '<path d="M12 3 4.5 6v5.5c0 4.6 3.2 8.4 7.5 9.5 4.3-1.1 7.5-4.9 7.5-9.5V6z"/>'
            '<path d="M12 8.5v4l2.5 1.5"/>'
        ),
        CardState.READY: _SVG.format('<circle cx="12" cy="12" r="9"/><path d="m8 12.5 3 3 5-6"/>'),
        CardState.PARTIAL: _SVG.format(
            '<path d="M12 3.5 2.8 19.5h18.4z"/><path d="M12 10v4"/><path d="M12 17h.01"/>'
        ),
        CardState.WORKING: _SVG.format('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'),
        CardState.PAUSED: _SVG.format(
            '<circle cx="12" cy="12" r="9"/><path d="M10 9v6"/><path d="M14 9v6"/>'
        ),
        CardState.UNKNOWN: _SVG.format('<circle cx="12" cy="12" r="9"/>'),
        CardState.TO_START: _SVG.format(
            '<circle cx="12" cy="12" r="9"/><path d="m10 8.5 5 3.5-5 3.5z"/>'
        ),
        "plus": _SVG.format('<path d="M12 5v14"/><path d="M5 12h14"/>'),
        "reply": _SVG.format('<path d="M9 14 4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 6 6v4"/>'),
        "wallet": _SVG.format(
            '<rect x="3" y="6" width="18" height="13" rx="2"/><path d="M16 12.5h2"/>'
            '<path d="M3 9h18"/>'
        ),
        "clock": _SVG.format('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'),
        "download": _SVG.format(
            '<path d="M12 4v11"/><path d="m7 10.5 5 5 5-5"/><path d="M5 20h14"/>'
        ),
        "check": _SVG.format('<circle cx="12" cy="12" r="9"/><path d="m8 12.5 3 3 5-6"/>'),
        "user": _SVG.format('<circle cx="12" cy="8" r="4"/><path d="M4 20a8 8 0 0 1 16 0"/>'),
        "logout": _SVG.format(
            '<path d="M14 4h4a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-4"/><path d="M10 16l-4-4 4-4"/>'
            '<path d="M6 12h10"/>'
        ),
        "help": _SVG.format(
            '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.5a2.5 2.5 0 0 1 4.9.7'
            'c0 1.6-2.5 2.3-2.5 3.8"/><path d="M12 17h.01"/>'
        ),
        "lock": _SVG.format(
            '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>'
        ),
    }
)

# A friendly empty-state picture: a lightbulb on a soft circle (no text, fixed markup).
EMPTY_ART = (
    '<svg class="empty-art" viewBox="0 0 120 120" width="120" height="120" '
    'aria-hidden="true" focusable="false">'
    '<circle cx="60" cy="60" r="56" fill="var(--info)"/>'
    '<path d="M60 26a22 22 0 0 0-13 39.8c2.3 1.7 3.6 4.3 3.6 7.1V76h18.8v-3.1'
    'c0-2.8 1.3-5.4 3.6-7.1A22 22 0 0 0 60 26z" fill="var(--card)" '
    'stroke="var(--accent)" stroke-width="3" stroke-linejoin="round"/>'
    '<path d="M51 84h18M53 91h14" stroke="var(--accent)" stroke-width="3" '
    'stroke-linecap="round"/>'
    '<path d="M60 44v14M53 51h14" stroke="var(--accent2)" stroke-width="3" '
    'stroke-linecap="round"/>'
    "</svg>"
)

# The error page's picture (D177): a signpost on a soft circle (no text, fixed markup).
ERROR_ART = (
    '<svg class="empty-art" viewBox="0 0 120 120" width="120" height="120" '
    'aria-hidden="true" focusable="false">'
    '<circle cx="60" cy="60" r="56" fill="var(--busy)"/>'
    '<path d="M60 30v66" stroke="var(--busyfg)" stroke-width="4" stroke-linecap="round"/>'
    '<path d="M38 38h40l8 8-8 8H38z" fill="var(--card)" stroke="var(--busyfg)" '
    'stroke-width="3" stroke-linejoin="round"/>'
    '<path d="M82 60H42l-8 8 8 8h40z" fill="var(--card)" stroke="var(--busyfg)" '
    'stroke-width="3" stroke-linejoin="round"/>'
    '<path d="M46 96h28" stroke="var(--busyfg)" stroke-width="4" stroke-linecap="round"/>'
    "</svg>"
)

_RAW: dict[str, dict[Language, str]] = {
    "card.waiting": {"en": "Waiting for you", "fa": "منتظر جواب تو"},
    "card.waiting.hint": {
        "en": "It needs something from you before it can go on.",
        "fa": "برای ادامه یه چیزی از تو لازم داره.",
    },
    "card.with_owner": {"en": "Waiting for the owner", "fa": "منتظر تأیید صاحب Automa"},
    "card.with_owner.hint": {
        "en": "The owner of Automa has to approve one step; nothing is needed from you.",
        "fa": "صاحب Automa باید یک کار رو تأیید کنه؛ از تو کاری لازم نیست.",
    },
    "card.ready": {"en": "Ready", "fa": "آماده"},
    "card.ready.hint": {
        "en": "Done, and every main file of the product was checked.",
        "fa": "تموم شد و همهٔ فایل‌های اصلی محصول بررسی شدن.",
    },
    "card.partial": {"en": "Incomplete", "fa": "ناقص"},
    "card.partial.hint": {
        "en": "The product is not complete; the project page says what is missing.",
        "fa": "محصول کامل نشد؛ صفحهٔ پروژه دقیق می‌گه چی کم مونده.",
    },
    "card.working": {"en": "Working", "fa": "در حال کار"},
    "card.working.hint": {
        "en": "Working on it right now.",
        "fa": "الان داره روش کار می‌کنه.",
    },
    "card.paused": {"en": "Stopped", "fa": "متوقف"},
    "card.paused.hint": {
        "en": "Stopped mid-way; it can be continued.",
        "fa": "وسط کار ایستاده؛ می‌شه ادامه داد.",
    },
    "card.to_start": {"en": "Ready to start", "fa": "منتظر شروع"},
    "card.to_start.hint": {
        "en": "Nothing has been spent on the work yet.",
        "fa": "هنوز کاری انجام نشده؛ هر وقت خواستی شروعش کن.",
    },
    "card.steps": {"en": "{done} of {total} steps", "fa": "{done} از {total} مرحله"},
    "card.no_plan": {"en": "No plan yet", "fa": "هنوز برنامه نداره"},
    "card.open": {"en": "Open", "fa": "باز کن"},
    "card.answer": {"en": "Answer", "fa": "جواب بده"},
    "card.why.asks": {
        "en": "{count} question(s) from the work",
        "fa": "{count} سؤال وسط کار",
    },
    "card.why.questions": {
        "en": "questions before the plan",
        "fa": "چند سؤال قبل از برنامه",
    },
    "card.why.approvals": {
        "en": "{count} request(s) to approve",
        "fa": "{count} درخواست برای تأیید",
    },
    "card.why.owner_approvals": {
        "en": "{count} request(s) for the owner of Automa to approve",
        "fa": "{count} درخواست منتظر تأیید صاحب Automa",
    },
    "waiting.heading": {"en": "Waiting for your answer", "fa": "منتظر جواب تو"},
    "waiting.lead": {
        "en": "These projects stand still until you answer.",
        "fa": "این پروژه‌ها تا جواب تو منتظر می‌مونن.",
    },
    "all.heading": {"en": "All projects", "fa": "همهٔ پروژه‌ها"},
    "empty.heading": {"en": "Let's build your first idea", "fa": "بیا اولین ایده‌ات رو بسازیم"},
    "empty.body": {
        "en": (
            "Write what you want in a few sentences. Automa asks whatever it needs, shows a "
            "plan and a cost range, and starts only when you press Start."
        ),
        "fa": (
            "چیزی که می‌خوای رو در چند جمله بنویس. Automa هر چی لازم داره ازت می‌پرسه، "
            "برنامه و بازهٔ هزینه رو نشون می‌ده، و فقط وقتی «شروع» رو بزنی کار رو شروع می‌کنه."
        ),
    },
    "empty.try": {"en": "Or start from an example:", "fa": "یا از یک نمونه شروع کن:"},
}

MESSAGES: Mapping[str, Mapping[Language, str]] = MappingProxyType(
    {k: MappingProxyType(v) for k, v in _RAW.items()}
)
check_messages(MESSAGES)


def ct(key: str, lang: Language, **params: int) -> str:
    """The card text ``key`` in ``lang`` with ``params`` (counts, localized digits)
    filled in. Unknown key/language or a wrong parameter set raise, like ``t``."""
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language: {lang!r}")
    template = MESSAGES[key][lang]
    if set(params) != template_fields(template):
        raise ValueError(f"{key}: wrong parameters {sorted(params)}")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in params.values()):
        raise TypeError("card text parameters are counts (int)")
    return template.format(**{k: localize_digits(str(v), lang) for k, v in params.items()})


def card_label(state: CardState, lang: Language, status: str = "") -> str:
    """The short label; an unknown state is the recorded ``status`` itself (the caller
    escapes it)."""
    return status if state is CardState.UNKNOWN else ct(f"card.{state.value}", lang)


def card_hint(state: CardState, lang: Language) -> str:
    """The plain sentence; ``""`` for an unknown state."""
    return "" if state is CardState.UNKNOWN else ct(f"card.{state.value}.hint", lang)

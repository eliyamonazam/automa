"""Human status chips (M13): each project status and task state as an icon, a short
word and one plain sentence, in a fixed colour tone. The icon is the name of a fixed
inline SVG in :data:`STATUS_ICONS` (M16, D161; emoji before), drawn the same on
every computer and loading nothing.

Pure: a mapping from the recorded status value to display text. An unknown value
is shown as recorded with a neutral tone and no sentence (never guessed). Text comes
from :data:`MESSAGES`, checked like the main translation table (same fields in both
languages). The icon and tone are fixed values, never recorded text, so they are
safe in HTML attributes and class names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from automa.l10n import LANGUAGES, Language
from automa.webui.i18n import check_messages, status_label, task_state_label

ChipTone = Literal["ok", "busy", "paused", "bad", "info", "none"]

# Inline SVG icons (24x24, stroke = currentColor), the same style as the home-page
# cards (D160). Fixed strings, never recorded text.
_SVG = (
    '<svg class="icon" viewBox="0 0 24 24" width="16" height="16" fill="none" '
    'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true" focusable="false">{}</svg>'
)
STATUS_ICONS: Mapping[str, str] = MappingProxyType(
    {
        "sparkle": _SVG.format(
            '<path d="M12 4v4M12 16v4M4 12h4M16 12h4"/><path d="m7 7 2 2"/>'
            '<path d="m15 15 2 2"/><path d="m17 7-2 2"/><path d="m9 15-2 2"/>'
        ),
        "pencil": _SVG.format('<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13.5 6.5 4 4"/>'),
        "clock": _SVG.format('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'),
        "pause": _SVG.format(
            '<circle cx="12" cy="12" r="9"/><path d="M10 9v6"/><path d="M14 9v6"/>'
        ),
        "check": _SVG.format('<circle cx="12" cy="12" r="9"/><path d="m8 12.5 3 3 5-6"/>'),
        "cross": _SVG.format(
            '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6"/><path d="m15 9-6 6"/>'
        ),
        "search": _SVG.format('<circle cx="11" cy="11" r="6"/><path d="m20 20-4.5-4.5"/>'),
        "ban": _SVG.format('<circle cx="12" cy="12" r="9"/><path d="m5.6 5.6 12.8 12.8"/>'),
        "queue": _SVG.format('<circle cx="12" cy="12" r="9" stroke-dasharray="3 3"/>'),
        "dot": _SVG.format('<circle cx="12" cy="12" r="3"/>'),
    }
)

# status value -> (icon name in STATUS_ICONS, tone)
PROJECT_LOOK: Mapping[str, tuple[str, ChipTone]] = MappingProxyType(
    {
        "new": ("sparkle", "info"),
        "planning": ("pencil", "info"),
        "running": ("clock", "busy"),
        "paused": ("pause", "paused"),
        "done": ("check", "ok"),
        "failed": ("cross", "bad"),
    }
)
TASK_LOOK: Mapping[str, tuple[str, ChipTone]] = MappingProxyType(
    {
        "none": ("queue", "info"),
        "pending": ("queue", "info"),
        "running": ("clock", "busy"),
        "verifying": ("search", "busy"),
        "passed": ("check", "ok"),
        "failed": ("cross", "bad"),
        "blocked": ("ban", "bad"),
    }
)
UNKNOWN_ICON = "dot"

_RAW: dict[str, dict[Language, str]] = {
    "hint.project.new": {
        "en": "The idea is saved; there is no plan yet.",
        "fa": "ایده ثبت شده؛ هنوز برنامه نداره.",
    },
    "hint.project.planning": {
        "en": "Making the plan.",
        "fa": "داره برنامه می‌ریزه.",
    },
    "hint.project.running": {
        "en": "Working on it right now.",
        "fa": "الان داره روش کار می‌کنه.",
    },
    "hint.project.paused": {
        "en": "Stopped part-way; it can continue.",
        "fa": "وسط کار ایستاده؛ می‌شه ادامه داد.",
    },
    "hint.project.done": {
        "en": "Every step is finished.",
        "fa": "همهٔ مرحله‌ها انجام شد.",
    },
    "hint.project.failed": {
        "en": "A step did not pass its check and work stopped.",
        "fa": "یک مرحله در بررسی قبول نشد و کار متوقف شد.",
    },
    "hint.task.none": {"en": "Not its turn yet.", "fa": "هنوز نوبتش نرسیده."},
    "hint.task.pending": {"en": "Waiting for its turn.", "fa": "منتظر نوبتشه."},
    "hint.task.running": {"en": "Being done right now.", "fa": "الان داره انجام می‌شه."},
    "hint.task.verifying": {
        "en": "Done; now being checked.",
        "fa": "انجام شده؛ الان داره بررسی می‌شه.",
    },
    "hint.task.passed": {
        "en": "Done and passed its check.",
        "fa": "انجام شد و بررسی رو قبول شد.",
    },
    "hint.task.failed": {
        "en": "Did not pass its check.",
        "fa": "در بررسی قبول نشد.",
    },
    "hint.task.blocked": {
        "en": "Cannot go on; see the details below.",
        "fa": "نمی‌تونه ادامه بده؛ جزئیات پایین صفحه‌ست.",
    },
}

check_messages(_RAW)
MESSAGES: Mapping[str, Mapping[Language, str]] = MappingProxyType(
    {k: MappingProxyType(dict(v)) for k, v in _RAW.items()}
)


@dataclass(frozen=True)
class Chip:
    icon: str  # fixed name in STATUS_ICONS (from the tables above)
    label: str  # plain text (unknown status: as recorded; the caller escapes it)
    hint: str  # plain text, "" for an unknown status
    tone: ChipTone  # fixed value


def _check(lang: Language) -> None:
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language: {lang!r}")


def project_chip(status: str, lang: Language) -> Chip:
    """The chip of a project ``status``."""
    _check(lang)
    look = PROJECT_LOOK.get(status)
    if look is None:
        return Chip(UNKNOWN_ICON, status, "", "none")
    icon, tone = look
    return Chip(icon, status_label(status, lang), MESSAGES[f"hint.project.{status}"][lang], tone)


def task_chip(state: str | None, lang: Language) -> Chip:
    """The chip of a task ``state`` (``None``: not started)."""
    _check(lang)
    key = state or "none"
    look = TASK_LOOK.get(key)
    if look is None:
        return Chip(UNKNOWN_ICON, str(state), "", "none")
    icon, tone = look
    return Chip(icon, task_state_label(state, lang), MESSAGES[f"hint.task.{key}"][lang], tone)

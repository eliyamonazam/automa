"""Human status chips (M13): each project status and task state as an icon, a short
word and one plain sentence, in a fixed colour tone.

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

# status value -> (icon, tone)
PROJECT_LOOK: Mapping[str, tuple[str, ChipTone]] = MappingProxyType(
    {
        "new": ("✨", "info"),
        "planning": ("📝", "info"),
        "running": ("⏳", "busy"),
        "paused": ("⏸️", "paused"),
        "done": ("✅", "ok"),
        "failed": ("❌", "bad"),
    }
)
TASK_LOOK: Mapping[str, tuple[str, ChipTone]] = MappingProxyType(
    {
        "none": ("🕒", "info"),
        "pending": ("🕒", "info"),
        "running": ("⏳", "busy"),
        "verifying": ("🔍", "busy"),
        "passed": ("✅", "ok"),
        "failed": ("❌", "bad"),
        "blocked": ("⛔", "bad"),
    }
)
UNKNOWN_ICON = "•"

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
    icon: str  # fixed value from the tables above
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

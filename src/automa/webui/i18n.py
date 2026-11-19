"""The web UI's translation table (D140).

Every interface string a translated page shows comes from :data:`MESSAGES`: one key,
one template per language, with the *same* named ``{params}`` in both (checked by
:func:`check_messages` and the tests). :func:`t` fills a template; whole numbers are
written with Persian digits in ``fa``. The result is plain text — the caller still
escapes it (``render.text``), so a parameter can never inject markup.

Recorded data (goals, titles, file names, ids) is never translated; only the
interface around it is. Pure: no files, no network.
"""

from __future__ import annotations

from collections.abc import Mapping
from string import Formatter
from types import MappingProxyType

from automa.l10n import LANGUAGES, Language, localize_digits

_RAW: dict[str, dict[Language, str]] = {
    # Page shell
    "overview.title": {"en": "Automa projects", "fa": "پروژه‌های Automa"},
    "overview.mode.actions": {
        "en": "Actions enabled: approve or reject requests on each project page.",
        "fa": "دکمه‌ها فعال‌اند: درخواست‌ها رو در صفحهٔ هر پروژه تأیید یا رد کن.",
    },
    "overview.mode.readonly": {
        "en": "Read-only view.",
        "fa": "فقط نمایش؛ از این صفحه چیزی تغییر نمی‌کنه.",
    },
    "overview.new_idea": {"en": "+ New idea", "fa": "+ ایدهٔ جدید"},
    "overview.hide_tests": {"en": "Hide test runs", "fa": "پنهان کردن آزمون‌ها"},
    "overview.tests_hidden": {
        "en": "{count} test runs hidden",
        "fa": "{count} آزمون پنهان شده",
    },
    "overview.spent_today": {"en": "Spent today:", "fa": "خرج امروز:"},
    "overview.this_month": {"en": "This month:", "fa": "این ماه:"},
    "overview.show_tests": {"en": "show them", "fa": "نمایش آزمون‌ها"},
    "overview.pending": {"en": "{count} pending", "fa": "{count} منتظر تأیید"},
    "overview.no_pending": {"en": "approvals: 0", "fa": "بدون درخواست تأیید"},
    "overview.cost": {"en": "cost: {amount}", "fa": "هزینه: {amount}"},
    "overview.created": {"en": "created {when}", "fa": "ساخته‌شده {when}"},
    "errors.unreadable": {"en": "Unreadable data", "fa": "داده‌هایی که خونده نشدن"},
    # Project status (ProjectStatus values)
    "status.new": {"en": "new", "fa": "تازه"},
    "status.planning": {"en": "planning", "fa": "در حال برنامه‌ریزی"},
    "status.running": {"en": "running", "fa": "در حال کار"},
    "status.paused": {"en": "paused", "fa": "متوقف"},
    "status.done": {"en": "done", "fa": "تموم شد"},
    "status.failed": {"en": "failed", "fa": "ناموفق"},
    # Welcome (empty home page, D138)
    "welcome.none.heading": {"en": "No projects yet", "fa": "هنوز پروژه‌ای نیست"},
    "welcome.none.body": {
        "en": "Start the page with actions enabled to plan a new idea here.",
        "fa": "برای برنامه‌ریزی یک ایدهٔ تازه از همین‌جا، صفحه رو با دکمه‌های فعال باز کن.",
    },
    "welcome.heading": {"en": "👋 Welcome", "fa": "👋 خوش اومدی"},
    "welcome.body": {
        "en": (
            "Describe an idea in a few sentences. Automa asks what it needs, shows a plan "
            "and a cost range, and starts only when you press Start."
        ),
        "fa": (
            "ایده‌ات رو در چند جمله بنویس. Automa هر چی لازم داره می‌پرسه، برنامه و بازهٔ "
            "هزینه رو نشون می‌ده، و فقط وقتی «شروع» رو بزنی کار رو شروع می‌کنه."
        ),
    },
    "welcome.examples": {"en": "For example:", "fa": "مثلاً:"},
    "welcome.example.1": {
        "en": "A one-page HTML tool that converts Persian digits to English digits and back",
        "fa": "یک ابزار یک‌صفحه‌ای HTML که ارقام فارسی رو به انگلیسی و برعکس تبدیل کنه",
    },
    "welcome.example.2": {
        "en": "A short report comparing three note-taking apps for students, with sources",
        "fa": "یک گزارش کوتاه با منبع که سه برنامهٔ یادداشت‌برداری رو برای دانشجوها مقایسه کنه",
    },
    "welcome.example.3": {
        "en": "A Python script that renames the photos in a folder by the date they were taken",
        "fa": "یک اسکریپت پایتون که عکس‌های یک پوشه رو با تاریخ گرفتنشون نام‌گذاری کنه",
    },
}

WELCOME_EXAMPLE_KEYS = ("welcome.example.1", "welcome.example.2", "welcome.example.3")


def template_fields(template: str) -> frozenset[str]:
    """The named fields of ``template``; positional, attribute, index, conversion or
    format-spec fields raise ``ValueError`` (only plain ``{name}`` is allowed)."""
    names = set()
    for _literal, field, spec, conversion in Formatter().parse(template):
        if field is None:
            continue
        if not field.isidentifier() or spec or conversion:
            raise ValueError(f"only plain named fields are allowed: {template!r}")
        names.add(field)
    return frozenset(names)


def check_messages(messages: Mapping[str, Mapping[str, str]]) -> None:
    """Every key has exactly the supported languages, non-empty, same fields."""
    for key, by_lang in messages.items():
        if set(by_lang) != set(LANGUAGES):
            raise ValueError(f"{key}: languages {sorted(by_lang)} != {sorted(LANGUAGES)}")
        fields = {lang: template_fields(by_lang[lang]) for lang in LANGUAGES}
        if any(not by_lang[lang].strip() for lang in LANGUAGES):
            raise ValueError(f"{key}: empty template")
        if len(set(fields.values())) != 1:
            raise ValueError(f"{key}: different fields per language {fields}")


check_messages(_RAW)
MESSAGES: Mapping[str, Mapping[Language, str]] = MappingProxyType(
    {k: MappingProxyType(dict(v)) for k, v in _RAW.items()}
)


def _param(value: object, lang: Language) -> str:
    if isinstance(value, bool):
        raise TypeError("a bool is not a message parameter")
    if isinstance(value, int):
        return localize_digits(str(value), lang)
    if isinstance(value, str):
        return value
    raise TypeError(f"message parameters are str or int, got {type(value).__name__}")


def t(key: str, lang: Language, **params: object) -> str:
    """The ``key`` message in ``lang`` with ``params`` filled in (plain text).

    An unknown key or language, or missing/extra parameters, raise — a typo must fail
    a test, not show a raw key to the owner."""
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language: {lang!r}")
    template = MESSAGES[key][lang]
    if set(params) != template_fields(template):
        raise ValueError(
            f"{key}: parameters {sorted(params)} != {sorted(template_fields(template))}"
        )
    return template.format(**{k: _param(v, lang) for k, v in params.items()})


def status_label(status: str, lang: Language) -> str:
    """A project status in ``lang``; an unknown value is shown as recorded."""
    key = f"status.{status}"
    return t(key, lang) if key in MESSAGES else status

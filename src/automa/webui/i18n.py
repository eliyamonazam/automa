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
    # Project page (D144)
    "project.goal": {"en": "Goal:", "fa": "هدف:"},
    "project.status": {"en": "status:", "fa": "وضعیت:"},
    "project.times": {
        "en": "created {created} · updated {updated}",
        "fa": "ساخته‌شده {created} · آخرین تغییر {updated}",
    },
    "project.files": {"en": "Files", "fa": "فایل‌ها"},
    "project.questions.before": {
        "en": "The Executive has questions:",
        "fa": "Automa چند تا سؤال داره:",
    },
    "project.questions.link": {"en": "answer them", "fa": "جواب بده"},
    "project.questions.after": {"en": "to make the plan.", "fa": "تا برنامه ساخته بشه."},
    "project.cost_range": {"en": "Cost range:", "fa": "بازهٔ هزینه:"},
    "approvals.heading": {
        "en": "Pending approvals ({count})",
        "fa": "درخواست‌های منتظر تأیید ({count})",
    },
    "approvals.none": {"en": "None.", "fa": "هیچ درخواستی نیست."},
    "approval.task": {
        "en": "task: {task} · requested {when}",
        "fa": "مرحله: {task} · درخواست: {when}",
    },
    "approval.reason": {"en": "reason:", "fa": "دلیل:"},
    "approval.readonly": {
        "en": "This page is read-only: decide in the terminal.",
        "fa": "این صفحه فقط نمایشه: در ترمینال تصمیم بگیر.",
    },
    "approval.confirm": {
        "en": "I reviewed the full arguments above: run {tool} once with exactly these arguments",
        "fa": "همهٔ آرگومان‌های بالا رو دیدم: {tool} فقط یک بار و دقیقاً با همین‌ها اجرا بشه",
    },
    "approval.note": {"en": "note", "fa": "یادداشت"},
    "approval.approve": {"en": "Approve", "fa": "تأیید"},
    "approval.reject": {"en": "Reject", "fa": "رد"},
    "plan.heading": {"en": "Plan", "fa": "برنامه"},
    "plan.none": {"en": "No plan.", "fa": "هنوز برنامه‌ای نیست."},
    "plan.version": {
        "en": "Plan v{version} ({count} tasks)",
        "fa": "برنامه (نسخهٔ {version}، {count} مرحله)",
    },
    "task.role": {"en": "Role: {role}", "fa": "نقش: {role}"},
    "task.after": {"en": "After: {after}", "fa": "بعد از: {after}"},
    "task.attempts": {"en": "Attempts: {count}", "fa": "تلاش‌ها: {count}"},
    "task.last": {"en": "Last result:", "fa": "آخرین نتیجه:"},
    "task.model": {"en": "Model: {model}", "fa": "مدل: {model}"},
    # Task states (TaskState values); "none" is a task not started yet
    "task.state.none": {"en": "not started", "fa": "شروع نشده"},
    "task.state.pending": {"en": "pending", "fa": "در صف"},
    "task.state.running": {"en": "running", "fa": "در حال کار"},
    "task.state.verifying": {"en": "verifying", "fa": "در حال بررسی"},
    "task.state.passed": {"en": "passed", "fa": "تموم شد"},
    "task.state.failed": {"en": "failed", "fa": "ناموفق"},
    "task.state.blocked": {"en": "blocked", "fa": "گیر کرده"},
    "progress.heading": {"en": "Progress", "fa": "پیشرفت"},
    "progress.passed": {"en": "tasks passed:", "fa": "مرحله‌های تموم‌شده:"},
    "progress.live": {
        "en": "running: this page refreshes every {seconds} s",
        "fa": "در حال کار: این صفحه هر {seconds} ثانیه تازه می‌شه",
    },
    "tech.summary": {"en": "Technical details", "fa": "جزئیات فنی"},
    "tech.events": {
        "en": "Latest events ({count}, newest first)",
        "fa": "آخرین رویدادها ({count}، جدیدترین اول)",
    },
    "tech.time": {"en": "Time", "fa": "زمان"},
    "tech.event": {"en": "Event", "fa": "رویداد"},
    "tech.details": {"en": "Details", "fa": "جزئیات"},
    "unusual.one": {"en": "1 step cost", "fa": "یک مرحله"},
    "unusual.many": {"en": "{count} steps cost", "fa": "{count} مرحله"},
    "unusual.times": {"en": ", {ratio}× the usual cost", "fa": "، {ratio} برابر معمول"},
    "unusual.line": {
        "en": (
            "Unusual spending: {what} much more than usual this run (largest "
            "{amount}{times}). Advisory only: nothing was stopped."
        ),
        "fa": (
            "خرج غیرعادی: {what} در این اجرا خیلی بیشتر از معمول خرج کرد (بیشترین "
            "{amount}{times}). فقط هشداره؛ چیزی متوقف نشد."
        ),
    },
    "run.heading": {"en": "Run", "fa": "اجرا"},
    "run.label": {"en": "Background run", "fa": "اجرای پس‌زمینه"},
    "run.started": {"en": "(started {when})", "fa": "(شروع: {when})"},
    "run.refreshing": {
        "en": "The page refreshes itself while the run is going.",
        "fa": "تا وقتی اجرا ادامه داره، این صفحه خودش تازه می‌شه.",
    },
    "run.stop.button": {"en": "Stop", "fa": "توقف"},
    "run.stop.help": {
        "en": (
            "pauses the run after its current step (no model call is cut off); "
            "the start button resumes it."
        ),
        "fa": (
            "اجرا رو بعد از گام فعلی نگه می‌داره (هیچ فراخوانی مدلی نصفه قطع نمی‌شه)؛ "
            "دکمهٔ شروع دوباره ادامه‌اش می‌ده."
        ),
    },
    "run.confirm": {
        "en": "I saw the cost range above and accept it (it is above confirm_above_usd)",
        "fa": "بازهٔ هزینهٔ بالا رو دیدم و قبولش دارم (از سقف confirm_above_usd بیشتره)",
    },
    "run.start.button": {"en": "Start run", "fa": "شروع"},
    "run.start.help": {
        "en": (
            "runs the plan in the background with the same budget guard and caps; "
            "every model call is counted."
        ),
        "fa": (
            "برنامه رو در پس‌زمینه اجرا می‌کنه، با همون محافظ بودجه و سقف‌ها؛ "
            "هزینهٔ هر فراخوانی مدل حساب می‌شه."
        ),
    },
    "run.not_startable": {
        "en": "Not startable now (no plan, already running or done).",
        "fa": "الان نمی‌شه شروع کرد (برنامه‌ای نیست، در حال اجراست یا تموم شده).",
    },
    "run.log.summary": {"en": "Technical details (run log)", "fa": "جزئیات فنی (گزارش اجرا)"},
    "run.log.file": {"en": "log file:", "fa": "فایل گزارش:"},
    # Background run states (RunState.summary, D125)
    "run.state.running": {"en": "running in the background", "fa": "در پس‌زمینه در حال کاره"},
    "run.state.stopping": {
        "en": "stopping: the run pauses after its current step",
        "fa": "در حال توقف: اجرا بعد از گام فعلی نگه داشته می‌شه",
    },
    "run.state.ended": {
        "en": "ended with exit code {code}: {meaning}",
        "fa": "تموم شد (کد {code}): {meaning}",
    },
    "run.exit.signal": {"en": "killed by signal {signal}", "fa": "با سیگنال {signal} بسته شد"},
    "run.exit.other": {"en": "stopped (see the log)", "fa": "متوقف شد (گزارش رو ببین)"},
    "run.exit.0": {
        "en": "finished: every task passed",
        "fa": "تموم شد: همهٔ مرحله‌ها درست انجام شدن",
    },
    "run.exit.1": {
        "en": "stopped: user error (see the log)",
        "fa": "متوقف شد: خطای ورودی (گزارش رو ببین)",
    },
    "run.exit.2": {
        "en": "stopped: configuration error (see the log)",
        "fa": "متوقف شد: خطای تنظیمات (گزارش رو ببین)",
    },
    "run.exit.3": {
        "en": "budget stop: the project is paused until you raise the cap or wait",
        "fa": "سقف هزینه پر شد: پروژه متوقفه تا سقف رو بیشتر کنی یا صبر کنی",
    },
    "run.exit.4": {"en": "planning failed", "fa": "برنامه‌ریزی ناموفق بود"},
    "run.exit.5": {
        "en": "provider error (network or gateway); run again to resume",
        "fa": "خطای درگاه یا اینترنت؛ برای ادامه دوباره شروع کن",
    },
    "run.exit.6": {
        "en": "finished, but some task failed for good",
        "fa": "تموم شد، ولی بعضی مرحله‌ها برای همیشه ناموفق موندن",
    },
    "run.exit.7": {
        "en": "paused: an approval is needed or another cause (see the log)",
        "fa": "متوقف شد: منتظر تأیید تو یا دلیل دیگه (گزارش رو ببین)",
    },
    "run.exit.11": {
        "en": "not started: the cost range grew above confirm_above_usd; reload and confirm",
        "fa": "شروع نشد: بازهٔ هزینه از confirm_above_usd بیشتر شد؛ صفحه رو تازه کن و تأیید کن",
    },
    "run.exit.12": {
        "en": "stopped on request: the project is paused; start again to resume",
        "fa": "به درخواست تو متوقف شد؛ برای ادامه دوباره شروع کن",
    },
    "run.exit.130": {
        "en": "interrupted: the project is paused; start again to resume",
        "fa": "قطع شد: پروژه متوقفه؛ برای ادامه دوباره شروع کن",
    },
    # New idea form, thinking page and questions form (D145)
    "idea.title": {"en": "Automa — new idea", "fa": "Automa — ایدهٔ جدید"},
    "idea.heading": {"en": "New idea", "fa": "ایدهٔ جدید"},
    "idea.intro": {
        "en": (
            "The Executive may ask a few questions, then makes a plan with a cost range. "
            "Nothing is run yet. Each step is a model call through the budget guard and may "
            "take a minute."
        ),
        "fa": (
            "مدیر پروژه ممکنه چند سؤال بپرسه، بعد یک برنامه با بازهٔ هزینه می‌سازه. "
            "هنوز هیچ کاری اجرا نمی‌شه. هر مرحله یک درخواست به مدل از مسیر محافظ بودجه‌ست "
            "و ممکنه یک دقیقه طول بکشه."
        ),
    },
    "idea.placeholder": {
        "en": "What should the project achieve?",
        "fa": "پروژه قراره چه کاری انجام بده؟",
    },
    "idea.ask": {"en": "Ask clarifying questions first", "fa": "اول چند سؤال روشن‌کننده بپرس"},
    "idea.submit": {"en": "Create plan", "fa": "ساختن برنامه"},
    "thinking.title": {"en": "Automa — thinking", "fa": "Automa — در حال فکر کردن"},
    "thinking.heading": {"en": "⏳ Thinking…", "fa": "⏳ در حال فکر کردن…"},
    "thinking.questions": {
        "en": (
            "The Executive is reading your idea and deciding whether to ask a few questions."
            " This usually takes under a minute; this page refreshes by itself and moves on"
            " when it is done."
        ),
        "fa": (
            "مدیر پروژه داره ایده‌ات رو می‌خونه و تصمیم می‌گیره چند سؤال بپرسه یا نه. "
            "معمولاً کمتر از یک دقیقه طول می‌کشه؛ این صفحه خودش تازه می‌شه و وقتی تموم شد "
            "می‌ره مرحلهٔ بعد."
        ),
    },
    "thinking.plan": {
        "en": (
            "The Executive is making the plan and the cost estimate. This usually takes"
            " under a minute; this page refreshes by itself and moves on when it is done."
        ),
        "fa": (
            "مدیر پروژه داره برنامه و پیش‌بینی هزینه رو می‌سازه. معمولاً کمتر از یک دقیقه "
            "طول می‌کشه؛ این صفحه خودش تازه می‌شه و وقتی تموم شد می‌ره مرحلهٔ بعد."
        ),
    },
    "thinking.elapsed": {
        "en": "{seconds} s so far · project",
        "fa": "{seconds} ثانیه گذشته · پروژهٔ",
    },
    "clarify.title": {
        "en": "Automa — {project} questions",
        "fa": "Automa — سؤال‌های {project}",
    },
    "clarify.heading": {"en": "Questions for", "fa": "سؤال‌های پروژهٔ"},
    "clarify.goal": {"en": "Goal:", "fa": "هدف:"},
    "clarify.intro": {
        "en": "Leave an answer empty to skip it. The plan is made after you send.",
        "fa": "اگه نخواستی به سؤالی جواب بدی، خالی بذارش. برنامه بعد از فرستادن ساخته می‌شه.",
    },
    "clarify.submit": {
        "en": "Send answers and make the plan",
        "fa": "فرستادن جواب‌ها و ساختن برنامه",
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


def task_state_label(state: str | None, lang: Language) -> str:
    """A task state in ``lang``; ``None`` is "not started", an unknown value is shown
    as recorded."""
    key = f"task.state.{state or 'none'}"
    return t(key, lang) if key in MESSAGES else str(state)


def exit_label(code: int, lang: Language) -> str:
    """What an ``automa run`` exit code means, in ``lang`` (same text as
    ``runs.EXIT_MEANINGS`` in English)."""
    if code < 0:
        return t("run.exit.signal", lang, signal=-code)
    key = f"run.exit.{code}"
    return t(key, lang) if key in MESSAGES else t("run.exit.other", lang)


def localize_number(value: str, lang: Language) -> str:
    """A formatted number (e.g. ``"3.5"``) with Persian digits and decimal separator
    in ``fa``."""
    return localize_digits(value.replace(".", "٫"), lang) if lang == "fa" else value

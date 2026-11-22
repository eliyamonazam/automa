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
from automa.tools.request_warnings import WARNING_TEXT

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
    "status.paused": {"en": "paused", "fa": "متوقف شد"},
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
    "task.missing_program": {
        "en": "A program this task needs is not installed on your computer. "
        "Install it in the Ubuntu (WSL) terminal with:",
        "fa": "برنامه‌ای که این کار لازم داره روی کامپیوترت نصب نیست. "
        "در ترمینال Ubuntu (WSL) با این دستور نصبش کن:",
    },
    "task.missing_program.then": {
        "en": "then continue the project.",
        "fa": "بعد پروژه رو ادامه بده.",
    },
    # Task states (TaskState values); "none" is a task not started yet
    "task.state.none": {"en": "not started", "fa": "شروع نشده"},
    "task.state.pending": {"en": "pending", "fa": "در صف"},
    "task.state.running": {"en": "running", "fa": "در حال کار"},
    "task.state.verifying": {"en": "verifying", "fa": "در حال بررسی"},
    "task.state.passed": {"en": "passed", "fa": "تموم شد"},
    "task.state.failed": {"en": "failed", "fa": "ناموفق"},
    "task.state.blocked": {"en": "blocked", "fa": "گیر کرده"},
    "activity.heading": {"en": "Right now", "fa": "همین الان"},
    "activity.working": {"en": "Working on: {task}", "fa": "داره روی این کار می‌کنه: {task}"},
    "activity.working_any": {"en": "Working on a step", "fa": "داره روی یک مرحله کار می‌کنه"},
    "activity.checking": {
        "en": "Checking the result of: {task}",
        "fa": "داره نتیجهٔ این مرحله رو بررسی می‌کنه: {task}",
    },
    "activity.checking_any": {
        "en": "Checking a step's result",
        "fa": "داره نتیجهٔ یک مرحله رو بررسی می‌کنه",
    },
    "activity.reviewing": {
        "en": "Final review of the whole product",
        "fa": "بازبینی نهایی کل محصول",
    },
    "activity.last": {"en": "Last action ({when}): {action}", "fa": "آخرین کار ({when}): {action}"},
    "activity.wrote": {"en": "wrote {arg}", "fa": "فایل {arg} رو نوشت"},
    "activity.edited": {"en": "edited {arg}", "fa": "فایل {arg} رو ویرایش کرد"},
    "activity.read": {"en": "read {arg}", "fa": "فایل {arg} رو خوند"},
    "activity.listed": {"en": "looked at the project's files", "fa": "فایل‌های پروژه رو نگاه کرد"},
    "activity.deleted": {"en": "deleted {arg}", "fa": "فایل {arg} رو پاک کرد"},
    "activity.ran": {"en": "ran {arg}", "fa": "برنامهٔ {arg} رو اجرا کرد"},
    "activity.fetched": {"en": "read a page on {arg}", "fa": "یک صفحه از {arg} رو خوند"},
    "activity.searched": {"en": "searched the web", "fa": "در وب جست‌وجو کرد"},
    "activity.github": {"en": "read from GitHub", "fa": "از GitHub خوند"},
    "activity.image": {"en": "made an image", "fa": "یک تصویر ساخت"},
    "project.near_cap": {
        "en": "It may reach the project's spending cap before it is finished. If you want "
        "it to go all the way, raise project_usd under [limits] in automa.toml before "
        "starting.",
        "fa": "ممکنه قبل از تموم شدن به سقف خرج پروژه بخوره. اگه می‌خوای تا آخر بره، قبل از "
        "شروع project_usd رو زیر [limits] در automa.toml بیشتر کن.",
    },
    "wizard.label": {"en": "Steps of a new idea", "fa": "قدم‌های ایدهٔ تازه"},
    "wizard.idea": {"en": "Idea", "fa": "ایده"},
    "wizard.ask": {"en": "Questions and what is needed", "fa": "سؤال‌ها و چیزهای لازم"},
    "wizard.plan": {"en": "Plan and cost", "fa": "برنامه و هزینه"},
    "files.folder": {"en": "folder", "fa": "پوشه"},
    "progress.heading": {"en": "Progress", "fa": "پیشرفت"},
    "progress.passed": {"en": "tasks passed:", "fa": "مرحله‌های تموم‌شده:"},
    "progress.live": {
        "en": "running: this page refreshes every {seconds} s",
        "fa": "در حال کار: این صفحه هر {seconds} ثانیه تازه می‌شه",
    },
    "tech.summary": {"en": "Technical details", "fa": "جزئیات فنی"},
    # New-version bar (D178)
    "update.ready": {
        "en": "A new version of Automa is ready",
        "fa": "نسخهٔ تازهٔ Automa آماده‌ست",
    },
    "update.whats_new": {"en": "What's new", "fa": "چه چیزهایی تازه‌ست"},
    "update.more": {"en": "and {count} more", "fa": "و {count} مورد دیگه"},
    "update.small": {
        "en": "{count} small improvements",
        "fa": "{count} بهبود کوچک",
    },
    "update.how": {
        "en": "To get it, double-click “Update Automa” on your desktop, then close this "
        "Automa window and open it again. Nothing of yours is deleted.",
        "fa": "برای گرفتنش روی میان‌بر «به‌روزرسانی Automa» روی دسکتاپ دابل‌کلیک کن، بعد "
        "این پنجرهٔ Automa رو ببند و دوباره باز کن. چیزی از تو پاک نمی‌شه.",
    },
    "update.terminal": {"en": "Or in Ubuntu:", "fa": "یا در Ubuntu بزن:"},
    "tech.project_id": {"en": "Project id:", "fa": "شناسهٔ پروژه:"},
    "tech.steps": {"en": "Steps", "fa": "مرحله‌ها"},
    "tech.col.id": {"en": "Id", "fa": "شناسه"},
    "tech.col.role": {"en": "Role", "fa": "نقش"},
    "tech.col.after": {"en": "After", "fa": "بعد از"},
    "tech.col.model": {"en": "Model", "fa": "مدل"},
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
    "run.missing.heading": {
        "en": "Before you start: some required things were not given",
        "fa": "قبل از شروع: چند چیز ضروری رو هنوز ندادی",
    },
    "run.missing.body": {
        "en": (
            "Without these, the parts that depend on them will stay unfinished; "
            "Automa will not invent them and will say what is missing at the end."
        ),
        "fa": (
            "بدون این‌ها بخش‌هایی که بهشون وابسته‌ان ناقص می‌مونن؛ Automa چیزی از "
            "خودش نمی‌سازه و آخر کار دقیق می‌گه چی کم مونده."
        ),
    },
    # Feasibility warning before "Start" (D164): fixed texts, chosen by code.
    "list.sep": {"en": ", ", "fa": "، "},
    "run.feas.heading": {
        "en": "Before you start: some parts depend on things outside Automa",
        "fa": "قبل از شروع: بعضی بخش‌ها به چیزهایی بیرون از Automa وابسته‌ان",
    },
    "run.feas.tasks": {"en": "(in: {tasks})", "fa": "(در: {tasks})"},
    "run.feas.live_web_data": {
        "en": "Live data from websites.",
        "fa": "دادهٔ زنده از سایت‌ها.",
    },
    "run.feas.live_web_data.alt": {
        "en": (
            "Some sites (stock markets, prices, heavy JavaScript) may not give reliable "
            "data to these tools, and retries cost money. Cheaper and surer: give the "
            "data yourself as an Excel or CSV file."
        ),
        "fa": (
            "بعضی سایت‌ها (بورس، قیمت‌ها، سایت‌های پرجاوااسکریپت) ممکنه با ابزارهای فعلی "
            "دادهٔ قابل اعتماد ندن و تلاش دوباره خرج داره. راه ارزون‌تر و مطمئن‌تر: داده "
            "رو خودت به‌صورت فایل Excel یا CSV بده."
        ),
    },
    "run.feas.login": {"en": "An account behind a login.", "fa": "حسابی که ورود می‌خواد."},
    "run.feas.login.alt": {
        "en": (
            "Automa never takes your password. Export the data from that account "
            "yourself and give it as a file."
        ),
        "fa": (
            "Automa هیچ‌وقت رمزت رو نمی‌گیره. داده رو خودت از اون حساب خروجی بگیر و به‌صورت فایل بده."
        ),
    },
    "run.feas.paid_api": {"en": "A paid service.", "fa": "یک سرویس پولی."},
    "run.feas.paid_api.alt": {
        "en": (
            "It works only if you set up its key yourself (never in the idea text). "
            "Otherwise give the data as a file."
        ),
        "fa": (
            "فقط وقتی کار می‌کنه که کلیدش رو خودت تنظیم کرده باشی (هیچ‌وقت در متن "
            "ایده ننویس). وگرنه داده رو به‌صورت فایل بده."
        ),
    },
    "run.feas.gui": {"en": "A desktop program with a window.", "fa": "یک برنامهٔ پنجره‌دار."},
    "run.feas.gui.alt": {
        "en": (
            "Automa cannot click in it. It makes the files or a script, and you do "
            "that step yourself."
        ),
        "fa": (
            "Automa نمی‌تونه داخلش کلیک کنه؛ فایل‌ها یا اسکریپت رو می‌سازه و اون قدم "
            "رو خودت انجام می‌دی."
        ),
    },
    "run.feas.body": {
        "en": (
            "You can still start: if a part cannot get real data, Automa asks you or "
            "says what is missing, and never invents it."
        ),
        "fa": (
            "باز هم می‌تونی شروع کنی: اگه بخشی دادهٔ واقعی نگیره، Automa ازت می‌پرسه "
            "یا دقیق می‌گه چی کم مونده؛ هیچ‌وقت از خودش نمی‌سازه."
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
    "idea.files": {"en": "Helpful files (optional):", "fa": "فایل‌های کمکی (اختیاری):"},
    "idea.files.hint": {
        "en": "For example a spreadsheet or CSV of your data, a sample or a logo. Up to 10 "
        "files, each up to 20 MB. Automa reads them as data and does not have to ask you "
        "for them later.",
        "fa": "مثلاً فایل اکسل یا جدول داده‌هات، یک نمونه یا لوگو. تا ۱۰ فایل، هر کدوم تا ۲۰ "
        "مگابایت. Automa این‌ها رو به‌عنوان داده می‌خونه و بعداً لازم نیست دوباره ازت بخواد.",
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
    "thinking.change": {
        "en": (
            "The Executive is adding the steps your change needs. This usually takes"
            " under a minute; this page refreshes by itself and moves on when it is done."
        ),
        "fa": (
            "داره مرحله‌هایی که تغییرت لازم داره رو به برنامه اضافه می‌کنه. معمولاً کمتر از"
            " یک دقیقه طول می‌کشه؛ این صفحه خودش تازه می‌شه و بعدش جلو می‌ره."
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
    "clarify.questions": {"en": "Questions", "fa": "سؤال‌ها"},
    "clarify.needs": {"en": "What I need from you", "fa": "چیزهایی که ازت لازم دارم"},
    "clarify.needs.intro": {
        "en": (
            "To do this right without inventing anything, I need these. Give what you "
            "can as text or a file; whatever you don't give, I won't make up."
        ),
        "fa": (
            "برای اینکه کار درست و بدون حدس دربیاد این‌ها رو لازم دارم. هر کدوم رو "
            "می‌تونی بنویسی یا فایلش رو بدی؛ هر چی رو ندی، از خودم نمی‌سازم."
        ),
    },
    "clarify.need.required": {"en": "Required", "fa": "ضروری"},
    "clarify.need.optional": {"en": "Optional", "fa": "اختیاری"},
    "clarify.need.why": {"en": "Why:", "fa": "چرا:"},
    "clarify.need.where": {"en": "Where to get it:", "fa": "از کجا بیارم:"},
    "clarify.need.answer": {
        "en": "Your answer (text, a link or a short note)",
        "fa": "جوابت (متن، لینک یا یک توضیح کوتاه)",
    },
    "clarify.need.files": {"en": "or attach a file:", "fa": "یا فایلش رو بده:"},
    "clarify.need.secret": {
        "en": "Never write passwords or keys here.",
        "fa": "رمز عبور یا کلید رو این‌جا ننویس.",
    },
    "clarify.submit": {
        "en": "Send answers and make the plan",
        "fa": "فرستادن جواب‌ها و ساختن برنامه",
    },
    # File browser (D146)
    "files.title": {"en": "Automa — {project} files", "fa": "Automa — فایل‌های {project}"},
    "files.heading": {"en": "Files of {project}", "fa": "فایل‌های پروژهٔ {project}"},
    "files.root": {"en": "files", "fa": "فایل‌ها"},
    "files.none": {"en": "No files yet.", "fa": "هنوز فایلی ساخته نشده."},
    "files.up": {"en": "↑ up", "fa": "↑ پوشهٔ بالاتر"},
    "files.empty": {"en": "Empty directory.", "fa": "این پوشه خالیه."},
    "files.download": {"en": "download", "fa": "دانلود"},
    "files.open": {"en": "▶ open", "fa": "▶ باز کردن"},
    "files.symlink": {"en": "symlink, not followed", "fa": "میان‌بر؛ دنبالش نمی‌کنیم"},
    "files.special": {"en": "not a regular file", "fa": "فایل معمولی نیست"},
    "files.showing": {
        "en": "Showing {shown} of {total} entries.",
        "fa": "{shown} مورد از {total} مورد نشون داده شده.",
    },
    "size.b": {"en": "{size} B", "fa": "{size} بایت"},
    "size.kib": {"en": "{size} KiB", "fa": "{size} کیلوبایت"},
    "size.mib": {"en": "{size} MiB", "fa": "{size} مگابایت"},
    "file.open_page": {
        "en": "▶ open the page (runs isolated: no network, no access to Automa)",
        "fa": "▶ باز کردن صفحه (جدا اجرا می‌شه: بدون اینترنت و بدون دسترسی به Automa)",
    },
    "file.truncated": {
        "en": "Only the first {size} are shown; download the file for all of it.",
        "fa": "فقط {size} اول نشون داده شده؛ برای دیدن همهٔ فایل دانلودش کن.",
    },
    "file.not_shown": {"en": "{note}: not shown here.", "fa": "{note}؛ اینجا نشون داده نمی‌شه."},
    "file.note.binary": {"en": "binary file", "fa": "فایل غیرمتنی"},
    "file.note.too_large": {
        "en": "image larger than {mib} MiB: download it",
        "fa": "عکس بزرگ‌تر از {mib} مگابایته؛ دانلودش کن",
    },
    "file.note.not_image": {
        "en": "the extension says image but the content is not that image type",
        "fa": "پسوند فایل می‌گه عکسه، ولی محتواش اون نوع عکس نیست",
    },
    "file.note.not_text": {"en": "not a UTF-8 text file", "fa": "متن سادهٔ قابل‌خوندن نیست"},
    # Result of a decision made from the project page (D146)
    "decided.approved": {
        "en": "{id}: approved; the call runs at most once, when the run continues",
        "fa": "{id}: تأیید شد؛ وقتی اجرا ادامه پیدا کنه، این کار حداکثر یک بار انجام می‌شه",
    },
    "decided.rejected": {
        "en": "{id}: rejected; the model is told when the run continues",
        "fa": "{id}: رد شد؛ وقتی اجرا ادامه پیدا کنه، به مدل گفته می‌شه",
    },
    # Target lines of a fetch_url / github_read request (D146)
    "target.host": {"en": "host: {value}", "fa": "میزبان: {value}"},
    "target.port": {"en": "port: {value}", "fa": "درگاه (پورت): {value}"},
    "target.path": {"en": "path: {value}", "fa": "مسیر: {value}"},
    "target.query": {"en": "query parameters: {value}", "fa": "پارامترهای نشانی: {value}"},
    "target.bad_url": {
        "en": "invalid URL ({error}); the tool will refuse it",
        "fa": "نشانی نامعتبر ({error})؛ ابزار ردش می‌کنه",
    },
    "target.api_url": {"en": "api url: {value}", "fa": "نشانی API: {value}"},
    "target.kind": {"en": "kind: {value}", "fa": "نوع: {value}"},
    "target.repo": {"en": "repo: {value}", "fa": "مخزن: {value}"},
    "target.bad_request": {
        "en": "invalid request ({error}); the tool will refuse it",
        "fa": "درخواست نامعتبر ({error})؛ ابزار ردش می‌کنه",
    },
    # Tool warnings on an approval card (D148); English from request_warnings
    "warn.fetch.http": {
        "en": WARNING_TEXT["fetch.http"],
        "fa": "رمزنگاری نشده (http): هر کسی در شبکه می‌تونه ببینه یا تغییرش بده",
    },
    "warn.fetch.ip_host": {
        "en": WARNING_TEXT["fetch.ip_host"],
        "fa": "میزبان یک نشانی IP است، نه یک نام",
    },
    "warn.fetch.non_ascii_host": {
        "en": WARNING_TEXT["fetch.non_ascii_host"],
        "fa": "نام میزبان حروف غیرلاتین داره؛ به‌شکل {wire_host} فرستاده می‌شه",
    },
    "warn.fetch.punycode": {
        "en": WARNING_TEXT["fetch.punycode"],
        "fa": "نام میزبان punycode است (xn--): ممکنه شبیه یک نام دیگه به نظر بیاد",
    },
    "warn.fetch.port": {"en": WARNING_TEXT["fetch.port"], "fa": "درگاه غیرمعمول {port}"},
    "warn.fetch.query": {
        "en": WARNING_TEXT["fetch.query"],
        "fa": (
            "بخش پارامترهای نشانی ({count} پارامتر) به {wire_host} فرستاده می‌شه: "
            "مطمئن شو رمز یا دادهٔ پروژه توش نیست"
        ),
    },
    "warn.fetch.long_path": {
        "en": WARNING_TEXT["fetch.long_path"],
        "fa": "مسیر خیلی بلنده: مطمئن شو دادهٔ پروژه توش نیست",
    },
    "warn.github.no_ref": {
        "en": WARNING_TEXT["github.no_ref"],
        "fa": (
            "شاخه یا نسخه مشخص نشده: شاخهٔ اصلی همون‌طور که موقع اجرای وظیفه هست "
            "خونده می‌شه، که ممکنه با الان فرق کنه"
        ),
    },
    "warn.github.root_path": {
        "en": WARNING_TEXT["github.root_path"],
        "fa": "مسیر، ریشهٔ مخزنه: فهرست فایل‌ها خونده می‌شه",
    },
    # Error pages (D147): titles, the back link and redirect pages
    "error.title.not_found": {"en": "Not found", "fa": "پیدا نشد"},
    "error.title.method_not_allowed": {"en": "Method not allowed", "fa": "این کار مجاز نیست"},
    "error.title.unexpected_host": {
        "en": "Unexpected Host header",
        "fa": "نشانی صفحه درست نیست",
    },
    "error.title.refused": {"en": "Refused", "fa": "انجام نشد"},
    "error.title.not_available": {"en": "Not available", "fa": "در دسترس نیست"},
    "error.title.project_unreadable": {
        "en": "Project unreadable",
        "fa": "پروژه خونده نشد",
    },
    "error.title.executive_failed": {
        "en": "The Executive could not finish",
        "fa": "برنامه‌ریزی تموم نشد",
    },
    "error.title.length_required": {"en": "Length required", "fa": "درخواست ناقصه"},
    "error.back": {"en": "← all projects", "fa": "→ همهٔ پروژه‌ها"},
    "error.technical": {"en": "Technical details:", "fa": "جزئیات فنی:"},
    "error.lead.generic": {
        "en": "Something went wrong.",
        "fa": "مشکلی پیش اومد؛ جزئیات فنی پایین‌تره.",
    },
    "redirect.done": {"en": "Done", "fa": "انجام شد"},
    "redirect.recorded": {"en": "Recorded", "fa": "ثبت شد"},
    "redirect.continue": {"en": "continue", "fa": "ادامه"},
    # Refusals (D147): the English text is exactly the message the code raised before
    "refusal.read_only": {
        "en": "This interface is read-only.",
        "fa": "این صفحه فقط برای دیدنه؛ از اینجا چیزی تغییر نمی‌کنه.",
    },
    "refusal.forms_only": {
        "en": "Only the forms of these pages are accepted.",
        "fa": "فقط فرم‌های همین صفحه‌ها پذیرفته می‌شن.",
    },
    "refusal.open_printed_address": {
        "en": "Open the address printed by automa web.",
        "fa": "همون نشانی‌ای رو باز کن که automa web در ترمینال نوشته.",
    },
    "refusal.duplicate_header": {
        "en": "duplicate {name} header",
        "fa": "درخواست سرآیند تکراری داره ({name})",
    },
    "refusal.chunked": {
        "en": "chunked bodies are not accepted",
        "fa": "این شکل فرستادن پذیرفته نمی‌شه",
    },
    "refusal.cross_site": {
        "en": "cross-site request refused",
        "fa": "درخواست از یک سایت دیگه اومده بود و رد شد",
    },
    "refusal.other_origin": {
        "en": "request from another origin refused",
        "fa": "درخواست از صفحه‌ای غیر از همین رابط اومده بود و رد شد",
    },
    "refusal.no_origin": {
        "en": "request without Origin or Referer refused",
        "fa": "معلوم نبود درخواست از کدوم صفحه اومده، پس رد شد",
    },
    "refusal.form_only": {
        "en": "only HTML form submissions are accepted",
        "fa": "فقط فرم‌های همین صفحه‌ها پذیرفته می‌شن",
    },
    "refusal.form_too_large": {"en": "form is too large", "fa": "فرم خیلی بزرگه"},
    "refusal.malformed_form": {
        "en": "malformed form",
        "fa": "فرم ناقص یا خرابه؛ صفحه رو دوباره باز کن",
    },
    "refusal.unexpected_field": {
        "en": "unexpected form field",
        "fa": "فرم یک بخش ناشناخته داره؛ صفحه رو دوباره باز کن",
    },
    "refusal.bad_token": {
        "en": "missing or wrong form token; reload the page",
        "fa": "کد امنیتی فرم درست نیست (مثلاً بعد از راه‌اندازی دوبارهٔ رابط)؛ صفحه رو دوباره باز کن",
    },
    "refusal.no_such_project": {"en": "no such project", "fa": "چنین پروژه‌ای نیست"},
    "refusal.no_project": {"en": "no project {id}", "fa": "پروژهٔ {id} وجود نداره"},
    "refusal.no_such_request": {
        "en": "no such approval request",
        "fa": "چنین درخواست تأییدی نیست",
    },
    "refusal.already_decided": {
        "en": "request {id} is already {status}",
        "fa": "درخواست {id} قبلاً بررسی شده (وضعیت: {status})",
    },
    "refusal.request_changed": {
        "en": "the request differs from the page you saw; reload it",
        "fa": "درخواست با چیزی که دیدی فرق کرده؛ صفحه رو دوباره باز کن",
    },
    "refusal.confirm_arguments": {
        "en": "tick the box to confirm you reviewed the full arguments",
        "fa": "اول تیک تأیید رو بزن تا معلوم باشه جزئیات کامل درخواست رو دیدی",
    },
    "refusal.note_too_long": {
        "en": "note is longer than {max} characters",
        "fa": "یادداشت از {max} نویسه بیشتره",
    },
    "refusal.goal_missing": {
        "en": "write the idea (goal) first",
        "fa": "اول ایده رو بنویس",
    },
    "refusal.goal_too_long": {
        "en": "the goal is longer than {max} characters",
        "fa": "متن ایده از {max} نویسه بیشتره",
    },
    "refusal.questions_changed": {
        "en": "the questions differ from the page you saw; reload it",
        "fa": "سؤال‌ها با چیزی که دیدی فرق کردن؛ صفحه رو دوباره باز کن",
    },
    "refusal.answer_too_long": {
        "en": "answer {n} is longer than {max} characters",
        "fa": "جواب {n} از {max} نویسه بیشتره",
    },
    "refusal.need_answer_too_long": {
        "en": "the answer to needed item {n} is longer than {max} characters",
        "fa": "جواب مورد لازم {n} از {max} نویسه بیشتره",
    },
    "refusal.executive_busy": {
        "en": "the Executive is already working on an idea; wait for it",
        "fa": "Automa الان داره روی یک ایدهٔ دیگه فکر می‌کنه؛ کمی صبر کن",
    },
    "refusal.has_plan": {
        "en": "project {id} already has a plan",
        "fa": "پروژهٔ {id} از قبل برنامه داره",
    },
    "refusal.already_answered": {
        "en": "project {id} was already answered",
        "fa": "به سؤال‌های پروژهٔ {id} قبلاً جواب داده شده",
    },
    "refusal.no_questions": {
        "en": "project {id} has no questions waiting",
        "fa": "پروژهٔ {id} سؤالی منتظر جواب نداره",
    },
    "refusal.not_running_here": {
        "en": (
            "no background run of project {id} is running here "
            "(a run started in a terminal is stopped there with Ctrl+C)"
        ),
        "fa": (
            "پروژهٔ {id} از این صفحه در حال اجرا نیست "
            "(اجرایی که در ترمینال شروع شده، همون‌جا با Ctrl+C متوقف می‌شه)"
        ),
    },
    "refusal.stop_failed": {
        "en": "could not request the stop: {reason}",
        "fa": "درخواست توقف ثبت نشد: {reason}",
    },
    "refusal.already_running": {
        "en": "project {id} is already running here",
        "fa": "پروژهٔ {id} همین الان در حال اجراست",
    },
    "refusal.no_plan": {
        "en": "project {id} has no plan yet",
        "fa": "پروژهٔ {id} هنوز برنامه نداره",
    },
    "refusal.marked_running": {
        "en": (
            "project {id} is marked running (maybe in a terminal); "
            "wait for it to finish (after a crash, resume it from the terminal)"
        ),
        "fa": (
            "پروژهٔ {id} در حال اجرا ثبت شده (شاید در ترمینال)؛ صبر کن تموم بشه "
            "(اگه برنامه بسته شده بود، از ترمینال ادامه‌اش بده)"
        ),
    },
    "refusal.already_done": {
        "en": "project {id} is already done",
        "fa": "پروژهٔ {id} قبلاً تموم شده",
    },
    "refusal.confirm_cost": {
        "en": (
            "the cost range is above confirm_above_usd: tick the box to confirm "
            "it, then start (nothing was run)"
        ),
        "fa": (
            "بازهٔ هزینه از حد تأیید (confirm_above_usd) بالاتره: تیک تأیید هزینه رو بزن "
            "و دوباره «شروع» رو بزن (چیزی اجرا نشد)"
        ),
    },
    "refusal.not_braked": {
        "en": "task {task} of project {id} is not waiting for a spending brake decision",
        "fa": "مرحلهٔ {task} از پروژهٔ {id} منتظر تصمیم دربارهٔ ترمز خرج نیست",
    },
    "refusal.brake_failed": {
        "en": "could not record the decision: {reason}",
        "fa": "تصمیم ثبت نشد: {reason}",
    },
    "refusal.start_failed": {
        "en": "could not start the run: {reason}",
        "fa": "اجرا شروع نشد: {reason}",
    },
    "refusal.address_too_long": {
        "en": "the address is too long",
        "fa": "نشانی صفحه خیلی بلنده",
    },
    "refusal.missing_path": {"en": "missing ?path=", "fa": "نشانی فایل مشخص نشده"},
    "refusal.invalid_query": {
        "en": "invalid query string",
        "fa": "نشانی صفحه خرابه",
    },
    "refusal.only_path": {
        "en": "only ?path= is accepted",
        "fa": "نشانی صفحه بخش ناشناخته داره",
    },
    "refusal.no_files": {
        "en": "this project has no files yet",
        "fa": "این پروژه هنوز فایلی نداره",
    },
    "refusal.image_types": {
        "en": "only .png, .jpg and .jpeg files are shown as images",
        "fa": "فقط فایل‌های .png، .jpg و .jpeg به‌شکل تصویر نشون داده می‌شن",
    },
    "refusal.bad_image": {
        "en": "the file is not a valid image of its extension's type",
        "fa": "این فایل با پسوندش جور نیست و تصویر سالمی نیست",
    },
    "refusal.preview_bad_image": {
        "en": "not a valid image of its extension's type",
        "fa": "این فایل با پسوندش جور نیست و تصویر سالمی نیست",
    },
    "refusal.preview_types": {
        "en": "only HTML pages and their CSS, scripts and images are previewed",
        "fa": "فقط صفحه‌های HTML و CSS، اسکریپت و تصویرهاشون پیش‌نمایش دارن",
    },
    # Leads in front of an untranslated technical message (D147)
    "refusal.idea.config": {
        "en": "The settings are not ready for planning.",
        "fa": "تنظیمات برای برنامه‌ریزی آماده نیست (مثلاً کلید یا مدل‌ها در automa.toml).",
    },
    "refusal.idea.budget": {
        "en": "A spending cap stopped the planning.",
        "fa": "سقف هزینه جلوی برنامه‌ریزی رو گرفت؛ سقف رو در automa.toml بیشتر کن یا صبر کن.",
    },
    "refusal.idea.planning": {
        "en": "The Executive could not make a usable plan.",
        "fa": "مدل نتونست برنامهٔ درستی بسازه؛ ایده رو روشن‌تر بنویس و دوباره امتحان کن.",
    },
    "refusal.idea.provider": {
        "en": "The model gateway did not answer properly.",
        "fa": "درگاه مدل درست جواب نداد؛ کمی بعد دوباره امتحان کن یا موجودی حساب رو ببین.",
    },
    "refusal.lead.store": {
        "en": "The project's data could not be read or written.",
        "fa": "داده‌های پروژه خونده یا نوشته نشد.",
    },
    "refusal.lead.files": {
        "en": "The file could not be read.",
        "fa": "فایل خونده نشد.",
    },
    "refusal.lead.run": {
        "en": "The run could not be prepared.",
        "fa": "اجرا آماده نشد.",
    },
    # Delivery: the product card (D157), retry (D156) and the zip (D158)
    "nav.projects": {"en": "Projects", "fa": "پروژه‌ها"},
    "nav.new": {"en": "New idea", "fa": "ایدهٔ جدید"},
    "deliver.ready.title": {"en": "✅ Your product is ready", "fa": "✅ محصولت آماده‌ست"},
    "deliver.partial.title": {
        "en": "⚠️ The product is not complete yet",
        "fa": "⚠️ محصول هنوز کامل نشده",
    },
    "deliver.promise.title": {"en": "🎯 What you will get", "fa": "🎯 چیزی که تحویل می‌گیری"},
    "deliver.derived": {
        "en": "The plan had no product description; this list comes from its final steps.",
        "fa": "برنامه شرح محصول نداشت؛ این فهرست از مرحله‌های آخرش برداشت شده.",
    },
    "deliver.files": {"en": "Main files", "fa": "فایل‌های اصلی"},
    "deliver.file.missing": {"en": "not made", "fa": "ساخته نشد"},
    "deliver.file.planned": {"en": "not made yet", "fa": "هنوز ساخته نشده"},
    "deliver.file.unchecked": {
        "en": "made, but did not pass its checks",
        "fa": "ساخته شده ولی از بررسی رد شد",
    },
    "deliver.open": {"en": "Open", "fa": "باز کردن"},
    "deliver.view": {"en": "View", "fa": "دیدن"},
    "deliver.download": {"en": "Download", "fa": "دانلود"},
    "deliver.zip": {
        "en": "⬇ Download everything (zip)",
        "fa": "⬇ دانلود همهٔ فایل‌ها (یک فایل فشرده)",
    },
    "deliver.zip.help": {
        "en": "All files plus DELIVERY.md, the report of what was built and checked.",
        "fa": "همهٔ فایل‌ها به‌علاوهٔ گزارش تحویل: چی ساخته شد و چی بررسی شد.",
    },
    "deliver.how": {"en": "How to use it", "fa": "روش استفاده"},
    "deliver.checked": {
        "en": "{passed} of {total} steps passed · {checks} checks passed",
        "fa": "{passed} از {total} مرحله قبول شد · {checks} مورد بررسی قبول شد",
    },
    "deliver.review": {"en": "Final review", "fa": "بازبینی نهایی"},
    "deliver.problems": {"en": "What is still missing", "fa": "چی هنوز کم مونده"},
    "deliver.blocked": {
        "en": "waits for an earlier step",
        "fa": "منتظر یک مرحلهٔ قبلیه",
    },
    "deliver.retry.body": {
        "en": (
            "The failed steps run again on the strongest model, with the problems the "
            "reviewer found as guidance ({attempts} more attempts each). This costs "
            "extra; the budget caps still apply."
        ),
        "fa": (
            "مرحله‌های ناموفق با قوی‌ترین مدل و با ایرادهایی که ارزیاب گرفت دوباره انجام "
            "می‌شن (هر کدوم {attempts} تلاش دیگه). این کار خرج اضافه داره؛ سقف‌های بودجه "
            "همچنان برقرارن."
        ),
    },
    "deliver.retry.confirm": {
        "en": "I accept that trying again costs extra",
        "fa": "قبول دارم که دوباره تلاش کردن خرج اضافه داره",
    },
    "deliver.retry.button": {
        "en": "🔁 Try again until the product is complete",
        "fa": "🔁 دوباره تلاش کن تا محصول کامل بشه",
    },
    "deliver.retry.cli": {
        "en": "From a terminal: {retry}, then {run}",
        "fa": "از ترمینال: {retry} و بعد {run}",
    },
    "ask.title": {
        "en": "❓ I need something from you to go on",
        "fa": "❓ برای ادامه یه چیزی ازت لازم دارم",
    },
    "ask.answer": {"en": "Your answer", "fa": "جوابت"},
    "ask.files": {"en": "Files (optional):", "fa": "فایل (اختیاری):"},
    "ask.files.needed": {"en": "Upload the file(s):", "fa": "فایل(ها) رو بفرست:"},
    "ask.confirm": {
        "en": "Going on may cost more than the confirmation limit; I accept",
        "fa": "ادامه ممکنه از حد تأیید هزینه بیشتر خرج داشته باشه؛ قبول دارم",
    },
    "ask.send": {"en": "Send and continue", "fa": "بفرست و ادامه بده"},
    "ask.decline.open": {"en": "I can't provide this", "fa": "نمی‌تونم این رو بدم"},
    "ask.decline.help": {
        "en": "The work goes on without it and says clearly what is missing because of it.",
        "fa": "کار بدون اون ادامه پیدا می‌کنه و دقیق می‌گه به خاطرش چی کم مونده.",
    },
    "ask.decline.reason": {"en": "Why (optional)", "fa": "چرا (اختیاری)"},
    "ask.decline": {"en": "Go on without it", "fa": "بدون اون ادامه بده"},
    "ask.cli": {
        "en": "Answer from a terminal: {answer} (or: {decline}), then automa run",
        "fa": "از ترمینال جواب بده: {answer} (یا: {decline})، بعد automa run",
    },
    "refusal.no_such_question": {"en": "no such question", "fa": "چنین سؤالی نیست"},
    "refusal.idea_files_too_many": {
        "en": "at most {max} files can be attached to an idea; nothing was created",
        "fa": "حداکثر {max} فایل می‌شه کنار ایده فرستاد؛ چیزی ساخته نشد",
    },
    "refusal.idea_file_too_large": {
        "en": "{name} is larger than {mb} MB; nothing was created",
        "fa": "فایل {name} بزرگ‌تر از {mb} مگابایته؛ چیزی ساخته نشد",
    },
    "refusal.idea_files_too_large": {
        "en": "the files are larger than {mb} MB together; nothing was created",
        "fa": "فایل‌ها روی هم بزرگ‌تر از {mb} مگابایتن؛ چیزی ساخته نشد",
    },
    "refusal.idea_file_not_saved": {
        "en": "could not save the file {name}; the project has no plan yet",
        "fa": "فایل {name} ذخیره نشد؛ پروژه هنوز برنامه نداره",
    },
    "refusal.answer_refused": {
        "en": "the answer was not saved: {reason}",
        "fa": "جواب ذخیره نشد: {reason}",
    },
    "change.title": {"en": "Want a change?", "fa": "تغییری می‌خوای؟"},
    "change.help": {
        "en": (
            "Write what should change (for example: make the header blue, add my "
            "Telegram link). The steps it needs are added and the work goes on; what "
            "is already right stays as it is."
        ),
        "fa": (
            "بنویس چی عوض بشه (مثلاً: سربرگ آبی بشه، لینک تلگرامم اضافه بشه). مرحله‌های "
            "لازم اضافه می‌شن و کار ادامه پیدا می‌کنه؛ چیزهایی که درستن دست نمی‌خورن."
        ),
    },
    "change.send": {"en": "Apply this change", "fa": "این تغییر رو اعمال کن"},
    "refusal.not_changeable": {
        "en": "project {id} must be finished (done or not completed) to ask for a change",
        "fa": "برای درخواست تغییر، پروژهٔ {id} باید تموم شده باشه",
    },
    "task.checks": {"en": "Acceptance checks", "fa": "معیارهای پذیرش"},
    "task.issues": {"en": "Reviewer's notes", "fa": "ایرادهای ارزیاب"},
    "refusal.not_retryable": {
        "en": "project {id} has no failed step to try again",
        "fa": "پروژهٔ {id} مرحلهٔ ناموفقی برای تلاش دوباره نداره",
    },
    "refusal.retry_failed": {
        "en": "could not prepare the new attempt: {reason}",
        "fa": "تلاش دوباره آماده نشد: {reason}",
    },
    "refusal.bundle_too_large": {
        "en": "the project's files are too many or too large for one zip; download them one by one",
        "fa": "فایل‌های پروژه برای یک فایل فشرده خیلی زیاد یا بزرگ‌ان؛ تک‌تک دانلودشون کن",
    },
    "refusal.bundle_changed": {
        "en": "a file changed while the zip was being made; try again",
        "fa": "یک فایل وسط ساختن فایل فشرده عوض شد؛ دوباره امتحان کن",
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

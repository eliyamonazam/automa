"""The family guide (M15, D189): ``GET /guide`` and ``docs/FAMILY-GUIDE.md``.

One step-by-step guide, in very simple language, for someone who has never used a
programming or AI tool: opening the link and saving it on the phone, signing in,
writing an idea (with stock-market and web examples), sending a file, answering the
questions, seeing the cost and pressing "Start", getting the result, and what to do
when something is stuck.

The guide must never describe a button that does not exist. So every UI label it
names is a reference such as ``{ui:run.start.button}`` that is filled in from the
same message tables the pages use (:mod:`automa.webui.i18n`, :mod:`automa.webui.cards`,
:mod:`automa.webui.spending`): renaming a button renames it here, and an unknown key
fails at import. The Markdown file is generated from the same content
(``python -m automa.webui.guide > docs/FAMILY-GUIDE.md``) and a test checks it is
current, so the page and the file can never drift apart.

Pure: no I/O except the ``__main__`` print. Everything shown is fixed text.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict

from automa.ideafiles import MAX_IDEA_FILE_BYTES, MAX_IDEA_FILES
from automa.l10n import LANGUAGES, Language, localize_digits
from automa.webui import cards, i18n, spending
from automa.webui.render import ICONS, page, text
from automa.webui.static import GUIDE_IMAGES, guide_image_path, guide_image_size

GUIDE_PATH = "/guide"

Text = Mapping[Language, str]

_TABLES: Mapping[str, Mapping[str, Mapping[Language, str]]] = {
    "ui": i18n.MESSAGES,
    "card": cards.MESSAGES,
    "spend": spending.MESSAGES,
}
_REF = re.compile(r"\{(ui|card|spend):([a-z0-9_.]+)\}")
_NUMBERS: Mapping[str, int] = {
    "max_files": MAX_IDEA_FILES,
    "max_mb": MAX_IDEA_FILE_BYTES // (1024 * 1024),
}
_NUM = re.compile(r"\{(" + "|".join(_NUMBERS) + r")\}")


class Section(BaseModel):
    """One part of the guide: a heading, numbered steps and an optional tip."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: Text
    steps: tuple[Text, ...]
    tip: Text | None = None


def _s(fa: str, en: str) -> Text:
    return {"fa": fa, "en": en}


TITLE = _s("راهنمای Automa", "Automa guide")
LEAD = _s(
    "Automa ایده‌ات رو به یک محصول واقعی تبدیل می‌کنه: یک فایل اکسل، یک نمودار، یک صفحهٔ "
    "وب یا یک برنامهٔ کوچک. لازم نیست چیزی از برنامه‌نویسی بدونی؛ فقط بنویس چی می‌خوای. "
    "اگه چیزی لازم داشته باشه ازت می‌پرسه، و هیچ‌وقت عدد یا نتیجهٔ ساختگی نمی‌ده.",
    "Automa turns your idea into a real product: a spreadsheet, a chart, a web page or a "
    "small program. You don't need to know anything about programming; just write what "
    "you want. If it needs something it asks you, and it never makes up numbers or results.",
)

EXAMPLES_HEADING = _s("چند نمونه ایده", "A few example ideas")
EXAMPLES: tuple[Text, ...] = (
    _s(
        "بورس: «فایل اکسلی که از سایت TSETMC گرفتم رو فرستادم. نمادهایی که حجم معاملهٔ "
        "امروزشون بیشتر از دو برابر میانگین یک ماه گذشته‌ست رو جدا کن و یک فایل اکسل و یک "
        "نمودار بده.»",
        "Stock market: “I've attached the Excel file I downloaded from TSETMC. Pick the "
        "symbols whose volume today is more than twice their last-month average and give "
        "me an Excel file and a chart.”",
    ),
    _s(
        "بورس: «پرتفوی من در این فایل اکسله (نماد، تعداد، قیمت خرید). سود و زیان هر سهم "
        "رو با قیمت‌های فایل دوم حساب کن و یک گزارش فارسی ساده بده.»",
        "Stock market: “My portfolio is in this Excel file (symbol, shares, buy price). "
        "Work out the profit or loss of each one with the prices in the second file and "
        "give me a simple report.”",
    ),
    _s(
        "وب: «یک صفحهٔ وب ساده برای معرفی کارگاه خیاطی‌ام، با اسم، چند عکس، ساعت کاری و دکمهٔ تماس.»",
        "Web: “A simple web page for my tailoring shop, with its name, a few photos, "
        "opening hours and a call button.”",
    ),
    _s(
        "برنامه: «یک اسکریپت پایتون که همهٔ فایل‌های اکسل یک پوشه رو یکی کنه.»",
        "Program: “A Python script that merges all the Excel files in a folder.”",
    ),
)

SECTIONS: tuple[Section, ...] = (
    Section(
        id="open",
        title=_s(
            "۱. باز کردن Automa و گذاشتنش روی گوشی", "1. Open Automa and keep it on your phone"
        ),
        steps=(
            _s(
                "لینکی که صاحب Automa بهت داده رو در مرورگر گوشی یا لپ‌تاپ باز کن "
                "(در اندروید Chrome، در آیفون Safari).",
                "Open the link the owner of this Automa gave you in your phone's or "
                "laptop's browser (Chrome on Android, Safari on iPhone).",
            ),
            _s(
                "اندروید: در Chrome دکمهٔ سه‌نقطهٔ بالای صفحه رو بزن و «افزودن به صفحهٔ "
                "اصلی» رو انتخاب کن. اگه زبان گوشیت انگلیسیه، اسمش Add to Home screen است.",
                "Android: in Chrome tap the three-dot button at the top and choose "
                "“Add to Home screen”.",
            ),
            _s(
                "آیفون: در Safari دکمهٔ اشتراک‌گذاری (مربعی که فلشش رو به بالاست) رو بزن و "
                "«افزودن به صفحهٔ اصلی» رو انتخاب کن. اگه زبان گوشیت انگلیسیه، اسمش "
                "Add to Home Screen است.",
                "iPhone: in Safari tap the Share button (a square with an arrow) and "
                "choose “Add to Home Screen”.",
            ),
            _s(
                "از این به بعد Automa مثل یک برنامه روی صفحهٔ گوشیت هست؛ فقط روش بزن.",
                "From now on Automa sits on your home screen like an app; just tap it.",
            ),
        ),
        tip=_s(
            "اسم دکمه‌ها در گوشی‌های مختلف ممکنه کمی فرق کنه. لینک رو برای کس دیگه‌ای نفرست.",
            "The button names can differ a little between phones. Don't send the link "
            "to anyone else.",
        ),
    ),
    Section(
        id="login",
        title=_s("۲. ورود", "2. Sign in"),
        steps=(
            _s(
                "{ui:login.name} و {ui:login.password} رو همون‌طور که صاحب Automa بهت "
                "گفته بنویس و {ui:login.submit} رو بزن.",
                "Type the {ui:login.name} and {ui:login.password} the owner gave you and "
                "press {ui:login.submit}.",
            ),
            _s(
                "اسم حسابت بالای صفحه نوشته می‌شه. برای بیرون رفتن، {ui:nav.logout} رو بزن.",
                "Your account name shows at the top. To leave, press {ui:nav.logout}.",
            ),
        ),
        tip=_s(
            "اگه چند بار رمز اشتباه بزنی، حساب چند دقیقه قفل می‌شه؛ صبر کن و دوباره امتحان "
            "کن. رمزت رو به هیچ‌کس نده و هیچ‌وقت داخل متن ایده ننویس.",
            "After a few wrong passwords the account locks for a few minutes; wait and "
            "try again. Never give your password to anyone or write it in an idea.",
        ),
    ),
    Section(
        id="idea",
        title=_s("۳. نوشتن ایده", "3. Write your idea"),
        steps=(
            _s(
                "دکمهٔ {ui:nav.new} رو بزن (در گوشی دکمهٔ + بالای صفحه).",
                "Press {ui:nav.new} (on a phone, the + button at the top).",
            ),
            _s(
                "در جعبهٔ بزرگ با زبان خودت بنویس چی می‌خوای: چه چیزی، از چه داده‌ای، و "
                "خروجی به چه شکلی (اکسل، نمودار، صفحهٔ وب، گزارش). هر چی دقیق‌تر، بهتر.",
                "In the big box write, in your own words, what you want: what, from which "
                "data, and in what form (Excel, chart, web page, report). The more exact, "
                "the better.",
            ),
            _s(
                "بهتره تیک {ui:idea.ask} روشن بمونه تا Automa قبل از هر کاری چیزهای مبهم رو بپرسه.",
                "Keep {ui:idea.ask} ticked so Automa asks about anything unclear first.",
            ),
            _s(
                "{ui:idea.submit} رو بزن. هنوز هیچ کاری انجام نمی‌شه و پولی خرج نمی‌شه "
                "جز چند لحظه فکر کردن برای سؤال‌ها و برنامه.",
                "Press {ui:idea.submit}. Nothing is done yet; only a little is spent "
                "thinking up the questions and the plan.",
            ),
        ),
    ),
    Section(
        id="files",
        title=_s("۴. فرستادن فایل", "4. Send a file"),
        steps=(
            _s(
                "اگه داده‌ات در فایله (مثلاً خروجی اکسل از سایت TSETMC، یا فهرستی که خودت "
                "داری)، در همون فرم زیر {ui:idea.files} فایل رو انتخاب کن.",
                "If your data is in a file (for example an Excel export from TSETMC, or a "
                "list of your own), pick it under {ui:idea.files} in the same form.",
            ),
            _s(
                "تا {max_files} فایل، هر کدوم تا {max_mb} مگابایت. Automa با همین داده‌ها "
                "کار می‌کنه و چیزی از خودش جای اون‌ها نمی‌ذاره.",
                "Up to {max_files} files, each up to {max_mb} MB. Automa works with this "
                "data and never puts made-up data in its place.",
            ),
        ),
        tip=_s(
            "Automa نمی‌تونه وارد حساب‌های تو (مثلاً کارگزاری) بشه. داده رو خودت از اون "
            "سایت به‌شکل فایل بگیر و این‌جا بفرست.",
            "Automa cannot sign in to your accounts (a broker, for example). Download the "
            "data from that site as a file yourself and send it here.",
        ),
    ),
    Section(
        id="questions",
        title=_s("۵. جواب دادن به سؤال‌ها", "5. Answer the questions"),
        steps=(
            _s(
                "اگه Automa سؤالی داشته باشه، در صفحهٔ اصلی زیر {card:waiting.heading} "
                "دکمهٔ {card:card.answer} رو می‌بینی؛ بزنش.",
                "If Automa has questions, the home page lists the project under "
                "{card:waiting.heading} with an {card:card.answer} button; press it.",
            ),
            _s(
                "به هر سؤالی که می‌دونی جواب بده؛ اگه نمی‌دونی خالی بذار.",
                "Answer what you know; leave the rest empty.",
            ),
            _s(
                "زیر {ui:clarify.needs} چیزهایی هست که Automa برای کار درست لازم داره؛ "
                "می‌تونی بنویسیش یا فایلش رو بدی.",
                "Under {ui:clarify.needs} are the things Automa needs to do the job "
                "right; type them or attach the file.",
            ),
            _s(
                "{ui:clarify.submit} رو بزن و چند لحظه صبر کن تا برنامه آماده بشه.",
                "Press {ui:clarify.submit} and wait a moment for the plan.",
            ),
        ),
    ),
    Section(
        id="start",
        title=_s("۶. دیدن هزینه و شروع", "6. See the cost and start"),
        steps=(
            _s(
                "در صفحهٔ پروژه برنامه رو می‌بینی: مرحله‌ها و {ui:project.cost_range} "
                "(به تومان). این یک تخمینه، نه قول دقیق.",
                "The project page shows the plan: its steps and the {ui:project.cost_range} "
                "(in toman). It is an estimate, not an exact promise.",
            ),
            _s(
                "اگه موافقی، {ui:run.start.button} رو بزن. Automa در پس‌زمینه کار می‌کنه و "
                "صفحه خودش تازه می‌شه؛ می‌تونی صفحه رو ببندی و بعداً برگردی.",
                "If you agree, press {ui:run.start.button}. Automa works in the background "
                "and the page refreshes itself; you can close it and come back later.",
            ),
            _s(
                "هر وقت خواستی، {ui:run.stop.button} کار رو بعد از قدم فعلی نگه می‌داره؛ "
                "با {ui:run.start.button} دوباره ادامه پیدا می‌کنه.",
                "Any time, {ui:run.stop.button} pauses after the current step; "
                "{ui:run.start.button} carries on.",
            ),
            _s(
                "خرج این ماهت رو با زدن آیکون کیف پول بالای صفحه ببین ({spend:title}).",
                "See this month's spending by tapping the wallet icon at the top ({spend:title}).",
            ),
        ),
        tip=_s(
            "هر حساب یک سقف ماهانه داره. اگه پر بشه کار تازه تا ماه بعد می‌ایسته، یا تا "
            "وقتی صاحب Automa سقفت رو بیشتر کنه.",
            "Each account has a monthly limit. When it is used up, new work waits until "
            "next month or until the owner raises your limit.",
        ),
    ),
    Section(
        id="during",
        title=_s("۷. وسط کار", "7. While it works"),
        steps=(
            _s(
                "اگه Automa وسط کار چیزی لازم داشته باشه، کارت پروژه {card:card.waiting} "
                "می‌شه و در صفحه {ui:ask.title} رو می‌بینی.",
                "If Automa needs something mid-way, the card says {card:card.waiting} and "
                "the page shows {ui:ask.title}.",
            ),
            _s(
                "جوابت یا فایل رو بده و {ui:ask.send} رو بزن. اگه نمی‌تونی، "
                "{ui:ask.decline.open} و بعد {ui:ask.decline} رو بزن؛ آخر کار دقیق می‌گه "
                "به خاطرش چی کم مونده.",
                "Give your answer or file and press {ui:ask.send}. If you can't, press "
                "{ui:ask.decline.open} then {ui:ask.decline}; at the end it says exactly "
                "what is missing because of it.",
            ),
            _s(
                "اگه کارت {card:card.with_owner} بگه، از تو کاری لازم نیست؛ صاحب Automa "
                "باید یک کار رو تأیید کنه.",
                "If the card says {card:card.with_owner}, there is nothing for you to do; "
                "the owner has to approve something.",
            ),
        ),
    ),
    Section(
        id="result",
        title=_s("۸. دیدن و دانلود نتیجه", "8. See and download the result"),
        steps=(
            _s(
                "وقتی تموم بشه، کارت پروژه {card:card.ready} می‌شه و در صفحه "
                "{ui:deliver.ready.title} رو می‌بینی، با فهرست فایل‌ها و {ui:deliver.how}.",
                "When it is done the card says {card:card.ready} and the page shows "
                "{ui:deliver.ready.title}, with the files and {ui:deliver.how}.",
            ),
            _s(
                "برای هر فایل {ui:deliver.open} یا {ui:deliver.download} رو بزن، یا همه "
                "رو با هم با {ui:deliver.zip} بگیر.",
                "Press {ui:deliver.open} or {ui:deliver.download} for each file, or get "
                "them all with {ui:deliver.zip}.",
            ),
            _s(
                "اگه چیزی باید عوض بشه، زیر {ui:change.title} بنویس و {ui:change.send} رو بزن.",
                "If something should change, write it under {ui:change.title} and press "
                "{ui:change.send}.",
            ),
        ),
    ),
    Section(
        id="stuck",
        title=_s("۹. وقتی چیزی گیر کرد", "9. When something is stuck"),
        steps=(
            _s(
                "{card:card.partial}: در صفحهٔ پروژه {ui:deliver.problems} رو بخون. اگه "
                "چیزی از تو لازمه بده، یا {ui:deliver.retry.button} رو بزن (خرج اضافه "
                "داره).",
                "{card:card.partial}: read {ui:deliver.problems} on the project page. "
                "Give what it needs from you, or press {ui:deliver.retry.button} (it "
                "costs extra).",
            ),
            _s(
                "{card:card.paused}: صفحهٔ پروژه دلیلش رو به زبان ساده می‌گه؛ معمولاً با "
                "{ui:run.start.button} ادامه پیدا می‌کنه.",
                "{card:card.paused}: the project page says why in plain words; usually "
                "{ui:run.start.button} carries on.",
            ),
            _s(
                "اگه سقف ماهانه‌ت پر شده یا حسابت قفل یا غیرفعال شده، به صاحب Automa بگو.",
                "If your monthly limit is used up, or your account is locked or disabled, "
                "tell the owner.",
            ),
            _s(
                "هر چیز دیگه: از صفحه عکس بگیر (اسکرین‌شات) و برای صاحب Automa بفرست. "
                "نگران نباش؛ با دکمه‌های این صفحه‌ها چیزی پاک نمی‌شه.",
                "Anything else: take a screenshot of the page and send it to the owner. "
                "Don't worry; no button on these pages deletes anything.",
            ),
        ),
    ),
)

# One sample picture per section that has one (D190): ``<section id>.png`` in
# ``automa/webui/static/guide/``, made by ``scripts/guide_screenshots.py`` and served
# at ``/static/guide/<id>.png``. The text is the picture's alt text and caption.
IMAGES: Mapping[str, Text] = {
    "login": _s("صفحهٔ ورود", "The sign-in page"),
    "idea": _s("فرم ایدهٔ جدید با یک ایدهٔ نوشته‌شده", "The new-idea form with an idea typed in"),
    "questions": _s(
        "صفحهٔ سؤال‌ها و چیزهایی که Automa لازم داره",
        "The questions page and what Automa needs from you",
    ),
    "start": _s(
        "صفحهٔ پروژه با برنامه، هزینه و دکمهٔ شروع", "A project page with the plan, cost and Start"
    ),
    "result": _s("صفحهٔ پروژه وقتی محصول آماده‌ست", "A project page when the product is ready"),
}
SAMPLE_NOTE = _s(
    "تصویر نمونه",
    "sample picture",
)

_ALL_TEXTS: tuple[Text, ...] = (
    TITLE,
    LEAD,
    SAMPLE_NOTE,
    *IMAGES.values(),
    EXAMPLES_HEADING,
    *EXAMPLES,
    *(t for s in SECTIONS for t in (s.title, *s.steps, *((s.tip,) if s.tip else ()))),
)


def label(ref: str, key: str, lang: Language) -> str:
    """The UI label ``key`` of table ``ref`` as shown on the page (no ``+`` prefix,
    no trailing colon); unknown keys raise ``KeyError``."""
    value = _TABLES[ref][key][lang].strip()
    if value.startswith("+ "):
        value = value[2:]
    return value.rstrip(":").strip()


def _check() -> None:
    """Every text has both languages, the same references in each, and only known
    references; every picture belongs to a section and is a served asset; run at
    import so a renamed key fails every test."""
    ids = {s.id for s in SECTIONS}
    if set(IMAGES) - ids or set(IMAGES) != set(GUIDE_IMAGES):
        raise ValueError("guide pictures must match the sections and the static assets")
    for item in _ALL_TEXTS:
        if set(item) != set(LANGUAGES):
            raise ValueError(f"guide text without both languages: {item!r}")
        refs = {lang: sorted(_REF.findall(item[lang])) for lang in LANGUAGES}
        if refs["fa"] != refs["en"]:
            raise ValueError(f"guide text with different labels per language: {item!r}")
        for ref, key in refs["fa"]:
            label(ref, key, "fa")
        nums = {lang: sorted(_NUM.findall(item[lang])) for lang in LANGUAGES}
        if nums["fa"] != nums["en"]:
            raise ValueError(f"guide text with different numbers per language: {item!r}")
        leftover = _NUM.sub("", _REF.sub("", item["fa"] + item["en"]))
        if "{" in leftover or "}" in leftover:
            raise ValueError(f"guide text with an unknown placeholder: {item!r}")


_check()


def _fill(
    template: str, lang: Language, ui: Callable[[str], str], plain: Callable[[str], str]
) -> str:
    """``template`` with ``plain`` applied to its text, ``ui`` to each quoted label."""
    out: list[str] = []
    pos = 0
    for m in _REF.finditer(template):
        out.append(plain(_numbers(template[pos : m.start()], lang)))
        quoted = f"«{label(m[1], m[2], lang)}»" if lang == "fa" else f"“{label(m[1], m[2], lang)}”"
        out.append(ui(quoted))
        pos = m.end()
    out.append(plain(_numbers(template[pos:], lang)))
    return "".join(out)


def _numbers(value: str, lang: Language) -> str:
    return _NUM.sub(lambda m: localize_digits(str(_NUMBERS[m[1]]), lang), value)


def fill_text(item: Text, lang: Language) -> str:
    """Plain text of a guide text with labels quoted (for tests and Markdown)."""
    return _fill(item[lang], lang, lambda s: s, lambda s: s)


def _html(item: Text, lang: Language) -> str:
    return _fill(item[lang], lang, lambda s: f'<b class="ui">{text(s)}</b>', text)


def render_guide(lang: Language) -> str:
    """The ``/guide`` page: fixed text only, readable without any project."""
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language: {lang!r}")
    toc = "".join(f'<li><a href="#{s.id}">{_html(s.title, lang)}</a></li>' for s in SECTIONS)
    examples = "".join(f'<li dir="auto">{_html(e, lang)}</li>' for e in EXAMPLES)
    parts = [
        f'<section class="hero guide-hero"><h1>{ICONS["help"]}{_html(TITLE, lang)}</h1>'
        f"<p>{_html(LEAD, lang)}</p></section>",
        f'<nav class="card guide-toc" aria-label="{text(TITLE[lang])}"><ul>{toc}</ul></nav>',
    ]
    for s in SECTIONS:
        steps = "".join(f"<li>{_html(step, lang)}</li>" for step in s.steps)
        tip = f'<p class="guide-tip">{_html(s.tip, lang)}</p>' if s.tip else ""
        shot = ""
        if s.id in IMAGES:
            alt = IMAGES[s.id][lang]
            width, height = guide_image_size(s.id)
            shot = (
                f'<figure class="guide-shot"><img src="{guide_image_path(s.id)}" '
                f'alt="{text(alt)}" width="{width}" height="{height}" loading="lazy">'
                f"<figcaption>{text(alt)} · {text(SAMPLE_NOTE[lang])}</figcaption></figure>"
            )
        extra = ""
        if s.id == "idea":
            extra = (
                f"<h3>{_html(EXAMPLES_HEADING, lang)}</h3>"
                f'<ul class="guide-examples">{examples}</ul>'
            )
        parts.append(
            f'<section class="card guide-step" id="{s.id}"><h2>{_html(s.title, lang)}</h2>'
            f"<ol>{steps}</ol>{extra}{shot}{tip}</section>"
        )
    return page(TITLE[lang], "".join(parts), lang=lang, self_images=True)


def guide_markdown(lang: Language = "fa") -> str:
    """``docs/FAMILY-GUIDE.md``: the same guide as Markdown."""
    note = (
        "> این فایل از `src/automa/webui/guide.py` ساخته می‌شه "
        "(`python -m automa.webui.guide > docs/FAMILY-GUIDE.md`)؛ دستی ویرایشش نکن. "
        "همین راهنما در Automa از دکمهٔ «راهنما» بالای صفحه (`/guide`) هم باز می‌شه."
        if lang == "fa"
        else "> Generated from `src/automa/webui/guide.py`; do not edit by hand. "
        "The same guide opens in Automa from “Guide” at the top (`/guide`)."
    )
    bold = lambda s: f"**{s}**"  # noqa: E731
    lines = [f"# {fill_text(TITLE, lang)}", "", note, "", fill_text(LEAD, lang), ""]
    for s in SECTIONS:
        lines += [f"## {fill_text(s.title, lang)}", ""]
        for n, step in enumerate(s.steps, 1):
            lines.append(f"{n}. {_fill(step[lang], lang, bold, lambda v: v)}")
        lines.append("")
        if s.id == "idea":
            lines += [f"### {fill_text(EXAMPLES_HEADING, lang)}", ""]
            lines += [f"- {fill_text(e, lang)}" for e in EXAMPLES]
            lines.append("")
        if s.id in IMAGES:
            alt = IMAGES[s.id][lang]
            src = f"../src/automa/webui/static/guide/{s.id}.png"
            lines += [
                f'<p align="center"><img src="{src}" alt="{alt}" width="260"><br>',
                f"<sub>{alt} · {SAMPLE_NOTE[lang]}</sub></p>",
                "",
            ]
        if s.tip:
            lines += [f"> {_fill(s.tip[lang], lang, bold, lambda v: v)}", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


if __name__ == "__main__":  # pragma: no cover - the docs generator
    sys.stdout.write(guide_markdown("fa"))

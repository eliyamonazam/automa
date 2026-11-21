"""Pure HTML rendering of the web UI views (D77).

Every piece of recorded text (goals, titles, reasons, arguments — some of it written by
a model) goes through :func:`text`: control, format and bidi characters are shown as
visible escapes (so they cannot reorder or hide what is being reviewed) and the result
is HTML-escaped. Pages have no scripts and no external resources, and a strict
Content-Security-Policy; forms appear only in action mode (D123) and post only to
this server. Nothing here reads files or the network.
"""

from __future__ import annotations

import html
import json
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from string import Formatter
from urllib.parse import quote

from automa.delivery import Delivery, DeliveryFile, DeliveryState, TaskDelivery
from automa.l10n import (
    DEFAULT_TIMEZONE,
    Language,
    localize_digits,
    relative_time,
    resolve_timezone,
)
from automa.plan import TaskStatus
from automa.plan.models import MAX_ANSWER_CHARS
from automa.project.store import MAX_GOAL_CHARS
from automa.retry import RETRY_ATTEMPTS
from automa.tools.approvals import MAX_NOTE_CHARS
from automa.tools.missing import install_command
from automa.webui.bundle import bundle_href
from automa.webui.files import (
    MAX_IMAGE_BYTES,
    FileContent,
    FileListing,
    FilesRoute,
    files_href,
)
from automa.webui.i18n import (
    MESSAGES,
    WELCOME_EXAMPLE_KEYS,
    exit_label,
    localize_number,
    t,
    task_state_label,
)
from automa.webui.ideas import ASK_VALUE
from automa.webui.journey import MESSAGES as JOURNEY_MESSAGES
from automa.webui.journey import Journey, build_journey, jt
from automa.webui.money import money, money_range, nusd_money
from automa.webui.preview import is_previewable, preview_href
from automa.webui.progress import REFRESH_SECONDS, CostProgress, ProgressView, UnusualSpending
from automa.webui.status import Chip, project_chip, task_chip
from automa.webui.stops import MESSAGES as STOP_MESSAGES
from automa.webui.stops import StopInfo, StopLine, describe_stop
from automa.webui.stops import st as stop_text
from automa.webui.views import ApprovalView, Overview, ProjectView, TaskRow, project_title

# Joiners needed by Persian text; every other Cf character (incl. bidi controls) is escaped.
_ALLOWED_FORMAT_CHARS = frozenset({"‌", "‍"})

CONTENT_SECURITY_POLICY = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; "
    "form-action 'none'; base-uri 'none'"
)
# Action mode (D123): forms may post only to this server; nothing else changes.
ACTION_CONTENT_SECURITY_POLICY = CONTENT_SECURITY_POLICY.replace(
    "form-action 'none'", "form-action 'self'"
)

# Local fonts only (the CSP loads nothing): Persian-capable faces first — Vazirmatn if
# installed, Segoe UI/Tahoma on Windows, Noto on Linux — then the system UI face (D129).
FONT_STACK = (
    '"Vazirmatn","Segoe UI",Tahoma,"Noto Sans Arabic","Noto Naskh Arabic",'
    '"Noto Sans",system-ui,-apple-system,sans-serif'
)
MONO_STACK = 'ui-monospace,"Cascadia Mono",Consolas,"DejaVu Sans Mono",monospace'

_STYLE = (
    """
:root{color-scheme:light dark;--bg:#f5f6fa;--card:#fff;--fg:#151826;--muted:#6b7285;
--line:#e6e8f0;--soft:#f0f2f8;--accent:#4f46e5;--accent2:#7c3aed;--accentb:#0ea5e9;
--code:#f2f4f9;--warn:#b54708;--err:#c01048;--chip:#eef0f6;--ok:#dcfae6;--okfg:#067647;
--okline:#12b76a;--bad:#fee4e2;--badfg:#b42318;--badline:#f04438;--busy:#fef0c7;
--busyfg:#93370d;--busyline:#f79009;--info:#e8edff;--infofg:#3538cd;--shadow:#1a1f3612;
--radius:16px}
@media (prefers-color-scheme: dark){:root{--bg:#0d0f16;--card:#161925;--fg:#eceef5;
--muted:#9aa2b6;--line:#262b3a;--soft:#1c2030;--accent:#8b85ff;--accent2:#b18cff;
--accentb:#38bdf8;--code:#1f2333;--warn:#fdb022;--err:#fd6f8e;--chip:#232838;--ok:#0f3d27;
--okfg:#75e0a7;--bad:#4a1a1a;--badfg:#fda29b;--busy:#4a3410;--busyfg:#fec84b;
--info:#1e2450;--infofg:#a4bcfd;--shadow:#00000066;--okline:#17b26a;--badline:#f97066;
--busyline:#fdb022;--radius:16px}}
*{box-sizing:border-box}
html{background:var(--bg)}
body{font-family:"""
    + FONT_STACK
    + """;font-size:16px;line-height:1.75;margin:0;color:var(--fg);background:var(--bg);
-webkit-font-smoothing:antialiased}
.topbar{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:18px;
padding:10px max(16px,calc((100% - 1100px)/2));background:color-mix(in srgb,var(--card) 88%,
transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.brand{font-weight:800;font-size:1.1em;text-decoration:none;color:var(--fg);
display:inline-flex;align-items:center;gap:8px}
.brand .logo{display:inline-flex;width:28px;height:28px;border-radius:9px;align-items:center;
justify-content:center;color:#fff;font-size:.9em;
background:linear-gradient(135deg,var(--accent),var(--accent2))}
.topbar nav{display:flex;gap:6px;margin-inline-start:auto}
.topbar nav a{color:var(--muted);text-decoration:none;padding:4px 12px;border-radius:999px;
font-weight:600;font-size:.95em}.topbar nav a:hover{background:var(--soft);color:var(--fg)}
.wrap{max-width:1100px;margin:0 auto;padding:18px 16px 48px}
h1{font-size:1.65em;margin:.3em 0;line-height:1.4}h2{font-size:1.2em;margin:1.3em 0 .5em}
h3{font-size:1.04em;margin:.4em 0}
a{color:var(--accent)}code,pre{font-family:"""
    + MONO_STACK
    + """;font-size:.92em}
code{background:var(--code);padding:1px 6px;border-radius:6px}
[dir=auto],pre{unicode-bidi:plaintext}
pre{white-space:pre-wrap;word-break:break-word;background:var(--code);padding:10px 12px;
border-radius:10px;text-align:start}
.hero{background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 55%,var(--accentb) 100%);
color:#fff;border-radius:22px;padding:22px 26px;margin:4px 0 16px;
box-shadow:0 14px 34px -12px #4f46e580}
.hero h1{margin:0 0 .25em;font-size:1.75em}.hero p{margin:.2em 0;opacity:.95}
.hero a{color:#fff}.hero .muted{color:#ffffffcc}.hero code{background:#ffffff26;color:#fff}
.hero .button{background:#fff;color:var(--accent);box-shadow:none}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;
margin:12px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
padding:12px 16px;box-shadow:0 1px 3px var(--shadow)}
.stat .label{color:var(--muted);font-size:.85em;display:block}
.stat .value{font-size:1.3em;font-weight:800}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
padding:16px 20px;margin:14px 0;box-shadow:0 1px 3px var(--shadow),0 8px 24px -18px var(--shadow)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.cards>.card{margin:0}.card h2,.card h3{margin-top:0}
.project{position:relative;overflow:hidden;transition:transform .15s,box-shadow .15s}
.project::before{content:"";position:absolute;inset:0 0 auto 0;height:4px;background:var(--accent)}
.project:hover{transform:translateY(-2px);box-shadow:0 12px 30px -16px var(--shadow)}
.project h2{font-size:1.08em;margin:.2em 0}.project h2 a{text-decoration:none;color:var(--fg)}
.project.ps-done::before{background:var(--okline)}
.project.ps-failed::before{background:var(--badline)}
.project.ps-running::before{background:var(--busyline)}
.project.ps-paused::before{background:#98a2b3}
.goal{margin:.3em 0;overflow-wrap:anywhere;white-space:pre-line}
.now{font-size:1.12em;font-weight:700;padding:12px 16px;border-radius:12px;
background:var(--info);color:var(--infofg);margin:6px 0 10px}
.now.ok{background:var(--ok);color:var(--okfg)}.now.bad{background:var(--bad);color:var(--badfg)}
.now.busy{background:var(--busy);color:var(--busyfg)}
.costcard{border-top:1px solid var(--line);margin-top:12px;padding-top:8px}
.costcard .big{font-size:1.9em;margin:.15em 0;color:var(--accent)}
.meta{color:var(--muted);font-size:.9em;margin:.2em 0}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid var(--line);
padding:8px 10px;text-align:start;vertical-align:top}th{color:var(--muted);font-weight:600;
font-size:.9em}
.scroll{overflow-x:auto}
.warn{color:var(--warn)}.err{color:var(--err)}
.badge{display:inline-flex;align-items:center;gap:4px;padding:1px 11px;border-radius:999px;
background:var(--chip);font-size:.82em;font-weight:700;vertical-align:middle;line-height:1.9}
.st-passed,.ps-done .badge.st,.badge.ps-done{background:var(--ok);color:var(--okfg)}
.st-failed,.st-blocked,.badge.ps-failed{background:var(--bad);color:var(--badfg)}
.st-running,.st-verifying,.badge.ps-running{background:var(--busy);color:var(--busyfg)}
.badge.ps-planning,.badge.ps-new{background:var(--info);color:var(--infofg)}
.badge.tone-paused{background:#d0d5dd;color:#1d2939}
.hero .badge{background:#ffffff2e;color:#fff}
.hint{color:var(--muted);font-size:.9em;margin:.2em 0}
.task{border-inline-start:4px solid var(--line)}.t-passed{border-inline-start-color:var(--okline)}
.t-failed,.t-blocked{border-inline-start-color:var(--badline)}
.t-running,.t-verifying{border-inline-start-color:var(--busyline)}
.chips{display:flex;flex-wrap:wrap;gap:6px 14px;color:var(--muted);font-size:.88em;margin:.4em 0}
meter{width:12em;vertical-align:middle}
.bar{height:12px;border-radius:999px;background:var(--chip);overflow:hidden;margin:8px 0}
.bar .fill{height:100%;border-radius:999px;
background:linear-gradient(90deg,var(--accent),var(--accentb))}
.steps{list-style:none;margin:14px 0 4px;padding:0;display:flex;flex-wrap:wrap;gap:0}
.steps li{flex:1 1 120px;position:relative;text-align:center;padding:0 6px 6px;
min-width:110px}
.steps li::before{content:"";position:absolute;top:17px;inset-inline-start:0;width:100%;
height:4px;background:var(--line);z-index:0}
.steps li:first-child::before{inset-inline-start:50%;width:50%}
.steps li:last-child::before{width:50%}
.steps li:only-child::before{display:none}
.steps .dot{position:relative;z-index:1;display:inline-flex;align-items:center;
justify-content:center;width:38px;height:38px;border-radius:50%;background:var(--card);
border:3px solid var(--line);font-weight:800}
.steps .name{display:block;font-size:.88em;margin-top:4px;overflow-wrap:anywhere}
.steps .when{display:block;font-size:.78em;color:var(--muted)}
.steps .s-done::before{background:var(--okline)}
.steps .s-done .dot{background:var(--okline);border-color:var(--okline);color:#fff}
.steps .s-now .dot{border-color:var(--busyline);background:var(--busy);color:var(--busyfg)}
.steps .s-now .name{font-weight:800}
.steps .s-fail .dot{background:var(--badline);border-color:var(--badline);color:#fff}
.deliver{border:1px solid var(--line);border-top:6px solid var(--accent);padding:18px 22px}
.deliver.ready{border-top-color:var(--okline);
background:linear-gradient(180deg,color-mix(in srgb,var(--ok) 55%,var(--card)),var(--card) 140px)}
.deliver.partial{border-top-color:var(--busyline);
background:linear-gradient(180deg,color-mix(in srgb,var(--busy) 50%,var(--card)),var(--card) 140px)}
.deliver h2{font-size:1.35em;margin:0 0 .3em}
.deliver .summary{font-size:1.05em;white-space:pre-line;overflow-wrap:anywhere}
.files-list{list-style:none;margin:10px 0;padding:0;display:grid;gap:8px}
.file-row{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;padding:10px 14px;
border:1px solid var(--line);border-radius:12px;background:var(--card)}
.file-row .fname{font-weight:700;overflow-wrap:anywhere}
.file-row .fmeta{color:var(--muted);font-size:.85em}
.file-row .acts{display:flex;gap:6px;margin-inline-start:auto;flex-wrap:wrap}
.file-row.missing{opacity:.75;border-style:dashed}
.howto{background:var(--soft);border-radius:12px;padding:10px 14px;margin:10px 0;
white-space:pre-line;overflow-wrap:anywhere}
.checks{list-style:none;margin:6px 0;padding:0}
.checks li{padding:2px 0;padding-inline-start:28px;position:relative}
.checks li::before{position:absolute;inset-inline-start:0;top:3px;width:20px;height:20px;
border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
font-size:.75em;font-weight:800;content:"•";background:var(--chip);color:var(--muted)}
.checks li.c-ok::before{content:"✓";background:var(--ok);color:var(--okfg)}
.checks li.c-bad::before{content:"✕";background:var(--bad);color:var(--badfg)}
.problems{margin:8px 0;padding-inline-start:20px}.problems li{margin:4px 0}
.retry{border-top:1px dashed var(--line);margin-top:14px;padding-top:12px}
.btn-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
textarea,input,button{font:inherit;color:inherit}
textarea{width:100%;background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:12px 14px;line-height:1.7}
textarea:focus,input:focus{outline:2px solid color-mix(in srgb,var(--accent) 45%,transparent);
outline-offset:1px}
input[type=text],input:not([type]){background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:5px 10px}
input[type=checkbox]{accent-color:var(--accent);width:1.05em;height:1.05em;vertical-align:middle}
button,.button{display:inline-flex;align-items:center;gap:6px;background:var(--accent);
color:#fff;border:1px solid transparent;border-radius:12px;padding:8px 18px;cursor:pointer;
font-weight:700;text-decoration:none;box-shadow:0 6px 16px -8px var(--accent);
transition:filter .15s,transform .15s}
button:hover,.button:hover{filter:brightness(1.08);transform:translateY(-1px)}
.button.secondary{background:var(--card);color:var(--fg);border-color:var(--line);box-shadow:none}
.button.small{padding:3px 12px;font-size:.88em;border-radius:10px}
details>summary{cursor:pointer;color:var(--muted)}
@media (prefers-color-scheme: dark){button,.button{color:#0d0f16}
.button.secondary{color:var(--fg)}}
@media (max-width:640px){body{font-size:15px}.wrap{padding:12px 10px 40px}
.hero{padding:16px 18px;border-radius:18px}.hero h1{font-size:1.35em}
.cards{grid-template-columns:1fr}
.steps{flex-direction:column}.steps li{flex:none;text-align:start;padding:0 0 8px 0;
min-width:0}
.steps li::before{display:none}.steps .name,.steps .when{display:inline;margin:0 8px}
.file-row .acts{margin-inline-start:0}}
"""
)


def visible(value: str) -> str:
    """``value`` with control/format characters as visible ``\\xNN``/``\\uNNNN``."""
    out = []
    for ch in value:
        if ch in "\n\t" or ch in _ALLOWED_FORMAT_CHARS:
            out.append(ch)
        elif ch.isprintable() and unicodedata.category(ch) != "Cf":
            out.append(ch)
        else:
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}")
    return "".join(out)


def normalize_newlines(value: str) -> str:
    """``\\r\\n`` and a lone ``\\r`` as ``\\n`` (D131): a browser form on Windows sends
    CRLF line breaks, which must show as line breaks, not as a visible ``\\x0d``."""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def text(value: object) -> str:
    """Safe HTML text for any recorded value (line breaks normalized, D131)."""
    return html.escape(visible(normalize_newlines(str(value))), quote=True)


def _time(value: datetime) -> str:
    return text(value.isoformat(timespec="seconds"))


def _project_href(project_id: str) -> str:
    return "/projects/" + quote(project_id, safe="")


@dataclass(frozen=True)
class ActionForms:
    """What the action-mode forms need (D123): the per-session CSRF token."""

    csrf_token: str


def page_csp(*, actions: bool = False, img_src: str | None = None) -> str:
    """The page's Content-Security-Policy; ``img_src`` (D128) is the one image source
    a file view may load (checked: ``http://`` and no space, quote or ``;``)."""
    csp = ACTION_CONTENT_SECURITY_POLICY if actions else CONTENT_SECURITY_POLICY
    if img_src is None:
        return csp
    if not img_src.startswith("http://") or any(c in img_src for c in " \t\r\n;,'\"<>"):
        raise ValueError(f"invalid image source: {img_src!r}")
    return csp.replace("img-src 'none'", f"img-src {img_src}")


def page(
    title: str,
    body: str,
    *,
    actions: bool = False,
    refresh: int | None = None,
    img_src: str | None = None,
    lang: Language = "en",
) -> str:
    """A complete HTML document around already-safe ``body``.

    ``lang`` (D140) sets ``<html lang>``; Persian pages are ``dir="rtl"``.

    In action mode the CSP allows forms to post to this server only, and the referrer
    policy is ``same-origin`` so the browser sends a real ``Origin`` with each post
    (with ``no-referrer`` it would send ``Origin: null``, which the server refuses).
    ``refresh`` (seconds) makes the browser reload the same page, with no script (D127).
    """
    csp = page_csp(actions=actions, img_src=img_src)
    referrer = "same-origin" if actions else "no-referrer"
    if lang not in ("fa", "en"):
        raise ValueError(f"unknown language: {lang!r}")
    direction = ' dir="rtl"' if lang == "fa" else ""
    reload = ""
    if refresh is not None:
        if isinstance(refresh, bool) or not isinstance(refresh, int) or not 1 <= refresh <= 3600:
            raise ValueError("refresh must be 1..3600 seconds")
        reload = f'<meta http-equiv="refresh" content="{refresh}">'
    return (
        f'<!doctype html>\n<html lang="{lang}"{direction}><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f'<meta name="referrer" content="{referrer}">{reload}'
        f"<title>{text(title)}</title><style>{_STYLE}</style></head>"
        f'<body>{_topbar(lang)}<main class="wrap">{body}</main></body></html>\n'
    )


def _topbar(lang: Language) -> str:
    """The app bar on every page: the name (home link) and the project list."""
    return (
        '<header class="topbar"><a class="brand" href="/">'
        '<span class="logo" aria-hidden="true">A</span>Automa</a>'
        f'<nav><a href="/">{text(t("nav.projects", lang))}</a></nav></header>'
    )


def _errors(errors: tuple[str, ...], lang: Language = "en") -> str:
    if not errors:
        return ""
    items = "".join(f'<li dir="auto">{text(e)}</li>' for e in errors)
    heading = text(t("errors.unreadable", lang))
    return f'<section class="card err"><h2>{heading}</h2><ul>{items}</ul></section>'


def _when(value: datetime, lang: Language, tz: tzinfo | None, now: datetime | None) -> str:
    """``now`` given: a relative local time (D140); else the recorded ISO time (plain
    text, escaped by the caller)."""
    if now is None:
        return value.isoformat(timespec="seconds")
    return relative_time(value, now, tz or resolve_timezone(DEFAULT_TIMEZONE), lang)


def render_overview(
    view: Overview,
    *,
    actions: bool = False,
    ideas: bool = False,
    rate: float | None = None,
    totals: tuple[float, float] | None = None,
    lang: Language = "en",
    tz: tzinfo | None = None,
    now: datetime | None = None,
) -> str:
    """The project list. ``ideas`` (action mode with a planner, D124) adds the
    "New idea" link; ``rate`` (toman per USD) shows costs first in toman (D133);
    ``totals`` is (spent today, this month); ``lang`` translates the page and
    ``now``/``tz`` show times relative and local (D140)."""
    mode = t("overview.mode.actions" if actions else "overview.mode.readonly", lang)
    title = t("overview.title", lang)
    new_idea = text(t("overview.new_idea", lang))
    hero = [f'<header class="hero"><h1>{text(title)}</h1>', f"<p>{text(mode)}</p>"]
    if actions and ideas and view.projects:
        hero.append(f'<p><a class="button" href="/new"><strong>{new_idea}</strong></a></p>')
    hero.append("</header>")
    parts = ["".join(hero)]
    if totals is not None:
        today, month = totals
        spent_today = text(t("overview.spent_today", lang))
        this_month = text(t("overview.this_month", lang))
        parts.append(
            f'<div class="stats"><div class="stat"><span class="label">{spent_today}</span> '
            f'<span class="value"><strong>{text(money(today, rate, lang=lang))}</strong>'
            f'</span></div><div class="stat"><span class="label">{this_month}</span> '
            f'<span class="value"><strong>{text(money(month, rate, lang=lang))}</strong>'
            "</span></div></div>"
        )
    if view.showing_tests:
        parts.append(f'<p class="muted"><a href="/">{text(t("overview.hide_tests", lang))}</a></p>')
    elif view.hidden_tests:
        hidden = t("overview.tests_hidden", lang, count=view.hidden_tests)
        parts.append(
            f'<p class="muted">{text(hidden)} · '
            f'<a href="/?tests=1">{text(t("overview.show_tests", lang))}</a></p>'
        )
    if not view.projects:
        parts.append(_welcome(actions=actions, ideas=ideas, lang=lang))
    else:
        cards = []
        for p in view.projects:
            waiting = (
                '<span class="badge warn">'
                f"{text(t('overview.pending', lang, count=p.pending_approvals))}</span>"
                if p.pending_approvals
                else text(t("overview.no_pending", lang))
            )
            cost = t("overview.cost", lang, amount=money(p.cost_usd, rate, lang=lang))
            created = t("overview.created", lang, when=_when(p.created_at, lang, tz, now))
            cards.append(
                f'<article class="card project ps-{_ps_class(p.status)}">'
                f'<h2 dir="auto"><a href="{text(_project_href(p.id))}">{text(p.title or p.id)}</a> '
                f"{_chip_html(project_chip(p.status, lang), f'ps-{_ps_class(p.status)}')}</h2>"
                f'<p class="muted"><code>{text(p.id)}</code></p>'
                f'<p class="goal" dir="auto">{text(p.goal_preview)}</p>'
                f"{_hint_html(project_chip(p.status, lang))}"
                f'<p class="chips"><span>{text(cost)}</span>'
                f"<span>{waiting}</span>"
                f"<span>{text(created)}</span></p></article>"
            )
        parts.append('<div class="cards">' + "".join(cards) + "</div>")
    parts.append(_errors(view.errors, lang))
    return page(title, "".join(parts), actions=actions, lang=lang)


WELCOME_EXAMPLES = tuple(MESSAGES[k]["en"] for k in WELCOME_EXAMPLE_KEYS)


def _welcome(*, actions: bool, ideas: bool, lang: Language = "en") -> str:
    """The empty home page: what Automa does and how to start (D138)."""
    if not (actions and ideas):
        return (
            f'<div class="card"><h2>{text(t("welcome.none.heading", lang))}</h2>'
            f"<p>{text(t('welcome.none.body', lang))}</p></div>"
        )
    examples = "".join(f'<li dir="auto">{text(t(k, lang))}</li>' for k in WELCOME_EXAMPLE_KEYS)
    new_idea = text(t("overview.new_idea", lang))
    return (
        f'<div class="card"><h2>{text(t("welcome.heading", lang))}</h2>'
        f"<p>{text(t('welcome.body', lang))}</p>"
        f'<p><a class="button" href="/new"><strong>{new_idea}</strong></a></p>'
        f'<p class="muted">{text(t("welcome.examples", lang))}</p><ul>{examples}</ul></div>'
    )


def _hidden(name: str, value: str) -> str:
    return f'<input type="hidden" name="{name}" value="{text(value)}">'


def _decision_forms(
    project_id: str, a: ApprovalView, forms: ActionForms, lang: Language = "en"
) -> str:
    def action(kind: str) -> str:
        path = f"/projects/{quote(project_id, safe='')}/approvals/{quote(a.id, safe='')}/{kind}"
        return text(path)

    common = _hidden("csrf", forms.csrf_token) + _hidden("fingerprint", a.fingerprint)
    label = text(t("approval.note", lang))
    note = f'<label>{label} <input name="note" maxlength="{MAX_NOTE_CHARS}" dir="auto"></label> '
    confirm = text(t("approval.confirm", lang, tool=a.tool))
    return (
        f'<form method="post" action="{action("approve")}">{common}'
        '<p><label><input type="checkbox" name="confirm" value="yes" required> '
        f"{confirm}</label></p>"
        f'<p>{note}<button type="submit">{text(t("approval.approve", lang))}</button></p></form>'
        f'<form method="post" action="{action("reject")}">{common}'
        f'<p>{note}<button type="submit">{text(t("approval.reject", lang))}</button></p></form>'
    )


When = Callable[[datetime], str]


def _when_fn(lang: Language, tz: tzinfo | None, now: datetime | None) -> When:
    """Plain time text (escaped by the caller): a relative local time when ``now`` is
    given (D144), else the recorded ISO time."""
    return lambda value: _when(value, lang, tz, now)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _approval(
    project_id: str,
    a: ApprovalView,
    forms: ActionForms | None,
    lang: Language = "en",
    when: When = _iso,
) -> str:
    args = json.dumps(a.arguments, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    lines = "".join(f'<li dir="auto">{text(line)}</li>' for line in a.request_lines)
    warns = "".join(f'<li class="warn" dir="auto">{text(w)}</li>' for w in a.warnings)
    target = f"<ul>{lines}{warns}</ul>" if lines or warns else ""
    if forms is None:
        decide = f"<p>{text(t('approval.readonly', lang))}</p>"
    else:
        decide = _decision_forms(project_id, a, forms, lang)
    meta = text(t("approval.task", lang, task=a.task_id or "-", when=when(a.created_at)))
    return (
        f'<article class="card"><h3>{text(a.id)} — {text(a.tool)} '
        f'<span class="badge">{text(a.risk)}</span></h3>'
        f"<p>{meta}</p>"
        f'<p>{text(t("approval.reason", lang))} <span dir="auto">{text(a.reason)}</span></p>'
        f'{target}<pre dir="ltr">{text(args)}</pre>{decide}'
        "</article>"
    )


def render_project(
    view: ProjectView,
    forms: ActionForms | None = None,
    notice: str | None = None,
    *,
    cost_range: str | None = None,
    questions_waiting: bool = False,
    run: RunPanel | None = None,
    progress: ProgressView | None = None,
    rate: float | None = None,
    task_costs: dict[str, float] | None = None,
    lang: Language = "en",
    tz: tzinfo | None = None,
    now: datetime | None = None,
    delivery: Delivery | None = None,
    retry: bool = False,
) -> str:
    """One project page. ``forms`` (action mode only) adds approve/reject forms;
    ``notice`` is a one-line result of the last decision, shown at the top;
    ``cost_range`` is the estimate line of a planned project and ``questions_waiting``
    links to the answer form of a new idea (D124); ``run`` (action mode with
    background runs, D125) adds the start button and the last background run;
    ``progress`` (D127) adds tasks passed, cost so far against the range, the latest
    events and, while running, a self-refresh of the page; ``rate`` (toman per USD)
    shows every amount first in rounded toman with dollars in parentheses (D133);
    ``lang`` translates the whole page and, with ``now``, times are relative local
    times in ``tz`` (D144)."""
    when = _when_fn(lang, tz, now)
    running = run is not None and run.running
    stop = _stop_line(view, progress, running, lang, tz, now)
    journey = build_journey(
        view,
        lang=lang,
        questions_waiting=questions_waiting,
        running=running,
        budget_stop=run is not None and not running and run.exit_code == 3,
        stop_now=(stop.text, stop.tone) if stop is not None else None,
    )
    resume = (
        stop is not None
        and stop.resume
        and journey.now == stop.text
        and not view.pending
        and forms is not None
        and run is not None
        and run.can_start
    )
    back = text(jt("nav.back", lang))
    parts = [f'<p><a href="/">{back}</a></p>']
    if notice:
        parts.append(f'<p class="badge" dir="auto">{text(notice)}</p>')
    parts.append(_hero_and_journey(view, journey, lang))
    if delivery is not None:
        retry_form = forms if retry and not running else None
        parts.append(_delivery_card(delivery, retry_form, lang))
    if resume and forms is not None and run is not None:
        parts.append(_start_form(view.id, run, forms, lang, stop_text("stop.continue", lang), ""))
    brake = progress.stop if progress is not None else None
    if (
        stop is not None
        and brake is not None
        and brake.kind == "task_brake"
        and brake.task is not None
        and brake.awaiting
        and journey.now == stop.text
        and not view.pending
        and forms is not None
        and run is not None
        and run.can_start
    ):
        parts.append(_brake_card(view, brake, run, forms, rate, lang))
    times = t("project.times", lang, created=when(view.created_at), updated=when(view.updated_at))
    parts += [
        '<section class="card">',
        f'<p class="meta"><strong>{text(t("project.goal", lang))}</strong></p>',
        f'<p class="goal" dir="auto">{text(view.goal)}</p>',
        f'<p class="meta">{text(t("project.status", lang))} '
        f"{_chip_html(project_chip(view.status, lang), f'ps-{_ps_class(view.status)}')} · "
        f"{text(times)}</p>",
        _cost_card(view, progress, task_costs or {}, rate, lang),
        f'<p><a href="{text(_project_href(view.id) + "/files")}">'
        f"{text(t('project.files', lang))}</a></p>",
    ]
    if questions_waiting and forms is not None:
        href = text(_project_href(view.id) + "/clarify")
        parts.append(
            f'<p class="warn">{text(t("project.questions.before", lang))} '
            f'<a href="{href}">{text(t("project.questions.link", lang))}</a> '
            f"{text(t('project.questions.after', lang))}</p>"
        )
    if cost_range is not None:
        parts.append(
            f"<p><strong>{text(t('project.cost_range', lang))}</strong> "
            f'<span dir="auto">{text(cost_range)}</span></p>'
        )
    parts.append("</section>")
    if progress is not None:
        parts.append(_progress(progress, rate, lang, when))
    if run is not None and forms is not None:
        parts.append(_run_panel(view.id, run, forms, lang, when))
    count = len(view.pending)
    if view.pending:
        parts.append(f"<h2>{text(t('approvals.heading', lang, count=count))}</h2>")
        parts.append("".join(_approval(view.id, a, forms, lang, when) for a in view.pending))
    if view.plan_version is None:
        parts.append(f"<h2>{text(t('plan.heading', lang))}</h2><p>{text(t('plan.none', lang))}</p>")
    else:
        checks = {d.id: d for d in delivery.tasks} if delivery is not None else {}
        cards = "".join(_task_card(task, lang, checks.get(task.id)) for task in view.tasks)
        heading = t("plan.version", lang, version=view.plan_version, count=len(view.tasks))
        parts.append(f'<h2>{text(heading)}</h2><div class="cards">{cards}</div>')
    errors = view.errors + (progress.errors if progress is not None else ())
    parts.append(_errors(errors, lang))
    refresh = REFRESH_SECONDS if progress is not None and progress.live else None
    title = f"Automa — {project_title(view.goal) or view.id}"
    return page(title, "".join(parts), actions=forms is not None, refresh=refresh, lang=lang)


_STEP_MARK = {"done": "✓", "fail": "✕"}


def _hero_and_journey(view: ProjectView, j: Journey, lang: Language) -> str:
    """The colourful header (title, status) and the journey (D141): the "now"
    sentence, a progress bar and one step per stage. Classes are fixed values; all
    text is escaped."""
    status = _ps_class(view.status)
    title = project_title(view.goal) or view.id
    steps = []
    for i, st in enumerate(j.steps, start=1):
        mark = _STEP_MARK.get(st.state, localize_digits(str(i), lang))
        steps.append(
            f'<li class="s-{st.state}"><span class="dot">{text(mark)}</span>'
            f'<span class="name" dir="auto">{text(st.name)}</span></li>'
        )
    percent = max(0, min(100, int(j.percent)))
    return (
        '<header class="hero">'
        f'<h1 dir="auto">{text(title)} '
        f"{_chip_html(project_chip(view.status, lang), f'ps-{status}')}</h1>"
        f'<p class="muted"><code>{text(view.id)}</code></p></header>'
        '<section class="card journey">'
        f'<p class="now {j.tone}" dir="auto">{text(j.now)}</p>'
        f'<div class="bar" role="progressbar" aria-valuenow="{percent}" aria-valuemin="0" '
        f'aria-valuemax="100"><div class="fill" style="width:{percent}%"></div></div>'
        f'<p class="meta">{text(jt("progress.percent", lang, percent=percent))}</p>'
        f'<ol class="steps">{"".join(steps)}</ol></section>'
    )


def _chip_html(chip: Chip, cls: str) -> str:
    """A status chip: icon (hidden from screen readers), label and, as a tooltip, the
    plain sentence. ``cls`` and the tone are fixed values; the label is escaped."""
    title = f' title="{text(chip.hint)}"' if chip.hint else ""
    return (
        f'<span class="badge {cls} tone-{chip.tone}"{title}>'
        f'<span class="ico" aria-hidden="true">{text(chip.icon)}</span> '
        f"{text(chip.label)}</span>"
    )


def _hint_html(chip: Chip) -> str:
    """The chip's plain sentence as a muted line ("" for an unknown status)."""
    return f'<p class="hint">{text(chip.hint)}</p>' if chip.hint else ""


_PROJECT_STATUSES = frozenset({"new", "planning", "running", "paused", "done", "failed"})


def _ps_class(status: str) -> str:
    """A fixed CSS class for a project status (never recorded text in an attribute)."""
    return status if status in _PROJECT_STATUSES else "none"


_STATE_CLASSES = frozenset({"pending", "running", "verifying", "passed", "failed", "blocked"})


def _state_class(state: str | None) -> str:
    """A fixed CSS class for a task state (never recorded text in an attribute)."""
    return state if state in _STATE_CLASSES else "none"


def _criteria_html(d: TaskDelivery | None, lang: Language) -> str:
    """The task's acceptance checks with what the last verdict said (D157)."""
    if d is None or not d.criteria:
        return ""
    items = "".join(
        f'<li class="{_CHECK_CLASS[c.passed]}" dir="auto">{text(c.description)}</li>'
        for c in d.criteria
    )
    issues = ""
    if d.issues and d.status is not TaskStatus.PASSED:
        notes = "".join(f'<li dir="auto">{text(i)}</li>' for i in d.issues)
        issues = (
            f'<p class="meta"><strong>{text(t("task.issues", lang))}</strong></p>'
            f'<ul class="problems">{notes}</ul>'
        )
    return (
        f"<details{' open' if d.status is not TaskStatus.PASSED else ''}>"
        f"<summary>{text(t('task.checks', lang))}</summary>"
        f'<ul class="checks">{items}</ul>{issues}</details>'
    )


_CHECK_CLASS = {True: "c-ok", False: "c-bad", None: "c-wait"}


def _task_card(task: TaskRow, lang: Language = "en", checks: TaskDelivery | None = None) -> str:
    """One task of the plan as a card (D129): the title in its own direction, the
    state as a fixed class and the rest as labelled chips (translated, D144)."""
    state = _state_class(task.state)
    chip = task_chip(task.state, lang)
    order = localize_digits(str(task.order), lang)
    after = ", ".join(task.depends_on) or "-"
    # The latest attempt's status uses the task-state words (unknown: as recorded).
    last = task_state_label(task.last_result, lang) if task.last_result else "-"
    return (
        f'<article class="card task t-{state}">'
        f'<h3><span class="badge">#{text(order)}</span> <code>{text(task.id)}</code> '
        f"{_chip_html(chip, f'st-{state}')}</h3>"
        f'<p class="goal" dir="auto">{text(task.title)}</p>'
        + _hint_html(chip)
        + '<p class="chips">'
        f"<span>{text(t('task.role', lang, role=task.role))}</span>"
        f'<span dir="auto">{text(t("task.after", lang, after=after))}</span>'
        f"<span>{text(t('task.attempts', lang, count=task.attempts))}</span>"
        f"<span>{text(t('task.last', lang))} "
        f'<span dir="auto">{text(last)}</span></span>'
        f"<span>{text(t('task.model', lang, model=task.model or '-'))}</span></p>"
        + _missing_program_html(task.missing_program, lang)
        + _criteria_html(checks, lang)
        + "</article>"
    )


def _delivery_card(d: Delivery, forms: ActionForms | None, lang: Language) -> str:
    """The product card (D157): what the user gets, its files with open/view/download,
    how to use it, what was checked and, for an incomplete product, what is missing
    and the "try again" form (D156, ``forms`` only when a retry is possible)."""
    state = d.state.value if d.state.value in _DELIVERY_STATES else "promise"
    title_key = {
        "ready": "deliver.ready.title",
        "partial": "deliver.partial.title",
    }.get(state, "deliver.promise.title")
    cls = state if state in ("ready", "partial") else "promise"
    parts = [
        f'<section class="card deliver {cls}">',
        f"<h2>{text(t(title_key, lang))}</h2>",
        f'<p class="summary" dir="auto">{text(d.summary)}</p>',
    ]
    if d.derived:
        parts.append(f'<p class="hint">{text(t("deliver.derived", lang))}</p>')
    done = {td.id for td in d.tasks if td.status is TaskStatus.PASSED}
    started = d.state in (DeliveryState.READY, DeliveryState.PARTIAL)
    if d.main_files:
        rows = [
            _file_row(d, f, f.task_id in done or f.task_id is None, started, lang)
            for f in d.main_files
        ]
        parts.append(
            f'<p class="meta"><strong>{text(t("deliver.files", lang))}</strong></p>'
            f'<ul class="files-list">{"".join(rows)}</ul>'
        )
    if d.how_to_use.strip():
        parts.append(
            f'<p class="meta"><strong>{text(t("deliver.how", lang))}</strong></p>'
            f'<div class="howto" dir="auto">{text(d.how_to_use.strip())}</div>'
        )
    if started:
        checks = sum(c.passed is True for td in d.tasks for c in td.criteria)
        line = t("deliver.checked", lang, passed=d.passed_tasks, total=len(d.tasks), checks=checks)
        parts.append(f'<p class="meta">{text(line)}</p>')
    problems = d.problem_tasks if d.state is DeliveryState.PARTIAL else ()
    if problems:
        items = []
        for td in problems:
            why = "; ".join(td.issues) if td.status is TaskStatus.FAILED else ""
            why = why or t("deliver.blocked", lang)
            items.append(
                f"<li><strong><bdi>{text(td.title)}</bdi></strong>: <bdi>{text(why)}</bdi></li>"
            )
        parts.append(
            f'<p class="meta"><strong>{text(t("deliver.problems", lang))}</strong></p>'
            f'<ul class="problems">{"".join(items)}</ul>'
        )
    if started and any(f.exists for f in d.main_files) or d.state is DeliveryState.READY:
        parts.append(
            f'<div class="btn-row"><a class="button" href="{text(bundle_href(d.project_id))}">'
            f"{text(t('deliver.zip', lang))}</a>"
            f'<span class="hint">{text(t("deliver.zip.help", lang))}</span></div>'
        )
    if d.state is DeliveryState.PARTIAL and d.project_status.value == "failed":
        parts.append(_retry_form(d.project_id, forms, lang))
    parts.append("</section>")
    return "".join(parts)


_DELIVERY_STATES = frozenset({"ready", "partial", "in_progress", "not_started"})


def _file_row(d: Delivery, f: DeliveryFile, accepted: bool, started: bool, lang: Language) -> str:
    """One main file: name, size or why it is missing, and open/view/download links."""
    pid = d.project_id
    name = text(f.path)
    if not f.exists:
        key = "deliver.file.missing" if started else "deliver.file.planned"
        return (
            f'<li class="file-row missing"><span class="fname" dir="auto">📄 {name}</span>'
            f'<span class="fmeta">{text(t(key, lang))}</span></li>'
        )
    meta = _size_text(f.size, lang)
    if started and not accepted:
        meta += " · " + t("deliver.file.unchecked", lang)
    acts = []
    if is_previewable(f.path):
        acts.append(
            f'<a class="button small" href="{text(preview_href(pid, f.path))}" target="_blank" '
            f'rel="noopener noreferrer">{text(t("deliver.open", lang))}</a>'
        )
    view = text(files_href(pid, FilesRoute.VIEW, f.path))
    down = text(files_href(pid, FilesRoute.DOWNLOAD, f.path))
    acts.append(
        f'<a class="button small secondary" href="{view}">{text(t("deliver.view", lang))}</a>'
    )
    acts.append(
        f'<a class="button small secondary" href="{down}">{text(t("deliver.download", lang))}</a>'
    )
    icon = "🌐" if f.is_html else "📄"
    return (
        f'<li class="file-row"><span class="fname" dir="auto">{icon} {name}</span>'
        f'<span class="fmeta">{text(meta)}</span><span class="acts">{"".join(acts)}</span></li>'
    )


def _retry_form(project_id: str, forms: ActionForms | None, lang: Language) -> str:
    """ "Try again" for a failed project (D156); the terminal command when read-only."""
    body = text(t("deliver.retry.body", lang, attempts=RETRY_ATTEMPTS))
    if forms is None:
        cli = _filled(
            MESSAGES["deliver.retry.cli"][lang],
            {"retry": f"automa retry {project_id}", "run": f"automa run {project_id}"},
        )
        return f'<div class="retry"><p class="hint">{body}</p><p class="hint">{cli}</p></div>'
    action = text(_project_href(project_id) + "/retry")
    return (
        f'<form class="retry" method="post" action="{action}">'
        f"{_hidden('csrf', forms.csrf_token)}<p>{body}</p>"
        '<p><label><input type="checkbox" name="confirm" value="yes" required> '
        f"{text(t('deliver.retry.confirm', lang))}</label></p>"
        f'<p><button type="submit">{text(t("deliver.retry.button", lang))}</button></p></form>'
    )


def _missing_program_html(program: str | None, lang: Language) -> str:
    """Plain words and the install command when a program is not installed (D152)."""
    if program is None:
        return ""
    return (
        f'<p class="hint missing-program">{text(t("task.missing_program", lang))} '
        f'<code dir="ltr">{text(install_command(program))}</code> '
        f"{text(t('task.missing_program.then', lang))}</p>"
    )


def _cost_card(
    view: ProjectView,
    progress: ProgressView | None,
    task_costs: dict[str, float],
    rate: float | None,
    lang: Language = "en",
) -> str:
    """The project's cost at a glance (D139): the total in large type (toman first
    when a rate is set), one plain sentence against the estimated range with a meter,
    planning vs tasks, and what each task cost."""
    parts = [
        '<div class="costcard">',
        f'<p class="big"><strong>{text(money(view.cost_usd, rate, lang=lang))}</strong></p>',
    ]
    c = progress.cost if progress is not None else None
    if c is not None:
        live = progress is not None and progress.live
        span = money_range(c.low_usd, c.high_usd, rate, lang=lang)
        key = "cost.live" if live and c.verdict != "above" else f"cost.{c.verdict}"
        line = jt(key, lang, range=span) if key in JOURNEY_MESSAGES else span
        cls = ' class="warn"' if c.verdict == "above" else ""
        parts.append(f"<p{cls}>{text(line)} {_range_meter(c)}</p>")
    elif view.cost_usd <= 0:
        parts.append(f'<p class="muted">{text(jt("cost.nothing", lang))}</p>')
    if view.cost_usd > 0:
        planning = min(max(view.planning_usd, 0.0), view.cost_usd)
        tasks = max(view.cost_usd - planning, 0.0)
        plan_s = jt("cost.planning", lang, amount=money(planning, rate, lang=lang))
        task_s = jt("cost.tasks", lang, amount=money(tasks, rate, lang=lang))
        parts.append(f'<p class="chips"><span>{text(plan_s)}</span><span>{text(task_s)}</span></p>')
    rows = [
        f'<tr><td dir="auto">{text(t.title or t.id)}</td>'
        f"<td>{text(money(task_costs[t.id], rate, lang=lang))}</td></tr>"
        for t in view.tasks
        if task_costs.get(t.id, 0) > 0
    ]
    if rows:
        parts.append(
            f"<details><summary>{text(jt('cost.each', lang))}</summary>"
            f"<table><tbody>{''.join(rows)}</tbody></table></details>"
        )
    parts.append("</div>")
    return "".join(parts)


def _progress(
    p: ProgressView, rate: float | None = None, lang: Language = "en", when: When = _iso
) -> str:
    """Tasks passed, the latest run's cost against its range and the latest events (D127)."""
    passed = localize_digits(f"{p.passed} / {p.total}", lang)
    parts = [
        f'<section class="card"><h2>{text(t("progress.heading", lang))}</h2>',
        f"<p>{text(t('progress.passed', lang))} <strong>{text(passed)}</strong>",
    ]
    if p.total:
        parts.append(f' <meter min="0" max="{p.total}" value="{p.passed}"></meter>')
    parts.append("</p>")
    if p.unusual is not None:
        parts.append(_unusual_line(p.unusual, rate, lang))
    if p.live:
        live = t("progress.live", lang, seconds=REFRESH_SECONDS)
        parts.append(f'<p class="badge">{text(live)}</p>')
    if p.events:
        rows = "".join(
            f"<tr><td>{text(when(e.ts))}</td><td>{text(e.type)}</td>"
            f'<td dir="auto">{text(e.summary or "-")}</td></tr>'
            for e in p.events
        )
        events = t("tech.events", lang, count=len(p.events))
        parts.append(
            f"<details><summary>{text(t('tech.summary', lang))}</summary>"
            f"<h3>{text(events)}</h3>"
            f'<div class="scroll"><table><thead><tr><th>{text(t("tech.time", lang))}</th>'
            f"<th>{text(t('tech.event', lang))}</th><th>{text(t('tech.details', lang))}</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div></details>"
        )
    parts.append("</section>")
    return "".join(parts)


def _unusual_line(u: UnusualSpending, rate: float | None = None, lang: Language = "en") -> str:
    """At most one line about unusual spending (D132): how many and the largest."""
    # A flagged sample is a model call or a judged attempt: both are "steps" here.
    what = t("unusual.one", lang) if u.count == 1 else t("unusual.many", lang, count=u.count)
    times = (
        t("unusual.times", lang, ratio=localize_number(f"{u.ratio:.1f}", lang))
        if u.ratio is not None
        else ""
    )
    line = t(
        "unusual.line", lang, what=what, amount=money(u.largest_usd, rate, lang=lang), times=times
    )
    return f'<p class="warn">{text(line)}</p>'


def _range_meter(c: CostProgress) -> str:
    """A meter of the cost so far over the estimated range (D131): its end is the top
    of the range (or the cost, when above), never the theoretical worst case."""
    top = max(c.high_usd, c.spent_usd)
    # Lower is better and the range's top splits the meter (D141): inside the range
    # the browser draws it green, above it red (it used to stay green when above).
    return (
        f'<meter min="0" low="{c.high_usd:.6f}" high="{c.high_usd:.6f}" optimum="0" '
        f'max="{top:.6f}" value="{c.spent_usd:.6f}"></meter>'
    )


def _stop_line(
    view: ProjectView,
    progress: ProgressView | None,
    running: bool,
    lang: Language,
    tz: tzinfo | None,
    now: datetime | None,
) -> StopLine | None:
    """The plain reason of the latest pause (D151), only for a paused project that
    nothing is running."""
    if progress is None or progress.stop is None or running or view.status != "paused":
        return None
    return describe_stop(
        progress.stop,
        lang=lang,
        tz=tz or resolve_timezone(DEFAULT_TIMEZONE),
        now=now or datetime.now(UTC),
    )


def _filled(template: str, values: dict[str, str]) -> str:
    """HTML of ``template`` with each field's (escaped) value in ``<bdi>``."""
    out: list[str] = []
    for literal, name, _spec, _conv in Formatter().parse(template):
        out.append(text(literal))
        if name is not None:
            out.append(f"<bdi>{text(values[name])}</bdi>")
    return "".join(out)


def _brake_card(
    view: ProjectView,
    brake: StopInfo,
    run: RunPanel,
    forms: ActionForms,
    rate: float | None,
    lang: Language,
) -> str:
    """The task brake's question "continue?" (D154): "Continue" and "Stop" post to
    ``/projects/<pid>/brake`` (same CSRF and same-origin checks). "Continue" carries
    the start form's cost confirmation box; after "Stop" only "Continue" is left."""
    task = brake.task or ""
    title = next((r.title for r in view.tasks if r.id == task and r.title.strip()), task)
    if brake.spent_nusd is not None and brake.limit_nusd is not None:
        values = {
            "task": title,
            "spent": nusd_money(brake.spent_nusd, rate, lang=lang),
            "limit": nusd_money(brake.limit_nusd, rate, lang=lang),
        }
        question = _filled(STOP_MESSAGES["stop.brake.ask"][lang], values)
    else:
        question = _filled(STOP_MESSAGES["stop.brake.ask_plain"][lang], {"task": title})
    action = text(_project_href(view.id) + "/brake")
    hidden = _hidden("csrf", forms.csrf_token) + _hidden("task", task)
    confirm = (
        '<p><label><input type="checkbox" name="confirm" value="yes" required> '
        f"{text(t('run.confirm', lang))}</label></p>"
        if run.needs_confirm
        else ""
    )
    go = (
        f'<form method="post" action="{action}">{hidden}'
        f"{_hidden('decision', 'continue')}{confirm}"
        f'<p><button type="submit">{text(stop_text("stop.brake.continue", lang))}</button> '
        f'<span class="muted">{text(stop_text("stop.brake.help", lang))}</span></p></form>'
    )
    halt = (
        ""
        if brake.declined
        else (
            f'<form method="post" action="{action}">{hidden}'
            f"{_hidden('decision', 'stop')}"
            f'<p><button type="submit">{text(stop_text("stop.brake.stop", lang))}</button></p>'
            "</form>"
        )
    )
    ask = "" if brake.declined else f'<p class="warn" dir="auto">{question}</p>'
    return f'<section class="card brake">{ask}{go}{halt}</section>'


def _start_form(
    project_id: str, run: RunPanel, forms: ActionForms, lang: Language, label: str, help_: str
) -> str:
    """The start form (D125): the run panel's "Start" and the "Continue" of a pause
    (D151) post the same fields to the same checked endpoint."""
    action = text(_project_href(project_id) + "/start")
    confirm = (
        '<p><label><input type="checkbox" name="confirm" value="yes" required> '
        f"{text(t('run.confirm', lang))}</label></p>"
        if run.needs_confirm
        else ""
    )
    after = f" {text(help_)}" if help_ else ""
    return (
        f'<form method="post" action="{action}">{_hidden("csrf", forms.csrf_token)}'
        f'{confirm}<p><button type="submit">{text(label)}</button>{after}</p></form>'
    )


@dataclass(frozen=True)
class RunPanel:
    """The background-run part of a project page (D125)."""

    can_start: bool
    needs_confirm: bool
    can_stop: bool = False
    state: str | None = None
    running: bool = False
    started_at: datetime | None = None
    log_name: str | None = None
    tail: tuple[str, ...] = ()
    exit_code: int | None = None
    stopping: bool = False


def _run_state(run: RunPanel, lang: Language) -> str:
    """The background run's state line: as recorded in English, translated from its
    structured fields otherwise (D144)."""
    if lang == "en" or run.state is None:
        return run.state or ""
    if run.running:
        return t("run.state.stopping" if run.stopping else "run.state.running", lang)
    if run.exit_code is None:
        return run.state
    meaning = exit_label(run.exit_code, lang)
    return t("run.state.ended", lang, code=run.exit_code, meaning=meaning)


def _run_panel(
    project_id: str,
    run: RunPanel,
    forms: ActionForms,
    lang: Language = "en",
    when: When = _iso,
) -> str:
    parts = [f'<section class="card"><h2>{text(t("run.heading", lang))}</h2>']
    if run.state is not None:
        started = (
            " " + text(t("run.started", lang, when=when(run.started_at)))
            if run.started_at is not None
            else ""
        )
        cls = "badge warn" if run.running else "badge"
        parts.append(
            f"<p>{text(t('run.label', lang))}{started}: "
            f'<span class="{cls}" dir="auto">{text(_run_state(run, lang))}</span></p>'
        )
        if run.log_name or run.tail:
            parts.append(_technical_log(run, lang))
        if run.running:
            parts.append(f"<p>{text(t('run.refreshing', lang))}</p>")
        if run.can_stop:
            stop = text(_project_href(project_id) + "/stop")
            parts.append(
                f'<form method="post" action="{stop}">{_hidden("csrf", forms.csrf_token)}'
                f'<p><button type="submit">{text(t("run.stop.button", lang))}</button> '
                f"{text(t('run.stop.help', lang))}</p></form>"
            )
    if run.can_start:
        parts.append(
            _start_form(
                project_id, run, forms, lang, t("run.start.button", lang), t("run.start.help", lang)
            )
        )
    elif run.state is None:
        parts.append(f"<p>{text(t('run.not_startable', lang))}</p>")
    parts.append("</section>")
    return "".join(parts)


def _technical_log(run: RunPanel, lang: Language = "en") -> str:
    """The raw run log, folded away (D131): it is the terminal output of the run (with
    its command hints), useful only when something went wrong."""
    parts = [f'<details class="tech"><summary>{text(t("run.log.summary", lang))}</summary>']
    if run.log_name:
        parts.append(
            f"<p>{text(t('run.log.file', lang))} "
            f'<code dir="ltr">web-runs/{text(run.log_name)}</code></p>'
        )
    if run.tail:
        parts.append('<pre dir="auto">' + text("\n".join(run.tail)) + "</pre>")
    parts.append("</details>")
    return "".join(parts)


def render_new_idea(forms: ActionForms, lang: Language = "en") -> str:
    """The "New idea" form (D124): the goal and whether to ask questions first."""
    body = (
        f'<p><a href="/">{text(jt("nav.back", lang))}</a></p>'
        f"<h1>{text(t('idea.heading', lang))}</h1>"
        f"<p>{text(t('idea.intro', lang))}</p>"
        f'<form class="card" method="post" action="/new">{_hidden("csrf", forms.csrf_token)}'
        f'<p><textarea name="goal" rows="8" cols="80" maxlength="{MAX_GOAL_CHARS}" '
        f'dir="auto" required placeholder="{text(t("idea.placeholder", lang))}"></textarea></p>'
        f'<p><label><input type="checkbox" name="ask" value="{ASK_VALUE}" checked> '
        f"{text(t('idea.ask', lang))}</label></p>"
        f'<p><button type="submit">{text(t("idea.submit", lang))}</button></p></form>'
    )
    return page(t("idea.title", lang), body, actions=True, lang=lang)


def _project_id_html(project_id: str, lang: Language) -> str:
    """A project id inside Persian text is recorded Latin data: kept left-to-right."""
    return text(project_id) if lang == "en" else f'<code dir="ltr">{text(project_id)}</code>'


def render_thinking(project_id: str, phase: str, elapsed_s: int, lang: Language = "en") -> str:
    """The self-refreshing page shown while the Executive works (D137)."""
    what = "thinking.questions" if phase == "questions" else "thinking.plan"
    elapsed = t("thinking.elapsed", lang, seconds=max(0, elapsed_s))
    body = (
        f'<p><a href="/">{text(jt("nav.back", lang))}</a></p>'
        f'<div class="card"><h1>{text(t("thinking.heading", lang))}</h1>'
        f"<p>{text(t(what, lang))}</p>"
        f'<p class="muted">{text(elapsed)} {_project_id_html(project_id, lang)}</p></div>'
    )
    return page(t("thinking.title", lang), body, actions=True, refresh=3, lang=lang)


def render_clarify(
    project_id: str,
    goal: str,
    questions: list[str],
    forms: ActionForms,
    lang: Language = "en",
) -> str:
    """One answer field per stored question (questions are model output: escaped)."""
    fields = "".join(
        f'<p dir="auto"><label>{text(localize_digits(str(i + 1), lang))}. {text(q)}<br>'
        f'<textarea name="a{i}" rows="3" cols="80" maxlength="{MAX_ANSWER_CHARS}" '
        'dir="auto"></textarea></label></p>'
        for i, q in enumerate(questions)
    )
    action = text(_project_href(project_id) + "/clarify")
    pid = _project_id_html(project_id, lang)
    back = f"← {pid}" if lang == "en" else f"→ {pid}"
    body = (
        f'<p><a href="{text(_project_href(project_id))}">{back}</a></p>'
        f"<h1>{text(t('clarify.heading', lang))} {pid}</h1>"
        f'<p class="meta"><strong>{text(t("clarify.goal", lang))}</strong></p>'
        f'<p class="goal" dir="auto">{text(goal)}</p>'
        f"<p>{text(t('clarify.intro', lang))}</p>"
        f'<form class="card" method="post" action="{action}">{_hidden("csrf", forms.csrf_token)}'
        f'{fields}<p><button type="submit">{text(t("clarify.submit", lang))}</button></p></form>'
    )
    return page(t("clarify.title", lang, project=project_id), body, actions=True, lang=lang)


def _size_text(size: int, lang: Language = "en") -> str:
    if size < 1024:
        return t("size.b", lang, size=size)
    if size < 1024 * 1024:
        return t("size.kib", lang, size=localize_number(f"{size / 1024:.1f}", lang))
    return t("size.mib", lang, size=localize_number(f"{size / (1024 * 1024):.1f}", lang))


def _crumbs(project_id: str, path: str, lang: Language = "en") -> str:
    """``files / a / b`` with a link per directory."""
    root = text(t("files.root", lang))
    links = [f'<a href="{text(files_href(project_id, FilesRoute.LIST))}">{root}</a>']
    parts = path.split("/") if path else []
    for i, part in enumerate(parts):
        sub = "/".join(parts[: i + 1])
        href = text(files_href(project_id, FilesRoute.LIST, sub))
        links.append(f'<a href="{href}" dir="auto">{text(part)}</a>')
    return " / ".join(links)


def render_files(listing: FileListing, *, actions: bool = False, lang: Language = "en") -> str:
    """One directory of a project's ``files/`` (D128). Symlinks and special files
    are listed but never linked. Interface text in ``lang`` (D146); file names as
    recorded."""
    pid = listing.project_id
    title = t("files.title", lang, project=pid)
    parts = [
        f'<p><a href="{text(_project_href(pid))}">← {text(pid)}</a></p>',
        f"<h1>{text(t('files.heading', lang, project=pid))}</h1>",
        f"<p>{_crumbs(pid, listing.dir, lang)}</p>",
    ]
    if listing.missing:
        parts.append(f"<p>{text(t('files.none', lang))}</p>")
        return page(title, "".join(parts), actions=actions, lang=lang)
    if listing.parent is not None:
        up = text(files_href(pid, FilesRoute.LIST, listing.parent))
        parts.append(f'<p><a href="{up}">{text(t("files.up", lang))}</a></p>')
    if not listing.entries:
        parts.append(f"<p>{text(t('files.empty', lang))}</p>")
    else:
        download, open_ = text(t("files.download", lang)), text(t("files.open", lang))
        rows = []
        for e in listing.entries:
            name = text(e.name)
            if e.kind == "dir":
                href = text(files_href(pid, FilesRoute.LIST, e.path))
                cell, size, extra = f'<a href="{href}" dir="auto">{name}/</a>', "-", ""
            elif e.kind == "file":
                view = text(files_href(pid, FilesRoute.VIEW, e.path))
                down = text(files_href(pid, FilesRoute.DOWNLOAD, e.path))
                cell = f'<a href="{view}" dir="auto">{name}</a>'
                size, extra = _size_text(e.size, lang), f'<a href="{down}">{download}</a>'
                if is_previewable(e.path):
                    show = text(preview_href(pid, e.path))
                    extra += (
                        f' · <a href="{show}" target="_blank" rel="noopener noreferrer">{open_}</a>'
                    )
            else:
                key = "files.symlink" if e.kind == "symlink" else "files.special"
                label = text(t(key, lang))
                cell = f'<span dir="auto">{name}</span> <span class="warn">({label})</span>'
                size, extra = "-", ""
            rows.append(f"<tr><td>{cell}</td><td>{text(size)}</td><td>{extra}</td></tr>")
        name_h, size_h = text(t("files.col.name", lang)), text(t("files.col.size", lang))
        parts.append(
            f'<div class="card scroll"><table><thead><tr><th>{name_h}</th><th>{size_h}</th>'
            f"<th></th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )
        if listing.total > len(listing.entries):
            shown = t("files.showing", lang, shown=len(listing.entries), total=listing.total)
            parts.append(f"<p>{text(shown)}</p>")
    return page(title, "".join(parts), actions=actions, lang=lang)


def _file_note(content: FileContent, lang: Language) -> str:
    """Why a file is not shown, in ``lang`` (D146)."""
    code = content.note_code
    if code == "too_large":
        return t("file.note.too_large", lang, mib=MAX_IMAGE_BYTES // (1024 * 1024))
    if code in ("not_image", "not_text"):
        return t(f"file.note.{code}", lang)
    if content.note is not None and lang == "en":
        return content.note
    return t("file.note.binary", lang)


def render_file(
    content: FileContent,
    *,
    actions: bool = False,
    img_src: str | None = None,
    lang: Language = "en",
) -> str:
    """The view page of one file (D128): escaped text, a checked image (loaded only
    from ``img_src``) or a download line. Interface text in ``lang`` (D146)."""
    pid, path = content.project_id, content.path
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    down = text(files_href(pid, FilesRoute.DOWNLOAD, path))
    download = text(t("files.download", lang))
    parts = [
        f'<p><a href="{text(_project_href(pid))}">← {text(pid)}</a> · '
        f"{_crumbs(pid, parent, lang)}</p>",
        f'<h1 dir="auto">{text(path.rsplit("/", 1)[-1])}</h1>',
        f'<p>{text(_size_text(content.size, lang))} · <a href="{down}">{download}</a></p>',
    ]
    if is_previewable(path):
        show = text(preview_href(pid, path))
        parts.append(
            f'<p><a class="button" href="{show}" target="_blank" rel="noopener noreferrer">'
            f"{text(t('file.open_page', lang))}</a></p>"
        )
    use_img = None
    if content.kind == "text" and content.text is not None:
        if content.truncated:
            shown = _size_text(len(content.text.encode()), lang)
            parts.append(f'<p class="warn">{text(t("file.truncated", lang, size=shown))}</p>')
        parts.append(f'<pre dir="auto">{text(content.text)}</pre>')
    elif content.kind == "image" and img_src is not None:
        src = text(files_href(pid, FilesRoute.RAW, path))
        dims = localize_digits(f"{content.width}×{content.height}", lang)
        parts.append(
            f"<p>{text(content.image_format or '')} {text(dims)}</p>"
            f'<p><img src="{src}" alt="{text(path)}" width="{content.width}" '
            f'height="{content.height}" style="max-width:100%;height:auto"></p>'
        )
        use_img = img_src
    else:
        note = text(t("file.not_shown", lang, note=_file_note(content, lang)))
        parts.append(f'<p class="warn">{note}</p>')
    return page(f"Automa — {path}", "".join(parts), actions=actions, img_src=use_img, lang=lang)

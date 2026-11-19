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
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote

from automa.plan.models import MAX_ANSWER_CHARS
from automa.project.store import MAX_GOAL_CHARS
from automa.tools.approvals import MAX_NOTE_CHARS
from automa.webui.files import FileContent, FileListing, FilesRoute, files_href
from automa.webui.ideas import ASK_VALUE
from automa.webui.money import money, money_range
from automa.webui.preview import is_previewable, preview_href
from automa.webui.progress import REFRESH_SECONDS, CostProgress, ProgressView, UnusualSpending
from automa.webui.views import ApprovalView, Overview, ProjectView, TaskRow

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
:root{color-scheme:light dark;--bg:#f5f6f8;--card:#fff;--fg:#1d1d1f;--muted:#5f6368;
--line:#e1e3e8;--accent:#2457c5;--code:#f0f1f4;--warn:#9a3d00;--err:#b00020;
--chip:#eceef2;--ok:#d6f5dd;--bad:#fbd9d9;--busy:#fff0c2}
@media (prefers-color-scheme: dark){:root{--bg:#121214;--card:#1c1c20;--fg:#e8e8ea;
--muted:#a3a7ae;--line:#2e2f35;--accent:#8ab4ff;--code:#26262b;--warn:#ffb070;
--err:#ff7b8a;--chip:#2c2d33;--ok:#1f4a2a;--bad:#5a2323;--busy:#5a4a12}}
*{box-sizing:border-box}
body{font-family:"""
    + FONT_STACK
    + """;font-size:16px;line-height:1.7;margin:0 auto;padding:16px;max-width:1100px;
color:var(--fg);background:var(--bg)}
h1{font-size:1.5em;margin:.4em 0}h2{font-size:1.2em;margin:1.2em 0 .5em}h3{font-size:1.05em}
a{color:var(--accent)}code,pre{font-family:"""
    + MONO_STACK
    + """}
[dir=auto],pre{unicode-bidi:plaintext}
pre{white-space:pre-wrap;word-break:break-word;background:var(--code);padding:8px 10px;
border-radius:8px;text-align:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:12px 16px;margin:12px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.cards>.card{margin:0}.card h2,.card h3{margin-top:0}
.goal{margin:.3em 0;overflow-wrap:anywhere;white-space:pre-line}
.meta{color:var(--muted);font-size:.9em;margin:.2em 0}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid var(--line);
padding:6px;text-align:start;vertical-align:top}
.scroll{overflow-x:auto}
.warn{color:var(--warn)}.err{color:var(--err)}
.badge{display:inline-block;padding:0 8px;border-radius:10px;background:var(--chip);
font-size:.9em}
.st-passed{background:var(--ok)}.st-failed,.st-blocked{background:var(--bad)}
.st-running,.st-verifying{background:var(--busy)}
.task{border-inline-start:4px solid var(--line)}.t-passed{border-inline-start-color:#2e9e4f}
.t-failed,.t-blocked{border-inline-start-color:#c62828}
.t-running,.t-verifying{border-inline-start-color:#c9a000}
.chips{display:flex;flex-wrap:wrap;gap:4px 12px;color:var(--muted);font-size:.9em;margin:.3em 0}
meter{width:12em;vertical-align:middle}
textarea,input,button{font:inherit;color:inherit}
textarea{width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:8px}
input[type=text],input:not([type]){background:var(--card);border:1px solid var(--line);
border-radius:6px;padding:2px 6px}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:4px 14px;
cursor:pointer}
@media (prefers-color-scheme: dark){button{color:#121214}}
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
) -> str:
    """A complete HTML document around already-safe ``body``.

    In action mode the CSP allows forms to post to this server only, and the referrer
    policy is ``same-origin`` so the browser sends a real ``Origin`` with each post
    (with ``no-referrer`` it would send ``Origin: null``, which the server refuses).
    ``refresh`` (seconds) makes the browser reload the same page, with no script (D127).
    """
    csp = page_csp(actions=actions, img_src=img_src)
    referrer = "same-origin" if actions else "no-referrer"
    reload = ""
    if refresh is not None:
        if isinstance(refresh, bool) or not isinstance(refresh, int) or not 1 <= refresh <= 3600:
            raise ValueError("refresh must be 1..3600 seconds")
        reload = f'<meta http-equiv="refresh" content="{refresh}">'
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f'<meta name="referrer" content="{referrer}">{reload}'
        f"<title>{text(title)}</title><style>{_STYLE}</style></head>"
        f"<body>{body}</body></html>\n"
    )


def _errors(errors: tuple[str, ...]) -> str:
    if not errors:
        return ""
    items = "".join(f'<li dir="auto">{text(e)}</li>' for e in errors)
    return f'<section class="card err"><h2>Unreadable data</h2><ul>{items}</ul></section>'


def render_overview(
    view: Overview, *, actions: bool = False, ideas: bool = False, rate: float | None = None
) -> str:
    """The project list. ``ideas`` (action mode with a planner, D124) adds the
    "New idea" link; ``rate`` (toman per USD) shows costs first in toman (D133)."""
    mode = (
        "Actions enabled: approve or reject requests on each project page."
        if actions
        else "Read-only view."
    )
    parts = ["<h1>Automa projects</h1>", f"<p>{mode}</p>"]
    if actions and ideas:
        parts.append('<p><a href="/new"><strong>+ New idea</strong></a></p>')
    if not view.projects:
        parts.append("<p>No projects yet.</p>")
    else:
        cards = []
        for p in view.projects:
            waiting = (
                f'<span class="badge warn">{p.pending_approvals} pending</span>'
                if p.pending_approvals
                else "approvals: 0"
            )
            cards.append(
                '<article class="card project">'
                f'<h2><a href="{text(_project_href(p.id))}">{text(p.id)}</a> '
                f'<span class="badge">{text(p.status)}</span></h2>'
                f'<p class="goal" dir="auto">{text(p.goal_preview)}</p>'
                f'<p class="chips"><span>cost: {text(money(p.cost_usd, rate))}</span>'
                f"<span>{waiting}</span>"
                f"<span>created {_time(p.created_at)}</span></p></article>"
            )
        parts.append('<div class="cards">' + "".join(cards) + "</div>")
    parts.append(_errors(view.errors))
    return page("Automa projects", "".join(parts), actions=actions)


def _hidden(name: str, value: str) -> str:
    return f'<input type="hidden" name="{name}" value="{text(value)}">'


def _decision_forms(project_id: str, a: ApprovalView, forms: ActionForms) -> str:
    def action(kind: str) -> str:
        path = f"/projects/{quote(project_id, safe='')}/approvals/{quote(a.id, safe='')}/{kind}"
        return text(path)

    common = _hidden("csrf", forms.csrf_token) + _hidden("fingerprint", a.fingerprint)
    note = f'<label>note <input name="note" maxlength="{MAX_NOTE_CHARS}" dir="auto"></label> '
    return (
        f'<form method="post" action="{action("approve")}">{common}'
        '<p><label><input type="checkbox" name="confirm" value="yes" required> '
        f"I reviewed the full arguments above: run {text(a.tool)} once with exactly "
        "these arguments</label></p>"
        f'<p>{note}<button type="submit">Approve</button></p></form>'
        f'<form method="post" action="{action("reject")}">{common}'
        f'<p>{note}<button type="submit">Reject</button></p></form>'
    )


def _approval(project_id: str, a: ApprovalView, forms: ActionForms | None) -> str:
    args = json.dumps(a.arguments, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    lines = "".join(f'<li dir="auto">{text(line)}</li>' for line in a.request_lines)
    warns = "".join(f'<li class="warn" dir="auto">{text(w)}</li>' for w in a.warnings)
    target = f"<ul>{lines}{warns}</ul>" if lines or warns else ""
    if forms is None:
        decide = "<p>This page is read-only: decide in the terminal.</p>"
    else:
        decide = _decision_forms(project_id, a, forms)
    return (
        f'<article class="card"><h3>{text(a.id)} — {text(a.tool)} '
        f'<span class="badge">{text(a.risk)}</span></h3>'
        f"<p>task: {text(a.task_id or '-')} · requested {_time(a.created_at)}</p>"
        f'<p>reason: <span dir="auto">{text(a.reason)}</span></p>{target}'
        f'<pre dir="ltr">{text(args)}</pre>{decide}'
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
) -> str:
    """One project page. ``forms`` (action mode only) adds approve/reject forms;
    ``notice`` is a one-line result of the last decision, shown at the top;
    ``cost_range`` is the estimate line of a planned project and ``questions_waiting``
    links to the answer form of a new idea (D124); ``run`` (action mode with
    background runs, D125) adds the start button and the last background run;
    ``progress`` (D127) adds tasks passed, cost so far against the range, the latest
    events and, while running, a self-refresh of the page; ``rate`` (toman per USD)
    shows every amount first in rounded toman with dollars in parentheses (D133)."""
    parts = ['<p><a href="/">← all projects</a></p>']
    if notice:
        parts.append(f'<p class="badge" dir="auto">{text(notice)}</p>')
    parts += [
        '<section class="card">',
        f"<h1>{text(view.id)}</h1>",
        '<p class="meta"><strong>Goal:</strong></p>',
        f'<p class="goal" dir="auto">{text(view.goal)}</p>',
        f'<p class="meta">status: <span class="badge">{text(view.status)}</span> · '
        f"created {_time(view.created_at)} · updated {_time(view.updated_at)}</p>",
        _spent_line(view, rate),
        f'<p><a href="{text(_project_href(view.id) + "/files")}">Files</a></p>',
    ]
    if questions_waiting and forms is not None:
        href = text(_project_href(view.id) + "/clarify")
        parts.append(
            f'<p class="warn">The Executive has questions: <a href="{href}">answer them</a> '
            "to make the plan.</p>"
        )
    if cost_range is not None:
        parts.append(f"<p><strong>Cost range:</strong> {text(cost_range)}</p>")
    parts.append("</section>")
    if progress is not None:
        parts.append(_progress(progress, rate))
    if run is not None and forms is not None:
        parts.append(_run_panel(view.id, run, forms))
    parts.append(f"<h2>Pending approvals ({len(view.pending)})</h2>")
    if view.pending:
        parts.append("".join(_approval(view.id, a, forms) for a in view.pending))
    else:
        parts.append("<p>None.</p>")
    if view.plan_version is None:
        parts.append("<h2>Plan</h2><p>No plan.</p>")
    else:
        cards = "".join(_task_card(t) for t in view.tasks)
        parts.append(
            f"<h2>Plan v{view.plan_version} ({len(view.tasks)} tasks)</h2>"
            f'<div class="cards">{cards}</div>'
        )
    errors = view.errors + (progress.errors if progress is not None else ())
    parts.append(_errors(errors))
    refresh = REFRESH_SECONDS if progress is not None and progress.live else None
    return page(f"Automa — {view.id}", "".join(parts), actions=forms is not None, refresh=refresh)


_STATE_CLASSES = frozenset({"pending", "running", "verifying", "passed", "failed", "blocked"})


def _state_class(state: str | None) -> str:
    """A fixed CSS class for a task state (never recorded text in an attribute)."""
    return state if state in _STATE_CLASSES else "none"


def _task_card(t: TaskRow) -> str:
    """One task of the plan as a card (D129): the title in its own direction, the
    state as a fixed class and the rest as labelled chips."""
    state = _state_class(t.state)
    return (
        f'<article class="card task t-{state}">'
        f'<h3><span class="badge">#{t.order}</span> <code>{text(t.id)}</code> '
        f'<span class="badge st-{state}">{text(t.state or "not started")}</span></h3>'
        f'<p class="goal" dir="auto">{text(t.title)}</p>'
        '<p class="chips">'
        f"<span>Role: {text(t.role)}</span>"
        f"<span>After: {text(', '.join(t.depends_on) or '-')}</span>"
        f"<span>Attempts: {t.attempts}</span>"
        f'<span>Last result: <span dir="auto">{text(t.last_result or "-")}</span></span>'
        f"<span>Model: {text(t.model or '-')}</span></p></article>"
    )


def _spent_line(view: ProjectView, rate: float | None) -> str:
    """What the project has cost so far, with planning and the tasks apart (D133)."""
    total = f"<p><strong>Spent so far:</strong> {text(money(view.cost_usd, rate))}"
    if view.cost_usd <= 0:
        return total + "</p>"
    planning = min(max(view.planning_usd, 0.0), view.cost_usd)
    tasks = max(view.cost_usd - planning, 0.0)
    return (
        f'{total}</p><p class="chips"><span>Planning (questions, plan, reviews): '
        f"{text(money(planning, rate))}</span>"
        f"<span>Doing the tasks: {text(money(tasks, rate))}</span></p>"
    )


def _progress(p: ProgressView, rate: float | None = None) -> str:
    """Tasks passed, the latest run's cost against its range and the latest events (D127)."""
    parts = [
        '<section class="card"><h2>Progress</h2>',
        f"<p>tasks passed: <strong>{p.passed} / {p.total}</strong>",
    ]
    if p.total:
        parts.append(f' <meter min="0" max="{p.total}" value="{p.passed}"></meter>')
    parts.append("</p>")
    if p.cost is not None:
        c = p.cost
        so_far = "so far" if p.live else "in total"
        verdict = c.verdict if not p.live or c.verdict == "above" else "running"
        cls = ' class="warn"' if c.verdict == "above" else ""
        other = f" (+ {text(money(c.other_usd, rate))} other calls)" if c.other_usd else ""
        parts.append(
            f"<p{cls}>latest run (estimated {_time(c.estimated_at)}): "
            f"<strong>{text(money(c.spent_usd, rate))}</strong> {so_far}{other} vs range "
            f"{text(money_range(c.low_usd, c.high_usd, rate))}: {text(verdict)}"
            f" {_range_meter(c)}</p>"
        )
    else:
        parts.append("<p>No cost range recorded yet (it is written when a run starts).</p>")
    if p.unusual is not None:
        parts.append(_unusual_line(p.unusual, rate))
    if p.live:
        parts.append(f'<p class="badge">running: this page refreshes every {REFRESH_SECONDS} s</p>')
    if p.events:
        rows = "".join(
            f"<tr><td>{_time(e.ts)}</td><td>{text(e.type)}</td>"
            f'<td dir="auto">{text(e.summary or "-")}</td></tr>'
            for e in p.events
        )
        parts.append(
            f"<h3>Latest events ({len(p.events)}, newest first)</h3>"
            '<div class="scroll"><table><thead><tr><th>Time</th><th>Event</th>'
            f"<th>Details</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    parts.append("</section>")
    return "".join(parts)


def _unusual_line(u: UnusualSpending, rate: float | None = None) -> str:
    """At most one line about unusual spending (D132): how many and the largest."""
    # A flagged sample is a model call or a judged attempt: both are "steps" here.
    what = "1 step cost" if u.count == 1 else f"{u.count} steps cost"
    times = f", {u.ratio:.1f}× the usual cost" if u.ratio is not None else ""
    return (
        f'<p class="warn">Unusual spending: {what} much more than usual this run '
        f"(largest {text(money(u.largest_usd, rate))}{times}). "
        "Advisory only: nothing was stopped.</p>"
    )


def _range_meter(c: CostProgress) -> str:
    """A meter of the cost so far over the estimated range (D131): its end is the top
    of the range (or the cost, when above), never the theoretical worst case."""
    top = max(c.high_usd, c.spent_usd)
    return (
        f'<meter min="0" low="{c.low_usd:.6f}" high="{c.high_usd:.6f}" '
        f'max="{top:.6f}" value="{c.spent_usd:.6f}"></meter>'
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


def _run_panel(project_id: str, run: RunPanel, forms: ActionForms) -> str:
    parts = ['<section class="card"><h2>Run</h2>']
    if run.state is not None:
        when = f" (started {_time(run.started_at)})" if run.started_at is not None else ""
        cls = "badge warn" if run.running else "badge"
        parts.append(f'<p>Background run{when}: <span class="{cls}">{text(run.state)}</span></p>')
        if run.log_name or run.tail:
            parts.append(_technical_log(run))
        if run.running:
            parts.append("<p>The page refreshes itself while the run is going.</p>")
        if run.can_stop:
            stop = text(_project_href(project_id) + "/stop")
            parts.append(
                f'<form method="post" action="{stop}">{_hidden("csrf", forms.csrf_token)}'
                '<p><button type="submit">Stop</button> pauses the run after its current '
                "step (no model call is cut off); the start button resumes it.</p></form>"
            )
    if run.can_start:
        action = text(_project_href(project_id) + "/start")
        confirm = (
            '<p><label><input type="checkbox" name="confirm" value="yes" required> '
            "I saw the cost range above and accept it (it is above confirm_above_usd)"
            "</label></p>"
            if run.needs_confirm
            else ""
        )
        parts.append(
            f'<form method="post" action="{action}">{_hidden("csrf", forms.csrf_token)}'
            f'{confirm}<p><button type="submit">Start run</button> runs the plan in the '
            "background with the same budget guard and caps; every model call is "
            "counted.</p></form>"
        )
    elif run.state is None:
        parts.append("<p>Not startable now (no plan, already running or done).</p>")
    parts.append("</section>")
    return "".join(parts)


def _technical_log(run: RunPanel) -> str:
    """The raw run log, folded away (D131): it is the terminal output of the run (with
    its command hints), useful only when something went wrong."""
    parts = ['<details class="tech"><summary>Technical details (run log)</summary>']
    if run.log_name:
        parts.append(f"<p>log file: <code>web-runs/{text(run.log_name)}</code></p>")
    if run.tail:
        parts.append('<pre dir="auto">' + text("\n".join(run.tail)) + "</pre>")
    parts.append("</details>")
    return "".join(parts)


def render_new_idea(forms: ActionForms) -> str:
    """The "New idea" form (D124): the goal and whether to ask questions first."""
    body = (
        '<p><a href="/">← all projects</a></p><h1>New idea</h1>'
        "<p>The Executive may ask a few questions, then makes a plan with a cost range. "
        "Nothing is run yet. Each step is a model call through the budget guard and may "
        "take a minute.</p>"
        f'<form class="card" method="post" action="/new">{_hidden("csrf", forms.csrf_token)}'
        f'<p><textarea name="goal" rows="8" cols="80" maxlength="{MAX_GOAL_CHARS}" '
        'dir="auto" required placeholder="What should the project achieve?"></textarea></p>'
        f'<p><label><input type="checkbox" name="ask" value="{ASK_VALUE}" checked> '
        "Ask clarifying questions first</label></p>"
        '<p><button type="submit">Create plan</button></p></form>'
    )
    return page("Automa — new idea", body, actions=True)


def render_clarify(project_id: str, goal: str, questions: list[str], forms: ActionForms) -> str:
    """One answer field per stored question (questions are model output: escaped)."""
    fields = "".join(
        f'<p dir="auto"><label>{i + 1}. {text(q)}<br>'
        f'<textarea name="a{i}" rows="3" cols="80" maxlength="{MAX_ANSWER_CHARS}" '
        'dir="auto"></textarea></label></p>'
        for i, q in enumerate(questions)
    )
    action = text(_project_href(project_id) + "/clarify")
    body = (
        f'<p><a href="{text(_project_href(project_id))}">← {text(project_id)}</a></p>'
        f"<h1>Questions for {text(project_id)}</h1>"
        '<p class="meta"><strong>Goal:</strong></p>'
        f'<p class="goal" dir="auto">{text(goal)}</p>'
        "<p>Leave an answer empty to skip it. The plan is made after you send.</p>"
        f'<form class="card" method="post" action="{action}">{_hidden("csrf", forms.csrf_token)}'
        f'{fields}<p><button type="submit">Send answers and make the plan</button></p></form>'
    )
    return page(f"Automa — {project_id} questions", body, actions=True)


def _size_text(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _crumbs(project_id: str, path: str) -> str:
    """``files / a / b`` with a link per directory."""
    links = [f'<a href="{text(files_href(project_id, FilesRoute.LIST))}">files</a>']
    parts = path.split("/") if path else []
    for i, part in enumerate(parts):
        sub = "/".join(parts[: i + 1])
        href = text(files_href(project_id, FilesRoute.LIST, sub))
        links.append(f'<a href="{href}" dir="auto">{text(part)}</a>')
    return " / ".join(links)


def render_files(listing: FileListing, *, actions: bool = False) -> str:
    """One directory of a project's ``files/`` (D128). Symlinks and special files
    are listed but never linked."""
    pid = listing.project_id
    parts = [
        f'<p><a href="{text(_project_href(pid))}">← {text(pid)}</a></p>',
        f"<h1>Files of {text(pid)}</h1>",
        f"<p>{_crumbs(pid, listing.dir)}</p>",
    ]
    if listing.missing:
        parts.append("<p>No files yet.</p>")
        return page(f"Automa — {pid} files", "".join(parts), actions=actions)
    if listing.parent is not None:
        up = text(files_href(pid, FilesRoute.LIST, listing.parent))
        parts.append(f'<p><a href="{up}">↑ up</a></p>')
    if not listing.entries:
        parts.append("<p>Empty directory.</p>")
    else:
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
                size, extra = _size_text(e.size), f'<a href="{down}">download</a>'
                if is_previewable(e.path):
                    show = text(preview_href(pid, e.path))
                    extra += (
                        f' · <a href="{show}" target="_blank" rel="noopener noreferrer">▶ open</a>'
                    )
            else:
                label = "symlink, not followed" if e.kind == "symlink" else "not a regular file"
                cell = f'<span dir="auto">{name}</span> <span class="warn">({label})</span>'
                size, extra = "-", ""
            rows.append(f"<tr><td>{cell}</td><td>{text(size)}</td><td>{extra}</td></tr>")
        parts.append(
            '<div class="card scroll"><table><thead><tr><th>Name</th><th>Size</th><th></th>'
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )
        if listing.total > len(listing.entries):
            parts.append(f"<p>Showing {len(listing.entries)} of {listing.total} entries.</p>")
    return page(f"Automa — {pid} files", "".join(parts), actions=actions)


def render_file(content: FileContent, *, actions: bool = False, img_src: str | None = None) -> str:
    """The view page of one file (D128): escaped text, a checked image (loaded only
    from ``img_src``) or a download line."""
    pid, path = content.project_id, content.path
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    down = text(files_href(pid, FilesRoute.DOWNLOAD, path))
    parts = [
        f'<p><a href="{text(_project_href(pid))}">← {text(pid)}</a> · {_crumbs(pid, parent)}</p>',
        f'<h1 dir="auto">{text(path.rsplit("/", 1)[-1])}</h1>',
        f'<p>{text(_size_text(content.size))} · <a href="{down}">download</a></p>',
    ]
    if is_previewable(path):
        show = text(preview_href(pid, path))
        parts.append(
            f'<p><a class="button" href="{show}" target="_blank" rel="noopener noreferrer">'
            "▶ open the page (runs isolated: no network, no access to Automa)</a></p>"
        )
    use_img = None
    if content.kind == "text" and content.text is not None:
        if content.truncated:
            parts.append(
                f'<p class="warn">Only the first {text(_size_text(len(content.text.encode())))} '
                "are shown; download the file for all of it.</p>"
            )
        parts.append(f'<pre dir="auto">{text(content.text)}</pre>')
    elif content.kind == "image" and img_src is not None:
        src = text(files_href(pid, FilesRoute.RAW, path))
        parts.append(
            f"<p>{text(content.image_format or '')} {content.width}×{content.height}</p>"
            f'<p><img src="{src}" alt="{text(path)}" width="{content.width}" '
            f'height="{content.height}" style="max-width:100%;height:auto"></p>'
        )
        use_img = img_src
    else:
        note = content.note or "binary file"
        parts.append(f'<p class="warn">{text(note)}: not shown here.</p>')
    return page(f"Automa — {path}", "".join(parts), actions=actions, img_src=use_img)

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
from automa.webui.ideas import ASK_VALUE
from automa.webui.views import ApprovalView, Overview, ProjectView

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

_STYLE = """
body{font-family:system-ui,sans-serif;margin:16px;max-width:1100px;color:#1d1d1f;background:#fff}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ddd;padding:6px;
text-align:start;vertical-align:top}pre{white-space:pre-wrap;word-break:break-word;
background:#f4f4f6;padding:8px}.warn{color:#9a3d00}.err{color:#b00020}
.badge{padding:1px 6px;border-radius:8px;background:#eee}
@media (prefers-color-scheme: dark){body{color:#e8e8ea;background:#151517}
th,td{border-color:#333}pre{background:#222226}.badge{background:#333}a{color:#8ab4ff}}
"""


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


def text(value: object) -> str:
    """Safe HTML text for any recorded value."""
    return html.escape(visible(str(value)), quote=True)


def _time(value: datetime) -> str:
    return text(value.isoformat(timespec="seconds"))


def _project_href(project_id: str) -> str:
    return "/projects/" + quote(project_id, safe="")


@dataclass(frozen=True)
class ActionForms:
    """What the action-mode forms need (D123): the per-session CSRF token."""

    csrf_token: str


def page(title: str, body: str, *, actions: bool = False) -> str:
    """A complete HTML document around already-safe ``body``.

    In action mode the CSP allows forms to post to this server only, and the referrer
    policy is ``same-origin`` so the browser sends a real ``Origin`` with each post
    (with ``no-referrer`` it would send ``Origin: null``, which the server refuses).
    """
    csp = ACTION_CONTENT_SECURITY_POLICY if actions else CONTENT_SECURITY_POLICY
    referrer = "same-origin" if actions else "no-referrer"
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta name="referrer" content="{referrer}">'
        f"<title>{text(title)}</title><style>{_STYLE}</style></head>"
        f"<body>{body}</body></html>\n"
    )


def _errors(errors: tuple[str, ...]) -> str:
    if not errors:
        return ""
    items = "".join(f"<li>{text(e)}</li>" for e in errors)
    return f'<section class="err"><h2>Unreadable data</h2><ul>{items}</ul></section>'


def render_overview(view: Overview, *, actions: bool = False, ideas: bool = False) -> str:
    """The project list. ``ideas`` (action mode with a planner, D124) adds the
    "New idea" link."""
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
        rows = []
        for p in view.projects:
            waiting = (
                f'<span class="badge warn">{p.pending_approvals} pending</span>'
                if p.pending_approvals
                else "0"
            )
            rows.append(
                f'<tr><td><a href="{text(_project_href(p.id))}">{text(p.id)}</a></td>'
                f'<td><span class="badge">{text(p.status)}</span></td>'
                f'<td dir="auto">{text(p.goal_preview)}</td>'
                f"<td>${p.cost_usd:.4f}</td><td>{waiting}</td>"
                f"<td>{_time(p.created_at)}</td></tr>"
            )
        parts.append(
            "<table><thead><tr><th>Project</th><th>Status</th><th>Goal</th><th>Cost</th>"
            "<th>Approvals</th><th>Created</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
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
    lines = "".join(f"<li>{text(line)}</li>" for line in a.request_lines)
    warns = "".join(f'<li class="warn">{text(w)}</li>' for w in a.warnings)
    target = f"<ul>{lines}{warns}</ul>" if lines or warns else ""
    if forms is None:
        decide = f"<p>Decide with: <code>automa approvals {text(a.id)}</code> in the terminal.</p>"
    else:
        decide = _decision_forms(project_id, a, forms)
    return (
        f"<article><h3>{text(a.id)} — {text(a.tool)} "
        f'<span class="badge">{text(a.risk)}</span></h3>'
        f"<p>task: {text(a.task_id or '-')} · requested {_time(a.created_at)}</p>"
        f'<p dir="auto">reason: {text(a.reason)}</p>{target}'
        f"<pre>{text(args)}</pre>{decide}"
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
) -> str:
    """One project page. ``forms`` (action mode only) adds approve/reject forms;
    ``notice`` is a one-line result of the last decision, shown at the top;
    ``cost_range`` is the estimate line of a planned project and ``questions_waiting``
    links to the answer form of a new idea (D124); ``run`` (action mode with
    background runs, D125) adds the start button and the last background run."""
    parts = ['<p><a href="/">← all projects</a></p>']
    if notice:
        parts.append(f'<p class="badge" dir="auto">{text(notice)}</p>')
    parts += [
        f"<h1>{text(view.id)}</h1>",
        f'<p dir="auto"><strong>Goal:</strong> {text(view.goal)}</p>',
        f'<p>status: <span class="badge">{text(view.status)}</span> · '
        f"cost: ${view.cost_usd:.4f} · created {_time(view.created_at)} · "
        f"updated {_time(view.updated_at)}</p>",
    ]
    if questions_waiting and forms is not None:
        href = text(_project_href(view.id) + "/clarify")
        parts.append(
            f'<p class="warn">The Executive has questions: <a href="{href}">answer them</a> '
            "to make the plan.</p>"
        )
    if cost_range is not None:
        parts.append(f"<p><strong>Cost range:</strong> {text(cost_range)}</p>")
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
        rows = "".join(
            f'<tr><td>{t.order}</td><td>{text(t.id)}</td><td dir="auto">{text(t.title)}</td>'
            f"<td>{text(t.role)}</td><td>{text(', '.join(t.depends_on) or '-')}</td>"
            f"<td>{t.attempts}</td><td>{text(t.last_result or '-')}</td></tr>"
            for t in view.tasks
        )
        parts.append(
            f"<h2>Plan v{view.plan_version} ({len(view.tasks)} tasks)</h2>"
            "<table><thead><tr><th>#</th><th>Task</th><th>Title</th><th>Role</th>"
            "<th>After</th><th>Attempts</th><th>Last result</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    parts.append(_errors(view.errors))
    return page(f"Automa — {view.id}", "".join(parts), actions=forms is not None)


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
    parts = ["<h2>Run</h2>"]
    if run.state is not None:
        when = f" (started {_time(run.started_at)})" if run.started_at is not None else ""
        cls = "badge warn" if run.running else "badge"
        parts.append(f'<p>Background run{when}: <span class="{cls}">{text(run.state)}</span></p>')
        if run.log_name:
            parts.append(f"<p>log: <code>web-runs/{text(run.log_name)}</code></p>")
        if run.tail:
            parts.append("<pre>" + text("\n".join(run.tail)) + "</pre>")
        if run.running:
            parts.append("<p>Reload this page to see progress.</p>")
        if run.can_stop:
            stop = text(_project_href(project_id) + "/stop")
            parts.append(
                f'<form method="post" action="{stop}">{_hidden("csrf", forms.csrf_token)}'
                '<p><button type="submit">Stop</button> pauses the run after its current '
                "step (no model call is cut off); the start button or "
                f"<code>automa run {text(project_id)}</code> resumes it.</p></form>"
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
            f'{confirm}<p><button type="submit">Start run</button> runs '
            f"<code>automa run {text(project_id)}</code> in the background with the same "
            "budget guard and caps; every model call is counted.</p></form>"
        )
    elif run.state is None:
        parts.append("<p>Not startable now (no plan, already running or done).</p>")
    return "".join(parts)


def render_new_idea(forms: ActionForms) -> str:
    """The "New idea" form (D124): the goal and whether to ask questions first."""
    body = (
        '<p><a href="/">← all projects</a></p><h1>New idea</h1>'
        "<p>The Executive may ask a few questions, then makes a plan with a cost range. "
        "Nothing is run yet. Each step is a model call through the budget guard and may "
        "take a minute.</p>"
        f'<form method="post" action="/new">{_hidden("csrf", forms.csrf_token)}'
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
        f'<p dir="auto"><strong>Goal:</strong> {text(goal)}</p>'
        "<p>Leave an answer empty to skip it. The plan is made after you send.</p>"
        f'<form method="post" action="{action}">{_hidden("csrf", forms.csrf_token)}'
        f'{fields}<p><button type="submit">Send answers and make the plan</button></p></form>'
    )
    return page(f"Automa — {project_id} questions", body, actions=True)

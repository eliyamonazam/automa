"""Pure HTML rendering of the web UI views (D77).

Every piece of recorded text (goals, titles, reasons, arguments — some of it written by
a model) goes through :func:`text`: control, format and bidi characters are shown as
visible escapes (so they cannot reorder or hide what is being reviewed) and the result
is HTML-escaped. Pages have no scripts, no forms, no external resources, and a strict
Content-Security-Policy. Nothing here reads files or the network.
"""

from __future__ import annotations

import html
import json
import unicodedata
from datetime import datetime
from urllib.parse import quote

from automa.webui.views import ApprovalView, Overview, ProjectView

# Joiners needed by Persian text; every other Cf character (incl. bidi controls) is escaped.
_ALLOWED_FORMAT_CHARS = frozenset({"‌", "‍"})

CONTENT_SECURITY_POLICY = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; "
    "form-action 'none'; base-uri 'none'"
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


def page(title: str, body: str) -> str:
    """A complete HTML document around already-safe ``body``."""
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{CONTENT_SECURITY_POLICY}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="referrer" content="no-referrer">'
        f"<title>{text(title)}</title><style>{_STYLE}</style></head>"
        f"<body>{body}</body></html>\n"
    )


def _errors(errors: tuple[str, ...]) -> str:
    if not errors:
        return ""
    items = "".join(f"<li>{text(e)}</li>" for e in errors)
    return f'<section class="err"><h2>Unreadable data</h2><ul>{items}</ul></section>'


def render_overview(view: Overview) -> str:
    parts = ["<h1>Automa projects</h1>", "<p>Read-only view.</p>"]
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
    return page("Automa projects", "".join(parts))


def _approval(a: ApprovalView) -> str:
    args = json.dumps(a.arguments, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    lines = "".join(f"<li>{text(line)}</li>" for line in a.request_lines)
    warns = "".join(f'<li class="warn">{text(w)}</li>' for w in a.warnings)
    target = f"<ul>{lines}{warns}</ul>" if lines or warns else ""
    return (
        f"<article><h3>{text(a.id)} — {text(a.tool)} "
        f'<span class="badge">{text(a.risk)}</span></h3>'
        f"<p>task: {text(a.task_id or '-')} · requested {_time(a.created_at)}</p>"
        f'<p dir="auto">reason: {text(a.reason)}</p>{target}'
        f"<pre>{text(args)}</pre>"
        f"<p>Decide with: <code>automa approvals {text(a.id)}</code> in the terminal.</p>"
        "</article>"
    )


def render_project(view: ProjectView) -> str:
    parts = [
        '<p><a href="/">← all projects</a></p>',
        f"<h1>{text(view.id)}</h1>",
        f'<p dir="auto"><strong>Goal:</strong> {text(view.goal)}</p>',
        f'<p>status: <span class="badge">{text(view.status)}</span> · '
        f"cost: ${view.cost_usd:.4f} · created {_time(view.created_at)} · "
        f"updated {_time(view.updated_at)}</p>",
    ]
    parts.append(f"<h2>Pending approvals ({len(view.pending)})</h2>")
    if view.pending:
        parts.append("".join(_approval(a) for a in view.pending))
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
    return page(f"Automa — {view.id}", "".join(parts))
